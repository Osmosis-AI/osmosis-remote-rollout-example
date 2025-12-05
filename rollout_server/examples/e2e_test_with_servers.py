"""End-to-end test with real servers.

This module contains E2E tests that verify the full system with real running servers.
These are NOT regular unit/integration tests - they require external services to be running.

Purpose:
    - Manual testing and validation with real servers
    - E2E verification of the complete rollout flow
    - Debugging and troubleshooting with actual HTTP communication

Usage:
    1. Start mock trainer: python -m rollout_server.tests.mocks.mock_trainer
    2. Start rollout server: python -m rollout_server.server
    3. Run these tests: pytest examples/e2e_test_with_servers.py -v

Note:
    - These tests are TOO HEAVY for regular pytest runs
    - They require external servers on ports 9000 and 9001
    - Tests will be skipped if servers are not available
    - For lightweight tests, see tests/integration/test_rollout_api.py
"""

import uuid
import pytest
import httpx


# Pytest fixtures

@pytest.fixture
def mock_trainer_url():
    """Mock trainer base URL."""
    return "http://localhost:9001"


@pytest.fixture
def rollout_server_url():
    """Rollout server base URL."""
    return "http://localhost:9000"


@pytest.fixture
async def async_client():
    """Provide async HTTP client."""
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
def example_payload(mock_trainer_url):
    """Create example rollout request payload with unique rollout_id."""
    return {
        "rollout_id": str(uuid.uuid4()),
        "server_url": mock_trainer_url,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful calculator assistant with access to calculator tools."
            },
            {
                "role": "user",
                "content": "Please calculate 5 plus 3."
            }
        ],
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "logprobs": True
        },
        "tokenizer_name": "Qwen/Qwen3-8B",
        "tokenizer_revision": "main",
        "max_turns": 10,
        "max_tokens_total": 8192
    }


# Helper function to check server availability

async def check_server_health(client: httpx.AsyncClient, url: str) -> bool:
    """Check if a server is available by calling its health endpoint."""
    try:
        response = await client.get(f"{url}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


# Integration tests

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_servers
async def test_mock_trainer_health(async_client, mock_trainer_url):
    """Test that mock trainer is available and healthy."""
    is_healthy = await check_server_health(async_client, mock_trainer_url)

    if not is_healthy:
        pytest.skip(
            f"Mock trainer not available at {mock_trainer_url}. "
            "Start it with: python -m rollout_server.tests.mocks.mock_trainer"
        )

    # If we reach here, server is healthy
    response = await async_client.get(f"{mock_trainer_url}/health", timeout=5.0)
    assert response.status_code == 200, f"Mock trainer health check failed with status {response.status_code}"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_servers
async def test_rollout_server_health(async_client, rollout_server_url):
    """Test that rollout server is available and healthy."""
    is_healthy = await check_server_health(async_client, rollout_server_url)

    if not is_healthy:
        pytest.skip(
            f"Rollout server not available at {rollout_server_url}. "
            "Start it with: cd rollout_server && python -m rollout_server.server"
        )

    # If we reach here, server is healthy
    response = await async_client.get(f"{rollout_server_url}/health", timeout=5.0)
    assert response.status_code == 200, f"Rollout server health check failed with status {response.status_code}"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.requires_servers
async def test_rollout_endpoint(async_client, rollout_server_url, mock_trainer_url, example_payload):
    """Test the /rollout endpoint with mock trainer (main integration test)."""

    # Check both servers are available
    mock_trainer_healthy = await check_server_health(async_client, mock_trainer_url)
    rollout_server_healthy = await check_server_health(async_client, rollout_server_url)

    if not mock_trainer_healthy:
        pytest.skip(
            f"Mock trainer not available at {mock_trainer_url}. "
            "Start it with: python -m rollout_server.tests.mocks.mock_trainer"
        )

    if not rollout_server_healthy:
        pytest.skip(
            f"Rollout server not available at {rollout_server_url}. "
            "Start it with: cd rollout_server && python -m rollout_server.server"
        )

    # Send rollout request
    response = await async_client.post(
        f"{rollout_server_url}/rollout",
        json=example_payload,
        timeout=30.0
    )

    # Assert response status
    assert response.status_code == 200, (
        f"Rollout request failed with status {response.status_code}: {response.text}"
    )

    # Parse and validate response
    result = response.json()

    # Assert basic structure
    assert "status" in result, "Response missing 'status' field"
    assert "finish_reason" in result, "Response missing 'finish_reason' field"
    assert "final_messages" in result, "Response missing 'final_messages' field"

    # Assert status is COMPLETED
    assert result["status"] == "COMPLETED", (
        f"Expected status 'COMPLETED', got '{result['status']}'"
    )

    # Assert we have messages
    assert len(result["final_messages"]) > 0, "Expected non-empty final_messages"

    # Validate message structure
    for msg in result["final_messages"]:
        assert "role" in msg, "Message missing 'role' field"
        assert msg["role"] in ["system", "user", "assistant", "tool"], (
            f"Invalid message role: {msg['role']}"
        )

    # Assert we have system and user messages at minimum
    roles = [msg["role"] for msg in result["final_messages"]]
    assert "system" in roles, "Missing system message in conversation"
    assert "user" in roles, "Missing user message in conversation"
    assert "assistant" in roles, "Missing assistant response in conversation"
