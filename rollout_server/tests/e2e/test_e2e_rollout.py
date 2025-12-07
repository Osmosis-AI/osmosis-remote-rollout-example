#!/usr/bin/env python3
"""End-to-end tests with real running servers.

These tests require the mock trainer and rollout server to be running.
Use pytest markers to skip these tests in CI/CD pipelines.

Usage:
    1. Start the test environment:
       ./scripts/start_test_environment.sh

    2. Run E2E tests:
       uv run pytest tests/e2e/ -v -m requires_servers

    3. Stop the test environment:
       ./scripts/stop_test_environment.sh

Or run with specific markers:
    uv run pytest tests/e2e/ -v -m "requires_servers"  # Only E2E tests
    uv run pytest tests/ -v -m "not requires_servers"  # Skip E2E tests
"""

import asyncio
import uuid
from typing import Any, Dict

import httpx
import pytest

# Server configuration
MOCK_TRAINER_URL = "http://localhost:9001"
ROLLOUT_SERVER_URL = "http://localhost:9000"


async def check_server_health(url: str) -> bool:
    """Check if a server is available."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/health", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


def servers_available() -> bool:
    """Check if both required servers are running."""
    async def check_both():
        mock_ok = await check_server_health(MOCK_TRAINER_URL)
        rollout_ok = await check_server_health(ROLLOUT_SERVER_URL)
        return mock_ok and rollout_ok

    return asyncio.get_event_loop().run_until_complete(check_both())


# Skip all tests in this module if servers are not running
pytestmark = [
    pytest.mark.requires_servers,
    pytest.mark.skipif(
        not servers_available(),
        reason="Requires running mock trainer and rollout server"
    ),
]


def create_rollout_request(
    messages: list[Dict[str, Any]],
    max_turns: int = 10
) -> Dict[str, Any]:
    """Create a standard rollout request payload."""
    return {
        "rollout_id": str(uuid.uuid4()),
        "server_url": MOCK_TRAINER_URL,
        "messages": messages,
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "logprobs": True
        },
        "tokenizer_name": "Qwen/Qwen3-8B",
        "tokenizer_revision": "main",
        "max_turns": max_turns,
        "max_tokens_total": 8192
    }


@pytest.mark.asyncio
async def test_e2e_health_endpoints():
    """Test that both servers respond to health checks."""
    async with httpx.AsyncClient() as client:
        # Check mock trainer
        response = await client.get(f"{MOCK_TRAINER_URL}/health", timeout=5.0)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # Check rollout server
        response = await client.get(f"{ROLLOUT_SERVER_URL}/health", timeout=5.0)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_e2e_tools_endpoint():
    """Test that rollout server returns tool definitions."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ROLLOUT_SERVER_URL}/tools", timeout=5.0)
        assert response.status_code == 200

        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) > 0

        # Check calculator tools are present
        tool_names = [t["function"]["name"] for t in data["tools"]]
        assert "add" in tool_names
        assert "subtract" in tool_names
        assert "multiply" in tool_names
        assert "divide" in tool_names


@pytest.mark.asyncio
async def test_e2e_simple_rollout_without_tools():
    """Test a simple rollout that doesn't use tools."""
    payload = create_rollout_request([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ])

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ROLLOUT_SERVER_URL}/rollout",
            json=payload,
            timeout=60.0
        )

        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "COMPLETED"
        assert result["finish_reason"] == "stop"
        assert len(result["final_messages"]) >= 3  # system, user, assistant


@pytest.mark.asyncio
async def test_e2e_rollout_with_calculator():
    """Test a rollout that uses calculator tools."""
    payload = create_rollout_request([
        {
            "role": "system",
            "content": "You are a helpful calculator assistant with access to calculator tools."
        },
        {
            "role": "user",
            "content": "Please calculate 5 plus 3."
        }
    ])

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ROLLOUT_SERVER_URL}/rollout",
            json=payload,
            timeout=60.0
        )

        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "COMPLETED"
        final_messages = result["final_messages"]

        # Should have: system, user, assistant (with tool call), tool, assistant (response)
        assert len(final_messages) >= 4

        # Check for tool message
        tool_messages = [m for m in final_messages if m["role"] == "tool"]
        assert len(tool_messages) >= 1


@pytest.mark.asyncio
async def test_e2e_rollout_max_turns():
    """Test that max_turns limit is respected."""
    payload = create_rollout_request(
        [
            {"role": "system", "content": "You are a calculator assistant."},
            {"role": "user", "content": "Calculate 1 plus 1."}
        ],
        max_turns=2
    )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ROLLOUT_SERVER_URL}/rollout",
            json=payload,
            timeout=60.0
        )

        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "COMPLETED"
        # Should be either "stop" (completed) or "max_turns" (hit limit)
        assert result["finish_reason"] in ["stop", "max_turns"]


@pytest.mark.asyncio
async def test_e2e_rollout_metrics():
    """Test that rollout returns metrics."""
    payload = create_rollout_request([
        {"role": "system", "content": "You are a calculator assistant."},
        {"role": "user", "content": "Add 5 and 3."}
    ])

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ROLLOUT_SERVER_URL}/rollout",
            json=payload,
            timeout=60.0
        )

        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "COMPLETED"
        assert "metrics" in result
        assert result["metrics"] is not None

        metrics = result["metrics"]
        assert metrics["num_llm_calls"] >= 1
        assert metrics["total_latency_ms"] > 0
        assert metrics["llm_latency_ms"] >= 0


if __name__ == "__main__":
    # Allow running as a script for quick testing
    import sys

    async def run_tests():
        """Run all tests manually."""
        print("Running E2E tests...")

        # Check servers first
        if not servers_available():
            print("ERROR: Servers not available. Start with:")
            print("  ./scripts/start_test_environment.sh")
            sys.exit(1)

        # Run tests
        tests = [
            test_e2e_health_endpoints,
            test_e2e_tools_endpoint,
            test_e2e_simple_rollout_without_tools,
            test_e2e_rollout_with_calculator,
            test_e2e_rollout_max_turns,
            test_e2e_rollout_metrics,
        ]

        for test in tests:
            print(f"\n{test.__name__}...")
            try:
                await test()
                print("  PASSED")
            except AssertionError as e:
                print(f"  FAILED: {e}")
            except Exception as e:
                print(f"  ERROR: {e}")

    asyncio.run(run_tests())

