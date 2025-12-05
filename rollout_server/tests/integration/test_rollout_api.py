"""Integration tests for rollout API.

These tests verify the /rollout endpoint with in-process mock servers.
No external services required - everything runs within pytest using TestClient.

Note: This file uses shared fixtures from conftest.py:
- mock_trainer_app: Mock trainer FastAPI app
- mock_trainer_client: TestClient with httpx monkey-patching
- tokenizer: Session-scoped tokenizer for performance
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from rollout_server.server import app as rollout_app
from rollout_server.schemas import Message, RolloutRequest


# Note: mock_trainer_app and mock_trainer_client fixtures are now in conftest.py
# This removes code duplication and ensures consistent mock behavior across tests


# Integration tests

@pytest.mark.asyncio
@pytest.mark.integration
async def test_rollout_endpoint_with_mock_trainer(mock_trainer_client):
    """Test /rollout endpoint with in-process mock trainer."""
    rollout_client = TestClient(rollout_app)

    # Prepare rollout request
    rollout_request = RolloutRequest(
        rollout_id=str(uuid.uuid4()),
        server_url="http://mock-trainer:9001",  # Will be intercepted by mock
        messages=[
            Message(
                role="system",
                content="You are a helpful calculator assistant with access to calculator tools."
            ),
            Message(
                role="user",
                content="Please calculate 5 plus 3."
            )
        ],
        sampling_params={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "logprobs": True
        },
        tokenizer_name="Qwen/Qwen3-8B",
        tokenizer_revision="main",
        max_turns=10,
        max_tokens_total=8192
    )

    # Send rollout request
    response = rollout_client.post(
        "/rollout",
        json=rollout_request.model_dump()
    )

    # Verify response
    assert response.status_code == 200, f"Request failed: {response.text}"

    result = response.json()
    assert result["status"] == "COMPLETED"
    assert "final_messages" in result
    assert len(result["final_messages"]) > 0

    # Verify conversation structure
    roles = [msg["role"] for msg in result["final_messages"]]
    assert "system" in roles
    assert "user" in roles
    assert "assistant" in roles


@pytest.mark.asyncio
@pytest.mark.integration
async def test_rollout_health_endpoint():
    """Test /health endpoint."""
    client = TestClient(rollout_app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_rollout_with_invalid_request():
    """Test /rollout endpoint with invalid request."""
    client = TestClient(rollout_app)

    # Send invalid request (missing required fields)
    response = client.post("/rollout", json={"invalid": "data"})

    # Should return 422 Unprocessable Entity for validation errors
    assert response.status_code == 422
