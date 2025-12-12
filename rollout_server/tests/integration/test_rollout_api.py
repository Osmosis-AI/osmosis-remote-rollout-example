"""Integration tests for the async-init rollout API.

These tests run the RolloutServer FastAPI app in-process and intercept its
trainer callbacks using an in-process mock trainer.
"""

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from rollout_server.server import app as rollout_app


@pytest.mark.integration
def test_init_endpoint_returns_202_and_tools(mock_trainer_with_completion):
    _client, tracker = mock_trainer_with_completion

    rollout_id = str(uuid.uuid4())
    payload = {
        "rollout_id": rollout_id,
        "server_url": "http://mock-trainer:9001",
        "messages": [
            {"role": "user", "content": "Please calculate 5 plus 3."},
        ],
        "completion_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 128,
            "logprobs": True,
        },
        "max_turns": 10,
        "max_tokens_total": 8192,
    }

    with TestClient(rollout_app) as rollout_client:
        resp = rollout_client.post("/init", json=payload)
        assert resp.status_code == 202, resp.text

        body = resp.json()
        assert body["rollout_id"] == rollout_id
        assert isinstance(body["tools"], list)
        assert len(body["tools"]) > 0

        # Wait for completion callback.
        ok = tracker["event"].wait(timeout=5.0)
        assert ok, "Timed out waiting for /v1/rollout/completed callback"

    assert len(tracker["responses"]) == 1
    completed = tracker["responses"][0]
    assert completed["rollout_id"] == rollout_id
    assert completed["status"] == "COMPLETED"
    assert isinstance(completed.get("final_messages"), list)
    assert len(completed["final_messages"]) >= 2

    metrics = completed.get("metrics") or {}
    assert metrics.get("num_llm_calls") == 2
    assert metrics.get("num_tool_calls") == 1


@pytest.mark.integration
def test_health_endpoint():
    with TestClient(rollout_app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


@pytest.mark.integration
def test_init_with_invalid_request_returns_422():
    with TestClient(rollout_app) as client:
        resp = client.post("/init", json={"invalid": "data"})
        assert resp.status_code == 422
