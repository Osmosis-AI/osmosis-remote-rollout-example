"""Integration tests for async-init completion callbacks.

These tests validate that RolloutServer:
- Returns 202 Accepted on /init with tools.
- Drives the conversation via /v1/chat/completions callbacks.
- Posts exactly one /v1/rollout/completed callback per rollout_id.
"""

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from rollout_server.server import app as rollout_app


@pytest.mark.integration
def test_single_turn_without_tools_completes(mock_trainer_with_completion):
    _client, tracker = mock_trainer_with_completion

    rollout_id = str(uuid.uuid4())
    payload = {
        "rollout_id": rollout_id,
        "server_url": "http://mock-trainer:9001",
        "messages": [{"role": "user", "content": "Hello"}],
        "completion_params": {"temperature": 0.7, "max_tokens": 64, "logprobs": True},
        "max_turns": 10,
        "max_tokens_total": 8192,
    }

    with TestClient(rollout_app) as rollout_client:
        resp = rollout_client.post("/init", json=payload)
        assert resp.status_code == 202

        ok = tracker["event"].wait(timeout=5.0)
        assert ok, "Timed out waiting for completion callback"

    assert len(tracker["responses"]) == 1
    completed = tracker["responses"][0]
    assert completed["rollout_id"] == rollout_id
    assert completed["status"] == "COMPLETED"

    metrics = completed.get("metrics") or {}
    assert metrics.get("num_llm_calls") == 1
    assert metrics.get("num_tool_calls") == 0


@pytest.mark.integration
def test_init_is_idempotent_by_rollout_id(mock_trainer_with_completion):
    _client, tracker = mock_trainer_with_completion

    rollout_id = str(uuid.uuid4())
    payload = {
        "rollout_id": rollout_id,
        "server_url": "http://mock-trainer:9001",
        "messages": [{"role": "user", "content": "Please calculate 5 plus 3."}],
        "completion_params": {"temperature": 0.7, "max_tokens": 64, "logprobs": True},
        "max_turns": 10,
        "max_tokens_total": 8192,
    }

    with TestClient(rollout_app) as rollout_client:
        resp1 = rollout_client.post("/init", json=payload)
        resp2 = rollout_client.post("/init", json=payload)

        assert resp1.status_code == 202
        assert resp2.status_code == 202
        assert resp1.json()["tools"] == resp2.json()["tools"]

        ok = tracker["event"].wait(timeout=5.0)
        assert ok, "Timed out waiting for completion callback"

    # Give the server a brief window; it should not post a second completion.
    time.sleep(0.5)
    assert len(tracker["responses"]) == 1

