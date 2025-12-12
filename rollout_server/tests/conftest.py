"""Shared pytest fixtures for rollout_server tests.

Fixtures are designed to be isolated per-test to avoid cross-test interference.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rollout_server.schemas import (
    CompletionsRequest,
    CompletionsResponse,
    RolloutResponse,
)


def _should_use_tools(last_message: Dict[str, Any]) -> bool:
    if last_message.get("role") != "user":
        return False
    content = last_message.get("content")
    if not isinstance(content, str):
        return False
    keywords = ["calculate", "add", "sum", "plus"]
    return any(k in content.lower() for k in keywords)


def _fake_token_ids(text: str) -> List[int]:
    # Deterministic, lightweight token id generation for tests.
    return [i for i in range(len(text))]


def _fake_prompt_token_ids(messages: List[Dict[str, Any]]) -> List[int]:
    # Make prompt length grow with number of messages.
    return [i for i in range(10 * max(1, len(messages)))]


@pytest.fixture
def rollout_completion_tracker() -> Dict[str, Any]:
    """Capture /v1/rollout/completed callbacks in a thread-safe way."""
    return {
        "event": threading.Event(),
        "responses": [],  # List[dict]
    }


def create_mock_trainer_app(
    tracker: Optional[Dict[str, Any]] = None,
) -> FastAPI:
    """Create a mock trainer app with completions + completion-callback endpoints."""
    app = FastAPI(title="Mock Trainer")

    @app.post("/v1/chat/completions")
    async def completions(request: CompletionsRequest) -> CompletionsResponse:
        messages = list(request.messages)
        last_message = messages[-1] if messages else {"role": "user", "content": ""}

        if _should_use_tools(last_message):
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": "I'll calculate that for you.",
                "tool_calls": [
                    {
                        "id": "call_test_add",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 5, "b": 3}'},
                    }
                ],
            }
            response_text = assistant_message["content"]
        elif last_message.get("role") == "tool":
            assistant_message = {
                "role": "assistant",
                "content": "The calculation is complete.",
            }
            response_text = assistant_message["content"]
        else:
            assistant_message = {
                "role": "assistant",
                "content": "OK.",
            }
            response_text = assistant_message["content"]

        response_token_ids = _fake_token_ids(response_text)
        prompt_token_ids = _fake_prompt_token_ids(messages)

        return CompletionsResponse(
            id=request.rollout_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": "stop",
                }
            ],
            token_ids=response_token_ids,
            logprobs=[0.0] * len(response_token_ids),
            prompt_token_ids=prompt_token_ids,
        )

    @app.post("/v1/rollout/completed")
    async def rollout_completed(response: RolloutResponse) -> Dict[str, Any]:
        if tracker is not None:
            tracker["responses"].append(response.model_dump(mode="json", exclude_none=True))
            tracker["event"].set()
        return {"status": "ok"}

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "healthy"}

    return app


def patch_httpx_for_mock_trainer(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch httpx.AsyncClient.post to route trainer callbacks to the mock app."""

    original_post = httpx.AsyncClient.post

    async def mock_post(self, url: str, **kwargs):
        if "/v1/chat/completions" in url:
            resp = client.post("/v1/chat/completions", **kwargs)
            return httpx.Response(
                status_code=resp.status_code,
                json=resp.json(),
                request=httpx.Request("POST", url),
            )
        if "/v1/rollout/completed" in url:
            resp = client.post("/v1/rollout/completed", **kwargs)
            return httpx.Response(
                status_code=resp.status_code,
                json=resp.json(),
                request=httpx.Request("POST", url),
            )
        return await original_post(self, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)


@pytest.fixture
def mock_trainer_with_completion(
    rollout_completion_tracker: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> Tuple[TestClient, Dict[str, Any]]:
    """Mock trainer that records /v1/rollout/completed callbacks."""

    app = create_mock_trainer_app(tracker=rollout_completion_tracker)
    client = TestClient(app)
    patch_httpx_for_mock_trainer(client, monkeypatch)
    return client, rollout_completion_tracker
