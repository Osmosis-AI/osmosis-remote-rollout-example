"""Mock trainer server for end-to-end testing.

This mock trainer implements:
- POST /v1/chat/completions: the LLM generation callback used by RolloutServer.
- POST /v1/rollout/completed: rollout completion callback endpoint.

It returns deterministic token IDs for testing without requiring a real model.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
import uvicorn

from rollout_server.schemas import CompletionUsage, CompletionsRequest, CompletionsResponse, RolloutResponse

app = FastAPI(title="Mock Trainer Server")

# In-memory storage of completed rollouts for debugging.
_COMPLETED: Dict[str, Dict[str, Any]] = {}


def _fake_token_ids(text: str) -> List[int]:
    return [i for i in range(len(text))]


def _fake_prompt_token_ids(messages: List[Dict[str, Any]]) -> List[int]:
    return [i for i in range(10 * max(1, len(messages)))]


def _should_use_tools(last_message: Dict[str, Any]) -> bool:
    if last_message.get("role") != "user":
        return False
    content = last_message.get("content")
    if not isinstance(content, str):
        return False
    keywords = ["calculate", "add", "sum", "plus"]
    return any(k in content.lower() for k in keywords)


@app.post("/v1/chat/completions")
async def completions(request: CompletionsRequest) -> CompletionsResponse:
    messages = list(request.messages)
    last_message = messages[-1] if messages else {"role": "user", "content": ""}

    assistant_message: Dict[str, Any]

    if _should_use_tools(last_message):
        assistant_message = {
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
    elif last_message.get("role") == "tool":
        assistant_message = {
            "role": "assistant",
            "content": "The calculation is complete.",
        }
    else:
        assistant_message = {
            "role": "assistant",
            "content": "OK.",
        }

    response_text = assistant_message.get("content") or ""
    response_token_ids = _fake_token_ids(str(response_text))
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
        usage=CompletionUsage(
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(response_token_ids),
            total_tokens=len(prompt_token_ids) + len(response_token_ids),
        ),
        token_ids=response_token_ids,
        logprobs=[0.0] * len(response_token_ids),
        prompt_token_ids=prompt_token_ids,
    )


@app.post("/v1/rollout/completed")
async def rollout_completed(response: RolloutResponse) -> Dict[str, Any]:
    payload = response.model_dump(mode="json", exclude_none=True)
    _COMPLETED[response.rollout_id] = payload
    return {"status": "ok"}


@app.get("/v1/rollout/completed/{rollout_id}")
async def get_completed_rollout(rollout_id: str) -> Dict[str, Any]:
    return _COMPLETED.get(rollout_id, {})


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy", "service": "mock-trainer"}


if __name__ == "__main__":
    import os

    port = int(os.getenv("MOCK_TRAINER_PORT", "9001"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
