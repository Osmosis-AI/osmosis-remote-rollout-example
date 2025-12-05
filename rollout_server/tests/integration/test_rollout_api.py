"""Integration tests for rollout API.

These tests verify the /rollout endpoint with in-process mock servers.
No external services required - everything runs within pytest using TestClient.
"""

import uuid
import time
from typing import List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from rollout_server.server import app as rollout_app
from rollout_server.schemas import (
    CompletionsRequest,
    CompletionsResponse,
    CompletionsChoice,
    Message,
    RolloutRequest,
)


# Mock trainer server fixture

@pytest.fixture
def mock_trainer_app():
    """Create a mock trainer FastAPI app for testing."""
    app = FastAPI(title="Mock Trainer")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    @app.post("/v1/completions")
    async def completions(request: CompletionsRequest) -> CompletionsResponse:
        """Mock completions endpoint that simulates LLM responses."""
        # Determine if we should use tools
        last_message = request.messages[-1]
        use_tools = False

        if last_message.role == "user":
            user_content = last_message.content
            if isinstance(user_content, str) and any(
                word in user_content.lower()
                for word in ["calculate", "add", "sum", "plus"]
            ):
                use_tools = True

        # Generate mock response
        if use_tools:
            response_text = "I'll calculate that for you."
            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": '{"a": 5, "b": 3}'
                        }
                    }
                ]
            }
        else:
            response_text = "The calculation is complete."
            assistant_message = {
                "role": "assistant",
                "content": response_text
            }

        # Tokenize response
        response_token_ids = tokenizer.encode(response_text, add_special_tokens=False)
        response_logprobs = [0.0] * len(response_token_ids)

        # Tokenize prompt
        prompt_text = tokenizer.apply_chat_template(
            [msg.model_dump() for msg in request.messages],
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

        return CompletionsResponse(
            id=f"mock-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model="mock-qwen3-8b",
            choices=[
                CompletionsChoice(
                    index=0,
                    message=Message(**assistant_message),
                    finish_reason="stop"
                )
            ],
            token_ids=response_token_ids,
            logprobs=response_logprobs,
            prompt_token_ids=prompt_token_ids
        )

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


@pytest.fixture
def mock_trainer_client(mock_trainer_app, monkeypatch):
    """Create TestClient for mock trainer and monkey-patch httpx calls."""
    client = TestClient(mock_trainer_app)

    # Monkey-patch httpx.AsyncClient to route calls to our mock trainer
    import httpx
    original_post = httpx.AsyncClient.post

    async def mock_post(self, url, **kwargs):
        # Route completions calls to our mock trainer
        if "/v1/completions" in url:
            # Use TestClient synchronously (it's designed for this)
            response = client.post("/v1/completions", **kwargs)
            # Convert to httpx.Response-like object
            mock_response = httpx.Response(
                status_code=response.status_code,
                json=response.json(),
                request=httpx.Request("POST", url)
            )
            return mock_response
        # Fall back to original for other URLs
        return await original_post(self, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    return client


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
