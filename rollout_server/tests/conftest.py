"""Shared pytest fixtures for rollout_server tests.

This module provides common fixtures used across integration and unit tests.
Fixtures are designed to be isolated per-test to avoid race conditions when
running tests in parallel (e.g., with pytest-xdist).
"""

import time
import uuid
from typing import List, Dict, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from transformers import AutoTokenizer, PreTrainedTokenizer

from rollout_server.schemas import (
    CompletionsRequest,
    CompletionsResponse,
    CompletionsChoice,
    Message,
)


# =============================================================================
# Tokenizer Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizer:
    """Shared tokenizer for all tests (session-scoped for performance).

    Uses Qwen/Qwen3-8B to match production training environment.
    Session-scoped because tokenizer loading is expensive (~2-5s).
    """
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# =============================================================================
# Mask Tracking Fixtures (Thread-Safe)
# =============================================================================


@pytest.fixture
def mask_tracker() -> Dict[str, List]:
    """Isolated mask tracker for each test (avoids race conditions).

    Use this instead of global mutable state for tracking response_masks
    in parallel test runs.

    Returns:
        Dict with 'masks' list that can be populated during test execution.
    """
    return {"masks": []}


# =============================================================================
# Mock Trainer Fixtures
# =============================================================================


@pytest.fixture
def mock_trainer_app(tokenizer: PreTrainedTokenizer) -> FastAPI:
    """Create a mock trainer FastAPI app for testing.

    This fixture creates an isolated mock trainer instance for each test,
    avoiding shared state between tests.

    Args:
        tokenizer: Session-scoped tokenizer fixture

    Returns:
        FastAPI app instance with /v1/completions endpoint
    """
    app = FastAPI(title="Mock Trainer")

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
def mock_trainer_client(mock_trainer_app: FastAPI, monkeypatch) -> TestClient:
    """Create TestClient for mock trainer and monkey-patch httpx calls.

    This fixture sets up the mock trainer and patches httpx.AsyncClient.post
    to route /v1/completions calls to the mock trainer.

    Args:
        mock_trainer_app: Mock trainer FastAPI app fixture
        monkeypatch: pytest monkeypatch fixture

    Returns:
        TestClient instance for the mock trainer
    """
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


@pytest.fixture
def mock_trainer_with_tracking(tokenizer: PreTrainedTokenizer, mask_tracker: Dict, monkeypatch):
    """Mock trainer that tracks response_mask values (thread-safe).

    This fixture creates a dedicated mock trainer that captures response_mask values
    for verification in tests. Uses isolated mask_tracker fixture to avoid race
    conditions in parallel test runs.

    Args:
        tokenizer: Session-scoped tokenizer fixture
        mask_tracker: Isolated mask tracker dict fixture
        monkeypatch: pytest monkeypatch fixture

    Yields:
        Tuple of (TestClient, masks list) for test assertions
    """
    import httpx

    # Create a fresh app with tracking built-in
    app = FastAPI(title="Mock Trainer with Tracking")

    @app.post("/v1/completions")
    async def completions_with_tracking(request: CompletionsRequest) -> CompletionsResponse:
        """Mock completions endpoint that tracks response_mask values."""
        # Track the response_mask
        mask_tracker["masks"].append(request.response_mask)

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

    # Create test client
    client = TestClient(app)

    # Monkey-patch httpx to route to our mock
    original_post = httpx.AsyncClient.post

    async def mock_post(self, url, **kwargs):
        if "/v1/completions" in url:
            response = client.post("/v1/completions", **kwargs)
            mock_response = httpx.Response(
                status_code=response.status_code,
                json=response.json(),
                request=httpx.Request("POST", url)
            )
            return mock_response
        return await original_post(self, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    yield client, mask_tracker["masks"]

    # Cleanup is automatic via fixture scoping

