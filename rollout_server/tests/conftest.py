"""Shared pytest fixtures for rollout_server tests.

This module provides common fixtures used across integration and unit tests.
Fixtures are designed to be isolated per-test to avoid race conditions when
running tests in parallel (e.g., with pytest-xdist).
"""

import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
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
# Constants
# =============================================================================

# Default tokenizer for tests - should match production
DEFAULT_TEST_TOKENIZER = os.getenv("TEST_TOKENIZER", "Qwen/Qwen3-8B")

# Whether to trust remote code for tokenizers in tests
# Set to "false" in CI environments for security
TEST_TRUST_REMOTE_CODE = os.getenv("TEST_TRUST_REMOTE_CODE", "true").lower() == "true"


# =============================================================================
# Tokenizer Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizer:
    """Shared tokenizer for all tests (session-scoped for performance).

    Uses Qwen/Qwen3-8B to match production training environment.
    Session-scoped because tokenizer loading is expensive (~2-5s).

    Note: trust_remote_code is controlled by TEST_TRUST_REMOTE_CODE env var.
    """
    tok = AutoTokenizer.from_pretrained(
        DEFAULT_TEST_TOKENIZER,
        trust_remote_code=TEST_TRUST_REMOTE_CODE
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# =============================================================================
# Mask Tracking Fixtures (Thread-Safe)
# =============================================================================


@pytest.fixture
def mask_tracker() -> Dict[str, List[Optional[List[int]]]]:
    """Isolated mask tracker for each test (avoids race conditions).

    Use this instead of global mutable state for tracking response_masks
    in parallel test runs.

    Returns:
        Dict with 'masks' list that can be populated during test execution.
    """
    return {"masks": []}


# =============================================================================
# Mock Trainer Factory (reduces code duplication)
# =============================================================================


def _should_use_tools(message: Message) -> bool:
    """Determine if a message should trigger tool use.

    Args:
        message: The last message in the conversation

    Returns:
        True if the message contains calculation-related keywords
    """
    if message.role != "user":
        return False

    user_content = message.content
    if not isinstance(user_content, str):
        return False

    calculation_keywords = ["calculate", "add", "sum", "plus"]
    return any(word in user_content.lower() for word in calculation_keywords)


def _generate_mock_response(
    use_tools: bool
) -> Tuple[str, Dict[str, Any]]:
    """Generate a mock LLM response.

    Args:
        use_tools: Whether to include tool calls in the response

    Returns:
        Tuple of (response_text, assistant_message_dict)
    """
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

    return response_text, assistant_message


def _create_completions_response(
    tokenizer: PreTrainedTokenizer,
    request: CompletionsRequest,
    response_text: str,
    assistant_message: Dict[str, Any]
) -> CompletionsResponse:
    """Create a CompletionsResponse from mock data.

    Args:
        tokenizer: Tokenizer for encoding
        request: The original request
        response_text: The response text
        assistant_message: The assistant message dict

    Returns:
        CompletionsResponse object
    """
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


def create_mock_trainer_app(
    tokenizer: PreTrainedTokenizer,
    mask_tracker: Optional[Dict[str, List[Optional[List[int]]]]] = None
) -> FastAPI:
    """Factory function to create a mock trainer FastAPI app.

    Args:
        tokenizer: Tokenizer for encoding/decoding
        mask_tracker: Optional dict with 'masks' list for tracking response_mask values

    Returns:
        FastAPI app instance with /v1/chat/completions endpoint
    """
    app = FastAPI(title="Mock Trainer")

    @app.post("/v1/chat/completions")
    async def completions(request: CompletionsRequest) -> CompletionsResponse:
        """Mock completions endpoint that simulates LLM responses."""
        # Track response_mask if tracker is provided
        if mask_tracker is not None:
            mask_tracker["masks"].append(request.response_mask)

        # Determine if we should use tools
        last_message = request.messages[-1]
        use_tools = _should_use_tools(last_message)

        # Generate mock response
        response_text, assistant_message = _generate_mock_response(use_tools)

        return _create_completions_response(
            tokenizer, request, response_text, assistant_message
        )

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


def patch_httpx_for_mock_trainer(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch httpx.AsyncClient to route calls to mock trainer.

    Args:
        client: TestClient for the mock trainer
        monkeypatch: pytest monkeypatch fixture
    """
    original_post = httpx.AsyncClient.post

    async def mock_post(self, url: str, **kwargs):
        if "/v1/chat/completions" in url:
            response = client.post("/v1/chat/completions", **kwargs)
            mock_response = httpx.Response(
                status_code=response.status_code,
                json=response.json(),
                request=httpx.Request("POST", url)
            )
            return mock_response
        return await original_post(self, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)


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
        FastAPI app instance with /v1/chat/completions endpoint
    """
    return create_mock_trainer_app(tokenizer)


@pytest.fixture
def mock_trainer_client(
    mock_trainer_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """Create TestClient for mock trainer and monkey-patch httpx calls.

    This fixture sets up the mock trainer and patches httpx.AsyncClient.post
    to route /v1/chat/completions calls to the mock trainer.

    Args:
        mock_trainer_app: Mock trainer FastAPI app fixture
        monkeypatch: pytest monkeypatch fixture

    Returns:
        TestClient instance for the mock trainer
    """
    client = TestClient(mock_trainer_app)
    patch_httpx_for_mock_trainer(client, monkeypatch)
    return client


@pytest.fixture
def mock_trainer_with_tracking(
    tokenizer: PreTrainedTokenizer,
    mask_tracker: Dict[str, List[Optional[List[int]]]],
    monkeypatch: pytest.MonkeyPatch
) -> Tuple[TestClient, List[Optional[List[int]]]]:
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
    app = create_mock_trainer_app(tokenizer, mask_tracker)
    client = TestClient(app)
    patch_httpx_for_mock_trainer(client, monkeypatch)

    yield client, mask_tracker["masks"]

    # Cleanup is automatic via fixture scoping

