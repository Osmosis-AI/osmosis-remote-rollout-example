"""Core rollout execution logic for the Remote Rollout Server.

This module contains the main execution logic that was extracted from server.py
to improve maintainability and separation of concerns.

The executor handles:
- Application state management (tokenizer cache, HTTP client)
- Tokenizer loading with LRU caching
- Core rollout execution with response_mask tracking
- Error handling and metrics collection
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from cachetools import LRUCache
from pydantic import ValidationError
from transformers import AutoTokenizer

from rollout_server.config import settings
from rollout_server.exceptions import TokenizerLoadError, ToolExecutionError
from rollout_server.schemas import (
    Message,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStatus,
)
from rollout_server.session import RolloutSession
from rollout_server.tools.calculator import execute_calculator_calls


logger = logging.getLogger(__name__)


# =============================================================================
# Application State
# =============================================================================


class AppState:
    """Application state managing shared resources.

    This class manages resources that are shared across requests:
    - HTTP client with connection pooling
    - Tokenizer cache with LRU eviction
    - Rate limiting semaphore

    Thread Safety:
    - tokenizer_cache uses asyncio.Lock for safe concurrent access
    - rollout_semaphore limits concurrent rollouts
    """

    def __init__(self) -> None:
        self.tokenizer: Optional[AutoTokenizer] = None
        self.http_client: Optional[httpx.AsyncClient] = None

        # LRU cache with configurable size (~1-2GB per tokenizer)
        # Async-safe lock for concurrent FastAPI requests
        self._tokenizer_cache_lock = asyncio.Lock()
        self.tokenizer_cache: LRUCache = LRUCache(maxsize=settings.tokenizer_cache_size)

        # Semaphore for rate limiting concurrent rollouts
        self._rollout_semaphore: Optional[asyncio.Semaphore] = None

    @property
    def rollout_semaphore(self) -> asyncio.Semaphore:
        """Lazy initialization of rollout semaphore.

        Must be created in async context (after event loop is running).
        """
        if self._rollout_semaphore is None:
            self._rollout_semaphore = asyncio.Semaphore(settings.max_concurrent_rollouts)
        return self._rollout_semaphore

    async def initialize(self) -> None:
        """Initialize async resources (call during app startup)."""
        self.http_client = httpx.AsyncClient(
            timeout=settings.http_client_timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

    async def cleanup(self) -> None:
        """Cleanup resources (call during app shutdown)."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None


# Global application state instance
app_state = AppState()


# =============================================================================
# Tokenizer Loading
# =============================================================================


def load_tokenizer(tokenizer_name: str, tokenizer_revision: Optional[str] = None) -> AutoTokenizer:
    """Load tokenizer matching the trainer's configuration.

    CRITICAL: Must use the EXACT same tokenizer as the trainer for token ID consistency.

    Args:
        tokenizer_name: Model name (e.g., "Qwen/Qwen3-8B")
        tokenizer_revision: Git revision (e.g., "main" or commit hash)

    Returns:
        Loaded tokenizer instance

    Reference: docs/rollout_server.md Section 2.2 (Tokenizer Alignment)

    Security Note:
        trust_remote_code is controlled by TOKENIZER_TRUST_REMOTE_CODE env var.
        Default is True for Qwen3 models. Set to "false" only for standard tokenizers.
    """
    logger.info(
        f"Loading tokenizer: {tokenizer_name} (revision={tokenizer_revision}, "
        f"trust_remote_code={settings.tokenizer_trust_remote_code})"
    )

    # NOTE: trust_remote_code allows execution of custom code from the model repository.
    # This is required for models like Qwen that have custom tokenizer implementations.
    # Controlled via TOKENIZER_TRUST_REMOTE_CODE environment variable for security.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        trust_remote_code=settings.tokenizer_trust_remote_code
    )

    # Ensure tokenizer has pad_token (required for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add default chat template for tokenizers that don't have one (e.g., GPT-2)
    if tokenizer.chat_template is None:
        logger.warning(f"Tokenizer {tokenizer_name} does not have chat_template, adding default template")
        tokenizer.chat_template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}assistant: "

    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    return tokenizer


async def get_or_load_tokenizer(
    rollout_id: str,
    tokenizer_name: Optional[str],
    tokenizer_revision: Optional[str]
) -> AutoTokenizer:
    """Get tokenizer from cache or load it.

    Uses double-checked locking pattern to minimize lock contention.

    Args:
        rollout_id: For logging purposes
        tokenizer_name: Model name or None for default
        tokenizer_revision: Git revision or None

    Returns:
        Tokenizer instance

    Raises:
        TokenizerLoadError: If tokenizer cannot be loaded
    """
    if tokenizer_name:
        cache_key = (tokenizer_name, tokenizer_revision)
    else:
        cache_key = ("default", None)
        logger.warning(f"[{rollout_id}] No tokenizer specified, using default Qwen3-8B")

    # First check without lock (fast path for cached tokenizers)
    tokenizer = app_state.tokenizer_cache.get(cache_key)
    if tokenizer is not None:
        logger.debug(f"[{rollout_id}] Using cached tokenizer: {cache_key[0]}")
        return tokenizer

    # Acquire lock only when we need to potentially load a new tokenizer
    async with app_state._tokenizer_cache_lock:
        # Double-check after acquiring lock
        tokenizer = app_state.tokenizer_cache.get(cache_key)
        if tokenizer is not None:
            logger.debug(f"[{rollout_id}] Using cached tokenizer (loaded by another request): {cache_key[0]}")
            return tokenizer

        logger.info(f"[{rollout_id}] Loading tokenizer: {cache_key[0]}")
        tokenizer_name_to_load = cache_key[0] if cache_key[0] != "default" else "Qwen/Qwen3-8B"

        try:
            # Load tokenizer in thread pool to avoid blocking event loop
            tokenizer = await asyncio.to_thread(
                load_tokenizer, tokenizer_name_to_load, cache_key[1]
            )
        except Exception as e:
            logger.error(f"[{rollout_id}] Failed to load tokenizer: {e}")
            raise TokenizerLoadError(f"Failed to load tokenizer: {tokenizer_name_to_load}") from e

        # Store in LRU cache
        app_state.tokenizer_cache[cache_key] = tokenizer
        logger.info(
            f"[{rollout_id}] Cached tokenizer. "
            f"Cache size: {len(app_state.tokenizer_cache)}/{app_state.tokenizer_cache.maxsize}"
        )

        return tokenizer


# =============================================================================
# Helper Functions
# =============================================================================


def parse_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse tool calls from assistant message.

    Args:
        message: Assistant message dict

    Returns:
        List of tool call dicts, or empty list if no tool calls
    """
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        logger.info(f"Parsed {len(tool_calls)} tool calls")
    return tool_calls


def create_metrics(
    start_time: float,
    llm_latency_ms: float,
    tool_latency_ms: float,
    num_llm_calls: int,
    num_tool_calls: int,
    prompt_tokens: int,
    response_tokens: int,
    max_context_tokens: int = 0
) -> RolloutMetrics:
    """Create RolloutMetrics with calculated total latency."""
    return RolloutMetrics(
        total_latency_ms=(time.time() - start_time) * 1000,
        llm_latency_ms=llm_latency_ms,
        tool_latency_ms=tool_latency_ms,
        num_llm_calls=num_llm_calls,
        num_tool_calls=num_tool_calls,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        max_context_tokens=max_context_tokens,
    )


# =============================================================================
# Core Rollout Execution
# =============================================================================


async def execute_rollout(request: RolloutRequest) -> RolloutResponse:
    """Execute a complete rollout with rate limiting.

    This is the main entry point for rollout execution.
    Wraps the core logic with semaphore-based rate limiting.

    Args:
        request: RolloutRequest from the API

    Returns:
        RolloutResponse with final messages and status
    """
    rollout_id = request.rollout_id

    # Rate limiting via semaphore
    async with app_state.rollout_semaphore:
        logger.debug(f"[{rollout_id}] Acquired rollout semaphore")
        return await _execute_rollout_core(request)


async def _execute_rollout_core(request: RolloutRequest) -> RolloutResponse:
    """Core rollout execution logic.

    This function implements the agent loop:
    1. Load tokenizer
    2. Create RolloutSession
    3. Loop: call LLM -> parse tools -> execute tools
    4. Return final messages

    Args:
        request: RolloutRequest from the API

    Returns:
        RolloutResponse with final messages and status
    """
    rollout_id = request.rollout_id
    session: Optional[RolloutSession] = None

    # Initialize metrics tracking
    start_time = time.time()
    total_llm_latency_ms = 0.0
    total_tool_latency_ms = 0.0
    num_llm_calls = 0
    num_tool_calls = 0
    prompt_tokens = 0
    response_tokens = 0

    try:
        # Load tokenizer (MUST match trainer's tokenizer!)
        try:
            tokenizer = await get_or_load_tokenizer(
                rollout_id,
                request.tokenizer_name,
                request.tokenizer_revision
            )
        except TokenizerLoadError as e:
            return RolloutResponse(
                rollout_id=rollout_id,
                status=RolloutStatus.ERROR,
                error_message=str(e),
                final_messages=[],
                extra_fields={"error_category": "tokenizer_error"},
                metrics=create_metrics(start_time, 0, 0, 0, 0, 0, 0)
            )

        # Create session (manages response_mask calculation)
        session = RolloutSession(
            rollout_id=rollout_id,
            tokenizer=tokenizer,
            server_url=request.server_url,
            http_client=app_state.http_client,
            callback_api_key=request.callback_api_key
        )

        # Initialize messages
        session.messages = [msg.model_dump() for msg in request.messages]

        # Track total tokens for max_tokens_total enforcement
        total_tokens = 0

        # Agent loop
        for turn in range(request.max_turns):
            logger.info(f"[{rollout_id}] Turn {turn + 1}/{request.max_turns}, total_tokens={total_tokens}")

            # 1. Call LLM (session handles response_mask calculation)
            try:
                llm_start = time.time()
                completion_response = await session.call_llm(
                    sampling_params=request.sampling_params.model_dump()
                )
                llm_end = time.time()

                # Update metrics
                total_llm_latency_ms += (llm_end - llm_start) * 1000
                num_llm_calls += 1
                prompt_tokens += len(completion_response.prompt_token_ids)
                response_tokens += len(completion_response.token_ids)

            except ValueError as e:
                # response_mask calculation error (e.g., negative token count)
                logger.error(f"[{rollout_id}] Response mask error: {e}")
                return RolloutResponse(
                    rollout_id=rollout_id,
                    status=RolloutStatus.ERROR,
                    error_message=f"Invalid conversation state (possible context truncation): {e}",
                    final_messages=[],
                    metrics=create_metrics(
                        start_time, total_llm_latency_ms, total_tool_latency_ms,
                        num_llm_calls, num_tool_calls, prompt_tokens, response_tokens
                    )
                )

            # 2. Track token usage for max_tokens_total enforcement
            total_tokens += len(completion_response.token_ids)
            if total_tokens >= request.max_tokens_total:
                logger.info(f"[{rollout_id}] Token limit reached: {total_tokens}/{request.max_tokens_total}")
                return RolloutResponse(
                    rollout_id=rollout_id,
                    status=RolloutStatus.COMPLETED,
                    final_messages=[Message(**msg) for msg in session.messages],
                    finish_reason="max_tokens",
                    metrics=create_metrics(
                        start_time, total_llm_latency_ms, total_tool_latency_ms,
                        num_llm_calls, num_tool_calls, prompt_tokens, response_tokens, total_tokens
                    )
                )

            # 3. Extract assistant message
            assistant_message = completion_response.choices[0].message.model_dump()
            session.append_assistant_message(assistant_message)

            # 4. Check for tool calls
            tool_calls = parse_tool_calls(assistant_message)
            if not tool_calls:
                # No tool calls - conversation done
                logger.info(f"[{rollout_id}] No tool calls, conversation complete")
                return RolloutResponse(
                    rollout_id=rollout_id,
                    status=RolloutStatus.COMPLETED,
                    final_messages=[Message(**msg) for msg in session.messages],
                    finish_reason="stop",
                    metrics=create_metrics(
                        start_time, total_llm_latency_ms, total_tool_latency_ms,
                        num_llm_calls, num_tool_calls, prompt_tokens, response_tokens, total_tokens
                    )
                )

            # 5. Execute tools
            try:
                tool_start = time.time()
                logger.info(f"[{rollout_id}] Executing {len(tool_calls)} tool calls")
                tool_results = await execute_calculator_calls(tool_calls)
                tool_end = time.time()

                # Update metrics
                total_tool_latency_ms += (tool_end - tool_start) * 1000
                num_tool_calls += len(tool_calls)

            except Exception as e:
                logger.error(f"[{rollout_id}] Tool execution failed: {e}")
                raise ToolExecutionError(f"Failed to execute tools: {e}") from e

            # 6. Append tool outputs (session tracks for next response_mask)
            session.append_tool_outputs(tool_results)

        # Max turns reached
        logger.info(f"[{rollout_id}] Max turns reached")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.COMPLETED,
            final_messages=[Message(**msg) for msg in session.messages],
            finish_reason="max_turns",
            metrics=create_metrics(
                start_time, total_llm_latency_ms, total_tool_latency_ms,
                num_llm_calls, num_tool_calls, prompt_tokens, response_tokens, total_tokens
            )
        )

    except httpx.HTTPStatusError as e:
        return _handle_http_error(rollout_id, e, start_time, total_llm_latency_ms,
                                   total_tool_latency_ms, num_llm_calls, num_tool_calls,
                                   prompt_tokens, response_tokens)

    except httpx.RequestError as e:
        logger.error(f"[{rollout_id}] Network error: {e}")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message="Network error communicating with trainer",
            final_messages=[],
            extra_fields={"error_category": "network_error"},
            metrics=create_metrics(
                start_time, total_llm_latency_ms, total_tool_latency_ms,
                num_llm_calls, num_tool_calls, prompt_tokens, response_tokens
            )
        )

    except ToolExecutionError as e:
        logger.error(f"[{rollout_id}] Tool execution error: {e}")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message=str(e),
            final_messages=[],
            extra_fields={"error_category": "tool_error"},
            metrics=create_metrics(
                start_time, total_llm_latency_ms, total_tool_latency_ms,
                num_llm_calls, num_tool_calls, prompt_tokens, response_tokens
            )
        )

    except ValidationError as e:
        logger.error(f"[{rollout_id}] Validation error: {e}")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message=f"Invalid request data: {e}",
            final_messages=[],
            extra_fields={"error_category": "validation_error"},
            metrics=create_metrics(
                start_time, total_llm_latency_ms, total_tool_latency_ms,
                num_llm_calls, num_tool_calls, prompt_tokens, response_tokens
            )
        )

    except Exception as e:
        # Unexpected internal error - DO NOT expose details
        logger.exception(f"[{rollout_id}] Unexpected internal error")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message="Internal server error",
            final_messages=[],
            extra_fields={"error_category": "internal_error"},
            metrics=create_metrics(
                start_time, total_llm_latency_ms, total_tool_latency_ms,
                num_llm_calls, num_tool_calls, prompt_tokens, response_tokens
            )
        )

    finally:
        # Cleanup session
        if session:
            try:
                await session.close()
            except Exception as close_error:
                logger.warning(f"[{rollout_id}] Error closing session: {close_error}")


def _handle_http_error(
    rollout_id: str,
    error: httpx.HTTPStatusError,
    start_time: float,
    llm_latency_ms: float,
    tool_latency_ms: float,
    num_llm_calls: int,
    num_tool_calls: int,
    prompt_tokens: int,
    response_tokens: int
) -> RolloutResponse:
    """Handle HTTP errors from trainer with categorization."""
    status_code = error.response.status_code
    logger.error(
        f"[{rollout_id}] HTTP error from trainer: "
        f"{status_code} - {error.response.text}"
    )

    # Categorize error to determine retry strategy
    if status_code == 429:
        error_category = "rate_limited"
        error_message = "Trainer rate limit exceeded (429)"
    elif status_code >= 500:
        error_category = "trainer_server_error"
        error_message = f"Trainer server error (status {status_code}) - retryable"
    elif status_code >= 400:
        error_category = "trainer_client_error"
        error_message = f"Trainer client error (status {status_code}) - check request"
    else:
        error_category = "trainer_error"
        error_message = f"Trainer returned error (status {status_code})"

    return RolloutResponse(
        rollout_id=rollout_id,
        status=RolloutStatus.ERROR,
        error_message=error_message,
        final_messages=[],
        extra_fields={"error_category": error_category, "http_status": status_code},
        metrics=create_metrics(
            start_time, llm_latency_ms, tool_latency_ms,
            num_llm_calls, num_tool_calls, prompt_tokens, response_tokens
        )
    )

