"""FastAPI RolloutServer implementing the Remote Rollout Protocol.

This server implements the callback-based protocol specified in:
- docs/rollout_server.md (Protocol Specification)
- docs/remote_rollout_design.md (System Architecture)

The server receives POST /rollout requests, drives the agent loop by calling
back to the trainer's /v1/completions endpoint, and returns the final messages.

CRITICAL: This implementation demonstrates CORRECT response_mask handling
as specified in docs/rollout_server.md Section 4.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from cachetools import LRUCache
from fastapi import FastAPI
from pydantic import ValidationError
from transformers import AutoTokenizer

from rollout_server.schemas import (
    RolloutRequest,
    RolloutResponse,
    RolloutStatus,
    RolloutMetrics,
    Message,
)
from rollout_server.session import RolloutSession
from rollout_server.tools.calculator import execute_calculator_calls


# =============================================================================
# Constants
# =============================================================================

# Default sampling parameter values
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 512


# =============================================================================
# Custom Exceptions
# =============================================================================


class RolloutError(Exception):
    """Base exception for rollout errors."""
    pass


class TokenizerLoadError(RolloutError):
    """Error loading tokenizer."""
    pass


class ToolExecutionError(RolloutError):
    """Error executing tools."""
    pass


class RateLimitExceededError(RolloutError):
    """Too many concurrent rollouts."""
    pass


class RolloutTimeoutError(RolloutError):
    """Rollout exceeded time limit."""
    pass

# Configure module-specific logger (avoid global basicConfig conflicts)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Configuration Constants
# =============================================================================

# Tokenizer cache size (each tokenizer ~1-2GB memory)
TOKENIZER_CACHE_SIZE = int(os.getenv("TOKENIZER_CACHE_SIZE", "5"))

# HTTP client timeout in seconds
HTTP_CLIENT_TIMEOUT = float(os.getenv("HTTP_CLIENT_TIMEOUT", "300.0"))

# Security: trust_remote_code for HuggingFace tokenizers
# Default False for security; set to "true" only for trusted model sources
TOKENIZER_TRUST_REMOTE_CODE = os.getenv("TOKENIZER_TRUST_REMOTE_CODE", "false").lower() == "true"

# Rate limiting: max concurrent rollouts per server instance
MAX_CONCURRENT_ROLLOUTS = int(os.getenv("MAX_CONCURRENT_ROLLOUTS", "100"))

# Rollout timeout in seconds (total time for entire rollout, not per-request)
ROLLOUT_TIMEOUT_SECONDS = float(os.getenv("ROLLOUT_TIMEOUT_SECONDS", "600.0"))


# =============================================================================
# Application State
# =============================================================================


class AppState:
    """Application state managing shared resources."""

    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.http_client: Optional[httpx.AsyncClient] = None

        # LRU cache with configurable size (~1-2GB per tokenizer)
        # Async-safe lock for concurrent FastAPI requests
        self._tokenizer_cache_lock = asyncio.Lock()
        self.tokenizer_cache: LRUCache = LRUCache(maxsize=TOKENIZER_CACHE_SIZE)

        # Semaphore for rate limiting concurrent rollouts
        self._rollout_semaphore: Optional[asyncio.Semaphore] = None

    @property
    def rollout_semaphore(self) -> asyncio.Semaphore:
        """Lazy initialization of rollout semaphore.

        Must be created in async context (after event loop is running).
        """
        if self._rollout_semaphore is None:
            self._rollout_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ROLLOUTS)
        return self._rollout_semaphore


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: Initialize shared resources
    logger.info("Starting RolloutServer...")

    # Initialize HTTP client with connection pooling (configurable timeout)
    app_state.http_client = httpx.AsyncClient(
        timeout=HTTP_CLIENT_TIMEOUT,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )

    logger.info("RolloutServer startup complete")

    yield

    # Shutdown: Cleanup resources
    logger.info("Shutting down RolloutServer...")

    if app_state.http_client:
        await app_state.http_client.aclose()

    logger.info("RolloutServer shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title="Remote Rollout Server",
    description="Reference implementation of the Remote Rollout Protocol",
    version="0.1.0",
    lifespan=lifespan
)


# =============================================================================
# Helper Functions
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
        Default is False for security. Set to "true" only for trusted model sources.
    """
    logger.info(
        f"Loading tokenizer: {tokenizer_name} (revision={tokenizer_revision}, "
        f"trust_remote_code={TOKENIZER_TRUST_REMOTE_CODE})"
    )

    # NOTE: trust_remote_code allows execution of custom code from the model repository.
    # This is required for models like Qwen that have custom tokenizer implementations.
    # Controlled via TOKENIZER_TRUST_REMOTE_CODE environment variable for security.
    # In production deployments:
    # 1. Review model repository code before use
    # 2. Run in sandboxed environments
    # 3. Use only verified/trusted model sources
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        trust_remote_code=TOKENIZER_TRUST_REMOTE_CODE
    )

    # Ensure tokenizer has pad_token (required for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add default chat template for tokenizers that don't have one (e.g., GPT-2)
    if tokenizer.chat_template is None:
        logger.warning(f"Tokenizer {tokenizer_name} does not have chat_template, adding default template")
        # Simple default template for testing
        tokenizer.chat_template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}assistant: "

    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    return tokenizer


def parse_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse tool calls from assistant message.

    Args:
        message: Assistant message dict

    Returns:
        List of tool call dicts, or empty list if no tool calls
    """
    tool_calls = message.get("tool_calls", [])
    if not tool_calls:
        return []

    logger.info(f"Parsed {len(tool_calls)} tool calls")
    return tool_calls


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rollout-server"}


@app.post("/rollout")
async def handle_rollout(request: RolloutRequest) -> RolloutResponse:
    """Execute a complete rollout with CORRECT response_mask handling.

    This endpoint implements the callback-based protocol where:
    1. Receive rollout request from OsmosisAgentLoop
    2. Drive agent loop: call LLM → parse tools → execute tools → loop
    3. Call back to trainer's /v1/completions for LLM generation
    4. Return final messages when done

    CRITICAL IMPLEMENTATION:
    Uses RolloutSession to ensure CORRECT response_mask calculation for
    multi-turn conversations with tools.

    Features:
    - Rate limiting via semaphore (MAX_CONCURRENT_ROLLOUTS)
    - Overall rollout timeout (ROLLOUT_TIMEOUT_SECONDS)
    - Per-request HTTP timeout (HTTP_CLIENT_TIMEOUT)

    Args:
        request: RolloutRequest with server_url, messages, sampling_params

    Returns:
        RolloutResponse with final_messages and status

    Reference:
        - docs/rollout_server.md Section 3.1 (Protocol Specification)
        - docs/rollout_server.md Section 5 (Reference Implementation)
    """
    rollout_id = request.rollout_id
    logger.info(f"[{rollout_id}] Received rollout request: max_turns={request.max_turns}")

    # Wrap entire rollout in semaphore (rate limiting) and timeout
    try:
        async with asyncio.timeout(ROLLOUT_TIMEOUT_SECONDS):
            return await _execute_rollout(request)
    except TimeoutError:
        logger.error(f"[{rollout_id}] Rollout timeout after {ROLLOUT_TIMEOUT_SECONDS}s")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message=f"Rollout timeout exceeded ({ROLLOUT_TIMEOUT_SECONDS}s)",
            final_messages=[]
        )


async def _execute_rollout(request: RolloutRequest) -> RolloutResponse:
    """Internal function to execute rollout with rate limiting.

    Separated from handle_rollout to allow clean timeout handling.
    """
    rollout_id = request.rollout_id

    # Rate limiting via semaphore
    async with app_state.rollout_semaphore:
        logger.debug(f"[{rollout_id}] Acquired rollout semaphore")
        return await _execute_rollout_inner(request)


async def _execute_rollout_inner(request: RolloutRequest) -> RolloutResponse:
    """Core rollout execution logic."""
    rollout_id = request.rollout_id
    session = None  # Initialize for cleanup in finally block

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
        # Use LRU cache to avoid reloading tokenizers on every request
        # Uses double-checked locking pattern to minimize lock contention
        try:
            if request.tokenizer_name:
                cache_key = (request.tokenizer_name, request.tokenizer_revision)
            else:
                # Fallback tokenizer for testing
                cache_key = ("default", None)
                logger.warning(f"[{rollout_id}] No tokenizer specified, using default Qwen3-8B")

            # First check without lock (fast path for cached tokenizers)
            # Note: LRUCache reads modify internal order, but this is acceptable
            # for a quick existence check - we'll verify again under lock
            tokenizer = app_state.tokenizer_cache.get(cache_key)
            if tokenizer is not None:
                logger.debug(f"[{rollout_id}] Using cached tokenizer: {cache_key[0]}")
            else:
                # Acquire lock only when we need to potentially load a new tokenizer
                async with app_state._tokenizer_cache_lock:
                    # Double-check after acquiring lock (another request may have loaded it)
                    tokenizer = app_state.tokenizer_cache.get(cache_key)
                    if tokenizer is not None:
                        logger.debug(f"[{rollout_id}] Using cached tokenizer (loaded by another request): {cache_key[0]}")
                    else:
                        logger.info(f"[{rollout_id}] Loading tokenizer: {cache_key[0]}")
                        # Load tokenizer in thread pool to avoid blocking event loop
                        tokenizer_name = cache_key[0] if cache_key[0] != "default" else "Qwen/Qwen3-8B"
                        tokenizer = await asyncio.to_thread(
                            load_tokenizer, tokenizer_name, cache_key[1]
                        )
                        # Store in LRU cache (automatic eviction when full)
                        app_state.tokenizer_cache[cache_key] = tokenizer
                        logger.info(
                            f"[{rollout_id}] Cached tokenizer. "
                            f"Cache size: {len(app_state.tokenizer_cache)}/{app_state.tokenizer_cache.maxsize}"
                        )

        except Exception as e:
            logger.error(f"[{rollout_id}] Failed to load tokenizer: {e}")
            raise TokenizerLoadError(f"Failed to load tokenizer: {request.tokenizer_name}") from e

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
                    metrics=RolloutMetrics(
                        total_latency_ms=(time.time() - start_time) * 1000,
                        llm_latency_ms=total_llm_latency_ms,
                        tool_latency_ms=total_tool_latency_ms,
                        num_llm_calls=num_llm_calls,
                        num_tool_calls=num_tool_calls,
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
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
                    metrics=RolloutMetrics(
                        total_latency_ms=(time.time() - start_time) * 1000,
                        llm_latency_ms=total_llm_latency_ms,
                        tool_latency_ms=total_tool_latency_ms,
                        num_llm_calls=num_llm_calls,
                        num_tool_calls=num_tool_calls,
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                        max_context_tokens=total_tokens,
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
                    metrics=RolloutMetrics(
                        total_latency_ms=(time.time() - start_time) * 1000,
                        llm_latency_ms=total_llm_latency_ms,
                        tool_latency_ms=total_tool_latency_ms,
                        num_llm_calls=num_llm_calls,
                        num_tool_calls=num_tool_calls,
                        prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens,
                        max_context_tokens=total_tokens,
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
                # Preserve original error message for traingate debugging
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
            metrics=RolloutMetrics(
                total_latency_ms=(time.time() - start_time) * 1000,
                llm_latency_ms=total_llm_latency_ms,
                tool_latency_ms=total_tool_latency_ms,
                num_llm_calls=num_llm_calls,
                num_tool_calls=num_tool_calls,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                max_context_tokens=total_tokens,
            )
        )

    except httpx.HTTPStatusError as e:
        # HTTP error from trainer - categorize for traingate retry logic
        status_code = e.response.status_code
        logger.error(
            f"[{rollout_id}] HTTP error from trainer: "
            f"{status_code} - {e.response.text}"
        )

        # Categorize error for traingate to determine retry strategy
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
            metrics=RolloutMetrics(
                total_latency_ms=(time.time() - start_time) * 1000,
                llm_latency_ms=total_llm_latency_ms,
                tool_latency_ms=total_tool_latency_ms,
                num_llm_calls=num_llm_calls,
                num_tool_calls=num_tool_calls,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
            )
        )

    except httpx.RequestError as e:
        # Network error - safe to expose (no sensitive data)
        logger.error(f"[{rollout_id}] Network error: {e}")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message="Network error communicating with trainer",
            final_messages=[],
            extra_fields={"error_category": "network_error"},
            metrics=RolloutMetrics(
                total_latency_ms=(time.time() - start_time) * 1000,
                llm_latency_ms=total_llm_latency_ms,
                tool_latency_ms=total_tool_latency_ms,
                num_llm_calls=num_llm_calls,
                num_tool_calls=num_tool_calls,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
            )
        )

    except TokenizerLoadError as e:
        # Tokenizer error - safe to expose
        logger.error(f"[{rollout_id}] Tokenizer error: {e}")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message=f"Failed to load tokenizer: {e}",
            final_messages=[],
            extra_fields={"error_category": "tokenizer_error"},
            metrics=RolloutMetrics(
                total_latency_ms=(time.time() - start_time) * 1000,
            )
        )

    except ToolExecutionError as e:
        # Tool error - safe to expose (error message preserved for debugging)
        logger.error(f"[{rollout_id}] Tool execution error: {e}")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message=str(e),  # Preserved original error for traingate debugging
            final_messages=[],
            extra_fields={"error_category": "tool_error"},
            metrics=RolloutMetrics(
                total_latency_ms=(time.time() - start_time) * 1000,
                llm_latency_ms=total_llm_latency_ms,
                tool_latency_ms=total_tool_latency_ms,
                num_llm_calls=num_llm_calls,
                num_tool_calls=num_tool_calls,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
            )
        )

    except ValidationError as e:
        # Pydantic validation error - safe to expose
        logger.error(f"[{rollout_id}] Validation error: {e}")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message=f"Invalid request data: {e}",
            final_messages=[],
            extra_fields={"error_category": "validation_error"},
            metrics=RolloutMetrics(
                total_latency_ms=(time.time() - start_time) * 1000,
                llm_latency_ms=total_llm_latency_ms,
                tool_latency_ms=total_tool_latency_ms,
                num_llm_calls=num_llm_calls,
                num_tool_calls=num_tool_calls,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
            )
        )

    except Exception as e:
        # Unexpected internal error - DO NOT expose details
        logger.exception(f"[{rollout_id}] Unexpected internal error")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message="Internal server error",  # Generic message
            final_messages=[],
            extra_fields={"error_category": "internal_error"},
            metrics=RolloutMetrics(
                total_latency_ms=(time.time() - start_time) * 1000,
                llm_latency_ms=total_llm_latency_ms,
                tool_latency_ms=total_tool_latency_ms,
                num_llm_calls=num_llm_calls,
                num_tool_calls=num_tool_calls,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
            )
        )

    finally:
        # Cleanup session (with error handling to not mask original exception)
        if session:
            try:
                await session.close()
            except Exception as close_error:
                logger.warning(f"[{rollout_id}] Error closing session: {close_error}")


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import os
    import uvicorn

    # Use environment variable or default to 9000 (avoid conflict with traingate's 8080-8130 range)
    port = int(os.getenv("ROLLOUT_SERVER_PORT", "9000"))

    uvicorn.run(
        "rollout_server.server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )
