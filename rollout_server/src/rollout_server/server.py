"""FastAPI RolloutServer implementing the Remote Rollout Protocol.

This server implements the callback-based protocol specified in:
- docs/rollout_server.md (Protocol Specification)
- docs/remote_rollout_design.md (System Architecture)

The server receives POST /rollout requests, drives the agent loop by calling
back to the trainer's /v1/completions endpoint, and returns the final messages.

CRITICAL: This implementation demonstrates CORRECT response_mask handling
as specified in docs/rollout_server.md Section 4.
"""

import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer
import httpx

from rollout_server.schemas import RolloutRequest, RolloutResponse, RolloutStatus, Message
from rollout_server.session import RolloutSession
from rollout_server.tools.calculator import execute_calculator_calls

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application State
# =============================================================================


class AppState:
    """Application state managing shared resources."""

    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.http_client: Optional[httpx.AsyncClient] = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: Initialize shared resources
    logger.info("Starting RolloutServer...")

    # Initialize HTTP client with connection pooling
    app_state.http_client = httpx.AsyncClient(
        timeout=300.0,
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
    """
    logger.info(f"Loading tokenizer: {tokenizer_name} (revision={tokenizer_revision})")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        trust_remote_code=True
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


def parse_tool_calls(message: dict) -> list:
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

    try:
        # Load tokenizer (MUST match trainer's tokenizer!)
        if request.tokenizer_name:
            tokenizer = load_tokenizer(request.tokenizer_name, request.tokenizer_revision)
        else:
            # Fallback tokenizer for testing
            if app_state.tokenizer is None:
                logger.warning("No tokenizer specified in request, using default Qwen3-8B")
                app_state.tokenizer = load_tokenizer("Qwen/Qwen3-8B")
            tokenizer = app_state.tokenizer

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

        # Agent loop
        for turn in range(request.max_turns):
            logger.info(f"[{rollout_id}] Turn {turn + 1}/{request.max_turns}")

            # 1. Call LLM (session handles response_mask calculation)
            completion_response = await session.call_llm(
                sampling_params=request.sampling_params.model_dump()
            )

            # 2. Extract assistant message
            assistant_message = completion_response.choices[0].message.model_dump()
            session.append_assistant_message(assistant_message)

            # 3. Check for tool calls
            tool_calls = parse_tool_calls(assistant_message)
            if not tool_calls:
                # No tool calls - conversation done
                logger.info(f"[{rollout_id}] No tool calls, conversation complete")
                return RolloutResponse(
                    rollout_id=rollout_id,
                    status=RolloutStatus.COMPLETED,
                    final_messages=[Message(**msg) for msg in session.messages],
                    finish_reason="stop"
                )

            # 4. Execute tools
            logger.info(f"[{rollout_id}] Executing {len(tool_calls)} tool calls")
            tool_results = await execute_calculator_calls(tool_calls)

            # 5. Append tool outputs (session tracks for next response_mask)
            session.append_tool_outputs(tool_results)

        # Max turns reached
        logger.info(f"[{rollout_id}] Max turns reached")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.COMPLETED,
            final_messages=[Message(**msg) for msg in session.messages],
            finish_reason="max_turns"
        )

    except httpx.HTTPStatusError as e:
        # HTTP error from trainer
        logger.error(f"[{rollout_id}] HTTP error from trainer: {e.response.status_code} - {e.response.text}")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message=f"Trainer HTTP error: {e.response.status_code}",
            final_messages=[]
        )

    except httpx.RequestError as e:
        # Network error
        logger.error(f"[{rollout_id}] Network error: {e}")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message=f"Network error: {str(e)}",
            final_messages=[]
        )

    except Exception as e:
        # Unexpected error
        logger.exception(f"[{rollout_id}] Unexpected error during rollout")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message=f"Unexpected error: {str(e)}",
            final_messages=[]
        )

    finally:
        # Cleanup session
        if 'session' in locals():
            await session.close()


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
