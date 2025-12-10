"""FastAPI RolloutServer implementing the Remote Rollout Protocol.

This server implements the callback-based protocol specified in:
- docs/rollout_server.md (Protocol Specification)
- docs/remote_rollout_design.md (System Architecture)

The server receives POST /rollout requests, drives the agent loop by calling
back to the trainer's /v1/chat/completions endpoint, and returns the final messages.

CRITICAL: This implementation demonstrates CORRECT response_mask handling
as specified in docs/rollout_server.md Section 4.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rollout_server.config import settings
from rollout_server.executor import app_state, execute_rollout
from rollout_server.schemas import (
    RolloutRequest,
    RolloutResponse,
    RolloutStatus,
    ToolsResponse,
)
from rollout_server.tools.calculator import CALCULATOR_TOOL_SCHEMAS


# Configure logging for all rollout_server modules
# This ensures all submodules (session, executor, etc.) inherit the configuration
root_logger = logging.getLogger("rollout_server")
if not root_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: Initialize shared resources
    logger.info("Starting RolloutServer...")
    await app_state.initialize()
    logger.info("RolloutServer startup complete")

    yield

    # Shutdown: Cleanup resources
    logger.info("Shutting down RolloutServer...")
    await app_state.cleanup()
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
# API Endpoints
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rollout-server"}


@app.get("/tools", response_model=ToolsResponse)
async def get_tools() -> ToolsResponse:
    """Return available tool schemas for LLM tool calling.

    This endpoint is called once at worker startup by the training cluster
    to discover what tools are available. The tool definitions are used for
    apply_chat_template() so the LLM knows what tools it can use.

    The schemas follow the OpenAI function calling format.

    Returns:
        ToolsResponse with list of tool definitions

    Example Response:
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number", "description": "First number"},
                                "b": {"type": "number", "description": "Second number"}
                            },
                            "required": ["a", "b"]
                        }
                    }
                },
                ...
            ]
        }

    Reference: docs/rollout_server.md Section 3.0
    """
    return ToolsResponse(tools=CALCULATOR_TOOL_SCHEMAS)


@app.post("/rollout")
async def handle_rollout(request: RolloutRequest) -> RolloutResponse:
    """Execute a complete rollout with CORRECT response_mask handling.

    This endpoint implements the callback-based protocol where:
    1. Receive rollout request from OsmosisAgentLoop
    2. Drive agent loop: call LLM -> parse tools -> execute tools -> loop
    3. Call back to trainer's /v1/chat/completions for LLM generation
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

    # Wrap entire rollout in timeout
    try:
        async with asyncio.timeout(settings.rollout_timeout_seconds):
            response = await execute_rollout(request)
            metrics_summary = response.metrics.model_dump() if response.metrics else {}
            logger.info(
                f"[{rollout_id}] Rollout completed: "
                f"status={response.status.value}, "
                f"finish_reason={response.finish_reason}, "
                f"error={response.error_message}, "
                f"metrics={metrics_summary}, "
                f"extra_fields={response.extra_fields}"
            )
            return response
    except TimeoutError:
        logger.error(f"[{rollout_id}] Rollout timeout after {settings.rollout_timeout_seconds}s")
        return RolloutResponse(
            rollout_id=rollout_id,
            status=RolloutStatus.ERROR,
            error_message=f"Rollout timeout exceeded ({settings.rollout_timeout_seconds}s)",
            final_messages=[]
        )


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    # Configure uvicorn with detailed access logging
    # access_log=True ensures each HTTP request is logged with timestamp
    uvicorn.run(
        "rollout_server.server:app",
        host="0.0.0.0",
        port=settings.server_port,
        log_level="info",
        reload=False,
        access_log=True,  # Enable access logging for each request
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "access": {
                    "format": '%(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            },
        },
    )
