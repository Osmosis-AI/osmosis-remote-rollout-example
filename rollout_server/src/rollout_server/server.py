"""FastAPI RolloutServer implementing the Remote Rollout Protocol.

This server implements an async-init protocol:
- Training calls POST /init to start a rollout (returns 202 with tools).
- RolloutServer drives the agent loop by calling back to
  {server_url}/v1/chat/completions for LLM generations.
- RolloutServer posts the final result to {server_url}/v1/rollout/completed.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rollout_server.config import settings
from rollout_server.executor import app_state, start_rollout
from rollout_server.schemas import (
    InitResponse,
    RolloutRequest,
)
from rollout_server.tools.calculator import CALCULATOR_TOOL_SCHEMAS


# Custom formatter that explicitly uses local time
class LocalTimeFormatter(logging.Formatter):
    """Formatter that uses local time with timezone info."""
    converter = time.localtime

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            # Add timezone abbreviation
            s = time.strftime(datefmt, ct)
            tz = time.strftime("%Z", ct)
            return f"{s} {tz}"
        else:
            t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
            tz = time.strftime("%Z", ct)
            return f"{t},{int(record.msecs):03d} {tz}"


# Configure logging for all rollout_server modules
# This ensures all submodules (session, executor, etc.) inherit the configuration
root_logger = logging.getLogger("rollout_server")
if not root_logger.handlers:
    handler = logging.StreamHandler()
    formatter = LocalTimeFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
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


@app.post("/init", response_model=InitResponse, status_code=202)
async def init_rollout(request: RolloutRequest) -> InitResponse:
    """Start a rollout (async-init protocol).

    Returns 202 Accepted with the tool definitions for this rollout, and starts
    the rollout asynchronously in the background.
    """
    rollout_id = request.rollout_id
    logger.info(f"[{rollout_id}] Received init request: max_turns={request.max_turns}")

    # Select tools for this rollout. This example always returns the calculator tools.
    tools = start_rollout(request=request, tools=CALCULATOR_TOOL_SCHEMAS)
    return InitResponse(rollout_id=rollout_id, tools=tools)


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
                    "()": "rollout_server.server.LocalTimeFormatter",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "access": {
                    "()": "rollout_server.server.LocalTimeFormatter",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
