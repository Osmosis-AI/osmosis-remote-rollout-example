"""FastAPI RolloutServer implementing the Remote Rollout Protocol.

This server implements an async-init protocol:
- Training calls POST /init to start a rollout (returns 202 with tools).
- RolloutServer drives the agent loop by calling back to
  {server_url}/v1/chat/completions for LLM generations.
- RolloutServer posts the final result to {server_url}/v1/rollout/completed.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

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
# Request/Response Logging Middleware
# =============================================================================


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests and outgoing responses."""

    # Paths where we skip response logging (only log request payload)
    SKIP_RESPONSE_LOG_PATHS = {"/tools", "/init"}

    async def dispatch(self, request: Request, call_next):
        # Skip logging entirely for health checks to reduce noise
        if request.url.path == "/health":
            return await call_next(request)

        # Read and log request body
        request_body = await request.body()
        try:
            request_json = json.loads(request_body) if request_body else {}
            request_body_str = json.dumps(request_json, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            request_body_str = request_body.decode("utf-8", errors="replace")

        logger.info(
            f">>> INCOMING REQUEST: {request.method} {request.url.path}\n"
            f"Body:\n{request_body_str}"
        )

        # Reconstruct request with body for downstream handlers
        async def receive():
            return {"type": "http.request", "body": request_body}

        request._receive = receive

        # Call the next handler and capture response
        response = await call_next(request)

        # For certain paths, skip response logging
        if request.url.path in self.SKIP_RESPONSE_LOG_PATHS:
            return response

        # Read response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        try:
            response_json = json.loads(response_body) if response_body else {}
            response_body_str = json.dumps(response_json, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            response_body_str = response_body.decode("utf-8", errors="replace")

        logger.info(
            f"<<< OUTGOING RESPONSE: {request.method} {request.url.path} - Status: {response.status_code}\n"
            f"Body:\n{response_body_str}"
        )

        # Return a new response with the same body
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


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

# Add request/response logging middleware
app.add_middleware(RequestResponseLoggingMiddleware)


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
