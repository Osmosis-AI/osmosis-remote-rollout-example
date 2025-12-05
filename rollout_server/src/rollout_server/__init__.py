"""Remote Rollout Server - Reference implementation of the Remote Rollout Protocol.

This package provides a FastAPI server that implements the callback-based protocol
for driving agent loops in remote rollout scenarios.

Public API:
- RolloutSession: Manages a single rollout session with response_mask tracking
- app: FastAPI application instance
- Schemas: Message, RolloutRequest, RolloutResponse, etc.
"""

from importlib.metadata import version, PackageNotFoundError

from rollout_server.schemas import (
    CompletionsChoice,
    CompletionsRequest,
    CompletionsResponse,
    Message,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStatus,
    SamplingParams,
    ToolCall,
    ToolCallFunction,
)
from rollout_server.server import app
from rollout_server.session import RolloutSession

__all__ = [
    # FastAPI app
    "app",
    # Session management
    "RolloutSession",
    # Schemas
    "CompletionsChoice",
    "CompletionsRequest",
    "CompletionsResponse",
    "Message",
    "RolloutMetrics",
    "RolloutRequest",
    "RolloutResponse",
    "RolloutStatus",
    "SamplingParams",
    "ToolCall",
    "ToolCallFunction",
]

# Single source of version truth: read from pyproject.toml via importlib.metadata
try:
    __version__ = version("rollout-server")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source)
    __version__ = "0.1.0-dev"

