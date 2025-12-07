"""Remote Rollout Server - Reference implementation of the Remote Rollout Protocol.

This package provides a FastAPI server that implements the callback-based protocol
for driving agent loops in remote rollout scenarios.

Public API:
- RolloutSession: Manages a single rollout session with response_mask tracking
- app: FastAPI application instance
- Schemas: Message, RolloutRequest, RolloutResponse, etc.
- Exceptions: RolloutError, TokenizerLoadError, ToolExecutionError, etc.
"""

from importlib.metadata import version, PackageNotFoundError

from rollout_server.exceptions import (
    RolloutError,
    TokenizerLoadError,
    ToolExecutionError,
    RateLimitExceededError,
    RolloutTimeoutError,
)
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
from rollout_server.executor import app_state, execute_rollout

__all__ = [
    # FastAPI app
    "app",
    # Session and executor
    "RolloutSession",
    "app_state",
    "execute_rollout",
    # Exceptions
    "RolloutError",
    "TokenizerLoadError",
    "ToolExecutionError",
    "RateLimitExceededError",
    "RolloutTimeoutError",
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

