"""Remote Rollout Server - Reference implementation of the Remote Rollout Protocol.

This package provides a FastAPI server implementing an async-init remote rollout
protocol:
- Training calls POST /init and receives tools (202 Accepted).
- RolloutServer drives the agent loop by calling back to
  {server_url}/v1/chat/completions.
- RolloutServer posts the final result to {server_url}/v1/rollout/completed.
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
    CompletionUsage,
    CompletionsChoice,
    CompletionsRequest,
    CompletionsResponse,
    InitResponse,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStatus,
)
from rollout_server.server import app
from rollout_server.executor import app_state, start_rollout

__all__ = [
    # FastAPI app
    "app",
    # Executor
    "app_state",
    "start_rollout",
    # Exceptions
    "RolloutError",
    "TokenizerLoadError",
    "ToolExecutionError",
    "RateLimitExceededError",
    "RolloutTimeoutError",
    # Schemas
    "CompletionUsage",
    "CompletionsChoice",
    "CompletionsRequest",
    "CompletionsResponse",
    "InitResponse",
    "RolloutMetrics",
    "RolloutRequest",
    "RolloutResponse",
    "RolloutStatus",
]

# Single source of version truth: read from pyproject.toml via importlib.metadata
try:
    __version__ = version("rollout-server")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source)
    __version__ = "0.1.0-dev"

