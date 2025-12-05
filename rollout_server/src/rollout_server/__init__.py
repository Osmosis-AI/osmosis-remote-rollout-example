"""Remote Rollout Server - Reference implementation of the Remote Rollout Protocol.

This package provides a FastAPI server that implements the callback-based protocol
for driving agent loops in remote rollout scenarios.

Public API:
- RolloutSession: Manages a single rollout session with response_mask tracking
- app: FastAPI application instance
- Schemas: Message, RolloutRequest, RolloutResponse, etc.
"""

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
]

__version__ = "0.1.0"

