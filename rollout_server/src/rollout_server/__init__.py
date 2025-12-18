"""Remote Rollout Server - Example implementation built on the Osmosis SDK.

This package intentionally delegates all remote rollout protocol handling to the
Osmosis Python SDK (`osmosis_ai.rollout`) and keeps only example agent logic.
"""

from importlib.metadata import version, PackageNotFoundError

from rollout_server.exceptions import (
    AgentLoopNotFoundError,
    OsmosisRolloutError,
    OsmosisServerError,
    OsmosisTimeoutError,
    OsmosisTransportError,
    OsmosisValidationError,
    ToolArgumentError,
    ToolExecutionError,
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
from rollout_server.server import CalculatorAgentLoop, app

__all__ = [
    # FastAPI app
    "app",
    # Example agent loop
    "CalculatorAgentLoop",
    # Exceptions (from osmosis_ai.rollout)
    "OsmosisRolloutError",
    "OsmosisTransportError",
    "OsmosisServerError",
    "OsmosisValidationError",
    "OsmosisTimeoutError",
    "AgentLoopNotFoundError",
    "ToolExecutionError",
    "ToolArgumentError",
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

