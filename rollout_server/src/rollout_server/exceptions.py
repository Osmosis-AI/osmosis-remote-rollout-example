"""Exceptions re-exported from the Osmosis remote rollout SDK.

This example server relies on `osmosis_ai.rollout` for protocol handling.
We expose the SDK exception types here for convenience.
"""

from osmosis_ai.rollout.core.exceptions import (
    AgentLoopNotFoundError,
    OsmosisRolloutError,
    OsmosisServerError,
    OsmosisTimeoutError,
    OsmosisTransportError,
    OsmosisValidationError,
    ToolArgumentError,
    ToolExecutionError,
)

__all__ = [
    "OsmosisRolloutError",
    "OsmosisTransportError",
    "OsmosisServerError",
    "OsmosisValidationError",
    "OsmosisTimeoutError",
    "AgentLoopNotFoundError",
    "ToolExecutionError",
    "ToolArgumentError",
]

