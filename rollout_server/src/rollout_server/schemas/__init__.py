"""Pydantic schemas for Remote Rollout Protocol.

These schemas define the public API contract between RolloutServer and the training cluster.
Based on the Remote Rollout Protocol specification in docs/rollout_server.md.

This module re-exports all public schemas from submodules for convenient access:

    from rollout_server.schemas import Message, RolloutRequest, RolloutResponse

Submodules:
    - constants: Shared constants (VALID_MESSAGE_ROLES, MAX_TOKENS_LIMIT, etc.)
    - messages: Message, ToolCall, ToolCallFunction, ToolFunction, ToolDefinition, ToolsResponse
    - params: SamplingParams
    - rollout: RolloutStatus, RolloutMetrics, RolloutRequest, RolloutResponse
    - completions: CompletionsChoice, CompletionsRequest, CompletionsResponse
"""

# Constants
from rollout_server.schemas.constants import (
    VALID_MESSAGE_ROLES,
    MAX_TOKENS_LIMIT,
)

# Message-related schemas
from rollout_server.schemas.messages import (
    Message,
    ToolCall,
    ToolCallFunction,
)

# Tool definition schemas (verl-compatible)
from rollout_server.schemas.tools import (
    OpenAIFunctionPropertySchema,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionCallSchema,
    OpenAIFunctionToolCall,
    ToolResponse,
)

# Sampling parameters
from rollout_server.schemas.params import SamplingParams

# Rollout schemas
from rollout_server.schemas.rollout import (
    RolloutStatus,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    InitResponse,
)

# Completions schemas
from rollout_server.schemas.completions import (
    CompletionsChoice,
    CompletionsRequest,
    CompletionsResponse,
)

__all__ = [
    # Constants
    "VALID_MESSAGE_ROLES",
    "MAX_TOKENS_LIMIT",
    # Messages
    "Message",
    "ToolCall",
    "ToolCallFunction",
    # Tool definitions (verl-compatible)
    "OpenAIFunctionPropertySchema",
    "OpenAIFunctionParametersSchema",
    "OpenAIFunctionSchema",
    "OpenAIFunctionToolSchema",
    "OpenAIFunctionParsedSchema",
    "OpenAIFunctionCallSchema",
    "OpenAIFunctionToolCall",
    "ToolResponse",
    # Params
    "SamplingParams",
    # Rollout
    "RolloutStatus",
    "RolloutMetrics",
    "RolloutRequest",
    "RolloutResponse",
    "InitResponse",
    # Completions
    "CompletionsChoice",
    "CompletionsRequest",
    "CompletionsResponse",
]

