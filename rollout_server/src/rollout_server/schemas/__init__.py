"""Pydantic schemas for the remote rollout protocol.

This example repo intentionally does NOT define its own copies of protocol
schemas. Instead, it imports the single source of truth from the Osmosis SDK:
`osmosis_ai.rollout.core.schemas`.
"""

from osmosis_ai.rollout.core.schemas import (
    CompletionUsage,
    CompletionsChoice,
    CompletionsRequest,
    CompletionsResponse,
    InitResponse,
    MessageDict,
    OpenAIFunctionCallSchema,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolCall,
    OpenAIFunctionToolSchema,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStatus,
    SamplingParamsDict,
    ToolResponse,
)

__all__ = [
    # Type aliases
    "MessageDict",
    "SamplingParamsDict",
    # Rollout
    "RolloutStatus",
    "RolloutMetrics",
    "RolloutRequest",
    "RolloutResponse",
    "InitResponse",
    # Completions
    "CompletionUsage",
    "CompletionsChoice",
    "CompletionsRequest",
    "CompletionsResponse",
    # Tool schemas
    "OpenAIFunctionPropertySchema",
    "OpenAIFunctionParametersSchema",
    "OpenAIFunctionSchema",
    "OpenAIFunctionToolSchema",
    "OpenAIFunctionParsedSchema",
    "OpenAIFunctionCallSchema",
    "OpenAIFunctionToolCall",
    "ToolResponse",
]

