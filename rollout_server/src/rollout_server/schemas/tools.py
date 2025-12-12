"""OpenAI-compatible tool definition schemas.

Re-exports tool schemas from verl.tools.schemas for convenience.
These schemas follow the OpenAI function calling format.
"""

from verl.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolCall,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

__all__ = [
    "OpenAIFunctionPropertySchema",
    "OpenAIFunctionParametersSchema",
    "OpenAIFunctionSchema",
    "OpenAIFunctionToolSchema",
    "OpenAIFunctionParsedSchema",
    "OpenAIFunctionCallSchema",
    "OpenAIFunctionToolCall",
    "ToolResponse",
]
