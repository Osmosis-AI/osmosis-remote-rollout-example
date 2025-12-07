"""Message and tool-related schemas for the Remote Rollout Protocol.

These schemas define the message structures used in conversations,
including tool calls and tool definitions.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rollout_server.schemas.constants import VALID_MESSAGE_ROLES


# =============================================================================
# Tool Definition Schemas (GET /tools endpoint)
# =============================================================================


class ToolFunction(BaseModel):
    """Function definition within a tool, following OpenAI tool format."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    """Tool definition following OpenAI tools format.

    Example:
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["a", "b"]
                }
            }
        }
    """

    type: str = "function"
    function: ToolFunction


class ToolsResponse(BaseModel):
    """Response from RolloutServer GET /tools endpoint.

    Contains the list of available tools that the LLM can use during rollout.
    This is fetched once at worker startup and used for apply_chat_template().

    Specification: docs/rollout_server.md Section 3.0
    """

    tools: List[ToolDefinition] = Field(default_factory=list)


# =============================================================================
# Message and Chat Schemas
# =============================================================================


class ToolCallFunction(BaseModel):
    """Function call details within a tool call."""

    name: str = Field(..., min_length=1, max_length=256)
    arguments: Union[str, Dict[str, Any]] = Field(...)


class ToolCall(BaseModel):
    """Tool call structure for assistant messages."""

    id: str = Field(..., min_length=1, max_length=256)
    type: Literal["function"] = "function"
    function: ToolCallFunction


class Message(BaseModel):
    """Chat message structure compatible with OpenAI-style payloads.

    Specification: docs/rollout_server.md Section 3.1, 3.2
    """

    role: str = Field(..., description="Message role: system, user, assistant, tool")
    content: Union[str, List[Dict[str, Any]], None] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = Field(None, max_length=256)

    # Allow optional keys like `name` or vendor-specific fields
    model_config = ConfigDict(extra="allow")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate role is one of the allowed values."""
        if v not in VALID_MESSAGE_ROLES:
            raise ValueError(
                f"Invalid role '{v}'. Must be one of: {sorted(VALID_MESSAGE_ROLES)}"
            )
        return v

