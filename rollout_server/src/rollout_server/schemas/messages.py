"""Message and tool-related schemas for the Remote Rollout Protocol.

These schemas define the message structures used in conversations,
including tool calls and tool definitions.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rollout_server.schemas.constants import VALID_MESSAGE_ROLES


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

