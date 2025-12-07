"""Pydantic schemas for Remote Rollout Protocol.

These schemas define the public API contract between RolloutServer and the training cluster.
Based on the Remote Rollout Protocol specification in docs/rollout_server.md.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Valid message roles
VALID_MESSAGE_ROLES = {"system", "user", "assistant", "tool", "function"}

# Maximum tokens limit for sampling params
MAX_TOKENS_LIMIT = 32768



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


# =============================================================================
# Sampling Parameters
# =============================================================================


class SamplingParams(BaseModel):
    """Sampling parameters for LLM generation.

    Specification: docs/rollout_server.md Section 3.1
    """

    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0)"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling top_p (0.0-1.0)"
    )
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=MAX_TOKENS_LIMIT,
        description=f"Maximum tokens to generate (1-{MAX_TOKENS_LIMIT})"
    )
    stop: Optional[List[str]] = None
    logprobs: bool = True


# =============================================================================
# Rollout Status and Metrics
# =============================================================================


class RolloutStatus(str, Enum):
    """Status returned by RolloutServer indicating rollout outcome."""

    COMPLETED = "COMPLETED"  # Rollout finished successfully
    ERROR = "ERROR"  # Rollout failed with error


class RolloutMetrics(BaseModel):
    """Metrics from rollout execution.

    num_turns Semantics
    ───────────────────
    `num_llm_calls` counts the number of LLM generation calls made during the rollout.
    This value is used by OsmosisAgentLoop to populate `AgentLoopOutput.num_turns`,
    which is consistent with verl's training pipeline expectations.

    Example:
        User: "Calculate 15*23"
        Turn 1: LLM generates "I'll use calculator" + tool_call  → num_llm_calls=1
        Turn 2: LLM generates "The result is 345"                → num_llm_calls=2

    Note: This differs from verl's local ToolAgentLoop which tracks
    `assistant_turns` and `user_turns` separately. For remote rollout,
    we use a single counter (num_llm_calls) for simplicity, as RolloutServer
    has full control over termination logic.

    Reference: traingate/integrations/verl/schemas.py - RolloutMetrics
    """

    total_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    num_llm_calls: int = 0  # Number of LLM generation calls (== num_turns in AgentLoopOutput)
    num_tool_calls: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    max_context_tokens: int = 0


# =============================================================================
# Rollout Request/Response (OsmosisAgentLoop <-> RolloutServer)
# =============================================================================


class RolloutRequest(BaseModel):
    """Request sent to POST /rollout to initiate a complete rollout.

    OsmosisAgentLoop sends this request and waits for RolloutServer to complete
    the entire agent loop. RolloutServer calls back to server_url/v1/chat/completions
    for LLM generation.

    Termination Control
    ───────────────────
    `max_turns` and `max_tokens_total` are **advisory parameters**. RolloutServer
    has full control over termination logic and may implement more sophisticated
    strategies (e.g., budget-based, task-completion-based, error-rate-based).

    This differs from verl's local ToolAgentLoop which uses `max_assistant_turns`
    and `max_user_turns` as separate counters. Remote rollout intentionally uses
    a single `max_turns` for simplicity, as RolloutServer can implement any
    termination logic internally.

    metadata Field
    ──────────────
    The `metadata` field allows passing optional fine-grained control parameters
    to RolloutServer. RolloutServer may choose to honor these or ignore them.

    Supported optional metadata keys (implementation-dependent):
    - `max_assistant_turns`: Max LLM generation calls (int)
    - `max_user_turns`: Max tool/user input turns (int)
    - `termination_strategy`: Custom termination logic identifier (str)
    - `budget_tokens`: Token budget for the entire rollout (int)

    Example:
        metadata = {
            "max_assistant_turns": 5,
            "max_user_turns": 5,
            "termination_strategy": "task_completion"
        }

    Note: These are optional hints. RolloutServer implementations are not
    required to support them. Core termination is controlled by `max_turns`
    and `max_tokens_total`.

    Specification: docs/rollout_server.md Section 3.1
    Reference: traingate/integrations/verl/schemas.py - RolloutRequest
    """

    rollout_id: str = Field(..., description="Unique rollout identifier (UUID format)")
    server_url: str = Field(..., description="Trainer's /v1/chat/completions endpoint URL")
    messages: List[Message] = Field(..., min_length=1)  # Initial conversation messages (at least 1)
    sampling_params: SamplingParams
    tool_server_url: Optional[str] = None

    max_turns: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Advisory: max LLM calls (RolloutServer may override)"
    )
    max_tokens_total: int = Field(
        default=8192,
        ge=1,
        le=1_000_000,
        description="Advisory: max total tokens"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional fine-grained control parameters"
    )

    # Callback authentication
    callback_api_key: Optional[str] = None  # API key for authenticating callbacks to server_url

    # Tokenizer information for validation
    tokenizer_name: Optional[str] = None  # e.g., "Qwen/Qwen3-8B"
    tokenizer_revision: Optional[str] = None  # e.g., "main" or git commit hash

    @field_validator('rollout_id')
    @classmethod
    def validate_rollout_id(cls, v: str) -> str:
        """Validate rollout_id is non-empty.

        Note: We accept any non-empty string format to be compatible with various
        training systems that use formats like "{job_id}-step{step}-idx{index}-{uuid[:8]}".
        """
        if not v or not v.strip():
            raise ValueError("rollout_id must be non-empty")
        # Limit length to prevent DoS
        if len(v) > 256:
            raise ValueError("rollout_id must be at most 256 characters")
        return v

    @field_validator('server_url')
    @classmethod
    def validate_server_url(cls, v: str) -> str:
        """Validate server_url is a valid HTTP/HTTPS URL.

        Note: This server is designed for internal training infrastructure use.
        The server_url is provided by trusted training components, not external users.
        Private/internal IPs are allowed since trainers typically run on private networks.
        """
        if not v.startswith(('http://', 'https://')):
            raise ValueError(
                f"server_url must be an HTTP/HTTPS URL, got: {v}"
            )

        # Remove trailing slash for consistency
        return v.rstrip('/')


class RolloutResponse(BaseModel):
    """Response from RolloutServer after completing the rollout.

    Contains the final messages and status. Token tracking data is accumulated
    in SessionManager via /v1/chat/completions requests during the rollout.

    num_turns in AgentLoopOutput
    ────────────────────────────
    OsmosisAgentLoop uses `metrics.num_llm_calls` from this response to populate
    `AgentLoopOutput.num_turns`. This ensures consistency with verl's training
    pipeline which expects `num_turns` to represent the number of LLM generations.

    finish_reason Values
    ────────────────────
    Standard values (RolloutServer should use these for consistency):
    - "stop": LLM generated response without tool calls (natural completion)
    - "max_turns": Reached max_turns limit
    - "max_tokens": Reached max_tokens_total limit
    - "error": Rollout failed (check error_message)
    - Custom values: RolloutServer may define additional finish reasons

    Specification: docs/rollout_server.md Section 3.1
    Reference: traingate/integrations/verl/schemas.py - RolloutResponse
    """

    rollout_id: str  # Echoed back for correlation
    status: RolloutStatus  # COMPLETED or ERROR
    final_messages: List[Message] = Field(default_factory=list)
    finish_reason: Optional[str] = None  # See docstring for standard values
    error_message: Optional[str] = None
    metrics: Optional[RolloutMetrics] = None  # Contains num_llm_calls for num_turns
    extra_fields: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Completions Request/Response (RolloutServer -> Trainer /v1/chat/completions)
# =============================================================================


class CompletionsChoice(BaseModel):
    """Single choice in completions response.

    Specification: docs/rollout_server.md Section 3.2
    """

    index: int = 0
    message: Message
    finish_reason: str = "stop"


# Maximum response_mask length before warning
MAX_RESPONSE_MASK_LENGTH = 100000


class CompletionsRequest(BaseModel):
    """OpenAI-compatible completions request with rollout_id extension.

    RolloutServer sends this to trainer's /v1/chat/completions endpoint.
    The rollout_id is used to route the request to the correct session.

    CRITICAL FIELD: response_mask
    ─────────────────────────────
    For multi-turn conversations with tools, RolloutServer MUST provide explicit
    response_mask values to indicate which tokens were added since the last LLM call.

    - response_mask[i] = 0: Token i is a tool/system output (not LLM-generated)
    - response_mask[i] = 1: Token i is LLM-generated (participates in PPO loss)

    When to provide response_mask:
    - Turn 1 (no tools yet): response_mask=None (OK to omit)
    - Turn 2+ (after tool execution): response_mask=[0] * num_tool_output_tokens

    Why this matters:
    Without explicit masks, the trainer falls back to fragile diff-based inference
    that WILL BREAK if you perform context truncation, summarization, or reordering.
    Incorrect masks corrupt training data and cause PPO to diverge or stagnate.

    Specification: docs/rollout_server.md Section 3.2, Section 4 (Response Mask)
    """

    model: str = "default"  # Ignored - uses loaded model
    messages: List[Message]
    rollout_id: str  # Custom extension for session routing

    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=MAX_TOKENS_LIMIT)
    stop: Optional[List[str]] = None
    logprobs: bool = True

    # CRITICAL: Explicit mask for tokens added since last LLM call
    # Must be provided by RolloutServer for turns after the first turn
    # See class docstring for detailed requirements
    response_mask: Optional[List[int]] = Field(
        default=None,
        description="Mask for tokens added since last LLM call (values must be 0 or 1)"
    )

    @field_validator('response_mask')
    @classmethod
    def validate_response_mask(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        """Validate response_mask contains only 0 or 1."""
        if v is None:
            return None

        # Check all values are 0 or 1
        invalid_values = [x for x in v if x not in (0, 1)]
        if invalid_values:
            raise ValueError(
                f"response_mask must contain only 0 or 1, "
                f"found invalid values: {set(invalid_values)}"
            )

        # Warn if mask is very large (possible DoS)
        if len(v) > MAX_RESPONSE_MASK_LENGTH:
            logger.warning(
                f"response_mask is very large ({len(v)} tokens), "
                f"possible performance issue or DoS attempt"
            )

        return v


class CompletionsResponse(BaseModel):
    """Extended OpenAI completions response with token tracking.

    Includes standard OpenAI fields plus extensions for token IDs and logprobs
    needed for training.

    Specification: docs/rollout_server.md Section 3.2
    """

    id: str  # Request ID
    object: str = "chat.completion"
    created: int  # Unix timestamp
    model: str = "default"
    choices: List[CompletionsChoice]

    # Extensions for training (trainer-specific)
    token_ids: List[int]  # Response token IDs
    logprobs: List[float]  # Log probabilities for response tokens
    prompt_token_ids: List[int]  # Prompt token IDs (for verification)
