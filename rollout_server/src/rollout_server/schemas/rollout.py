"""Rollout init/completion schemas for the Remote Rollout Protocol.

These schemas define the contract between the training-side agent loop and
RolloutServer. The protocol uses an async-init flow:

- Training -> RolloutServer: POST /init (returns 202 with tools)
- RolloutServer -> Training: POST {server_url}/v1/chat/completions (LLM callback)
- RolloutServer -> Training: POST {server_url}/v1/rollout/completed (completion callback)
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# Protocol transmission uses dicts for messages and completion params to remain
# compatible with the OpenAI message format (including vendor-specific fields).
MessageDict = Dict[str, Any]
CompletionParamsDict = Dict[str, Any]


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
# Rollout Request/Response
# =============================================================================


class RolloutRequest(BaseModel):
    """Request sent to POST /init to start a rollout (async-init protocol).

    The rollout continues asynchronously:
    - RolloutServer calls back to server_url/v1/chat/completions for LLM generation.
    - RolloutServer posts the final RolloutResponse to server_url/v1/rollout/completed.

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

    rollout_id: str = Field(..., description="Unique rollout identifier")
    server_url: str = Field(
        ...,
        description=(
            "Trainer base URL. RolloutServer calls {server_url}/v1/chat/completions "
            "and posts completion to {server_url}/v1/rollout/completed"
        ),
    )
    messages: List[MessageDict] = Field(..., min_length=1)
    completion_params: CompletionParamsDict
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
    api_key: Optional[str] = None  # Bearer token for authenticating callbacks to server_url

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
    """Payload posted by RolloutServer to server_url/v1/rollout/completed.

    Contains the final messages and status. Token tracking for training is
    accumulated on the training side via /v1/chat/completions requests.

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
    final_messages: List[MessageDict] = Field(default_factory=list)
    finish_reason: Optional[str] = None  # See docstring for standard values
    error_message: Optional[str] = None
    metrics: Optional[RolloutMetrics] = None  # Contains num_llm_calls for num_turns
    extra_fields: Dict[str, Any] = Field(default_factory=dict)


class InitResponse(BaseModel):
    """Response body for POST /init (202 Accepted).

    Provides the tool definitions available for this rollout.
    """

    rollout_id: str
    tools: List[Dict[str, Any]] = Field(default_factory=list)

