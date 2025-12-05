"""Pydantic schemas for Remote Rollout Protocol.

These schemas define the public API contract between RolloutServer and the training cluster.
Based on the Remote Rollout Protocol specification in docs/rollout_server.md.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Message and Chat Schemas
# =============================================================================


class Message(BaseModel):
    """Chat message structure compatible with OpenAI-style payloads.

    Specification: docs/rollout_server.md Section 3.1, 3.2
    """

    role: str  # system, user, assistant, tool
    content: Union[str, List[Dict[str, Any]]]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    # Allow optional keys like `name` or vendor-specific fields
    model_config = ConfigDict(extra="allow")


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
        le=32768,
        description="Maximum tokens to generate (1-32768)"
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

    Specification: Aligns with TrainGate's RolloutMetrics schema
    """

    total_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    num_llm_calls: int = 0
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
    the entire agent loop. RolloutServer calls back to server_url/v1/completions
    for LLM generation.

    Specification: docs/rollout_server.md Section 3.1
    """

    rollout_id: str = Field(..., description="Unique rollout identifier (UUID format)")
    server_url: str = Field(..., description="Trainer's /v1/completions endpoint URL")
    messages: List[Message] = Field(..., min_length=1)  # Initial conversation messages (at least 1)
    sampling_params: SamplingParams
    tool_server_url: Optional[str] = None

    max_turns: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum agent loop iterations (1-100)"
    )
    max_tokens_total: int = Field(
        default=8192,
        ge=1,
        le=1_000_000,
        description="Maximum total tokens (1-1,000,000)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Callback authentication
    callback_api_key: Optional[str] = None  # API key for authenticating callbacks to server_url

    # Tokenizer information for validation
    tokenizer_name: Optional[str] = None  # e.g., "Qwen/Qwen3-8B"
    tokenizer_revision: Optional[str] = None  # e.g., "main" or git commit hash

    @field_validator('rollout_id')
    @classmethod
    def validate_rollout_id(cls, v: str) -> str:
        """Validate rollout_id is non-empty.

        Note: We accept any non-empty string format to be compatible with traingate,
        which uses formats like "{job_id}-step{step}-idx{index}-{uuid[:8]}".
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
        """Validate server_url is a valid HTTP/HTTPS URL."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError(
                f"server_url must be an HTTP/HTTPS URL, got: {v}"
            )
        # Remove trailing slash for consistency
        return v.rstrip('/')


class RolloutResponse(BaseModel):
    """Response from RolloutServer after completing the rollout.

    Contains the final messages and status. Token tracking data is accumulated
    in SessionManager via /v1/completions requests during the rollout.

    Specification: docs/rollout_server.md Section 3.1
    """

    rollout_id: str  # Echoed back for correlation
    status: RolloutStatus  # COMPLETED or ERROR
    final_messages: List[Message] = Field(default_factory=list)
    finish_reason: Optional[str] = None  # "stop", "max_turns", etc.
    error_message: Optional[str] = None
    metrics: Optional[RolloutMetrics] = None
    extra_fields: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Completions Request/Response (RolloutServer -> Trainer /v1/completions)
# =============================================================================


class CompletionsChoice(BaseModel):
    """Single choice in completions response.

    Specification: docs/rollout_server.md Section 3.2
    """

    index: int = 0
    message: Message
    finish_reason: str = "stop"


class CompletionsRequest(BaseModel):
    """OpenAI-compatible completions request with rollout_id extension.

    RolloutServer sends this to trainer's /v1/completions endpoint.
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
    max_tokens: int = Field(default=512, ge=1, le=32768)
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
        if len(v) > 100000:
            import logging
            logging.getLogger(__name__).warning(
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
