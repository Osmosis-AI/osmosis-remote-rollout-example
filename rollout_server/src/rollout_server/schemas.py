"""Pydantic schemas for Remote Rollout Protocol.

These schemas define the public API contract between RolloutServer and the training cluster.
Based on the Remote Rollout Protocol specification in docs/rollout_server.md.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


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

    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop: Optional[List[str]] = None
    logprobs: Optional[int] = 1


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

    rollout_id: str  # Unique rollout identifier (UUID)
    server_url: str  # Trainer's /v1/completions endpoint URL
    messages: List[Message]  # Initial conversation messages
    sampling_params: SamplingParams
    tool_server_url: Optional[str] = None
    max_turns: int = 10
    max_tokens_total: int = 8192
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Callback authentication
    callback_api_key: Optional[str] = None  # API key for authenticating callbacks to server_url

    # Tokenizer information for validation
    tokenizer_name: Optional[str] = None  # e.g., "Qwen/Qwen3-8B"
    tokenizer_revision: Optional[str] = None  # e.g., "main" or git commit hash


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
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop: Optional[List[str]] = None
    logprobs: bool = True

    # CRITICAL: Explicit mask for tokens added since last LLM call
    # Must be provided by RolloutServer for turns after the first turn
    # See class docstring for detailed requirements
    response_mask: Optional[List[int]] = None


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
