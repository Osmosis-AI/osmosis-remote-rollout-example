"""Completions request/response schemas for the Remote Rollout Protocol.

These schemas define the contract for RolloutServer -> Trainer communication
via the /v1/chat/completions endpoint.
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from rollout_server.schemas.constants import MAX_TOKENS_LIMIT, MAX_RESPONSE_MASK_LENGTH
from rollout_server.schemas.messages import Message


logger = logging.getLogger(__name__)


class CompletionsChoice(BaseModel):
    """Single choice in completions response.

    Specification: docs/rollout_server.md Section 3.2
    """

    index: int = 0
    message: Message
    finish_reason: str = "stop"


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

