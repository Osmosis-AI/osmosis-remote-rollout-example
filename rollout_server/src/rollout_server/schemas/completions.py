"""Completions callback schemas for the Remote Rollout Protocol.

These schemas define the contract for RolloutServer -> Trainer communication
via the training-side /v1/chat/completions endpoint.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from rollout_server.schemas.constants import MAX_TOKENS_LIMIT

MessageDict = Dict[str, Any]


class CompletionsChoice(BaseModel):
    """Single choice in completions response.

    Specification: docs/rollout_server.md Section 3.2
    """

    index: int = 0
    message: MessageDict
    finish_reason: str = "stop"


class CompletionsRequest(BaseModel):
    """OpenAI-compatible completions request with rollout_id extension.

    RolloutServer sends this to trainer's /v1/chat/completions endpoint.
    The rollout_id is used to route the request to the correct session.

    Tokenization and training-time token tracking are handled on the trainer side.
    RolloutServer should send the full message list representing the current
    conversation state.

    Specification: docs/rollout_server.md Section 3.2
    """

    model: str = "default"  # Ignored - uses loaded model
    messages: List[MessageDict]
    rollout_id: str  # Custom extension for session routing

    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    # NOTE: In current verl implementation, max_tokens is IGNORED by trainer.
    # verl calculates max_tokens = config.max_model_len - len(prompt_ids) internally.
    # Kept for protocol compatibility.
    max_tokens: int = Field(default=512, ge=1, le=MAX_TOKENS_LIMIT)
    stop: Optional[List[str]] = None
    logprobs: bool = True


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

