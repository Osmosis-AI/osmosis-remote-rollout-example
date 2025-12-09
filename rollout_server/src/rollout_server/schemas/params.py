"""Sampling parameters schema for the Remote Rollout Protocol.

This module defines the SamplingParams model used to configure LLM generation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from rollout_server.schemas.constants import MAX_TOKENS_LIMIT


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
    # NOTE: In the current verl implementation, this field is IGNORED by the trainer.
    # verl's vLLMHttpServer.generate() calculates max_tokens internally as:
    #   max_tokens = config.max_model_len - len(prompt_ids)
    # This field is kept for protocol compatibility and future implementations
    # that may support external max_tokens control.
    max_tokens: int = Field(
        default=512,
        ge=1,
        le=MAX_TOKENS_LIMIT,
        description=f"Maximum tokens to generate (1-{MAX_TOKENS_LIMIT}). "
                    f"NOTE: Currently ignored by verl trainer; uses config.max_model_len instead."
    )
    stop: Optional[List[str]] = None
    logprobs: bool = True

