"""RolloutSession - Manages rollout session with CORRECT response_mask tracking.

This module implements the reference pattern from docs/rollout_server.md Section 4.2
for calculating response_mask in multi-turn conversations.

CRITICAL IMPLEMENTATION NOTE:
This class demonstrates the CORRECT way to calculate response_mask for multi-turn
conversations. The response_mask is essential for PPO training correctness - incorrect
masks corrupt training data.

Key Pattern (from docs/rollout_server.md:305-350):
- Track last_prompt_length between LLM calls
- Calculate num_new_tokens = current_prompt - last_prompt
- response_mask = [0] * num_new_tokens (for tool outputs)
- Update last_prompt_length after each LLM call

Reference: docs/rollout_server.md Section 4 (Response Mask Management)
"""

from __future__ import annotations

import logging
import warnings
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import httpx

from rollout_server.schemas import CompletionsRequest, CompletionsResponse, Message

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class RolloutSession:
    """Manages a single rollout session with CORRECT response_mask tracking.

    This class implements the reference pattern from docs/rollout_server.md:305-350
    for calculating response_mask in multi-turn conversations.

    **Why This Matters:**
    - response_mask determines which tokens participate in PPO training loss
    - mask=1: LLM-generated tokens (used for policy gradient)
    - mask=0: Tool/system tokens (excluded from loss)

    **Without explicit masks**, the training cluster falls back to fragile diff-based
    inference that WILL FAIL if you:
    - Truncate context (remove old messages)
    - Summarize conversation history
    - Reorder messages
    - Re-tokenize with different settings

    **Consequence of Incorrect Masks:**
    - False positives (LLM tokens marked as 0): Model can't learn → training stagnates
    - False negatives (tool tokens marked as 1): Model learns to predict tool outputs → training diverges

    Reference: docs/rollout_server.md Section 4.1-4.3
    """

    def __init__(
        self,
        rollout_id: str,
        tokenizer: "PreTrainedTokenizerBase",
        server_url: str,
        http_client: Optional[httpx.AsyncClient] = None,
        callback_api_key: Optional[str] = None
    ) -> None:
        """Initialize a rollout session.

        Args:
            rollout_id: Unique rollout identifier (UUID)
            tokenizer: Tokenizer instance (must match trainer's tokenizer!)
            server_url: Trainer's /v1/completions endpoint URL
            http_client: Optional httpx.AsyncClient for HTTP requests
            callback_api_key: Optional API key for authenticating callbacks to server_url
        """
        self.rollout_id = rollout_id
        self.tokenizer = tokenizer
        self.server_url = server_url
        self._owns_client = http_client is None  # Track if we created the client
        self.http_client = http_client or httpx.AsyncClient(timeout=300.0)
        self.callback_api_key = callback_api_key
        self.messages: List[Dict[str, Any]] = []
        self.last_prompt_length = 0  # CRITICAL for response_mask calculation
        self.turn_count = 0

        logger.info(f"[{rollout_id}] Created RolloutSession with server_url={server_url}")

    async def __aenter__(self) -> "RolloutSession":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.close()

    async def call_llm(self, sampling_params: dict) -> CompletionsResponse:
        """Call LLM with CORRECT response_mask calculation.

        This is the CRITICAL method that demonstrates correct mask handling.

        **Implementation Pattern** (from docs/rollout_server.md:305-350):
        1. Tokenize current messages to get prompt length
        2. Calculate mask for tokens added since last call
           - If last_prompt_length > 0: new tokens are tool outputs (mask=0)
           - If last_prompt_length == 0: first turn, no mask needed
        3. Call trainer's /v1/completions with EXPLICIT mask
        4. Update last_prompt_length with current_prompt + LLM response

        Args:
            sampling_params: Temperature, top_p, max_tokens, etc.

        Returns:
            CompletionsResponse from trainer with token_ids and logprobs

        Reference: docs/rollout_server.md Section 4.2
        """
        # 1. Tokenize current messages to get prompt length
        current_prompt = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True
        )
        current_prompt_length = len(current_prompt)

        # 2. Calculate mask for tokens added since last call (CRITICAL!)
        if self.last_prompt_length > 0:
            # Tokens added between calls = tool outputs from previous turn
            num_new_tokens = current_prompt_length - self.last_prompt_length

            # CRITICAL: Negative token count indicates serious error
            if num_new_tokens < 0:
                raise ValueError(
                    f"[{self.rollout_id}] Negative token count detected: "
                    f"current_prompt_length={current_prompt_length}, "
                    f"last_prompt_length={self.last_prompt_length}. "
                    f"This indicates message deletion, context truncation, or tokenizer mismatch. "
                    f"Context truncation is not supported by RolloutSession - "
                    f"use explicit mask tracking instead."
                )

            if num_new_tokens > 0:
                response_mask = [0] * num_new_tokens  # Tool outputs = mask 0
                logger.debug(
                    f"[{self.rollout_id}] Turn {self.turn_count + 1}: "
                    f"Calculated response_mask for {num_new_tokens} tool output tokens"
                )
            else:
                # num_new_tokens == 0: No new tokens added (unusual but valid)
                response_mask = None
                logger.debug(
                    f"[{self.rollout_id}] Turn {self.turn_count + 1}: "
                    f"No new tokens added since last call (num_new_tokens=0)"
                )
        else:
            # First turn - no tool outputs yet
            response_mask = None
            logger.debug(f"[{self.rollout_id}] Turn 1: No response_mask (first turn)")

        # 3. Call trainer's /v1/completions with EXPLICIT mask
        request = CompletionsRequest(
            rollout_id=self.rollout_id,
            messages=[Message(**msg) for msg in self.messages],
            response_mask=response_mask,  # CRITICAL!
            **sampling_params
        )

        logger.info(
            f"[{self.rollout_id}] Calling /v1/completions: "
            f"prompt_length={current_prompt_length}, "
            f"response_mask={'None' if response_mask is None else f'[{len(response_mask)} zeros]'}"
        )

        # Build headers with optional authentication
        headers = {}
        if self.callback_api_key:
            headers["Authorization"] = f"Bearer {self.callback_api_key}"

        response = await self.http_client.post(
            f"{self.server_url}/v1/completions",
            json=request.model_dump(),
            headers=headers
        )
        response.raise_for_status()
        response_data = response.json()
        completion_response = CompletionsResponse(**response_data)

        # 4. Update tracking with LLM response length
        llm_token_count = len(completion_response.token_ids)
        self.last_prompt_length = current_prompt_length + llm_token_count
        self.turn_count += 1

        logger.info(
            f"[{self.rollout_id}] Turn {self.turn_count}: "
            f"Received {llm_token_count} LLM tokens, "
            f"updated last_prompt_length to {self.last_prompt_length}"
        )

        return completion_response

    def append_tool_outputs(self, tool_results: List[dict]):
        """Append tool results to conversation.

        IMPORTANT: Don't update last_prompt_length here!
        It will be calculated in next call_llm() call.

        Args:
            tool_results: List of tool message dicts with role="tool"

        Reference: docs/rollout_server.md Section 4.2
        """
        self.messages.extend(tool_results)
        logger.debug(
            f"[{self.rollout_id}] Appended {len(tool_results)} tool results. "
            f"Total messages: {len(self.messages)}"
        )

    def append_assistant_message(self, message: dict):
        """Append assistant message to conversation.

        Args:
            message: Assistant message dict with role="assistant"
        """
        self.messages.append(message)
        logger.debug(f"[{self.rollout_id}] Appended assistant message")

    async def close(self):
        """Close HTTP client only if we created it (not shared).

        This method is idempotent - safe to call multiple times.
        """
        if self._owns_client and self.http_client:
            await self.http_client.aclose()
            self.http_client = None  # Prevent double-close


# =============================================================================
# DEPRECATED: Alternative Implementation
# =============================================================================


class RolloutSessionExplicit:
    """DEPRECATED: Alternative implementation with explicit token tracking.

    ⚠️  WARNING: This class is deprecated due to incorrect token counting logic.
    ⚠️  DO NOT USE. Use RolloutSession instead.

    KNOWN BUGS:
    - Uses tokenizer.encode() instead of apply_chat_template()
    - Token counts will not match chat format, causing mask length mismatches
    - Will corrupt training data if used in production

    This class is kept for reference only and will be removed in a future version.
    If you need explicit token tracking, implement it correctly using
    apply_chat_template() as shown in RolloutSession.

    Reason for deprecation:
    The original implementation in append_tool_outputs() used:
        token_ids = self.tokenizer.encode(content, add_special_tokens=False)
    This does not account for chat template formatting, leading to incorrect
    token counts and response_mask length mismatches.

    Reference: See RolloutSession for correct implementation pattern.
    """

    def __init__(self, *args, **kwargs):
        """Emit deprecation warning and raise NotImplementedError."""
        warnings.warn(
            "RolloutSessionExplicit is deprecated due to token counting bugs. "
            "Use RolloutSession instead. "
            "See session.py docstring for details on why this class is unsafe.",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError(
            "RolloutSessionExplicit is deprecated. Use RolloutSession instead."
        )
