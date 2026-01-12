"""Example RolloutServer built on the Osmosis remote rollout Python SDK.

This module keeps only example "agent logic" (a calculator loop) and delegates
all protocol handling to the Osmosis SDK.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Dict, Any, Optional

try:
    from osmosis_ai.rollout import (
        RolloutAgentLoop,
        RolloutContext,
        RolloutResult,
        RolloutRequest,
        create_app,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Cannot import `osmosis_ai.rollout`. Install the Osmosis SDK:\n\n"
        "  pip install osmosis-ai[server]>=0.2.9\n"
    ) from e

from tools import CALCULATOR_TOOL_SCHEMAS, execute_calculator_calls
from rewards import compute_reward

# Optional: Debug logging only, can be ignored
logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def get_last_assistant_content(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Get the content of the last assistant message.

    Args:
        messages: List of conversation messages.

    Returns:
        The content of the last assistant message, or None if not found.
    """
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return None


def compute_reward_from_messages(
    messages: List[Dict[str, Any]], ground_truth: Optional[str]
) -> Optional[float]:
    """Compute reward from messages if ground_truth is available.

    Args:
        messages: Final conversation messages.
        ground_truth: Expected answer from metadata.

    Returns:
        Reward score (0.0 or 1.0), or None if ground_truth not provided.
    """
    if not ground_truth:
        return None

    solution_str = get_last_assistant_content(messages)
    if not solution_str:
        # Optional: Debug logging only, can be ignored
        logger.debug("No assistant message found for reward computation")
        return 0.0

    try:
        reward = compute_reward(solution_str, ground_truth)
        # Optional: Debug logging only, can be ignored
        logger.info(f"Computed reward: {reward} (ground_truth={ground_truth})")
        return reward
    except Exception as e:
        # Optional: Debug logging only, can be ignored
        logger.warning(f"Reward computation failed: {e}")
        return None


# =============================================================================
# Agent Loop
# =============================================================================


class CalculatorAgentLoop(RolloutAgentLoop):
    """Minimal agent loop demonstrating tool calls with the remote rollout SDK."""

    name = "calculator"

    def get_tools(self, request: RolloutRequest):
        return CALCULATOR_TOOL_SCHEMAS

    async def run(self, ctx: RolloutContext) -> RolloutResult:
        messages = list(ctx.request.messages)
        finish_reason = "stop"
        total_completion_tokens = 0

        for _turn in range(ctx.request.max_turns):
            # Optional: Debug logging only, can be ignored
            # Uses SDK's debug logging, no-op if ROLLOUT_DEBUG_DIR not set
            ctx.log_event(
                "pre_llm",
                turn=_turn,
                num_messages=len(messages),
                messages_summary=[
                    {
                        "index": i,
                        "role": msg.get("role", "?"),
                        "content_preview": str(msg.get("content", ""))[:100],
                        "has_tool_calls": "tool_calls" in msg,
                        "tool_call_id": msg.get("tool_call_id"),
                    }
                    for i, msg in enumerate(messages)
                ],
            )

            result = await ctx.chat(messages, **ctx.request.completion_params)
            messages.append(result.message)

            # Optional: Debug logging only, can be ignored
            ctx.log_event(
                "llm_response",
                turn=_turn,
                result_message={
                    "role": result.message.get("role"),
                    "content_preview": str(result.message.get("content", ""))[:100],
                    "has_tool_calls": result.has_tool_calls,
                    "tool_calls_count": len(result.tool_calls) if result.tool_calls else 0,
                },
                finish_reason=result.finish_reason,
                num_messages_after_append=len(messages),
            )

            usage = result.usage or {}
            try:
                total_completion_tokens += int(usage.get("completion_tokens") or 0)
            except Exception:
                pass

            if total_completion_tokens >= ctx.request.max_tokens_total:
                finish_reason = "max_tokens"
                break

            if not result.has_tool_calls:
                finish_reason = result.finish_reason or "stop"
                break

            tool_calls = result.tool_calls
            tool_start = time.monotonic()
            tool_results = await execute_calculator_calls(tool_calls)
            latency_ms = (time.monotonic() - tool_start) * 1000.0

            per_call_latency = latency_ms / max(1, len(tool_calls))
            for _ in tool_calls:
                ctx.record_tool_call(latency_ms=per_call_latency)

            messages.extend(tool_results)

            # Optional: Debug logging only, can be ignored
            ctx.log_event(
                "tool_results",
                turn=_turn,
                num_tool_results=len(tool_results),
                tool_results_summary=[
                    {
                        "role": tr.get("role"),
                        "tool_call_id": tr.get("tool_call_id"),
                        "content_preview": str(tr.get("content", ""))[:50],
                    }
                    for tr in tool_results
                ],
                num_messages_after_extend=len(messages),
            )
        else:
            finish_reason = "max_turns"

        # Compute reward if ground_truth is provided in metadata
        ground_truth = ctx.request.metadata.get("ground_truth")
        reward = compute_reward_from_messages(messages, ground_truth)

        # Optional: Debug logging only, can be ignored
        ctx.log_event(
            "rollout_complete",
            finish_reason=finish_reason,
            reward=reward,
            ground_truth=ground_truth,
            total_turns=_turn + 1,
            final_messages_count=len(messages),
            last_assistant_content=get_last_assistant_content(messages),
        )

        return ctx.complete(messages, finish_reason=finish_reason, reward=reward)


# Agent loop instance for CLI usage (osmosis serve -m server:agent_loop)
agent_loop = CalculatorAgentLoop()

# FastAPI application provided by the SDK.
# Debug logging is controlled via ROLLOUT_DEBUG_DIR environment variable.
# When set, each rollout writes traces to {debug_dir}/{timestamp}/{rollout_id}.jsonl
debug_dir = os.getenv("ROLLOUT_DEBUG_DIR")
app = create_app(agent_loop, debug_dir=debug_dir)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("ROLLOUT_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("ROLLOUT_SERVER_PORT", "9000"))

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
        access_log=True,
    )
