"""Example RolloutServer built on the Osmosis remote rollout Python SDK.

This module keeps only example "agent logic" (a calculator loop) and delegates
all protocol handling to the Osmosis SDK.
"""

from __future__ import annotations

import os
import time

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
        "  pip install osmosis-ai[server]==0.2.7\n"
    ) from e

from tools import CALCULATOR_TOOL_SCHEMAS, execute_calculator_calls


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
            result = await ctx.chat(messages, **ctx.request.completion_params)
            messages.append(result.message)

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
        else:
            finish_reason = "max_turns"

        return ctx.complete(messages, finish_reason=finish_reason)


# Agent loop instance for CLI usage (osmosis serve -m server:agent_loop)
agent_loop = CalculatorAgentLoop()

# FastAPI application provided by the SDK.
app = create_app(agent_loop)


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
