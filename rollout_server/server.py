"""Example RolloutServer built on the Osmosis remote rollout Python SDK.

This module keeps only example "agent logic" (a calculator loop) and delegates
all protocol handling to the Osmosis SDK.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Callable, Coroutine, Dict, List

try:
    from osmosis_ai.rollout import (
        RolloutAgentLoop,
        RolloutContext,
        RolloutResult,
        RolloutRequest,
        create_app,
    )
    from osmosis_ai.rollout.core.exceptions import ToolArgumentError
    from osmosis_ai.rollout.core.schemas import (
        OpenAIFunctionParametersSchema,
        OpenAIFunctionPropertySchema,
        OpenAIFunctionSchema,
        OpenAIFunctionToolSchema,
    )
    from osmosis_ai.rollout.tools import (
        create_tool_error_result,
        create_tool_result,
        execute_tool_calls,
        get_tool_call_info,
        serialize_tool_result,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Cannot import `osmosis_ai.rollout`. Install the Osmosis SDK:\n\n"
        "  pip install osmosis-ai[server]==0.2.7\n"
    ) from e

logger = logging.getLogger(__name__)

# =============================================================================
# Calculator Tools
# =============================================================================

TOOL_DELAY_MIN_SECONDS = 0.1
TOOL_DELAY_MAX_SECONDS = 0.5

ToolFunction = Callable[..., Coroutine[Any, Any, float]]


async def add(a: float, b: float) -> float:
    """Add two numbers."""
    delay = random.uniform(TOOL_DELAY_MIN_SECONDS, TOOL_DELAY_MAX_SECONDS)
    await asyncio.sleep(delay)
    return a + b


async def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    delay = random.uniform(TOOL_DELAY_MIN_SECONDS, TOOL_DELAY_MAX_SECONDS)
    await asyncio.sleep(delay)
    return a - b


async def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    delay = random.uniform(TOOL_DELAY_MIN_SECONDS, TOOL_DELAY_MAX_SECONDS)
    await asyncio.sleep(delay)
    return round(a * b, 4)


async def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    delay = random.uniform(TOOL_DELAY_MIN_SECONDS, TOOL_DELAY_MAX_SECONDS)
    await asyncio.sleep(delay)
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


CALCULATOR_TOOLS: Dict[str, ToolFunction] = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}


async def _execute_calculator_call(tool_call: Dict[str, Any]) -> Dict[str, str]:
    """Execute a single calculator tool call."""
    try:
        tool_call_id, function_name, arguments = get_tool_call_info(tool_call)
    except ToolArgumentError as e:
        tool_call_id = e.tool_call_id or tool_call.get("id", "unknown")
        return create_tool_error_result(tool_call_id, str(e))

    tool_fn = CALCULATOR_TOOLS.get(function_name)
    if tool_fn is None:
        return create_tool_error_result(
            tool_call_id,
            f"Unknown function '{function_name}'. Available: {sorted(CALCULATOR_TOOLS.keys())}",
        )

    try:
        result = await tool_fn(**arguments)
        return create_tool_result(tool_call_id, serialize_tool_result(result))
    except TypeError as e:
        return create_tool_error_result(tool_call_id, f"Invalid arguments: {e}")
    except ValueError as e:
        return create_tool_error_result(tool_call_id, str(e))
    except Exception:
        logger.exception("Unexpected error executing %s", function_name)
        return create_tool_error_result(tool_call_id, "Tool execution failed")


async def execute_calculator_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Execute multiple calculator tool calls concurrently."""
    return await execute_tool_calls(tool_calls, _execute_calculator_call)


def _number_property(description: str) -> OpenAIFunctionPropertySchema:
    return OpenAIFunctionPropertySchema(type="number", description=description)


CALCULATOR_TOOL_SCHEMAS: List[OpenAIFunctionToolSchema] = [
    OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="add",
            description="Add two numbers",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={"a": _number_property("First number"), "b": _number_property("Second number")},
                required=["a", "b"],
            ),
        ),
    ),
    OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="subtract",
            description="Subtract two numbers",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={"a": _number_property("First number"), "b": _number_property("Second number")},
                required=["a", "b"],
            ),
        ),
    ),
    OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="multiply",
            description="Multiply two numbers",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={"a": _number_property("First number"), "b": _number_property("Second number")},
                required=["a", "b"],
            ),
        ),
    ),
    OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="divide",
            description="Divide two numbers",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={"a": _number_property("Numerator"), "b": _number_property("Denominator (cannot be zero)")},
                required=["a", "b"],
            ),
        ),
    ),
]


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


# FastAPI application provided by the SDK.
app = create_app(CalculatorAgentLoop())


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
