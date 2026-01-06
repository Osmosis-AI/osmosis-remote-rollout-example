"""Calculator tools for the example RolloutServer."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable, Coroutine, Dict, List

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
