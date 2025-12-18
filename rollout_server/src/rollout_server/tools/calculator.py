"""Simple async calculator tools with random delays.

This module is intentionally small and focuses only on "tool business logic".
All tool parsing/serialization utilities come from the Osmosis rollout SDK:
`osmosis_ai.rollout.tools`.
"""

import asyncio
import logging
import random
from typing import Any, Callable, Coroutine, Dict, List

logger = logging.getLogger(__name__)

from osmosis_ai.rollout.core.exceptions import ToolArgumentError
from osmosis_ai.rollout.tools import (
    create_tool_error_result,
    create_tool_result,
    execute_tool_calls,
    get_tool_call_info,
    serialize_tool_result,
)


# =============================================================================
# Constants
# =============================================================================

# Simulated delay range for tool execution (seconds)
TOOL_DELAY_MIN_SECONDS = 0.1
TOOL_DELAY_MAX_SECONDS = 0.5

# Type alias for async tool functions
ToolFunction = Callable[..., Coroutine[Any, Any, float]]


# =============================================================================
# Calculator Operations
# =============================================================================


async def add(a: float, b: float) -> float:
    """Add two numbers with random delay.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    delay = random.uniform(TOOL_DELAY_MIN_SECONDS, TOOL_DELAY_MAX_SECONDS)
    await asyncio.sleep(delay)
    result = a + b
    logger.debug(f"add({a}, {b}) = {result} (delay: {delay:.2f}s)")
    return result


async def subtract(a: float, b: float) -> float:
    """Subtract two numbers with random delay.

    Args:
        a: First number
        b: Second number

    Returns:
        Difference of a and b
    """
    delay = random.uniform(TOOL_DELAY_MIN_SECONDS, TOOL_DELAY_MAX_SECONDS)
    await asyncio.sleep(delay)
    result = a - b
    logger.debug(f"subtract({a}, {b}) = {result} (delay: {delay:.2f}s)")
    return result


async def multiply(a: float, b: float) -> float:
    """Multiply two numbers with random delay."""
    delay = random.uniform(TOOL_DELAY_MIN_SECONDS, TOOL_DELAY_MAX_SECONDS)
    await asyncio.sleep(delay)
    result = round(a * b, 4)
    logger.debug(f"multiply({a}, {b}) = {result} (delay: {delay:.2f}s)")
    return result


async def divide(a: float, b: float) -> float:
    """Divide two numbers with random delay.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Quotient of a and b

    Raises:
        ValueError: If b is zero
    """
    delay = random.uniform(TOOL_DELAY_MIN_SECONDS, TOOL_DELAY_MAX_SECONDS)
    await asyncio.sleep(delay)

    if b == 0:
        logger.warning(f"divide({a}, {b}) - Division by zero!")
        raise ValueError("Division by zero")

    result = a / b
    logger.debug(f"divide({a}, {b}) = {result} (delay: {delay:.2f}s)")
    return result


# Registry of available calculator tools
CALCULATOR_TOOLS: Dict[str, ToolFunction] = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}


# =============================================================================
# Tool Execution
# =============================================================================


async def execute_calculator_call(tool_call: Dict[str, Any]) -> Dict[str, str]:
    """Execute a calculator tool call.

    Args:
        tool_call: Tool call dict with structure:
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "add",
                    "arguments": {"a": 15, "b": 23}  # Can be dict or JSON string
                }
            }

    Returns:
        Tool result dict with structure:
            {
                "role": "tool",
                "content": "38",
                "tool_call_id": "call_123"
            }
    """
    try:
        tool_call_id, function_name, arguments = get_tool_call_info(tool_call)
    except ToolArgumentError as e:
        tool_call_id = e.tool_call_id or tool_call.get("id", "unknown")
        return create_tool_error_result(tool_call_id, str(e))

    tool_fn = CALCULATOR_TOOLS.get(function_name)
    if tool_fn is None:
        logger.warning("Unknown calculator function: %s", function_name)
        return create_tool_error_result(
            tool_call_id,
            f"Unknown function '{function_name}'. Available: {sorted(CALCULATOR_TOOLS.keys())}",
        )

    try:
        logger.info("Executing %s with arguments %s", function_name, arguments)
        result = await tool_fn(**arguments)
        return create_tool_result(tool_call_id, serialize_tool_result(result))
    except TypeError as e:
        logger.error("Invalid arguments for %s: %s - %s", function_name, arguments, e)
        return create_tool_error_result(
            tool_call_id,
            f"Invalid arguments for {function_name}: {e}",
        )
    except ValueError as e:
        logger.error("Calculation error in %s: %s", function_name, e)
        return create_tool_error_result(tool_call_id, str(e))
    except Exception as e:
        logger.exception("Unexpected error executing %s: %s", function_name, e)
        return create_tool_error_result(tool_call_id, "Tool execution failed")


async def execute_calculator_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Execute multiple calculator tool calls concurrently.

    Args:
        tool_calls: List of tool call dicts

    Returns:
        List of tool result dicts
    """
    return await execute_tool_calls(tool_calls, execute_calculator_call)


# =============================================================================
# Tool Schema (for LLM tool description)
# =============================================================================

from osmosis_ai.rollout.core.schemas import (
    OpenAIFunctionPropertySchema,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)


def _make_number_property(description: str) -> OpenAIFunctionPropertySchema:
    """Helper to create a number property schema."""
    return OpenAIFunctionPropertySchema(type="number", description=description)


CALCULATOR_TOOL_SCHEMAS: List[OpenAIFunctionToolSchema] = [
    OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="add",
            description="Add two numbers",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "a": _make_number_property("First number"),
                    "b": _make_number_property("Second number"),
                },
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
                properties={
                    "a": _make_number_property("First number"),
                    "b": _make_number_property("Second number to subtract from first"),
                },
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
                properties={
                    "a": _make_number_property("First number"),
                    "b": _make_number_property("Second number"),
                },
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
                properties={
                    "a": _make_number_property("Numerator"),
                    "b": _make_number_property("Denominator (cannot be zero)"),
                },
                required=["a", "b"],
            ),
        ),
    ),
]
