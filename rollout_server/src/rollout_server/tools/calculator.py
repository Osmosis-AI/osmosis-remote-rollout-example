"""Simple async calculator tools with random delays.

These tools demonstrate async tool execution without requiring external services
like MCP servers. Each operation includes a random delay to simulate real-world
async tool calls.

Usage:
    tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "add",
            "arguments": {"a": 15, "b": 23}
        }
    }

    result = await execute_calculator_call(tool_call)
    # Returns: {"role": "tool", "content": "38", "tool_call_id": "call_123"}
"""

import asyncio
import json
import logging
import random
from typing import Any, Callable, Coroutine, Dict, List, Union

logger = logging.getLogger(__name__)


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
    """Multiply two numbers with random delay.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
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


def _serialize_result(result: Any) -> str:
    """Serialize tool result to string for consistent output.

    Args:
        result: The result to serialize

    Returns:
        JSON-serialized string representation
    """
    if isinstance(result, (int, float)):
        # For numeric results, use simple string representation
        # This preserves precision better for simple numbers
        return str(result)
    elif isinstance(result, str):
        return result
    else:
        # For complex objects, use JSON serialization
        return json.dumps(result)


def _create_tool_result(tool_call_id: str, content: str) -> Dict[str, str]:
    """Create a standardized tool result dict.

    Args:
        tool_call_id: The ID of the tool call
        content: The content of the result

    Returns:
        Tool result dict with role, content, and tool_call_id
    """
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id
    }


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
    tool_call_id: str = tool_call.get("id", "unknown")
    function_data = tool_call.get("function", {})
    function_name: str = function_data.get("name", "")
    arguments: Union[str, Dict[str, Any]] = function_data.get("arguments", {})

    # Parse arguments if they're a JSON string
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse arguments as JSON: {arguments}")
            return _create_tool_result(
                tool_call_id,
                f"Error: Invalid JSON arguments: {str(e)}"
            )

    # Check if function exists
    if function_name not in CALCULATOR_TOOLS:
        logger.warning(f"Unknown calculator function: {function_name}")
        return _create_tool_result(
            tool_call_id,
            f"Error: Unknown function '{function_name}'. Available: {list(CALCULATOR_TOOLS.keys())}"
        )

    # Execute tool
    try:
        logger.info(f"Executing {function_name} with arguments {arguments}")
        result = await CALCULATOR_TOOLS[function_name](**arguments)

        return _create_tool_result(tool_call_id, _serialize_result(result))

    except TypeError as e:
        # Invalid arguments for function
        logger.error(f"Invalid arguments for {function_name}: {arguments} - {e}")
        return _create_tool_result(
            tool_call_id,
            f"Error: Invalid arguments for {function_name}: {str(e)}"
        )

    except ValueError as e:
        # Calculation error (e.g., division by zero)
        logger.error(f"Calculation error in {function_name}: {e}")
        return _create_tool_result(tool_call_id, f"Error: {str(e)}")

    except Exception as e:
        # Unexpected error - don't expose internal details
        logger.exception(f"Unexpected error executing {function_name}")
        return _create_tool_result(
            tool_call_id,
            "Error: Tool execution failed"
        )


async def execute_calculator_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Execute multiple calculator tool calls concurrently.

    Args:
        tool_calls: List of tool call dicts

    Returns:
        List of tool result dicts
    """
    tasks = [execute_calculator_call(tool_call) for tool_call in tool_calls]
    return await asyncio.gather(*tasks)


# =============================================================================
# Tool Schema (for LLM tool description)
# =============================================================================

from rollout_server.schemas import (
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
