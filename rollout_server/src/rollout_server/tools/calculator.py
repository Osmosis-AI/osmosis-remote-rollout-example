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
import random
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


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
    delay = random.uniform(0.1, 0.5)
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
    delay = random.uniform(0.1, 0.5)
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
    delay = random.uniform(0.1, 0.5)
    await asyncio.sleep(delay)
    result = a * b
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
    delay = random.uniform(0.1, 0.5)
    await asyncio.sleep(delay)

    if b == 0:
        logger.warning(f"divide({a}, {b}) - Division by zero!")
        raise ValueError("Division by zero")

    result = a / b
    logger.debug(f"divide({a}, {b}) = {result} (delay: {delay:.2f}s)")
    return result


# Registry of available calculator tools
CALCULATOR_TOOLS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}


# =============================================================================
# Tool Execution
# =============================================================================


async def execute_calculator_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
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
    tool_call_id = tool_call["id"]
    function_name = tool_call["function"]["name"]
    arguments = tool_call["function"]["arguments"]

    # Parse arguments if they're a JSON string
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse arguments as JSON: {arguments}")
            return {
                "role": "tool",
                "content": f"Error: Invalid JSON arguments: {str(e)}",
                "tool_call_id": tool_call_id
            }

    # Check if function exists
    if function_name not in CALCULATOR_TOOLS:
        logger.warning(f"Unknown calculator function: {function_name}")
        return {
            "role": "tool",
            "content": f"Error: Unknown function '{function_name}'. Available: {list(CALCULATOR_TOOLS.keys())}",
            "tool_call_id": tool_call_id
        }

    # Execute tool
    try:
        logger.info(f"Executing {function_name} with arguments {arguments}")
        result = await CALCULATOR_TOOLS[function_name](**arguments)

        return {
            "role": "tool",
            "content": str(result),
            "tool_call_id": tool_call_id
        }

    except TypeError as e:
        # Invalid arguments for function
        logger.error(f"Invalid arguments for {function_name}: {arguments} - {e}")
        return {
            "role": "tool",
            "content": f"Error: Invalid arguments for {function_name}: {str(e)}",
            "tool_call_id": tool_call_id
        }

    except ValueError as e:
        # Calculation error (e.g., division by zero)
        logger.error(f"Calculation error in {function_name}: {e}")
        return {
            "role": "tool",
            "content": f"Error: {str(e)}",
            "tool_call_id": tool_call_id
        }

    except Exception as e:
        # Unexpected error
        logger.exception(f"Unexpected error executing {function_name}")
        return {
            "role": "tool",
            "content": f"Error: Unexpected error: {str(e)}",
            "tool_call_id": tool_call_id
        }


async def execute_calculator_calls(tool_calls: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
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


CALCULATOR_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "Subtract two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number to subtract from first"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "Divide two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Numerator"},
                    "b": {"type": "number", "description": "Denominator (cannot be zero)"}
                },
                "required": ["a", "b"]
            }
        }
    }
]
