"""Calculator example demonstrating multi-turn conversations with tools.

This example shows how to use the RolloutServer for multi-turn conversations
that involve calculator tool calls. Demonstrates the correct response_mask
handling for tool outputs.

Prerequisites:
    1. Start mock trainer: uv run python -m tests.mocks.mock_trainer
    2. Start rollout server: uv run python -m rollout_server.server
"""

import asyncio
import uuid
import httpx


async def calculator_example():
    """Run a multi-turn rollout with calculator tools.
    
    This example demonstrates:
    1. Sending a math problem that triggers tool calls
    2. Server automatically executes calculator tools
    3. Server returns complete conversation with tool results
    4. Correct response_mask handling (tool outputs marked as 0)
    """

    # Create rollout request with math problem
    rollout_request = {
        "rollout_id": str(uuid.uuid4()),
        "server_url": "http://localhost:9001",  # Mock trainer endpoint
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful calculator assistant. Use the calculator tools to solve math problems."
            },
            {
                "role": "user",
                "content": "Please calculate 15 multiplied by 23, then add 50 to the result."
            }
        ],
        "sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "logprobs": True
        },
        "tokenizer_name": "Qwen/Qwen3-8B",
        "tokenizer_revision": "main",
        "max_turns": 10,  # Allow multiple turns for multi-step calculation
        "max_tokens_total": 8192
    }

    print("=" * 60)
    print("Calculator Example - Multi-turn with Tools")
    print("=" * 60)
    print(f"Rollout ID: {rollout_request['rollout_id']}")
    print(f"User prompt: {rollout_request['messages'][1]['content']}")
    print("-" * 60)

    # Send request to RolloutServer
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:9000/rollout",
                json=rollout_request,
                timeout=300.0
            )

            if response.status_code == 200:
                result = response.json()
                print_result(result)
            else:
                print(f"✗ Rollout failed with status: {response.status_code}")
                print(f"  Response: {response.text}")
                
        except httpx.ConnectError:
            print("✗ Connection failed!")
            print("  Make sure both servers are running:")
            print("  1. Mock trainer: uv run python -m tests.mocks.mock_trainer")
            print("  2. Rollout server: uv run python -m rollout_server.server")


def print_result(result: dict):
    """Pretty print the rollout result."""
    
    status = result.get("status", "UNKNOWN")
    finish_reason = result.get("finish_reason", "unknown")
    
    if status == "COMPLETED":
        print("✓ Rollout completed successfully!")
    else:
        print(f"✗ Rollout status: {status}")
        if "error_message" in result:
            print(f"  Error: {result['error_message']}")
        return
    
    print(f"  Finish reason: {finish_reason}")
    
    # Print metrics if available
    metrics = result.get("metrics")
    if metrics:
        print("\nMetrics:")
        print(f"  Total latency: {metrics.get('total_latency_ms', 0):.1f}ms")
        print(f"  LLM latency: {metrics.get('llm_latency_ms', 0):.1f}ms")
        print(f"  Tool latency: {metrics.get('tool_latency_ms', 0):.1f}ms")
        print(f"  LLM calls: {metrics.get('num_llm_calls', 0)}")
        print(f"  Tool calls: {metrics.get('num_tool_calls', 0)}")
        print(f"  Prompt tokens: {metrics.get('prompt_tokens', 0)}")
        print(f"  Response tokens: {metrics.get('response_tokens', 0)}")
    
    # Print conversation
    messages = result.get("final_messages", [])
    print(f"\nConversation ({len(messages)} messages):")
    print("-" * 60)
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id")
        
        # Format role with color indication
        role_display = {
            "system": "📋 System",
            "user": "👤 User",
            "assistant": "🤖 Assistant",
            "tool": "🔧 Tool"
        }.get(role, role)
        
        print(f"\n[{i+1}] {role_display}:")
        
        if content:
            # Truncate long content
            if len(content) > 200:
                print(f"    {content[:200]}...")
            else:
                print(f"    {content}")
        
        # Show tool calls if present
        if tool_calls:
            print(f"    Tool calls:")
            for tc in tool_calls:
                func = tc.get("function", {})
                print(f"      - {func.get('name', 'unknown')}({func.get('arguments', '{}')})")
        
        # Show tool call ID for tool responses
        if tool_call_id:
            print(f"    (response to: {tool_call_id})")
    
    print("\n" + "=" * 60)


async def calculator_simple_example():
    """Simpler example with a single calculation."""
    
    rollout_request = {
        "rollout_id": f"calc-simple-{uuid.uuid4().hex[:8]}",
        "server_url": "http://localhost:9001",
        "messages": [
            {
                "role": "system",
                "content": "You are a calculator. Use tools to compute results."
            },
            {
                "role": "user",
                "content": "What is 42 + 58?"
            }
        ],
        "sampling_params": {
            "temperature": 0.0,  # Deterministic for testing
            "top_p": 1.0,
            "max_tokens": 256,
            "logprobs": False
        },
        "tokenizer_name": "Qwen/Qwen3-8B",
        "max_turns": 5,
        "max_tokens_total": 4096
    }

    print("\n" + "=" * 60)
    print("Simple Calculator Example")
    print("=" * 60)
    print(f"Question: {rollout_request['messages'][1]['content']}")
    print("-" * 60)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:9000/rollout",
                json=rollout_request,
                timeout=60.0
            )

            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") == "COMPLETED":
                    # Extract final answer from last assistant message
                    messages = result.get("final_messages", [])
                    for msg in reversed(messages):
                        if msg.get("role") == "assistant" and msg.get("content"):
                            print(f"✓ Answer: {msg['content']}")
                            break
                    
                    # Show tool usage
                    tool_count = sum(1 for m in messages if m.get("role") == "tool")
                    if tool_count > 0:
                        print(f"  (Used {tool_count} tool call(s))")
                else:
                    print(f"✗ Error: {result.get('error_message', 'Unknown error')}")
                    
        except httpx.ConnectError:
            print("✗ Servers not running. Start them first.")


if __name__ == "__main__":
    print("Running calculator examples...")
    print("Make sure servers are running:")
    print("  1. uv run python -m tests.mocks.mock_trainer")
    print("  2. uv run python -m rollout_server.server")
    print()
    
    asyncio.run(calculator_example())
    asyncio.run(calculator_simple_example())

