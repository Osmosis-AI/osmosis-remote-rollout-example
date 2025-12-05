"""Complete end-to-end demo with mock trainer.

This example demonstrates the full Remote Rollout Protocol flow:
1. Starts mock trainer server (simulates training cluster)
2. Starts rollout server
3. Sends rollout requests and shows the complete flow
4. Demonstrates response_mask handling

This is an EDUCATIONAL example showing how all pieces fit together.
For actual testing, use the test suite in tests/.
"""

import asyncio
import subprocess
import sys
import time
import httpx
import uuid
from contextlib import asynccontextmanager


# Server configuration
MOCK_TRAINER_PORT = 9001
ROLLOUT_SERVER_PORT = 9000
STARTUP_TIMEOUT = 30  # seconds


@asynccontextmanager
async def managed_servers():
    """Context manager that starts and stops both servers.
    
    Yields when both servers are healthy.
    Cleans up servers on exit.
    """
    processes = []
    
    try:
        print("Starting servers...")
        
        # Start mock trainer
        print(f"  Starting mock trainer on port {MOCK_TRAINER_PORT}...")
        trainer_proc = subprocess.Popen(
            [sys.executable, "-m", "tests.mocks.mock_trainer"],
            env={**dict(__import__("os").environ), "MOCK_TRAINER_PORT": str(MOCK_TRAINER_PORT)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        processes.append(("mock_trainer", trainer_proc))
        
        # Start rollout server
        print(f"  Starting rollout server on port {ROLLOUT_SERVER_PORT}...")
        rollout_proc = subprocess.Popen(
            [sys.executable, "-m", "rollout_server.server"],
            env={**dict(__import__("os").environ), "ROLLOUT_SERVER_PORT": str(ROLLOUT_SERVER_PORT)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        processes.append(("rollout_server", rollout_proc))
        
        # Wait for servers to be healthy
        await wait_for_servers()
        
        print("✓ Both servers are healthy!")
        print("-" * 60)
        
        yield
        
    finally:
        print("\nShutting down servers...")
        for name, proc in processes:
            if proc.poll() is None:  # Still running
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    print(f"  ✓ {name} stopped")
                except subprocess.TimeoutExpired:
                    proc.kill()
                    print(f"  ✓ {name} killed")


async def wait_for_servers():
    """Wait for both servers to be healthy."""
    
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        
        while time.time() - start_time < STARTUP_TIMEOUT:
            try:
                # Check mock trainer
                trainer_resp = await client.get(
                    f"http://localhost:{MOCK_TRAINER_PORT}/health",
                    timeout=2.0
                )
                trainer_ok = trainer_resp.status_code == 200
                
                # Check rollout server
                rollout_resp = await client.get(
                    f"http://localhost:{ROLLOUT_SERVER_PORT}/health",
                    timeout=2.0
                )
                rollout_ok = rollout_resp.status_code == 200
                
                if trainer_ok and rollout_ok:
                    return
                    
            except httpx.RequestError:
                pass
            
            await asyncio.sleep(0.5)
        
        raise TimeoutError(f"Servers failed to start within {STARTUP_TIMEOUT}s")


async def demo_single_turn():
    """Demo: Single-turn conversation (no tools)."""
    
    print("\n" + "=" * 60)
    print("DEMO 1: Single-turn Conversation (No Tools)")
    print("=" * 60)
    
    request = {
        "rollout_id": f"demo-single-{uuid.uuid4().hex[:8]}",
        "server_url": f"http://localhost:{MOCK_TRAINER_PORT}",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! What's your name?"}
        ],
        "sampling_params": {
            "temperature": 0.7,
            "max_tokens": 256
        },
        "tokenizer_name": "Qwen/Qwen3-8B",
        "max_turns": 5,
        "max_tokens_total": 4096
    }
    
    print(f"\n📤 Request:")
    print(f"   User: {request['messages'][1]['content']}")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:{ROLLOUT_SERVER_PORT}/rollout",
            json=request,
            timeout=60.0
        )
        result = response.json()
    
    print(f"\n📥 Response:")
    print(f"   Status: {result['status']}")
    print(f"   Finish reason: {result.get('finish_reason', 'N/A')}")
    
    # Get assistant response
    for msg in result.get("final_messages", []):
        if msg["role"] == "assistant":
            print(f"   Assistant: {msg['content'][:100]}...")
            break
    
    print("\n   💡 Key Point: No tools called, single LLM turn")


async def demo_multi_turn_with_tools():
    """Demo: Multi-turn conversation with calculator tools."""
    
    print("\n" + "=" * 60)
    print("DEMO 2: Multi-turn with Calculator Tools")
    print("=" * 60)
    
    request = {
        "rollout_id": f"demo-tools-{uuid.uuid4().hex[:8]}",
        "server_url": f"http://localhost:{MOCK_TRAINER_PORT}",
        "messages": [
            {
                "role": "system",
                "content": "You are a calculator assistant. Use tools to compute."
            },
            {
                "role": "user",
                "content": "Please multiply 7 by 8."
            }
        ],
        "sampling_params": {
            "temperature": 0.7,
            "max_tokens": 512
        },
        "tokenizer_name": "Qwen/Qwen3-8B",
        "max_turns": 10,
        "max_tokens_total": 8192
    }
    
    print(f"\n📤 Request:")
    print(f"   User: {request['messages'][1]['content']}")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:{ROLLOUT_SERVER_PORT}/rollout",
            json=request,
            timeout=120.0
        )
        result = response.json()
    
    print(f"\n📥 Response:")
    print(f"   Status: {result['status']}")
    print(f"   Finish reason: {result.get('finish_reason', 'N/A')}")
    
    # Print conversation flow
    messages = result.get("final_messages", [])
    print(f"\n📜 Conversation Flow ({len(messages)} messages):")
    
    for i, msg in enumerate(messages):
        role = msg["role"]
        if role == "system":
            continue  # Skip system message
        
        icon = {"user": "👤", "assistant": "🤖", "tool": "🔧"}.get(role, "❓")
        
        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    print(f"   {icon} Assistant calls: {func.get('name')}({func.get('arguments')})")
            else:
                content = msg.get("content", "")[:80]
                print(f"   {icon} Assistant: {content}...")
        elif role == "tool":
            print(f"   {icon} Tool result: {msg.get('content', '')}")
        elif role == "user":
            print(f"   {icon} User: {msg.get('content', '')}")
    
    # Show metrics
    metrics = result.get("metrics", {})
    if metrics:
        print(f"\n📊 Metrics:")
        print(f"   LLM calls: {metrics.get('num_llm_calls', 0)}")
        print(f"   Tool calls: {metrics.get('num_tool_calls', 0)}")
        print(f"   Total latency: {metrics.get('total_latency_ms', 0):.0f}ms")
    
    print("\n   💡 Key Point: Multi-turn with tools, response_mask correctly marks tool outputs")


async def demo_response_mask_explanation():
    """Explain response_mask handling visually."""
    
    print("\n" + "=" * 60)
    print("DEMO 3: Response Mask Explanation")
    print("=" * 60)
    
    print("""
    The response_mask is CRITICAL for correct PPO training.
    
    It tells the trainer which tokens to include in the loss calculation:
    - mask = 1: LLM-generated tokens (include in training)
    - mask = 0: Tool/system tokens (exclude from training)
    
    Example flow:
    
    Turn 1 (first LLM call):
    ┌─────────────────────────────────────────────────────────────┐
    │ Prompt: [User: "Calculate 5 + 3"]                           │
    │ response_mask: None (first turn, no tool outputs yet)       │
    │                                                             │
    │ LLM Response: "I'll use the calculator"                     │
    │ All tokens get mask = 1 (LLM-generated)                     │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    Turn 2 (after tool execution):
    ┌─────────────────────────────────────────────────────────────┐
    │ Prompt: [...previous..., Tool: "8"]                         │
    │                          └─────┘                            │
    │                          ↓                                  │
    │ response_mask: [0, 0, ...]  ← Tool output tokens!           │
    │                                                             │
    │ LLM Response: "The result is 8"                             │
    │ All new tokens get mask = 1                                 │
    └─────────────────────────────────────────────────────────────┘
    
    Final mask structure:
    
    Tokens:     [LLM response 1] [Tool output] [LLM response 2]
    Mask:       [1, 1, 1, 1, 1]  [0, 0]        [1, 1, 1, 1, 1]
                └─ Train on ─┘   └─ Skip ─┘    └─ Train on ─┘
    
    Without correct masks:
    ❌ Model learns to "predict" tool outputs (divergence)
    ❌ Model doesn't learn from its responses (stagnation)
    """)


async def main():
    """Run all demos."""
    
    print("=" * 60)
    print("MOCK TRAINER EXAMPLE - Complete E2E Demo")
    print("=" * 60)
    print()
    print("This example demonstrates the full Remote Rollout flow.")
    print("It starts both servers, runs demos, and cleans up.")
    print()
    
    async with managed_servers():
        # Run demos sequentially
        await demo_single_turn()
        await asyncio.sleep(1)
        
        await demo_multi_turn_with_tools()
        await asyncio.sleep(1)
        
        await demo_response_mask_explanation()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

