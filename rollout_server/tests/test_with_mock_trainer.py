"""Integration test using mock trainer.

This script demonstrates how to test the /rollout endpoint with a mock trainer.

Usage:
    1. Start mock trainer: python -m rollout_server.tests.mocks.mock_trainer
    2. Start rollout server: python -m rollout_server.server
    3. Run this test: python -m rollout_server.tests.test_with_mock_trainer
"""

import asyncio
import uuid
import httpx


EXAMPLE_PAYLOAD = {
    "rollout_id": str(uuid.uuid4()),
    "server_url": "http://localhost:9001",  # Mock trainer endpoint
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful calculator assistant with access to calculator tools."
        },
        {
            "role": "user",
            "content": "Please calculate 5 plus 3."
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
    "max_turns": 10,
    "max_tokens_total": 8192
}


async def test_rollout():
    """Test the /rollout endpoint with mock trainer."""

    print("=" * 60)
    print("Testing /rollout endpoint with mock trainer")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # Test 1: Check mock trainer health
        try:
            health_response = await client.get("http://localhost:9001/health", timeout=5.0)
            if health_response.status_code == 200:
                print("✓ Mock trainer is running")
            else:
                print("✗ Mock trainer health check failed")
                return
        except Exception as e:
            print(f"✗ Mock trainer not available: {e}")
            print("\nPlease start the mock trainer first:")
            print("  python -m rollout_server.tests.mocks.mock_trainer")
            return

        # Test 2: Check rollout server health
        try:
            health_response = await client.get("http://localhost:9000/health", timeout=5.0)
            if health_response.status_code == 200:
                print("✓ Rollout server is running")
            else:
                print("✗ Rollout server health check failed")
                return
        except Exception as e:
            print(f"✗ Rollout server not available: {e}")
            print("\nPlease start the rollout server first:")
            print("  cd rollout_server && python -m rollout_server.server")
            return

        # Test 3: Send rollout request
        print("\nSending rollout request...")
        print(f"Rollout ID: {EXAMPLE_PAYLOAD['rollout_id']}")

        try:
            response = await client.post(
                "http://localhost:9000/rollout",
                json=EXAMPLE_PAYLOAD,
                timeout=30.0
            )

            if response.status_code == 200:
                result = response.json()
                print("\n✓ Rollout completed successfully!")
                print(f"  Status: {result['status']}")
                print(f"  Finish reason: {result['finish_reason']}")
                print(f"  Final messages: {len(result['final_messages'])}")

                print("\nConversation:")
                for i, msg in enumerate(result['final_messages']):
                    role = msg['role']
                    content = msg.get('content', '')
                    tool_calls = msg.get('tool_calls', [])

                    print(f"\n[{i+1}] {role.upper()}")
                    if content:
                        print(f"    Content: {content}")
                    if tool_calls:
                        print(f"    Tool calls: {len(tool_calls)}")
                        for tc in tool_calls:
                            func = tc.get('function', {})
                            print(f"      - {func.get('name')}: {func.get('arguments')}")

            else:
                print(f"\n✗ Rollout failed: {response.status_code}")
                print(f"Error: {response.text}")

        except Exception as e:
            print(f"\n✗ Request failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_rollout())
