"""Basic example demonstrating RolloutServer usage.

This example shows a simple single-turn conversation without tools.
Demonstrates the basic protocol flow.
"""

import asyncio
import uuid
import httpx


async def basic_example():
    """Run a basic single-turn rollout."""

    # Create rollout request
    rollout_request = {
        "rollout_id": str(uuid.uuid4()),
        "server_url": "http://localhost:9001",  # Mock trainer endpoint
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you today?"}
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

    # Send request to RolloutServer
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:9000/rollout",
            json=rollout_request,
            timeout=300.0
        )

        if response.status_code == 200:
            result = response.json()
            print("✓ Rollout completed successfully!")
            print(f"  Rollout ID: {result['rollout_id']}")
            print(f"  Status: {result['status']}")
            print(f"  Finish reason: {result['finish_reason']}")
            print(f"  Final messages: {len(result['final_messages'])}")

            # Print conversation
            print("\nConversation:")
            for msg in result['final_messages']:
                role = msg['role']
                content = msg['content']
                print(f"  {role}: {content}")

        else:
            print(f"✗ Rollout failed: {response.status_code}")
            print(f"  Error: {response.text}")


if __name__ == "__main__":
    print("Running basic rollout example...")
    print("=" * 60)

    asyncio.run(basic_example())
