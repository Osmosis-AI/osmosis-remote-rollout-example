"""Basic example demonstrating the async-init rollout flow.

This example starts a rollout via POST /init and then polls the mock trainer for
the completion callback.

Prerequisites:
  1) Start mock trainer: uv run python -m tests.mocks.mock_trainer
  2) Start rollout server: uv run python -m rollout_server.server

Or use:
  ./scripts/start_test_environment.sh
"""

import asyncio
import uuid

import httpx


ROLLOUT_SERVER_URL = "http://localhost:9000"
MOCK_TRAINER_URL = "http://localhost:9001"


async def basic_example() -> None:
    rollout_id = str(uuid.uuid4())

    init_request = {
        "rollout_id": rollout_id,
        "server_url": MOCK_TRAINER_URL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you today?"},
        ],
        "completion_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 128,
            "logprobs": True,
        },
        "max_turns": 10,
        "max_tokens_total": 8192,
    }

    async with httpx.AsyncClient() as client:
        init_resp = await client.post(
            f"{ROLLOUT_SERVER_URL}/init",
            json=init_request,
            timeout=10.0,
        )
        init_resp.raise_for_status()

        if init_resp.status_code != 202:
            raise RuntimeError(f"Unexpected status code: {init_resp.status_code}")

        init_data = init_resp.json()
        print("✓ Init accepted")
        print(f"  rollout_id: {init_data['rollout_id']}")
        print(f"  tools: {len(init_data.get('tools', []))}")

        # Poll mock trainer for completion callback.
        completed = None
        for _ in range(100):
            r = await client.get(
                f"{MOCK_TRAINER_URL}/v1/rollout/completed/{rollout_id}",
                timeout=5.0,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("rollout_id") == rollout_id and data.get("status"):
                completed = data
                break
            await asyncio.sleep(0.1)

        if completed is None:
            raise TimeoutError("Timed out waiting for completion callback")

        print("\n✓ Rollout completed")
        print(f"  status: {completed.get('status')}")
        print(f"  finish_reason: {completed.get('finish_reason')}")

        print("\nConversation:")
        for msg in completed.get("final_messages", []):
            role = msg.get("role")
            content = msg.get("content")
            if content is None:
                content = ""
            print(f"  {role}: {content}")


if __name__ == "__main__":
    asyncio.run(basic_example())
