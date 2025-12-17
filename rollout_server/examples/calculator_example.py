"""Calculator example demonstrating multi-turn conversations with tools.

This example starts a rollout via POST /v1/rollout/init and then polls the mock trainer for
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


def _print_conversation(messages: list[dict]) -> None:
    for i, msg in enumerate(messages, start=1):
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        print(f"[{i}] {role}: {content}")
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            for tc in tool_calls:
                fn = (tc.get("function") or {}).get("name")
                args = (tc.get("function") or {}).get("arguments")
                print(f"    tool_call: {fn}({args})")
        if msg.get("tool_call_id"):
            print(f"    tool_call_id: {msg['tool_call_id']}")


async def calculator_example() -> None:
    rollout_id = str(uuid.uuid4())

    init_request = {
        "rollout_id": rollout_id,
        "server_url": MOCK_TRAINER_URL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful calculator assistant. Use tools to compute results.",
            },
            {"role": "user", "content": "Please calculate 5 plus 3."},
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
            f"{ROLLOUT_SERVER_URL}/v1/rollout/init",
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

        metrics = completed.get("metrics") or {}
        if metrics:
            print("\nMetrics:")
            print(f"  num_llm_calls: {metrics.get('num_llm_calls')}")
            print(f"  num_tool_calls: {metrics.get('num_tool_calls')}")
            print(f"  total_latency_ms: {metrics.get('total_latency_ms')}")

        print("\nConversation:")
        _print_conversation(completed.get("final_messages", []))


if __name__ == "__main__":
    asyncio.run(calculator_example())
