"""Complete end-to-end demo with the mock trainer.

This example:
  1) Starts the mock trainer server
  2) Starts the RolloutServer
  3) Sends /init requests
  4) Polls the mock trainer for the completion callback

For automated coverage, use the test suite in tests/.
"""

import asyncio
import os
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager

import httpx


MOCK_TRAINER_PORT = 9001
ROLLOUT_SERVER_PORT = 9000
STARTUP_TIMEOUT = 30


@asynccontextmanager
async def managed_servers():
    processes = []
    try:
        trainer_proc = subprocess.Popen(
            [sys.executable, "-m", "tests.mocks.mock_trainer"],
            env={**os.environ, "MOCK_TRAINER_PORT": str(MOCK_TRAINER_PORT)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append(("mock_trainer", trainer_proc))

        rollout_proc = subprocess.Popen(
            [sys.executable, "-m", "rollout_server.server"],
            env={**os.environ, "ROLLOUT_SERVER_PORT": str(ROLLOUT_SERVER_PORT)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append(("rollout_server", rollout_proc))

        await wait_for_servers()
        yield

    finally:
        for _name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


async def wait_for_servers() -> None:
    async with httpx.AsyncClient() as client:
        start = time.time()
        while time.time() - start < STARTUP_TIMEOUT:
            try:
                trainer_ok = (await client.get(f"http://localhost:{MOCK_TRAINER_PORT}/health", timeout=2.0)).status_code == 200
                rollout_ok = (await client.get(f"http://localhost:{ROLLOUT_SERVER_PORT}/health", timeout=2.0)).status_code == 200
                if trainer_ok and rollout_ok:
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(0.2)
        raise TimeoutError("Servers failed to start")


async def run_rollout(init_request: dict) -> dict:
    rollout_id = init_request["rollout_id"]
    async with httpx.AsyncClient() as client:
        init_resp = await client.post(
            f"http://localhost:{ROLLOUT_SERVER_PORT}/init",
            json=init_request,
            timeout=10.0,
        )
        init_resp.raise_for_status()
        if init_resp.status_code != 202:
            raise RuntimeError(f"Unexpected /init status: {init_resp.status_code}")

        # Poll mock trainer for completion callback.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            r = await client.get(
                f"http://localhost:{MOCK_TRAINER_PORT}/v1/rollout/completed/{rollout_id}",
                timeout=5.0,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("rollout_id") == rollout_id and data.get("status"):
                return data
            await asyncio.sleep(0.1)

    raise TimeoutError("Timed out waiting for completion callback")


async def main() -> None:
    async with managed_servers():
        # Single-turn (no tools)
        single = {
            "rollout_id": f"demo-single-{uuid.uuid4().hex[:8]}",
            "server_url": f"http://localhost:{MOCK_TRAINER_PORT}",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            "completion_params": {"temperature": 0.7, "max_tokens": 64, "logprobs": True},
            "max_turns": 5,
            "max_tokens_total": 4096,
        }
        result = await run_rollout(single)
        print("Single-turn result:")
        print(f"  status={result.get('status')} finish_reason={result.get('finish_reason')}")

        # Multi-turn (tools)
        tools = {
            "rollout_id": f"demo-tools-{uuid.uuid4().hex[:8]}",
            "server_url": f"http://localhost:{MOCK_TRAINER_PORT}",
            "messages": [
                {"role": "system", "content": "You are a calculator assistant. Use tools to compute."},
                {"role": "user", "content": "Please calculate 5 plus 3."},
            ],
            "completion_params": {"temperature": 0.7, "max_tokens": 128, "logprobs": True},
            "max_turns": 10,
            "max_tokens_total": 8192,
        }
        result = await run_rollout(tools)
        print("Multi-turn result:")
        print(f"  status={result.get('status')} finish_reason={result.get('finish_reason')}")


if __name__ == "__main__":
    asyncio.run(main())
