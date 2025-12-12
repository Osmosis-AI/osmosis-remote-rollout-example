#!/usr/bin/env python3
"""Interactive demo script showing the async-init rollout flow with running servers.

Usage:
  1) Start mock trainer: uv run python -m tests.mocks.mock_trainer
  2) Start rollout server: uv run python -m rollout_server.server
  3) Run: uv run python examples/e2e_test_with_servers.py

Or use:
  ./scripts/start_test_environment.sh
  uv run python examples/e2e_test_with_servers.py
  ./scripts/stop_test_environment.sh
"""

import asyncio
import json
import uuid
import sys
from datetime import datetime

import httpx


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


MOCK_TRAINER_URL = "http://localhost:9001"
ROLLOUT_SERVER_URL = "http://localhost:9000"


def log_header(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.RESET}\n")


def log_step(step: str, description: str = "") -> None:
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {Colors.CYAN}▶{Colors.RESET} {Colors.BOLD}{step}{Colors.RESET}")
    if description:
        print(f"  {Colors.DIM}{description}{Colors.RESET}")


def log_success(message: str) -> None:
    print(f"  {Colors.GREEN}✓{Colors.RESET} {message}")


def log_error(message: str) -> None:
    print(f"  {Colors.RED}✗{Colors.RESET} {message}")


def log_json(label: str, data: dict) -> None:
    print(f"  {Colors.BLUE}{label}:{Colors.RESET}")
    formatted = json.dumps(data, indent=2)
    for line in formatted.split("\n"):
        print(f"    {Colors.DIM}{line}{Colors.RESET}")


async def check_health(client: httpx.AsyncClient, url: str, name: str) -> bool:
    log_step(f"Checking {name} health", f"GET {url}/health")
    try:
        r = await client.get(f"{url}/health", timeout=5.0)
        if r.status_code == 200:
            log_success(f"{name} is healthy")
            return True
        log_error(f"{name} returned {r.status_code}")
        return False
    except Exception as e:
        log_error(f"{name} health check failed: {e}")
        return False


async def poll_completion(client: httpx.AsyncClient, rollout_id: str, timeout_s: float = 10.0) -> dict:
    deadline = datetime.now().timestamp() + timeout_s
    while datetime.now().timestamp() < deadline:
        r = await client.get(
            f"{MOCK_TRAINER_URL}/v1/rollout/completed/{rollout_id}",
            timeout=5.0,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("rollout_id") == rollout_id and data.get("status"):
            return data
        await asyncio.sleep(0.1)
    raise TimeoutError("Timed out waiting for completion callback")


async def main() -> None:
    log_header("E2E Demo: Async-init Remote Rollout")

    async with httpx.AsyncClient() as client:
        if not await check_health(client, MOCK_TRAINER_URL, "Mock Trainer"):
            return
        if not await check_health(client, ROLLOUT_SERVER_URL, "Rollout Server"):
            return

        rollout_id = str(uuid.uuid4())
        init_payload = {
            "rollout_id": rollout_id,
            "server_url": MOCK_TRAINER_URL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful calculator assistant with access to calculator tools.",
                },
                {
                    "role": "user",
                    "content": "Please calculate 5 plus 3.",
                },
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

        log_header("Step 1: POST /init")
        log_json("init_request", init_payload)

        init_resp = await client.post(
            f"{ROLLOUT_SERVER_URL}/init",
            json=init_payload,
            timeout=10.0,
        )
        if init_resp.status_code != 202:
            log_error(f"/init returned {init_resp.status_code}: {init_resp.text}")
            return

        init_data = init_resp.json()
        log_success("Init accepted")
        log_json("init_response", init_data)

        log_header("Step 2: Wait for /v1/rollout/completed")
        completed = await poll_completion(client, rollout_id, timeout_s=10.0)
        log_success("Completion received")
        log_json("rollout_completed", completed)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
