#!/usr/bin/env python3
"""End-to-end tests with real running servers.

These tests require the mock trainer and rollout server to be running.

Usage:
  1) Start the test environment:
     ./scripts/start_test_environment.sh

  2) Run E2E tests:
     uv run pytest tests/e2e/ -v -m requires_servers

  3) Stop the test environment:
     ./scripts/stop_test_environment.sh
"""

import asyncio
import time
import uuid
from typing import Any, Dict

import httpx
import pytest


MOCK_TRAINER_URL = "http://localhost:9001"
ROLLOUT_SERVER_URL = "http://localhost:9000"


async def check_server_health(url: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{url}/health", timeout=2.0)
            return r.status_code == 200
    except Exception:
        return False


def servers_available() -> bool:
    async def check_both() -> bool:
        mock_ok = await check_server_health(MOCK_TRAINER_URL)
        rollout_ok = await check_server_health(ROLLOUT_SERVER_URL)
        return mock_ok and rollout_ok

    return asyncio.run(check_both())


pytestmark = [
    pytest.mark.requires_servers,
    pytest.mark.skipif(
        not servers_available(),
        reason="Requires running mock trainer and rollout server",
    ),
]


def create_init_request(messages: list[Dict[str, Any]], max_turns: int = 10) -> Dict[str, Any]:
    return {
        "rollout_id": str(uuid.uuid4()),
        "server_url": MOCK_TRAINER_URL,
        "messages": messages,
        "completion_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 128,
            "logprobs": True,
        },
        "max_turns": max_turns,
        "max_tokens_total": 8192,
    }


async def wait_for_completion(rollout_id: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            r = await client.get(
                f"{MOCK_TRAINER_URL}/v1/rollout/completed/{rollout_id}",
                timeout=2.0,
            )
            r.raise_for_status()
            data = r.json()
            if data.get("rollout_id") == rollout_id and data.get("status"):
                return data
            await asyncio.sleep(0.1)
    raise TimeoutError("Timed out waiting for completion callback")


@pytest.mark.asyncio
async def test_e2e_health_endpoints():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{MOCK_TRAINER_URL}/health", timeout=2.0)
        assert r.status_code == 200
        r = await client.get(f"{ROLLOUT_SERVER_URL}/health", timeout=2.0)
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_e2e_init_returns_tools():
    payload = create_init_request([
        {"role": "user", "content": "Hello"},
    ])

    async with httpx.AsyncClient() as client:
        r = await client.post(f"{ROLLOUT_SERVER_URL}/init", json=payload, timeout=10.0)
        assert r.status_code == 202
        data = r.json()
        assert data["rollout_id"] == payload["rollout_id"]
        assert isinstance(data.get("tools"), list)
        assert len(data["tools"]) > 0


@pytest.mark.asyncio
async def test_e2e_rollout_without_tools():
    payload = create_init_request([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"},
    ])

    async with httpx.AsyncClient() as client:
        r = await client.post(f"{ROLLOUT_SERVER_URL}/init", json=payload, timeout=10.0)
        assert r.status_code == 202

    completed = await wait_for_completion(payload["rollout_id"], timeout_s=10.0)
    assert completed["status"] == "COMPLETED"
    assert completed.get("finish_reason") == "stop"
    assert len(completed.get("final_messages", [])) >= 2


@pytest.mark.asyncio
async def test_e2e_rollout_with_tools():
    payload = create_init_request([
        {"role": "system", "content": "You are a calculator assistant."},
        {"role": "user", "content": "Please calculate 5 plus 3."},
    ])

    async with httpx.AsyncClient() as client:
        r = await client.post(f"{ROLLOUT_SERVER_URL}/init", json=payload, timeout=10.0)
        assert r.status_code == 202

    completed = await wait_for_completion(payload["rollout_id"], timeout_s=10.0)
    assert completed["status"] == "COMPLETED"

    msgs = completed.get("final_messages", [])
    assert any(m.get("role") == "tool" for m in msgs)

    metrics = completed.get("metrics") or {}
    assert metrics.get("num_llm_calls", 0) >= 2
    assert metrics.get("num_tool_calls", 0) >= 1


@pytest.mark.asyncio
async def test_e2e_max_turns_respected():
    payload = create_init_request(
        [
            {"role": "system", "content": "You are a calculator assistant."},
            {"role": "user", "content": "Please calculate 5 plus 3."},
        ],
        max_turns=1,
    )

    async with httpx.AsyncClient() as client:
        r = await client.post(f"{ROLLOUT_SERVER_URL}/init", json=payload, timeout=10.0)
        assert r.status_code == 202

    completed = await wait_for_completion(payload["rollout_id"], timeout_s=10.0)
    assert completed["status"] == "COMPLETED"
    assert completed.get("finish_reason") in ["stop", "max_turns", "max_tokens"]
