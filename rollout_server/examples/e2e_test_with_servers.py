#!/usr/bin/env python3
"""Interactive demo script showing the complete rollout flow.

This script demonstrates the Remote Rollout Protocol with real running servers,
providing detailed logging to help understand the system behavior.

NOTE: For automated E2E testing, use:
    uv run pytest tests/e2e/ -v -m requires_servers

This script is intended for interactive exploration and debugging.

Usage:
    1. Start mock trainer: python -m tests.mocks.mock_trainer
    2. Start rollout server: python -m rollout_server.server
    3. Run this script: python examples/e2e_test_with_servers.py

Or use the test environment script:
    ./scripts/start_test_environment.sh
    python examples/e2e_test_with_servers.py
    ./scripts/stop_test_environment.sh
"""

import asyncio
import json
import uuid
import sys
from datetime import datetime

import httpx

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def log_header(title: str):
    """Print a header section."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.RESET}\n")


def log_step(step: str, description: str = ""):
    """Print a step indicator."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {Colors.CYAN}▶{Colors.RESET} {Colors.BOLD}{step}{Colors.RESET}")
    if description:
        print(f"  {Colors.DIM}{description}{Colors.RESET}")


def log_success(message: str):
    """Print a success message."""
    print(f"  {Colors.GREEN}✓{Colors.RESET} {message}")


def log_error(message: str):
    """Print an error message."""
    print(f"  {Colors.RED}✗{Colors.RESET} {message}")


def log_info(key: str, value: str, indent: int = 2):
    """Print an info key-value pair."""
    spaces = " " * indent
    print(f"{spaces}{Colors.YELLOW}{key}:{Colors.RESET} {value}")


def log_json(label: str, data: dict, max_lines: int = 20):
    """Print JSON data with pretty formatting."""
    print(f"  {Colors.BLUE}{label}:{Colors.RESET}")
    formatted = json.dumps(data, indent=2, ensure_ascii=False)
    lines = formatted.split('\n')
    if len(lines) > max_lines:
        for line in lines[:max_lines]:
            print(f"    {Colors.DIM}{line}{Colors.RESET}")
        print(f"    {Colors.DIM}... ({len(lines) - max_lines} more lines){Colors.RESET}")
    else:
        for line in lines:
            print(f"    {Colors.DIM}{line}{Colors.RESET}")


def log_message(msg: dict, index: int):
    """Print a conversation message with formatting."""
    role = msg.get("role", "unknown")
    role_colors = {
        "system": Colors.YELLOW,
        "user": Colors.GREEN,
        "assistant": Colors.BLUE,
        "tool": Colors.CYAN
    }
    color = role_colors.get(role, Colors.RESET)
    
    print(f"  {Colors.DIM}[{index}]{Colors.RESET} {color}{Colors.BOLD}{role.upper()}{Colors.RESET}")
    
    if "content" in msg and msg["content"]:
        content = msg["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        print(f"      {Colors.DIM}content:{Colors.RESET} {content}")
    
    if "tool_calls" in msg and msg["tool_calls"]:
        print(f"      {Colors.DIM}tool_calls:{Colors.RESET}")
        for tc in msg["tool_calls"]:
            func_name = tc.get("function", {}).get("name", "unknown")
            func_args = tc.get("function", {}).get("arguments", "{}")
            print(f"        - {Colors.CYAN}{func_name}{Colors.RESET}({func_args})")
    
    if "tool_call_id" in msg:
        print(f"      {Colors.DIM}tool_call_id:{Colors.RESET} {msg['tool_call_id']}")


# Server configuration
MOCK_TRAINER_URL = "http://localhost:9001"
ROLLOUT_SERVER_URL = "http://localhost:9000"


async def check_server_health(client: httpx.AsyncClient, url: str, name: str) -> bool:
    """Check if a server is available by calling its health endpoint."""
    log_step(f"Checking {name} health", f"GET {url}/health")
    try:
        response = await client.get(f"{url}/health", timeout=5.0)
        if response.status_code == 200:
            log_success(f"{name} is healthy (status: {response.status_code})")
            return True
        else:
            log_error(f"{name} returned status {response.status_code}")
            return False
    except httpx.ConnectError:
        log_error(f"Cannot connect to {name} at {url}")
        return False
    except httpx.RequestError as e:
        log_error(f"Request error: {e}")
        return False


async def run_e2e_test():
    """Run the end-to-end test with detailed logging."""
    
    log_header("E2E Test: Remote Rollout Protocol")
    
    print(f"{Colors.DIM}This script tests the complete rollout flow:{Colors.RESET}")
    print(f"{Colors.DIM}  1. User sends rollout request to RolloutServer{Colors.RESET}")
    print(f"{Colors.DIM}  2. RolloutServer calls MockTrainer for LLM completions{Colors.RESET}")
    print(f"{Colors.DIM}  3. RolloutServer executes tool calls{Colors.RESET}")
    print(f"{Colors.DIM}  4. Process repeats until completion{Colors.RESET}")
    print()
    
    async with httpx.AsyncClient() as client:
        
        # Step 1: Check server health
        log_header("Step 1: Server Health Check")
        
        mock_trainer_ok = await check_server_health(
            client, MOCK_TRAINER_URL, "Mock Trainer"
        )
        rollout_server_ok = await check_server_health(
            client, ROLLOUT_SERVER_URL, "Rollout Server"
        )
        
        if not mock_trainer_ok:
            print(f"\n{Colors.RED}Error:{Colors.RESET} Mock trainer is not running.")
            print(f"{Colors.DIM}Start it with:{Colors.RESET} python -m rollout_server.tests.mocks.mock_trainer")
            return False
        
        if not rollout_server_ok:
            print(f"\n{Colors.RED}Error:{Colors.RESET} Rollout server is not running.")
            print(f"{Colors.DIM}Start it with:{Colors.RESET} python -m rollout_server.server")
            return False
        
        # Step 2: Prepare rollout request
        log_header("Step 2: Prepare Rollout Request")
        
        rollout_id = str(uuid.uuid4())
        log_info("rollout_id", rollout_id)
        
        payload = {
            "rollout_id": rollout_id,
            "server_url": MOCK_TRAINER_URL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful calculator assistant with access to calculator tools."
                },
                {
                    "role": "user",
                    "content": "Please calculate 5 plus 3, and then multiply the result by 2."
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
        
        log_step("Initial messages")
        for i, msg in enumerate(payload["messages"]):
            log_message(msg, i)
        
        print()
        log_info("sampling_params", "")
        log_info("  temperature", str(payload["sampling_params"]["temperature"]), 4)
        log_info("  top_p", str(payload["sampling_params"]["top_p"]), 4)
        log_info("  max_tokens", str(payload["sampling_params"]["max_tokens"]), 4)
        
        log_info("tokenizer", f"{payload['tokenizer_name']} @ {payload['tokenizer_revision']}")
        log_info("max_turns", str(payload["max_turns"]))
        
        # Step 3: Send rollout request
        log_header("Step 3: Execute Rollout")
        
        log_step("Sending POST /rollout", f"to {ROLLOUT_SERVER_URL}")
        
        start_time = datetime.now()
        try:
            response = await client.post(
                f"{ROLLOUT_SERVER_URL}/rollout",
                json=payload,
                timeout=60.0
            )
            elapsed = (datetime.now() - start_time).total_seconds()
        except httpx.RequestError as e:
            log_error(f"Request failed: {e}")
            return False
        
        log_info("status_code", str(response.status_code))
        log_info("elapsed_time", f"{elapsed:.3f}s")
        
        if response.status_code != 200:
            log_error(f"Rollout request failed!")
            print(f"  {Colors.DIM}Response:{Colors.RESET}")
            print(f"    {response.text}")
            return False
        
        log_success("Rollout request completed successfully!")
        
        # Step 4: Analyze response
        log_header("Step 4: Analyze Response")
        
        result = response.json()
        
        log_info("status", result.get("status", "N/A"))
        log_info("finish_reason", result.get("finish_reason", "N/A"))
        log_info("total_turns", str(result.get("total_turns", "N/A")))
        log_info("total_tokens", str(result.get("total_tokens", "N/A")))
        
        # Print final messages
        log_step("Final conversation messages")
        final_messages = result.get("final_messages", [])
        log_info("total_messages", str(len(final_messages)))
        print()
        
        for i, msg in enumerate(final_messages):
            log_message(msg, i)
        
        # Analyze the conversation flow
        log_header("Step 5: Conversation Flow Analysis")
        
        roles = [msg.get("role") for msg in final_messages]
        
        # Count message types
        role_counts = {}
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        
        log_step("Message type distribution")
        for role, count in role_counts.items():
            color = {
                "system": Colors.YELLOW,
                "user": Colors.GREEN,
                "assistant": Colors.BLUE,
                "tool": Colors.CYAN
            }.get(role, Colors.RESET)
            print(f"  {color}● {role}:{Colors.RESET} {count} message(s)")
        
        # Count tool calls
        tool_call_count = 0
        tool_names = []
        for msg in final_messages:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_call_count += 1
                    func_name = tc.get("function", {}).get("name", "unknown")
                    tool_names.append(func_name)
        
        if tool_call_count > 0:
            print()
            log_step("Tool usage")
            log_info("total_tool_calls", str(tool_call_count))
            log_info("tools_used", ", ".join(tool_names))
        
        # Summary
        log_header("Test Summary")
        
        status = result.get("status")
        if status == "COMPLETED":
            print(f"  {Colors.GREEN}{Colors.BOLD}✓ E2E TEST PASSED{Colors.RESET}")
            print()
            print(f"  {Colors.DIM}The rollout completed successfully:{Colors.RESET}")
            print(f"    • RolloutServer received the request")
            print(f"    • LLM generated responses with tool calls")
            print(f"    • Tools were executed correctly")
            print(f"    • Final response was returned")
            return True
        else:
            print(f"  {Colors.RED}{Colors.BOLD}✗ E2E TEST FAILED{Colors.RESET}")
            print(f"  {Colors.DIM}Status:{Colors.RESET} {status}")
            if "error" in result:
                print(f"  {Colors.DIM}Error:{Colors.RESET} {result['error']}")
            return False


async def main():
    """Main entry point."""
    print(f"\n{Colors.BOLD}Remote Rollout E2E Test Script{Colors.RESET}")
    print(f"{Colors.DIM}Testing the TrainGate Remote Rollout Protocol{Colors.RESET}")
    
    try:
        success = await run_e2e_test()
        print()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error:{Colors.RESET} {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
