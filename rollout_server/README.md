# Remote Rollout Server - SDK Example

Example RolloutServer built on the Osmosis Python SDK (`osmosis-ai[server]`).

## Overview

Remote rollout separates trajectory generation from training infrastructure:

- **Training side**: hosts LLM inference (`/v1/chat/completions`) and receives rollout callback (`/v1/rollout/completed`)
- **RolloutServer** (this project): implements agent logic (calculator tools) and delegates protocol to the SDK

## Project Structure

```
rollout_server/
├── server.py         # Agent loop + FastAPI app
├── tools.py          # Calculator tool definitions
├── pyproject.toml
├── uv.lock
└── README.md
```

## Quick Start

```bash
# From project root
cd rollout_server

# Install dependencies
uv sync
```

### Option 1: Using SDK CLI (Recommended)

The SDK provides `osmosis serve` command with built-in validation and features:

```bash
# Start server with validation (default port 9000)
uv run osmosis serve -m server:agent_loop

# Specify port
uv run osmosis serve -m server:agent_loop -p 8080

# Enable debug logging
uv run osmosis serve -m server:agent_loop --log ./rollout_logs

# Enable auto-reload for development
uv run osmosis serve -m server:agent_loop --reload

# Validate agent loop without starting server
uv run osmosis validate -m server:agent_loop
```

### Option 2: Direct Python

```bash
# Start server directly (default port 9000)
uv run python -m server
```

### Option 3: Using uvicorn directly

```bash
# Start server with uvicorn
uv run uvicorn server:app --host 0.0.0.0 --port 9000

# Enable auto-reload for development
uv run uvicorn server:app --host 0.0.0.0 --port 9000 --reload
```

## Protocol

### 1) Init rollout

`POST /v1/rollout/init` - Training sends init request, receives `202 Accepted` with available tools.

### 2) LLM callback

RolloutServer calls `POST {server_url}/v1/chat/completions` for each generation.

### 3) Completion callback

RolloutServer posts final result to `POST {server_url}/v1/rollout/completed`.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ROLLOUT_SERVER_HOST` | `0.0.0.0` | Server bind host |
| `ROLLOUT_SERVER_PORT` | `9000` | Server port |
| `ROLLOUT_DEBUG_DIR` | (disabled) | Enable debug logging to this directory |

## Debug Logging

The server uses the SDK's built-in debug logging via `ctx.log_event()`. When enabled, each rollout writes execution traces to JSONL files.

### Enable Debug Logging

```bash
# Using SDK CLI (recommended)
uv run osmosis serve -m server:agent_loop --log ./rollout_logs

# Or via environment variable (works with any start method)
ROLLOUT_DEBUG_DIR=./rollout_logs uv run python -m server
ROLLOUT_DEBUG_DIR=./rollout_logs uv run uvicorn server:app --port 9000
```

### Output Structure

```
rollout_logs/
├── 1703270400/                    # Timestamp when server started
│   ├── rollout-abc123.jsonl
│   └── rollout-def456.jsonl
```

### Logged Events

The agent loop logs these events (see `server.py`):

| Event | Description |
|-------|-------------|
| `pre_llm` | State before each LLM call (turn, message count, summaries) |
| `llm_response` | LLM response details (tool calls, finish reason) |
| `tool_results` | Tool execution results |
| `rollout_complete` | Final state (reward, total turns, finish reason) |

Example JSONL output:

```jsonl
{"event": "pre_llm", "rollout_id": "abc123", "turn": 0, "num_messages": 1, ...}
{"event": "llm_response", "rollout_id": "abc123", "turn": 0, "has_tool_calls": true, ...}
{"event": "tool_results", "rollout_id": "abc123", "turn": 0, "num_tool_results": 1, ...}
{"event": "rollout_complete", "rollout_id": "abc123", "finish_reason": "stop", "reward": 1.0, ...}
```

**Note:** `ctx.log_event()` is a no-op when `ROLLOUT_DEBUG_DIR` is not set, so there's zero overhead in production.

## Dependencies

- `osmosis-ai[server]>=0.2.7`
