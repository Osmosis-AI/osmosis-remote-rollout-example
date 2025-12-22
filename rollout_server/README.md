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

### Option 1: Direct Python

```bash
# Start server directly (default port 9000)
uv run python -m server
```

### Option 2: Using uvicorn directly

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

## Dependencies

- `osmosis-ai[server]>=0.2.7`
