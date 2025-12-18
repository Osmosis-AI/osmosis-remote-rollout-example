# Remote Rollout Server - SDK Example

Example RolloutServer built on the Osmosis Python SDK (`osmosis-ai[server]`).

## Overview

Remote rollout separates trajectory generation from training infrastructure:

- **Training side**: hosts LLM inference (`/v1/chat/completions`) and receives rollout callback (`/v1/rollout/completed`)
- **RolloutServer** (this project): implements agent logic (calculator tools) and delegates protocol to the SDK

## Project Structure

```
rollout_server/
├── server.py         # Agent loop + calculator tools (~250 lines)
├── pyproject.toml
├── uv.lock
└── README.md
```

## Quick Start

```bash
# From project root
cd rollout_server

# Install
uv sync

# Start server (default port 9000)
uv run python -m server
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

- `osmosis-ai[server]==0.2.7`
