# Osmosis Remote Rollout Example

This repository contains reference implementations for the Remote Rollout Protocol.

## Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| **RolloutServer** | [`rollout_server/`](rollout_server/) | Reference implementation of a RolloutServer for the Remote Rollout Protocol |
| **ToolServer** | [`tool_server/`](tool_server/) | MCP-compatible tool server (coming soon) |

## Quick Start

```bash
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

## Test Mode

Test mode allows you to validate your agent loop locally using external LLM providers (OpenAI, Anthropic, etc.) without deploying to TrainGate.

### Dataset

This project includes `multiply.parquet` as a test dataset. Test datasets require these columns:

| Column | Description |
|--------|-------------|
| `system_prompt` | System prompt for the LLM |
| `user_prompt` | User message to start conversation |
| `ground_truth` | Expected output (for reward computation) |

### Running Tests

```bash
# Batch test with GPT-4o (default)
uv run osmosis test -m server:agent_loop -d multiply.parquet

# Use Claude
uv run osmosis test -m server:agent_loop -d multiply.parquet --model anthropic/claude-sonnet-4-20250514

# Test subset of data
uv run osmosis test -m server:agent_loop -d multiply.parquet --limit 10

# Save results to JSON
uv run osmosis test -m server:agent_loop -d multiply.parquet -o results.json
```

### Interactive Debugging

Step through agent execution to debug issues:

```bash
# Start interactive mode
uv run osmosis test -m server:agent_loop -d multiply.parquet --interactive

# Start at specific row
uv run osmosis test -m server:agent_loop -d multiply.parquet --interactive --row 5
```

Interactive commands:

| Command | Description |
|---------|-------------|
| `n` | Execute next LLM call |
| `c` | Continue to completion |
| `m` | Show message history |
| `t` | Show available tools |
| `q` | Quit session |

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --agent` | required | Module path (`server:agent_loop`) |
| `-d, --dataset` | required | Dataset file (.json, .jsonl, .parquet) |
| `--model` | `gpt-4o` | Model name (e.g., `anthropic/claude-sonnet-4-20250514`) |
| `--max-turns` | `10` | Max agent turns per row |
| `--temperature` | - | LLM sampling temperature |
| `--limit` | all | Max rows to test |
| `--offset` | `0` | Rows to skip |
| `-i, --interactive` | - | Enable interactive mode |
| `--row` | - | Initial row (with `--interactive`) |
| `-o, --output` | - | Output JSON file |
| `-q, --quiet` | - | Suppress progress output |
| `--debug` | - | Enable debug output |

## Dependencies

- `osmosis-ai[server]>=0.2.8`
