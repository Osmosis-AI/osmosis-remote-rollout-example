# Remote Rollout Server - SDK Example

Example RolloutServer built on the Osmosis Python SDK (`osmosis-ai[server]`).

## Overview

Remote rollout separates trajectory generation from training infrastructure:

- **Training side**: hosts LLM inference (`/v1/chat/completions`) and receives rollout callback (`/v1/rollout/completed`)
- **RolloutServer** (this project): implements agent logic (calculator tools) and delegates protocol to the SDK

## Project Structure

```
├── server.py         # Agent loop + FastAPI app
├── tools.py          # Calculator tool definitions
├── rewards.py        # Reward computation
├── test_data.jsonl   # Example dataset for test mode
├── pyproject.toml
├── uv.lock
└── README.md
```

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

## Test Mode

SDK supports local testing without TrainGate using external LLM providers via [LiteLLM](https://docs.litellm.ai/docs/providers).

### Quick Start

```bash
# Test with OpenAI GPT-4o (default)
uv run osmosis test -m server:agent_loop -d test_data.jsonl

# Test with Anthropic Claude
uv run osmosis test -m server:agent_loop -d test_data.jsonl --model anthropic/claude-sonnet-4-20250514

# Interactive debugging (step through each LLM call)
uv run osmosis test -m server:agent_loop -d test_data.jsonl --interactive

# Start at specific row
uv run osmosis test -m server:agent_loop -d test_data.jsonl --interactive --row 5
```

### Dataset Format

Create a JSONL file with required columns:

```jsonl
{"system_prompt": "You are a calculator assistant.", "user_prompt": "What is 25 + 17?", "ground_truth": "42"}
{"system_prompt": "You are a calculator assistant.", "user_prompt": "What is 100 / 4?", "ground_truth": "25"}
```

See `test_data.jsonl` for a working example.

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --module` | (required) | Module path to agent loop |
| `-d, --dataset` | (required) | Path to dataset file |
| `--model` | `gpt-4o` | LLM model (LiteLLM format) |
| `--max-turns` | `10` | Maximum agent turns per row |
| `--limit` | all | Maximum rows to test |
| `--offset` | `0` | Rows to skip |
| `-i, --interactive` | - | Enable step-by-step debugging |
| `--row N` | - | Initial row for interactive mode |
| `-o, --output` | - | Write results to JSON file |
| `--debug` | - | Enable debug output |

### Supported Providers

- OpenAI: `gpt-4o`, `gpt-4-turbo`
- Anthropic: `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-3-haiku-20240307`
- Groq: `groq/llama-3.1-70b-versatile`
- Ollama: `ollama/llama2` (with `--base-url http://localhost:11434`)

Set API keys via environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) or `--api-key`.

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

- `osmosis-ai[server]>=0.2.11`
