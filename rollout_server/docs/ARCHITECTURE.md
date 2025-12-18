# System Architecture

## Overview

This RolloutServer drives an agent loop externally using an async-init protocol.

This repo is an **example server**: it delegates the protocol implementation to the
Osmosis rollout Python SDK (`osmosis_ai.rollout`) and keeps only example agent logic.

High-level flow:

1. Training calls `POST /v1/rollout/init` on RolloutServer (202 Accepted + tools).
2. RolloutServer calls back to the training side for LLM generations:
   - `POST {server_url}/v1/chat/completions`
3. RolloutServer executes tools, appends tool messages, and repeats until completion.
4. RolloutServer posts the final transcript once:
   - `POST {server_url}/v1/rollout/completed`

## Components

### FastAPI server (`src/rollout_server/server.py`)

- Exposes:
  - `GET /health`
  - `POST /v1/rollout/init` (returns 202)
- Uses the SDK app factory (`osmosis_ai.rollout.create_app`) for:
  - background task management
  - concurrency limiting
  - idempotency by `rollout_id`
  - callbacks to `{server_url}/v1/chat/completions` and `{server_url}/v1/rollout/completed`
  - metrics collection

### Agent loop (`src/rollout_server/server.py`)

- This repo provides a minimal `CalculatorAgentLoop` implementation.
- The agent loop:
  - calls the trainer's `/v1/chat/completions` via the SDK client (`OsmosisLLMClient`)
  - executes tool calls (calculator) and appends tool messages

### Osmosis rollout SDK (`osmosis_ai.rollout`)

- The SDK provides:
  - the FastAPI server factory: `create_app()`
  - the HTTP client: `OsmosisLLMClient`
  - protocol schemas: `osmosis_ai.rollout.core.schemas`
  - tool utilities: `osmosis_ai.rollout.tools`

### Tools (`src/rollout_server/tools/`)

This reference implementation includes local calculator tools:

- `add`, `subtract`, `multiply`, `divide`

Tool calls are executed asynchronously and tool results are appended as messages:

```json
{ "role": "tool", "content": "8", "tool_call_id": "call_123" }
```

### Schemas (`src/rollout_server/schemas/`)

This repo re-exports protocol schemas from the SDK (`osmosis_ai.rollout.core.schemas`):

- `/v1/rollout/init` request/response payloads
- `/v1/chat/completions` request/response payloads
- `/v1/rollout/completed` callback payload

## Critical requirement: append-only messages

Multi-turn rollouts must maintain an append-only message history:

- Do not truncate, summarize, reorder, or rewrite earlier messages.
- Always append new assistant/tool messages to the end.

This ensures the training side can reliably accumulate token tracking data across turns.
