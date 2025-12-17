# System Architecture

## Overview

This RolloutServer drives an agent loop externally using an async-init protocol.

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
- Starts rollouts asynchronously in background tasks.

### Executor (`src/rollout_server/executor.py`)

- Owns shared resources:
  - HTTP client with connection pooling
  - Concurrency limiting semaphore
  - Rollout task registry keyed by `rollout_id` (idempotency)
- Implements the control loop:
  - Call `/v1/chat/completions`
  - Parse `tool_calls`
  - Execute tools
  - Post `/v1/rollout/completed`

### Tools (`src/rollout_server/tools/`)

This reference implementation includes local calculator tools:
- `add`, `subtract`, `multiply`, `divide`

Tool calls are executed asynchronously and tool results are appended as messages:

```json
{"role": "tool", "content": "8", "tool_call_id": "call_123"}
```

### Schemas (`src/rollout_server/schemas/`)

Pydantic models defining:
- `/v1/rollout/init` request/response payloads
- `/v1/chat/completions` request/response payloads
- `/v1/rollout/completed` callback payload

## Critical requirement: append-only messages

Multi-turn rollouts must maintain an append-only message history:
- Do not truncate, summarize, reorder, or rewrite earlier messages.
- Always append new assistant/tool messages to the end.

This ensures the training side can reliably accumulate token tracking data across turns.
