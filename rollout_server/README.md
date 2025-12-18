# Remote Rollout Server - SDK-based Example

An example RolloutServer for the Remote Rollout Protocol, built on the Osmosis Python SDK (`osmosis_ai.rollout`).

## Overview

Remote rollout separates trajectory generation (agent loop) from training infrastructure:

- **Training side**: hosts the LLM inference endpoint (`/v1/chat/completions`) and receives the final rollout callback (`/v1/rollout/completed`).
- **RolloutServer** (this project): provides example agent logic (tools + loop) and delegates protocol handling to the SDK (`create_app`, `OsmosisLLMClient`, schemas).

## Dependency

This repo depends on the published Osmosis SDK package:

- `osmosis-ai[server]==0.2.7`

## Protocol

### 1) Start a rollout

**Endpoint**: `POST /v1/rollout/init`

Training sends an init request and receives `202 Accepted` with the tools available for this rollout.

### 2) LLM generation callback

RolloutServer calls the training side:

- `POST {server_url}/v1/chat/completions`

### 3) Completion callback

RolloutServer posts the final result once:

- `POST {server_url}/v1/rollout/completed`

### Authentication

If `api_key` is provided in the `/v1/rollout/init` request, RolloutServer includes it as a Bearer token in both callback requests:

```
Authorization: Bearer <api_key>
```

## Running locally

All commands below must be run from the `rollout_server/` directory.

### Install

```bash
uv sync
```

### Start servers (mock trainer + RolloutServer)

```bash
./scripts/start_test_environment.sh
```

Stop when done:

```bash
./scripts/stop_test_environment.sh
```

### Run examples

```bash
uv run python examples/basic_example.py
uv run python examples/calculator_example.py
uv run python examples/e2e_test_with_servers.py
```

## Testing

```bash
uv run pytest
```

## Key requirements

- **Append-only messages**: never truncate, summarize, reorder, or rewrite earlier messages.
- **Tool message format**: tool responses must include `tool_call_id` matching the tool call `id`.
- **Idempotency**: `rollout_id` is an idempotency key for `/v1/rollout/init`.

## Documentation

- `docs/REQUEST_FLOW_EXAMPLE.md`
- `docs/ARCHITECTURE.md`
- `docs/TESTING.md`
- `docs/DEPLOYMENT.md`
