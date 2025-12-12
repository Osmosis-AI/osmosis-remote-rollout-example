# Testing Guide

This guide explains how to run tests for the async-init RolloutServer protocol.

## Test Suite Overview

| Type | Location | External Dependencies |
|------|----------|-----------------------|
| **Unit** | `tests/unit/` | None |
| **Integration** | `tests/integration/` | None (in-process mock trainer) |
| **E2E** | `tests/e2e/` | Requires running servers |

## Run tests

All commands below must be run from the `rollout_server/` directory.

```bash
uv run pytest
```

If you prefer not to rely on an installed package, you can run with an explicit module path:

```bash
PYTHONPATH=src python -m pytest
```

## Integration tests

Integration tests run RolloutServer in-process and intercept its trainer callbacks using an in-process mock trainer.

What gets validated:
- `POST /init` returns `202 Accepted` and tool definitions.
- RolloutServer calls `POST /v1/chat/completions` to drive the loop.
- RolloutServer posts exactly one `POST /v1/rollout/completed` callback.

## E2E testing with real servers

### One-command environment

```bash
./scripts/start_test_environment.sh
```

Then run the interactive demo:

```bash
uv run python examples/e2e_test_with_servers.py
```

Stop services:

```bash
./scripts/stop_test_environment.sh
```

### Manual API testing

1) Open the FastAPI docs UI at `http://localhost:9000/docs`.
2) Use `POST /init` with a payload like:

```json
{
  "rollout_id": "test-rollout-123",
  "server_url": "http://localhost:9001",
  "messages": [
    {"role": "user", "content": "Please calculate 5 plus 3."}
  ],
  "completion_params": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 128,
    "logprobs": true
  },
  "max_turns": 10,
  "max_tokens_total": 8192
}
```

3) Fetch the completed rollout from the mock trainer:

```bash
curl http://localhost:9001/v1/rollout/completed/test-rollout-123
```
