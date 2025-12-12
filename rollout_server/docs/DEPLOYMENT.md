# Deployment Guide

This document describes how to deploy the RolloutServer.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROLLOUT_SERVER_PORT` | `9000` | HTTP port for RolloutServer |
| `HTTP_CLIENT_TIMEOUT` | `300.0` | Timeout (seconds) for callbacks to the training side |
| `MAX_CONCURRENT_ROLLOUTS` | `100` | Concurrency limit for active rollouts |
| `ROLLOUT_TIMEOUT_SECONDS` | `600.0` | Upper bound for a single rollout execution |

## Run

```bash
cd rollout_server
uv run python -m rollout_server.server
```

## Callback endpoints

RolloutServer must be able to reach the training side at:
- `{server_url}/v1/chat/completions`
- `{server_url}/v1/rollout/completed`

If `api_key` is provided in the `/init` request, RolloutServer sends:

```
Authorization: Bearer <api_key>
```

## Operational notes

- Ensure the service can make outbound HTTP requests to the training cluster.
- Ensure the training cluster can accept inbound requests at `/v1/rollout/completed`.
- Keep the conversation history append-only across turns.
