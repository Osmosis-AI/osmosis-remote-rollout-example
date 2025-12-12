# Request Flow Example: `/init` ↔ callbacks

This document walks through the end-to-end flow used by the RolloutServer async-init protocol.

## Scenario

- User asks: _"Please calculate 5 plus 3, and then multiply the result by 2."_
- Training side calls `POST /init` on RolloutServer.
- RolloutServer drives the agent loop by calling the trainer's `POST /v1/chat/completions` endpoint.
- When finished, RolloutServer posts a single completion callback to `POST /v1/rollout/completed`.

## 1) Training → RolloutServer: `POST /init`

```json
{
  "rollout_id": "demo-1234",
  "server_url": "http://localhost:9001",
  "api_key": null,
  "messages": [
    {"role": "system", "content": "You are a helpful calculator assistant with access to calculator tools."},
    {"role": "user", "content": "Please calculate 5 plus 3, and then multiply the result by 2."}
  ],
  "completion_params": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "stop": null,
    "logprobs": true
  },
  "tool_server_url": null,
  "max_turns": 10,
  "max_tokens_total": 8192,
  "metadata": {}
}
```

### `202 Accepted` response

RolloutServer returns tools for this rollout:

```json
{
  "rollout_id": "demo-1234",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
          "type": "object",
          "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"}
          },
          "required": ["a", "b"]
        }
      }
    }
  ]
}
```

After returning `202`, RolloutServer executes the rollout asynchronously.

## 2) RolloutServer → Trainer (turn 1): `POST {server_url}/v1/chat/completions`

RolloutServer calls the trainer for the next assistant message:

```json
{
  "model": "default",
  "rollout_id": "demo-1234",
  "messages": [
    {"role": "system", "content": "You are a helpful calculator assistant with access to calculator tools."},
    {"role": "user", "content": "Please calculate 5 plus 3, and then multiply the result by 2."}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "stop": null,
  "logprobs": true
}
```

If `api_key` was provided in the `/init` request, RolloutServer includes:

```
Authorization: Bearer <api_key>
```

### Trainer response (example)

```json
{
  "id": "demo-1234",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'll calculate that for you.",
        "tool_calls": [
          {
            "id": "call_abcd1234",
            "type": "function",
            "function": {"name": "add", "arguments": "{\"a\": 5, \"b\": 3}"}
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "token_ids": [0, 1, 2],
  "logprobs": [0.0, 0.0, 0.0],
  "prompt_token_ids": [0, 1, 2, 3]
}
```

## 3) RolloutServer executes tool call(s)

RolloutServer executes the requested tool and appends a tool message:

```json
{"role": "tool", "content": "8", "tool_call_id": "call_abcd1234"}
```

## 4) RolloutServer → Trainer (turn 2): `POST {server_url}/v1/chat/completions`

RolloutServer sends the updated conversation (append-only):

```json
{
  "model": "default",
  "rollout_id": "demo-1234",
  "messages": [
    {"role": "system", "content": "You are a helpful calculator assistant with access to calculator tools."},
    {"role": "user", "content": "Please calculate 5 plus 3, and then multiply the result by 2."},
    {
      "role": "assistant",
      "content": "I'll calculate that for you.",
      "tool_calls": [
        {
          "id": "call_abcd1234",
          "type": "function",
          "function": {"name": "add", "arguments": "{\"a\": 5, \"b\": 3}"}
        }
      ]
    },
    {"role": "tool", "content": "8", "tool_call_id": "call_abcd1234"}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "stop": null,
  "logprobs": true
}
```

Trainer returns the final assistant message (no tool calls), and RolloutServer ends the rollout.

## 5) RolloutServer → Trainer: `POST {server_url}/v1/rollout/completed`

RolloutServer posts the final transcript once:

```json
{
  "rollout_id": "demo-1234",
  "status": "COMPLETED",
  "final_messages": [
    {"role": "system", "content": "You are a helpful calculator assistant with access to calculator tools."},
    {"role": "user", "content": "Please calculate 5 plus 3, and then multiply the result by 2."},
    {
      "role": "assistant",
      "content": "I'll calculate that for you.",
      "tool_calls": [
        {
          "id": "call_abcd1234",
          "type": "function",
          "function": {"name": "add", "arguments": "{\"a\": 5, \"b\": 3}"}
        }
      ]
    },
    {"role": "tool", "content": "8", "tool_call_id": "call_abcd1234"},
    {"role": "assistant", "content": "The calculation is complete."}
  ],
  "finish_reason": "stop",
  "metrics": {
    "num_llm_calls": 2,
    "num_tool_calls": 1,
    "total_latency_ms": 150.0
  },
  "extra_fields": {}
}
```

If the rollout fails, RolloutServer posts a callback with `status="ERROR"` and an `error_message`.

## Key requirements

- **Append-only messages**: Do not truncate, summarize, reorder, or rewrite earlier messages.
- **Tool result format**: Tool responses should include `tool_call_id` matching the corresponding tool call.
- **Idempotency**: `rollout_id` is an idempotency key; repeated `/init` must not start duplicate rollouts.
- **Authentication**: If `api_key` is provided, include it as a Bearer token in both callback endpoints.
