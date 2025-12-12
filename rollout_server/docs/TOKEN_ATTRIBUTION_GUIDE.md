# Token Attribution & Append-only Conversation Guide

This guide explains the most important correctness requirement for remote rollout:
**the message history must be append-only across turns**.

In Traingate's remote rollout implementation, the training side is responsible for:
- Tokenization
- Training-time token tracking
- Constructing the PPO `response_mask`

Because of this, RolloutServer must maintain a stable, append-only conversation so the
training side can align prompt tokens across multiple LLM calls.

## Why append-only matters

Remote rollout is multi-turn:

- Turn 1: RolloutServer calls the training side for an assistant message.
- Turn 2+: RolloutServer appends tool results and calls again.

The training side records tokens across these calls. If earlier messages change, then
it can no longer align the new prompt with the previously recorded prompt token prefix.

## Requirements

### 1) Append-only message history

Do not:
- Truncate history
- Summarize earlier content
- Reorder messages
- Rewrite earlier messages

Do:
- Always append new assistant/tool messages to the end.

### 2) Tool response formatting

Tool responses must be appended as messages with:
- `role`: `"tool"`
- `content`: the tool result text
- `tool_call_id`: the corresponding tool call `id`

Example:

```json
{"role": "tool", "content": "8", "tool_call_id": "call_abcd1234"}
```

### 3) Idempotency

`rollout_id` is an idempotency key for `POST /init`. Repeated `/init` with the same
`rollout_id` must not start duplicate rollouts.

### 4) Completion callback

RolloutServer must post exactly one completion callback to:

- `POST {server_url}/v1/rollout/completed`

Use `status="COMPLETED"` or `status="ERROR"`.

## Common pitfalls

- **Context modification**: changing earlier messages breaks token alignment.
- **Missing `tool_call_id`**: tool results cannot be associated with the correct call.
- **Multiple completions for one rollout**: posting more than once can confuse downstream consumers.
- **Not posting completion on error**: the training side may wait until timeout.

## Debugging checklist

- Confirm RolloutServer returns `202 Accepted` on `POST /init`.
- Confirm RolloutServer is calling `POST {server_url}/v1/chat/completions`.
- Confirm RolloutServer posts `POST {server_url}/v1/rollout/completed`.
- Inspect the final transcript for correct tool message structure.

