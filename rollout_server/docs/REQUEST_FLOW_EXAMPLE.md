# Request Flow Example: `/rollout` ↔ `/v1/chat/completions`

This document walks through the end-to-end message flow used in `examples/e2e_test_with_servers.py` with the mock trainer (`tests/mocks/mock_trainer.py`). It shows every request/response payload, including the `response_mask` that protects PPO training data.

## Scenario

- User asks: _"Please calculate 5 plus 3, and then multiply the result by 2."_  
- RolloutServer forwards LLM calls to the mock trainer at `/v1/chat/completions`.  
- Calculator tools live in `src/rollout_server/tools/calculator.py`.
- In this walkthrough the assistant issues **two tool calls** (add then multiply) and returns the final answer **16**. All request/response fields mirror the real payload shapes used by the server and mock trainer.

## 1) Client → RolloutServer: POST `/rollout`

```json
{
  "rollout_id": "demo-1234",            // any UUID
  "server_url": "http://localhost:9001",// mock trainer
  "messages": [
    {"role": "system", "content": "You are a helpful calculator assistant with access to calculator tools."},
    {"role": "user", "content": "Please calculate 5 plus 3, and then multiply the result by 2."}
  ],
  "sampling_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512, "logprobs": true},
  "tokenizer_name": "Qwen/Qwen3-8B",
  "tokenizer_revision": "main",
  "max_turns": 10,
  "max_tokens_total": 8192
}
```

RolloutServer now drives the loop.

## 2) RolloutServer → Trainer (turn 1): POST `/v1/chat/completions`

First LLM call (no prior tool output) ⇒ `response_mask` is omitted/`null`.

```json
{
  "rollout_id": "demo-1234",
  "messages": [
    {"role": "system", "content": "You are a helpful calculator assistant with access to calculator tools."},
    {"role": "user", "content": "Please calculate 5 plus 3, and then multiply the result by 2."}
  ],
  "response_mask": null,               // first turn → no tool tokens yet
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "logprobs": true
}
```

Mock trainer responds with a tool call:

```json
{
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
  "token_ids": [ ... ],                // LLM tokens (mask=1)
  "prompt_token_ids": [ ... ]
}
```

## 3) RolloutServer executes tool call #1

RolloutServer runs calculator `add(5, 3)` → result `"8"` and appends a tool message:

```json
{"role": "tool", "content": "8", "tool_call_id": "call_abcd1234"}
```

Before the next LLM call, RolloutServer tokenizes the new prompt and computes the mask for **new tokens** (the tool output). If `"8"` tokenizes to 1 token, the mask will be:

```json
[0]   // one 0 per tool-output token
```

### Understanding `response_mask`

The `response_mask` field indicates which tokens were added to the prompt **since the last LLM call**:
- `0`: Token is a tool/system output (not LLM-generated, excluded from PPO loss)
- `1`: Token is LLM-generated (participates in PPO training)

In this example:
- Turn 1: No tokens were added before the first LLM call → `response_mask = null`
- Turn 2: Tool output `"8"` was added (1 token) → `response_mask = [0]`

**Critical requirement**: The trainer enforces strict mask validation. If `response_mask` is missing or its length doesn't match the number of new tokens, the request will be rejected with HTTP 422. This prevents training data corruption from fragile diff-based inference.

## 4) RolloutServer → Trainer (turn 2): POST `/v1/chat/completions`

Second LLM call includes the assistant tool call message + the tool result. The explicit mask marks the tool tokens as `0`. The mock trainer now returns a **second tool call** to complete the multiplication.

```json
{
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
  "response_mask": [0],               // tool output tokens only
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "logprobs": true
}
```

### How the Trainer Processes This Request

When the trainer receives this request:

1. **Tokenizes the messages**: Applies `tokenizer.apply_chat_template()` with the provided tools
2. **Validates response_mask**: Checks that `len(response_mask) == num_new_tokens` (tokens added since turn 1)
3. **Extracts tool tokens**: Uses prompt diff to identify the new tokens (the tool output `"8"`)
4. **Records tokens with masks**:
   - Tool tokens: `response_ids=[token1, ...]`, `response_mask=[0, ...]` (from request)
   - LLM tokens: `response_ids=[tokenX, ...]`, `response_mask=[1, 1, ...]` (auto-assigned)
5. **Returns response**: The LLM's generated tokens with logprobs

If `response_mask` is missing or length mismatches, the trainer rejects with HTTP 422.

Mock trainer returns the second tool call:

```json
{
  "id": "demo-1234",                   // echoed request_id/rollout_id
  "object": "chat.completion",
  "created": 1730000000,
  "model": "default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Continuing the calculation.",
        "tool_calls": [
          {
            "id": "call_efgh5678",
            "type": "function",
            "function": {"name": "multiply", "arguments": "{\"a\": 8, \"b\": 2}"}
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "token_ids": [ ... ],                // LLM tokens (mask=1)
  "logprobs": [ ... ],                 // one logprob per response token
  "prompt_token_ids": [ ... ]          // prompt tokens for verification
}
```

## 5) RolloutServer executes tool call #2

RolloutServer runs calculator `multiply(8, 2)` → result `"16"` and appends the tool message:

```json
{"role": "tool", "content": "16", "tool_call_id": "call_efgh5678"}
```

Mask for the new tool tokens (assume `"16"` tokenizes to 1 token):

```json
[0]
```

## 6) RolloutServer → Trainer (turn 3): POST `/v1/chat/completions`

Third LLM call includes both tool interactions. The mock trainer now returns the final natural-language answer.

```json
{
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
    {"role": "tool", "content": "8", "tool_call_id": "call_abcd1234"},
    {
      "role": "assistant",
      "content": "Continuing the calculation.",
      "tool_calls": [
        {
          "id": "call_efgh5678",
          "type": "function",
          "function": {"name": "multiply", "arguments": "{\"a\": 8, \"b\": 2}"}
        }
      ]
    },
    {"role": "tool", "content": "16", "tool_call_id": "call_efgh5678"}
  ],
  "response_mask": [0],               // mask for tool output "16"
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "logprobs": true
}
```

Mock trainer returns the final answer:

```json
{
  "id": "demo-1234",
  "object": "chat.completion",
  "created": 1730000001,
  "model": "default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "5 plus 3 equals 8. Multiplying 8 by 2 gives 16."
      },
      "finish_reason": "stop"
    }
  ],
  "token_ids": [ ... ],                // LLM tokens (mask=1)
  "logprobs": [ ... ],
  "prompt_token_ids": [ ... ]
}
```

## 7) RolloutServer → Client: final `/rollout` response

RolloutServer returns the full transcript and metrics:

```json
{
  "status": "COMPLETED",
  "finish_reason": "stop",
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
    {
      "role": "assistant",
      "content": "Continuing the calculation.",
      "tool_calls": [
        {
          "id": "call_efgh5678",
          "type": "function",
          "function": {"name": "multiply", "arguments": "{\"a\": 8, \"b\": 2}"}
        }
      ]
    },
    {"role": "tool", "content": "16", "tool_call_id": "call_efgh5678"},
    {"role": "assistant", "content": "5 plus 3 equals 8. Multiplying 8 by 2 gives 16."}
  ],
  "metrics": {
    "num_llm_calls": 3,
    "num_tool_calls": 2,
    "total_latency_ms": 1500
  }
}
```

## Key Takeaways

### `response_mask` Requirements

- **Turn 1**: `response_mask = null` (or omitted) because no tokens were added to the prompt before the first LLM call
- **Turn 2+**: `response_mask` **MUST** be provided, containing exactly one mask value (0 or 1) per token added since the last LLM call
  - Tool output tokens: mask value = `0` (excluded from PPO training loss)
  - System/formatting tokens: mask value = `0` or `1` depending on whether they should participate in training
  - Length mismatches are rejected by the trainer with HTTP 422 error

### Trainer Behavior

- The trainer tokenizes incoming messages using the same tokenizer as RolloutServer (`tokenizer_name` + `tokenizer_revision`)
- For turn 2+, it uses prompt length diff to detect new tokens, then applies the provided `response_mask` to those tokens
- LLM-generated tokens in the response automatically receive mask value `1` (always participate in PPO training)
- The complete sequence (tool tokens + LLM tokens) is accumulated in the session for final AgentLoopOutput

### Why Explicit Masks Matter

- **Without explicit masks**: The trainer falls back to fragile diff-based inference that breaks when RolloutServer performs context truncation, summarization, or reordering
- **With explicit masks**: RolloutServer has full control over which tokens participate in training, enabling advanced features like:
  - Multi-turn conversations with proper token attribution
  - Context window management without corrupting training data
  - Custom token masking strategies (e.g., masking system prompts)

### Implementation Pattern

For longer tool chains, the pattern repeats:
1. Tool execution adds tokens → RolloutServer computes mask for those tokens
2. Next LLM call includes `response_mask` matching the new token count
3. Trainer validates, records tokens with masks, generates LLM response
4. Repeat until task completion or termination condition

