# Request Flow Example: `/rollout` ↔ `/v1/chat/completions`

This document walks through the end-to-end message flow used in `examples/e2e_test_with_servers.py` with the mock trainer (`tests/mocks/mock_trainer.py`). It shows every request/response payload, including the `response_mask` that protects PPO training data.

## Scenario

- User asks: _"Please calculate 5 plus 3, and then multiply the result by 2."_  
- RolloutServer forwards LLM calls to the mock trainer at `/v1/chat/completions`.  
- Calculator tools live in `src/rollout_server/tools/calculator.py`.

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

## 3) RolloutServer executes tool calls

RolloutServer runs calculator `add(5, 3)` → result `"8"` and appends a tool message:

```json
{"role": "tool", "content": "8", "tool_call_id": "call_abcd1234"}
```

Before the next LLM call, RolloutServer tokenizes the new prompt and computes the mask for **new tokens** (the tool output). If `"8"` tokenizes to 2 tokens, the mask will be:

```json
[0, 0]   // one 0 per tool-output token
```

## 4) RolloutServer → Trainer (turn 2): POST `/v1/chat/completions`

Second LLM call includes the assistant tool call message + the tool result. The explicit mask marks the tool tokens as `0`.

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
  "response_mask": [0, 0],            // tool output tokens only
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "logprobs": true
}
```

Mock trainer returns a plain answer (no further tools):

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm a mock LLM. The calculation result should be in the previous message."
      },
      "finish_reason": "stop"
    }
  ],
  "token_ids": [ ... ],                // LLM tokens (mask=1)
  "prompt_token_ids": [ ... ]
}
```

## 5) RolloutServer → Client: final `/rollout` response

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
    {"role": "assistant", "content": "I'm a mock LLM. The calculation result should be in the previous message."}
  ],
  "metrics": {
    "num_llm_calls": 2,
    "num_tool_calls": 1,
    "total_latency_ms": 1234
  }
}
```

## Key Takeaways

- **Turn 1**: `response_mask = null` because no tool outputs precede the first LLM call.  
- **Turn 2**: `response_mask` length equals the number of **tool-output tokens** appended between turns.  
- The mask only covers tokens added since the previous LLM call; all new LLM tokens are implicitly `1`.  
- Using the exact tokenizer (`tokenizer_name` + `tokenizer_revision`) is mandatory so mask length matches tokenization.  
- The same pattern applies to longer tool chains: each time tools append tokens, the next LLM call sends a mask of zeros matching those tokens.

