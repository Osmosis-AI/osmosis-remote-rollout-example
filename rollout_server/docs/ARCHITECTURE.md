# System Architecture

## Overview

The Remote Rollout Server is a reference implementation that demonstrates the callback-based protocol for agent trajectory generation in distributed RL training environments.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Training Infrastructure                             │
│                                                                              │
│  ┌─────────────────┐                               ┌─────────────────────┐   │
│  │  OsmosisAgent   │                               │   Training Cluster  │   │
│  │     Loop        │                               │     (GPU Node)      │   │
│  │                 │                               │                     │   │
│  │  • Orchestrates │                               │  • vLLM/SGLang      │   │
│  │    rollouts     │                               │  • PPO Training     │   │
│  │  • Distributes  │                               │  • /v1/chat/completions  │   │
│  │    work         │                               └──────────▲──────────┘   │
│  └────────┬────────┘                                          │              │
│           │                                                   │              │
│           │ POST /rollout                POST /v1/chat/completions │              │
│           │                              (with response_mask) │              │
│           ▼                                        ┌──────────┘              │
│  ┌─────────────────────────────────────────────────┼──────────────────────┐  │
│  │                      RolloutServer (External)   │                      │  │
│  │                                                 │                      │  │
│  │  ┌──────────────────────────────────────────────┼──────────────────┐   │  │
│  │  │                        FastAPI Server        │                  │   │  │
│  │  │                                              │                  │   │  │
│  │  │   POST /rollout                              │                  │   │  │
│  │  │      │                                       │                  │   │  │
│  │  │      ▼                                       │                  │   │  │
│  │  │  ┌───────────────────────────────────────────┼──────────────┐   │   │  │
│  │  │  │                    Agent Loop             │              │   │   │  │
│  │  │  │                                           │              │   │   │  │
│  │  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │   │  │
│  │  │  │  │ call_llm()  │─▶│ parse_tools │─▶│execute_tools│       │   │   │  │
│  │  │  │  │             │  │             │  │             │       │   │   │  │
│  │  │  │  │  callback ──┼──┼─────────────┼──┼─────────────┼───────┘   │   │  │
│  │  │  │  │  to trainer │  │             │  │  Calculator │           │   │  │
│  │  │  │  └─────────────┘◀─└─────────────┘◀─└─────────────┘           │   │  │
│  │  │  │        │                               │                     │   │  │
│  │  │  └────────┼───────────────────────────────┼─────────────────────┘   │  │
│  │  │           │                               │                         │  │
│  │  │  ┌────────▼────────┐    ┌─────────────────▼───────────────┐         │  │
│  │  │  │ RolloutSession  │    │       Tool Executor             │         │  │
│  │  │  │                 │    │                                 │         │  │
│  │  │  │ • Token tracking│    │  • add(a, b)                    │         │  │
│  │  │  │ • response_mask │    │  • subtract(a, b)               │         │  │
│  │  │  │   calculation   │    │  • multiply(a, b)               │         │  │
│  │  │  │ • HTTP client   │    │  • divide(a, b)                 │         │  │
│  │  │  └─────────────────┘    └─────────────────────────────────┘         │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. FastAPI Server (`server.py`)

The main entry point that exposes the `/rollout` and `/tools` endpoints.

**Responsibilities:**

- Expose `GET /tools` for tool definitions (called once at worker startup)
- Handle incoming rollout requests
- Manage tokenizer caching (LRU cache with configurable size)
- Drive the agent loop
- Return final messages with metrics

**Key Features:**

- Async-safe tokenizer cache with double-checked locking
- Configurable via environment variables
- Comprehensive error handling
- Metrics collection during rollout

### 2. RolloutSession (`session.py`)

**THE CRITICAL COMPONENT** - Manages response_mask calculation for multi-turn conversations.

**Responsibilities:**

- Track token positions between LLM calls
- Calculate response_mask for tool output tokens
- Communicate with trainer's `/v1/chat/completions` endpoint
- Maintain conversation state

**Key Implementation:**

```python
class RolloutSession:
    def __init__(self, ...):
        self.last_prompt_length = 0  # CRITICAL for mask calculation

    async def call_llm(self, sampling_params):
        # 1. Tokenize current messages using chat template
        current_prompt = self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=True, tokenize=True
        )
        current_prompt_length = len(current_prompt)

        # 2. Calculate mask for new tokens (tool outputs)
        num_new_tokens = current_prompt_length - self.last_prompt_length
        response_mask = [0] * num_new_tokens if num_new_tokens > 0 else None

        # 3. Call trainer with mask
        response = await self.http_client.post(
            f"{self.server_url}/v1/chat/completions",
            json={"messages": self.messages, "response_mask": response_mask, ...}
        )

        # 4. Update tracking
        self.last_prompt_length = current_prompt_length + len(response.json()["token_ids"])
```

### 3. Tool Executor (`tools/calculator.py`)

Async calculator tools that simulate real-world tool execution.

**Tools Available:**

- `add(a, b)` - Addition
- `subtract(a, b)` - Subtraction
- `multiply(a, b)` - Multiplication
- `divide(a, b)` - Division with error handling

**Features:**

- Random delays (10-100ms) to simulate real tool latency
- Parallel execution using `asyncio.gather()`
- JSON argument parsing

### 4. Schemas (`schemas/`)

Modular Pydantic models for protocol data structures, organized by responsibility.

**Module Structure:**

| Module | Contents |
|--------|----------|
| `schemas/__init__.py` | Re-exports all public schemas |
| `schemas/messages.py` | `Message`, `ToolCall`, `ToolDefinition`, `ToolsResponse` |
| `schemas/params.py` | `SamplingParams` |
| `schemas/rollout.py` | `RolloutRequest`, `RolloutResponse`, `RolloutMetrics`, `RolloutStatus` |
| `schemas/completions.py` | `CompletionsRequest`, `CompletionsResponse`, `CompletionsChoice` |
| `schemas/constants.py` | Shared constants (`VALID_MESSAGE_ROLES`, `MAX_TOKENS_LIMIT`, etc.) |

**Key Models:**

- `RolloutRequest` - Input to `/rollout` endpoint (includes `callback_api_key` for auth)
- `RolloutResponse` - Output with final messages, status, and metrics
- `RolloutMetrics` - Performance metrics (latency, token counts, `num_llm_calls`)
- `Message` - Conversation message format with tool_calls support
- `SamplingParams` - LLM sampling configuration
- `ToolsResponse` - Response format for `GET /tools` endpoint

### 5. Configuration (`config.py`)

Centralized configuration management with environment variable support.

**Key Settings:**

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `server_port` | `ROLLOUT_SERVER_PORT` | 9000 | Server port |
| `tokenizer_cache_size` | `TOKENIZER_CACHE_SIZE` | 5 | Max tokenizers in LRU cache |
| `http_client_timeout` | `HTTP_CLIENT_TIMEOUT` | 300.0 | HTTP timeout (seconds) |
| `tokenizer_trust_remote_code` | `TOKENIZER_TRUST_REMOTE_CODE` | true | Allow custom tokenizer code |
| `max_concurrent_rollouts` | `MAX_CONCURRENT_ROLLOUTS` | 100 | Rate limiting |

### 6. Executor (`executor.py`)

Core rollout execution logic extracted from server.py for maintainability.

**Responsibilities:**

- Application state management (`AppState` class)
- Tokenizer loading with async-safe LRU caching
- Core rollout execution with `response_mask` tracking
- Error handling and metrics collection

**Key Components:**

- `AppState`: Manages HTTP client, tokenizer cache, rate limiting semaphore
- `execute_rollout()`: Main entry point with rate limiting
- `get_or_load_tokenizer()`: Async-safe tokenizer loading with double-checked locking

## Data Flow

### 0. Tool Discovery Flow (Worker Startup)

```
OsmosisAgentLoopWorker               RolloutServer
     │                                   │
     │   GET /tools                      │
     ├──────────────────────────────────▶│
     │                                   │
     │   {"tools": [...]}                │
     │◀──────────────────────────────────┤
     │                                   │
     │   Cache tools for                 │
     │   apply_chat_template()           │
     │                                   │
```

This happens **once per worker** at startup. The tool definitions are cached and used for all rollouts to ensure the LLM knows what tools are available.

### 1. Rollout Request Flow

```
OsmosisAgentLoop                    RolloutServer                      Trainer
     │                                   │                                │
     │   POST /rollout                   │                                │
     │   {rollout_id, messages,          │                                │
     │    server_url, sampling_params}   │                                │
     ├──────────────────────────────────▶│                                │
     │                                   │                                │
     │                                   │  Load/cache tokenizer          │
     │                                   │  Create RolloutSession         │
     │                                   │                                │
     │                                   │  POST /v1/chat/completions          │
     │                                   │  {rollout_id, messages,        │
     │                                   │   response_mask, ...}          │
     │                                   ├───────────────────────────────▶│
     │                                   │                                │
     │                                   │  {choices, token_ids,          │
     │                                   │   prompt_token_ids}            │
     │                                   │◀───────────────────────────────┤
     │                                   │                                │
     │                                   │  Parse tool calls              │
     │                                   │  Execute tools locally         │
     │                                   │  (repeat until no tools)       │
     │                                   │                                │
     │   {status: COMPLETED,             │                                │
     │    final_messages,                │                                │
     │    finish_reason, metrics}        │                                │
     │◀──────────────────────────────────┤                                │
     │                                   │                                │
```

### 2. Response Mask Calculation

```
Turn 1 (Initial):
  messages = [system, user]
  Tokenize → [1,2,3,4,5] (len=5)
  last_prompt_length = 0
  response_mask = None (first turn)
  LLM returns → [6,7,8,9,10]
  Update: last_prompt_length = 5 + 5 = 10

Turn 2 (After tool output):
  messages = [system, user, assistant, tool_output]
  Tokenize → [1,2,3,4,5,6,7,8,9,10,11,12,13] (len=13)
  num_new_tokens = 13 - 10 = 3 (tool output tokens)
  response_mask = [0, 0, 0]  ← Mark tool tokens!
  LLM returns → [14,15,16]
  Update: last_prompt_length = 13 + 3 = 16
```

## Termination Conditions

### RolloutServer's Control

RolloutServer has **full control** over when to terminate a rollout. The `max_turns` and `max_tokens_total` parameters in `RolloutRequest` are **advisory**.

### Standard Termination Conditions

| Condition | `finish_reason` | Description |
|-----------|-----------------|-------------|
| No tool calls | `"stop"` | LLM generated response without tool calls (natural completion) |
| Turn limit | `"max_turns"` | Reached `max_turns` limit |
| Token limit | `"max_tokens"` | Reached `max_tokens_total` limit |
| Error | `"error"` | Rollout failed (set `error_message`) |

### Comparison with Local Mode

Remote rollout uses a **single `max_turns`** counter, while verl's local `ToolAgentLoop` uses separate counters:

| Parameter | Remote Mode | Local Mode (`ToolAgentLoop`) |
|-----------|-------------|------------------------------|
| Turn limit | `max_turns` (single) | `max_assistant_turns` + `max_user_turns` (separate) |
| Token limit | `max_tokens_total` | `response_length` |

**Why the difference?**

Remote mode intentionally simplifies turn counting because:
1. RolloutServer can implement any termination logic internally
2. Decoupling agent logic from training is the design goal
3. Training correctness depends on `AgentLoopOutput`, not termination strategy

### num_turns Semantics

`RolloutMetrics.num_llm_calls` represents the number of LLM generation calls:

```
User: "Calculate 15*23"
  Turn 1: LLM → "I'll use calculator" + tool_call    → num_llm_calls = 1
  Turn 2: LLM → "The result is 345"                  → num_llm_calls = 2

Final: AgentLoopOutput.num_turns = metrics.num_llm_calls = 2
```

OsmosisAgentLoop uses this value to populate `AgentLoopOutput.num_turns` for training metrics.

### Optional Fine-Grained Control

If callers need local-mode-style control, they can pass hints in `metadata`:

```python
RolloutRequest(
    ...,
    metadata={
        "max_assistant_turns": 5,
        "max_user_turns": 5,
        "termination_strategy": "task_completion"
    }
)
```

**Note**: These are optional hints. This implementation does not currently use them, but custom RolloutServer implementations may choose to support them.

Reference: Remote Rollout Design Documentation, Section 11

## Configuration

### Environment Variables

| Variable                      | Default | Description                    |
| ----------------------------- | ------- | ------------------------------ |
| `ROLLOUT_SERVER_PORT`         | `9000`  | Server port                    |
| `TOKENIZER_CACHE_SIZE`        | `5`     | Max tokenizers in LRU cache    |
| `HTTP_CLIENT_TIMEOUT`         | `300.0` | HTTP request timeout (seconds) |
| `TOKENIZER_TRUST_REMOTE_CODE` | `true`  | Allow custom tokenizer code    |

### Trainer Callback Ports

When connecting to the training cluster:
- **Default callback port**: `8081` (base port for OsmosisAgentLoopWorker)
- **Port allocation**: `base_port + worker_index` (e.g., worker_0 → 8081, worker_1 → 8082)
- **Docker exposed range**: `8080-8130` (ensure firewall allows outbound to this range)
- **Endpoint**: `/v1/chat/completions` (OpenAI-compatible)

### Resource Requirements

| Resource | Recommendation | Notes                               |
| -------- | -------------- | ----------------------------------- |
| Memory   | 4-8GB          | Each tokenizer ~1-2GB               |
| CPU      | 2+ cores       | Async I/O benefits from concurrency |
| Network  | Low latency    | Critical for trainer callbacks      |

## Error Handling

### Error Categories

1. **TokenizerLoadError** - Failed to load tokenizer from HuggingFace
2. **ToolExecutionError** - Tool execution failed
3. **HTTP Errors** - Network/communication issues with trainer
4. **ValidationError** - Invalid request data

### Error Response Format

```json
{
  "rollout_id": "...",
  "status": "ERROR",
  "error_message": "Human-readable error description",
  "final_messages": []
}
```

## Thread Safety

The implementation is designed for async FastAPI environments:

1. **Tokenizer Cache** - Uses `asyncio.Lock()` with double-checked locking
2. **HTTP Client** - Shared `httpx.AsyncClient` with connection pooling
3. **Session State** - Each request creates its own `RolloutSession`

## See Also

- [RESPONSE_MASK_GUIDE.md](RESPONSE_MASK_GUIDE.md) - Deep dive on response_mask
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment guide
- [TESTING.md](TESTING.md) - Test suite documentation
