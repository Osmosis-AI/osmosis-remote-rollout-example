# System Architecture

**Last Updated**: 2025-12-05

## Overview

The Remote Rollout Server is a reference implementation that demonstrates the callback-based protocol for agent trajectory generation in distributed RL training environments.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Training Infrastructure                            │
│  ┌─────────────────┐                              ┌─────────────────────┐   │
│  │  OsmosisAgent   │   POST /rollout              │   Training Cluster  │   │
│  │     Loop        │ ─────────────────────────────▶│     (GPU Node)     │   │
│  │                 │                              │                     │   │
│  │  • Orchestrates │                              │  • vLLM/SGLang      │   │
│  │    rollouts     │                              │  • PPO Training     │   │
│  │  • Distributes  │                              │  • /v1/completions  │   │
│  │    work         │                              └──────────▲──────────┘   │
│  └─────────────────┘                                         │              │
│           │                                                  │              │
│           │ POST /rollout                                    │              │
│           ▼                                                  │              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      RolloutServer (External)                       │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                        FastAPI Server                         │  │   │
│  │  │                                                               │  │   │
│  │  │   POST /rollout                                               │  │   │
│  │  │      │                                                        │  │   │
│  │  │      ▼                                                        │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │   │
│  │  │  │                    Agent Loop                           │  │  │   │
│  │  │  │                                                         │  │  │   │
│  │  │  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │  │  │   │
│  │  │  │  │ call_llm()  │──▶│ parse_tools │──▶│execute_tools│    │  │  │   │
│  │  │  │  │             │   │             │   │             │    │  │  │   │
│  │  │  │  │ ▲ callback  │   │             │   │  Calculator │    │  │  │   │
│  │  │  │  │ │ to trainer│   │             │   │  Tools      │    │  │  │   │
│  │  │  │  └─┴───────────┘◀──└─────────────┘◀──└─────────────┘    │  │  │   │
│  │  │  │        │                                    │           │  │  │   │
│  │  │  │        │  POST /v1/completions              │           │  │  │   │
│  │  │  │        │  (with response_mask) ─────────────┼───────────┼──┼──┼───┘
│  │  │  │        │                                    │           │  │  │
│  │  │  └────────┼────────────────────────────────────┼───────────┘  │  │
│  │  │           │                                    │              │  │
│  │  │  ┌────────▼────────┐     ┌─────────────────────▼───────────┐  │  │
│  │  │  │ RolloutSession  │     │       Tool Executor             │  │  │
│  │  │  │                 │     │                                 │  │  │
│  │  │  │ • Token tracking│     │  • add(a, b)                    │  │  │
│  │  │  │ • response_mask │     │  • subtract(a, b)               │  │  │
│  │  │  │   calculation   │     │  • multiply(a, b)               │  │  │
│  │  │  │ • HTTP client   │     │  • divide(a, b)                 │  │  │
│  │  │  └─────────────────┘     └─────────────────────────────────┘  │  │
│  │  └───────────────────────────────────────────────────────────────┘  │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. FastAPI Server (`server.py`)

The main entry point that exposes the `/rollout` endpoint.

**Responsibilities:**

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
- Communicate with trainer's `/v1/completions` endpoint
- Maintain conversation state

**Key Implementation:**

```python
class RolloutSession:
    def __init__(self, ...):
        self.last_prompt_length = 0  # CRITICAL for mask calculation

    async def call_llm(self, sampling_params):
        # 1. Tokenize current messages
        current_prompt_length = len(tokenize(messages))

        # 2. Calculate mask for new tokens (tool outputs)
        num_new_tokens = current_prompt_length - self.last_prompt_length
        response_mask = [0] * num_new_tokens if num_new_tokens > 0 else None

        # 3. Call trainer with mask
        response = await call_trainer(response_mask=response_mask)

        # 4. Update tracking
        self.last_prompt_length = current_prompt_length + len(response.token_ids)
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

### 4. Schemas (`schemas.py`)

Pydantic models for protocol data structures.

**Key Models:**

- `RolloutRequest` - Input to `/rollout` endpoint
- `RolloutResponse` - Output with final messages and status
- `RolloutMetrics` - Performance metrics (latency, token counts)
- `Message` - Conversation message format
- `SamplingParams` - LLM sampling configuration

## Data Flow

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
     │                                   │  POST /v1/completions          │
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

## Configuration

### Environment Variables

| Variable                      | Default | Description                    |
| ----------------------------- | ------- | ------------------------------ |
| `ROLLOUT_SERVER_PORT`         | `9000`  | Server port                    |
| `TOKENIZER_CACHE_SIZE`        | `5`     | Max tokenizers in LRU cache    |
| `HTTP_CLIENT_TIMEOUT`         | `300.0` | HTTP request timeout (seconds) |
| `TOKENIZER_TRUST_REMOTE_CODE` | `false` | Allow custom tokenizer code    |

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
