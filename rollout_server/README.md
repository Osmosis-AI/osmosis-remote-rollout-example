# Remote Rollout Server - Reference Implementation

A reference implementation of the Remote Rollout Protocol, demonstrating correct `response_mask` handling for multi-turn conversations with tool use.

## Overview

This package provides:
1. **Reference implementation** for external developers implementing RolloutServer
2. **End-to-end testing infrastructure** for remote rollout systems
3. **Protocol compliance examples** based on official documentation

**Critical Focus**: Demonstrates CORRECT `response_mask` handling - the #1 source of bugs and training data corruption in remote rollout implementations.

## What is Remote Rollout?

Remote Rollout is an architecture that separates agent trajectory generation from training infrastructure:

- **Training cluster** (GPU): Runs LLM inference (vLLM/SGLang) and PPO training
- **RolloutServer** (external): Drives the agent loop (tool parsing, execution, state management)

**Key Benefits**:
- Decouple agent logic from training infrastructure
- Teams iterate independently
- Standard OpenAI-compatible `/v1/chat/completions` endpoint
- Support multi-turn conversations with tools

## Quick Start

### Installation

**IMPORTANT:** All commands below must be run from the `rollout_server/` directory.

```bash
# Navigate to the rollout_server directory
cd rollout_server

# Install dependencies and set up the package
uv sync
```

This will:
- Install all dependencies
- Build and install the `rollout-server` package in development mode
- Create a virtual environment in `.venv/`

### Start the Server

```bash
# From the rollout_server/ directory:
uv run python -m rollout_server.server
# Server starts on http://0.0.0.0:9000

# Or specify custom port:
ROLLOUT_SERVER_PORT=9100 uv run python -m rollout_server.server
```

### Port Configuration

| Service | Default Port | Description |
|---------|--------------|-------------|
| **RolloutServer** | `9000` | Receives POST /rollout requests from OsmosisAgentLoop |
| **Trainer callback** | `8081` | OsmosisAgentLoopWorker's /v1/chat/completions endpoint |
| **Trainer port range** | `8080-8130` | Docker exposed range for multiple workers |

**Note**: The trainer uses port `8081 + worker_index` for each worker. Ensure your firewall allows outbound connections to the trainer's port range.

You should see output like:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     RolloutServer startup complete
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```

### Basic Usage

```python
import httpx

# Send rollout request
response = await httpx.post(
    "http://localhost:9000/rollout",
    json={
        "rollout_id": "test-123",
        "server_url": "http://trainer:8081",  # Trainer's /v1/chat/completions endpoint
        "messages": [
            {"role": "user", "content": "Calculate 15 * 23"}
        ],
        "sampling_params": {
            "temperature": 0.7,
            "max_tokens": 512
        },
        "tokenizer_name": "Qwen/Qwen3-8B",
        "max_turns": 10
    }
)

print(response.json())
```

### Run Tests

```bash
# From the rollout_server/ directory:

# Run all tests
uv run pytest

# Run integration tests only
uv run pytest tests/integration/

# Run with coverage
uv run pytest --cov=rollout_server
```

## Architecture

### Protocol Flow

```
1. OsmosisAgentLoop sends POST /rollout request
   ↓
2. RolloutServer drives agent loop:
   a. Call trainer's /v1/chat/completions (with response_mask!)
   b. Parse tool calls from LLM response
   c. Execute tools (calculator in this example)
   d. Append tool outputs to conversation
   e. Repeat until no more tool calls
   ↓
3. RolloutServer returns final messages
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **RolloutSession** | `src/rollout_server/session.py` | **CRITICAL** - Manages response_mask calculation |
| **FastAPI Server** | `src/rollout_server/server.py` | POST /rollout and GET /tools endpoints |
| **Executor** | `src/rollout_server/executor.py` | Core rollout execution logic |
| **Calculator Tools** | `src/rollout_server/tools/calculator.py` | Async tools with random delays |
| **Schemas** | `src/rollout_server/schemas/` | Protocol data structures (modular) |
| **Config** | `src/rollout_server/config.py` | Centralized configuration |

### CRITICAL: Response Mask Handling

**The `response_mask` is the most critical part of this implementation.**

- `response_mask[i] = 1`: Token i is LLM-generated (used for PPO training loss)
- `response_mask[i] = 0`: Token i is tool/system output (excluded from training loss)

**Without explicit masks**, the trainer falls back to fragile diff-based inference that WILL BREAK if you:
- Truncate context (remove old messages)
- Summarize conversation history
- Reorder messages
- Re-tokenize with different settings

**Consequence of Incorrect Masks:**
- **False positives** (LLM tokens marked as 0): Model can't learn → training stagnates
- **False negatives** (tool tokens marked as 1): Model learns to predict tool outputs → training diverges

See [`docs/RESPONSE_MASK_GUIDE.md`](docs/RESPONSE_MASK_GUIDE.md) for detailed explanation.

## Implementation Pattern

The correct response_mask calculation pattern:

```python
class RolloutSession:
    def __init__(self, rollout_id, tokenizer, server_url):
        self.rollout_id = rollout_id
        self.tokenizer = tokenizer
        self.server_url = server_url
        self.messages = []
        self.last_prompt_length = 0  # CRITICAL for mask calculation

    async def call_llm(self, sampling_params):
        # 1. Tokenize current messages
        current_prompt = self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=True, tokenize=True
        )
        current_prompt_length = len(current_prompt)

        # 2. Calculate mask for tokens added since last call (CRITICAL!)
        if self.last_prompt_length > 0:
            num_new_tokens = current_prompt_length - self.last_prompt_length
            response_mask = [0] * num_new_tokens if num_new_tokens > 0 else None
        else:
            response_mask = None  # First turn

        # 3. Call trainer with EXPLICIT mask
        response = await httpx.post(
            f"{self.server_url}/v1/chat/completions",
            json={
                "rollout_id": self.rollout_id,
                "messages": self.messages,
                "response_mask": response_mask,  # CRITICAL!
                **sampling_params
            }
        )

        # 4. Update tracking
        llm_token_count = len(response["token_ids"])
        self.last_prompt_length = current_prompt_length + llm_token_count

        return response

    def append_tool_outputs(self, tool_results):
        """Append tool results. Don't update last_prompt_length here!"""
        self.messages.extend(tool_results)
```

**Key Points:**
1. Track `last_prompt_length` between LLM calls
2. Calculate `num_new_tokens` = current - last (these are tool outputs)
3. Provide `response_mask = [0] * num_new_tokens` for tool tokens
4. Update `last_prompt_length` after each LLM call
5. **Don't update `last_prompt_length` when appending tool outputs!**

### num_turns Semantics

`RolloutMetrics.num_llm_calls` is used by OsmosisAgentLoop to populate `AgentLoopOutput.num_turns`:

```
User: "Calculate 15*23"
  Turn 1: LLM → "I'll use calculator" + tool_call    → num_llm_calls = 1
  Turn 2: LLM → "The result is 345"                  → num_llm_calls = 2

Final: AgentLoopOutput.num_turns = metrics.num_llm_calls = 2
```

**Note**: This differs from verl's local `ToolAgentLoop` which tracks `assistant_turns` and `user_turns` separately. Remote rollout uses a single counter for simplicity.

### Termination Control

`max_turns` and `max_tokens_total` in `RolloutRequest` are **advisory parameters**. RolloutServer has full control over termination logic and may implement more sophisticated strategies.

Optional fine-grained control via `metadata` field:
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

See the protocol documentation for detailed comparison with local mode.

## Documentation

| Document | Description |
|----------|-------------|
| **[README.md](README.md)** | Quick start and overview (this file) |
| **[docs/RESPONSE_MASK_GUIDE.md](docs/RESPONSE_MASK_GUIDE.md)** | **CRITICAL** - Deep dive on response_mask |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | System design, components, and data flow |
| **[docs/TESTING.md](docs/TESTING.md)** | Complete testing documentation |
| **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** | Production deployment guide |

## Examples

| Example | Description |
|---------|-------------|
| `examples/basic_example.py` | Single-turn conversation (no tools) |
| `examples/calculator_example.py` | Multi-turn with calculator tools |
| `examples/mock_trainer_example.py` | Complete end-to-end demo with mock trainer |

## Testing

### Quick Start

```bash
cd rollout_server

# Run all automated tests (unit + integration)
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=rollout_server --cov-report=term-missing
```

### Test Types

| Type | Location | Description |
|------|----------|-------------|
| **Unit Tests** | `tests/unit/` | Fast, isolated component tests |
| **Integration Tests** | `tests/integration/` | In-process tests with mocked trainer |
| **E2E Tests** | `tests/e2e/` | Requires running servers (ports 9000, 9001) |

### Testing with Mock Trainer

For E2E testing, use the one-command test environment:

```bash
# Start mock trainer + rollout server
./scripts/start_test_environment.sh

# Run E2E tests
uv run pytest tests/e2e/ -v -m requires_servers

# Stop when done
./scripts/stop_test_environment.sh
```

For comprehensive testing documentation, see **[docs/TESTING.md](docs/TESTING.md)**.

## Protocol Reference

### GET /tools (Tool Definitions)

The trainer calls this endpoint once at worker startup to fetch available tool definitions. These tools are passed to `apply_chat_template()` so the LLM knows what tools it can use.

**Request**: No body required.

**Response**:
```json
{
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

### RolloutRequest (OsmosisAgentLoop → RolloutServer)

```python
{
    "rollout_id": "550e8400-e29b-41d4-a716-446655440000",
    "server_url": "http://trainer-node-1:8081",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    "sampling_params": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "logprobs": true
    },
    "tokenizer_name": "Qwen/Qwen3-8B",
    "tokenizer_revision": "main",
    "max_turns": 10,
    "max_tokens_total": 8192,
    "callback_api_key": "secret-key-123"  # Optional: API key for callback authentication
}
```

**Important Fields**:
- `callback_api_key` (optional): API key for authenticating callbacks to `server_url/v1/chat/completions`. When provided, RolloutServer includes `Authorization: Bearer <callback_api_key>` header in all callback requests.

### CompletionsRequest (RolloutServer → Trainer)

```python
{
    "model": "default",
    "rollout_id": "550e8400-e29b-41d4-a716-446655440000",
    "messages": [...],
    "response_mask": [0, 0, 0],  # CRITICAL! Tool output tokens
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "logprobs": true
}
```

**Authentication Header** (when `callback_api_key` provided in RolloutRequest):

```
Authorization: Bearer <callback_api_key>
```

## Common Pitfalls

### ❌ Forgetting response_mask After Tool Execution

```python
# Wrong
messages.extend(tool_results)
response = await call_llm(messages)  # Missing response_mask!
```

```python
# Correct - calculate token diff using chat template (recommended approach)
messages.extend(tool_results)
# The mask is calculated by comparing tokenized lengths before and after tool outputs
# See RolloutSession.call_llm() for the reference implementation
num_tool_tokens = current_prompt_length - last_prompt_length
response_mask = [0] * num_tool_tokens
response = await call_llm(messages, response_mask=response_mask)  # ✓
```

### ❌ Wrong Mask Timing

```python
# Wrong
response1 = await call_llm(messages, response_mask=[0,0,0])  # ❌ No tool outputs yet!
messages.extend(tool_results)
response2 = await call_llm(messages, response_mask=None)  # ❌ Should provide mask here!
```

```python
# Correct
response1 = await call_llm(messages, response_mask=None)  # ✓ First turn
messages.extend(tool_results)
response2 = await call_llm(messages, response_mask=[0,0,0])  # ✓ Tool output mask
```

### ❌ Mask Length Mismatch

```python
# Wrong
tool_tokens = tokenizer.encode(tool_output_text)  # len=150
response_mask = [0] * 100  # ❌ Wrong length!
```

```python
# Correct
tool_tokens = tokenizer.encode(tool_output_text)
response_mask = [0] * len(tool_tokens)  # ✓ Exact match
```

## Requirements

- Python 3.11+
- FastAPI 0.115+
- httpx 0.27+
- transformers 4.57+
- pydantic 2.9+

## Development

### Project Structure

```
osmosis-remote-rollout-example/
└── rollout_server/
    ├── src/rollout_server/       # Source code
    │   ├── __init__.py
    │   ├── server.py             # FastAPI server (POST /rollout, GET /tools)
    │   ├── session.py            # RolloutSession (CRITICAL for response_mask)
    │   ├── executor.py           # Core rollout execution logic
    │   ├── config.py             # Centralized configuration
    │   ├── exceptions.py         # Custom exception classes
    │   ├── schemas/              # Protocol data structures (modular)
    │   │   ├── __init__.py       # Re-exports all schemas
    │   │   ├── messages.py       # Message, ToolCall, ToolDefinition
    │   │   ├── params.py         # SamplingParams
    │   │   ├── rollout.py        # RolloutRequest, RolloutResponse
    │   │   ├── completions.py    # CompletionsRequest, CompletionsResponse
    │   │   └── constants.py      # Shared constants
    │   └── tools/
    │       ├── __init__.py
    │       └── calculator.py     # Async calculator tools
    ├── examples/                 # Usage examples
    ├── tests/                    # Test suite
    │   ├── unit/                 # Unit tests
    │   ├── integration/          # In-process integration tests
    │   ├── e2e/                  # End-to-end tests (requires servers)
    │   └── mocks/                # Mock infrastructure
    ├── docs/                     # Documentation
    ├── scripts/                  # Utility scripts
    ├── pyproject.toml            # uv configuration
    └── README.md                 # This file
```

### Scripts

```bash
# Start server
./scripts/start_server.sh

# Run tests directly with pytest
uv run pytest

# Run specific tests
uv run pytest tests/integration/
```

## License

Apache 2.0

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
