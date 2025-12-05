# Remote Rollout Server - Reference Implementation

A reference implementation of the TrainGate Remote Rollout Protocol, demonstrating correct `response_mask` handling for multi-turn conversations with tool use.

## Overview

This package provides:
1. **Reference implementation** for external developers implementing RolloutServer
2. **End-to-end testing infrastructure** for traingate's remote rollout system
3. **Protocol compliance examples** based on official documentation

**Critical Focus**: Demonstrates CORRECT `response_mask` handling - the #1 source of bugs and training data corruption in remote rollout implementations.

## What is Remote Rollout?

Remote Rollout is an architecture that separates agent trajectory generation from training infrastructure:

- **Training cluster** (GPU): Runs LLM inference (vLLM/SGLang) and PPO training
- **RolloutServer** (external): Drives the agent loop (tool parsing, execution, state management)

**Key Benefits**:
- Decouple agent logic from training infrastructure
- Teams iterate independently
- Standard OpenAI-compatible `/v1/completions` endpoint
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
# Server starts on http://0.0.0.0:9000 (default port, avoids traingate's 8080-8130 range)

# Or specify custom port:
ROLLOUT_SERVER_PORT=9100 uv run python -m rollout_server.server
```

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
        "server_url": "http://trainer:8081",  # Trainer's /v1/completions endpoint
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
   a. Call trainer's /v1/completions (with response_mask!)
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
| **FastAPI Server** | `src/rollout_server/server.py` | POST /rollout endpoint |
| **Calculator Tools** | `src/rollout_server/tools/calculator.py` | Async tools with random delays |
| **Schemas** | `src/rollout_server/schemas.py` | Protocol data structures |

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

The correct response_mask calculation pattern (from `docs/rollout_server.md:305-350`):

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
            f"{self.server_url}/v1/completions",
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

## Documentation

- **[README.md](README.md)** (this file) - Quick start and overview
- **[docs/RESPONSE_MASK_GUIDE.md](docs/RESPONSE_MASK_GUIDE.md)** - **CRITICAL** - Deep dive on response_mask
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and data flow
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Production deployment guide

## Examples

| Example | Description |
|---------|-------------|
| `examples/basic_example.py` | Single-turn conversation (no tools) |
| `examples/calculator_example.py` | Multi-turn with calculator tools |
| `examples/mock_trainer_example.py` | Complete end-to-end demo with mock trainer |

## Testing

### Testing with Mock Trainer

The `/rollout` endpoint requires a trainer's `/v1/completions` endpoint to function. For standalone testing without a real training cluster, use the mock trainer:

#### Quick Start: One-Command Test Environment

```bash
cd rollout_server

# Start both mock trainer and rollout server
./scripts/start_test_environment.sh

# Run E2E tests
uv run pytest examples/e2e_test_with_servers.py -v

# Stop the environment when done
./scripts/stop_test_environment.sh
```

This script automatically:
- Starts the mock trainer on port 9001
- Starts the rollout server on port 9000
- Verifies both services are healthy
- Saves logs to `logs/` directory
- Creates a PID file for easy cleanup

#### Step 1: Start the Mock Trainer

```bash
cd rollout_server

# Start mock trainer (port 9001 by default)
uv run python -m tests.mocks.mock_trainer

# Or specify custom port:
MOCK_TRAINER_PORT=9002 uv run python -m tests.mocks.mock_trainer
```

You should see:
```
============================================================
Starting Mock Trainer Server on port 9001
This server simulates the trainer's /v1/completions endpoint
for testing RolloutServer independently.
============================================================
```

#### Step 2: Start the Rollout Server

In a new terminal:

```bash
cd rollout_server

# Start rollout server (port 9000 by default)
uv run python -m rollout_server.server

# Or specify custom port:
ROLLOUT_SERVER_PORT=9100 uv run python -m rollout_server.server
```

#### Step 3: Run Tests

**Option A: Run E2E Tests**

```bash
cd rollout_server
uv run pytest examples/e2e_test_with_servers.py -v
```

**Option B: FastAPI Docs UI**

1. Open http://localhost:9000/docs in your browser
2. Navigate to `POST /rollout`
3. Click "Try it out"
4. Use this test payload:

```json
{
  "rollout_id": "test-rollout-001",
  "server_url": "http://localhost:9001",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful calculator assistant with access to calculator tools."
    },
    {
      "role": "user",
      "content": "Please calculate 5 plus 3."
    }
  ],
  "sampling_params": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "logprobs": true
  },
  "tokenizer_name": "Qwen/Qwen3-8B",
  "max_turns": 10,
  "max_tokens_total": 8192
}
```

**IMPORTANT**: The `server_url` field MUST point to the mock trainer (e.g., `http://localhost:9001`).

5. Click "Execute"
6. Check the response - you should see status: "COMPLETED"

**Option C: Using curl**

```bash
curl -X POST http://localhost:9000/rollout \
  -H "Content-Type: application/json" \
  -d '{
    "rollout_id": "test-rollout-001",
    "server_url": "http://localhost:9001",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful calculator assistant with access to calculator tools."
      },
      {
        "role": "user",
        "content": "Please calculate 5 plus 3."
      }
    ],
    "sampling_params": {
      "temperature": 0.7,
      "top_p": 0.9,
      "max_tokens": 512,
      "logprobs": true
    },
    "tokenizer_name": "Qwen/Qwen3-8B",
    "max_turns": 10,
    "max_tokens_total": 8192
  }'
```

#### Expected Successful Response

```json
{
  "rollout_id": "test-rollout-001",
  "status": "COMPLETED",
  "finish_reason": "stop",
  "final_messages": [
    {
      "role": "system",
      "content": "You are a helpful calculator assistant with access to calculator tools."
    },
    {
      "role": "user",
      "content": "Please calculate 5 plus 3."
    },
    {
      "role": "assistant",
      "content": "I'll calculate that for you.",
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "add",
            "arguments": "{\"a\": 5, \"b\": 3}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "content": "8",
      "tool_call_id": "call_abc123"
    },
    {
      "role": "assistant",
      "content": "I'm a mock LLM. The calculation result should be in the previous message."
    }
  ]
}
```

### Common Testing Errors and Solutions

#### Error: Connection Refused / Network Error

**Symptom**:
```json
{
  "status": "ERROR",
  "error_message": "Network error: ..."
}
```

**Cause**: The `server_url` in your request points to an endpoint that's not running.

**Solution**:
- Make sure the mock trainer is running: `curl http://localhost:9001/health`
- Verify `server_url` in your payload is `http://localhost:9001`

#### Error: 422 Validation Error

**Symptom**: FastAPI returns validation errors about missing fields.

**Solution**: Check that your payload includes all required fields:
- `rollout_id` (string)
- `server_url` (string, valid URL)
- `messages` (array of message objects)
- `sampling_params` (object with temperature, top_p, max_tokens)

### Test Suite

This project has two types of tests:

#### 1. Automated Tests (tests/)

**Integration Tests** - Fast, in-process tests using FastAPI TestClient:
- `tests/integration/test_rollout_api.py` - Full /rollout endpoint with mocked trainer
- No external servers required
- Runs in 2-3 seconds
- **Use these for regular development and CI/CD**

Run automated tests:
```bash
cd rollout_server

# Run all automated tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=rollout_server --cov-report=term-missing
```

#### 2. E2E Tests (examples/)

**End-to-End Tests** - Tests with real running servers:
- `examples/e2e_test_with_servers.py` - Requires servers on ports 9000 and 9001
- Tests actual HTTP communication
- **Use these for manual validation and debugging only**

Run E2E tests:
```bash
cd rollout_server

# Terminal 1: Start mock trainer
uv run python -m tests.mocks.mock_trainer

# Terminal 2: Start rollout server
uv run python -m rollout_server.server

# Terminal 3: Run E2E tests
uv run pytest examples/e2e_test_with_servers.py -v
```

#### Mock Infrastructure

- `tests/mocks/mock_trainer.py` - Standalone mock trainer for E2E testing
- Integration tests use in-process mocks (no standalone server needed)

### Additional Testing Documentation

For more detailed testing information, see:
- [`tests/README.md`](tests/README.md) - Complete test suite documentation
- [`docs/RESPONSE_MASK_GUIDE.md`](docs/RESPONSE_MASK_GUIDE.md) - Response mask validation

## Protocol Reference

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
    "max_tokens_total": 8192
}
```

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

## Common Pitfalls

### ❌ Forgetting response_mask After Tool Execution

```python
# Wrong
messages.extend(tool_results)
response = await call_llm(messages)  # Missing response_mask!
```

```python
# Correct
messages.extend(tool_results)
num_tool_tokens = len(tokenize_tool_results(tool_results))
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
    │   ├── server.py             # FastAPI server
    │   ├── session.py            # RolloutSession (CRITICAL)
    │   ├── schemas.py            # Protocol data structures
    │   └── tools/
    │       ├── __init__.py
    │       └── calculator.py     # Async calculator tools
    ├── examples/                 # Usage examples
    ├── tests/                    # Test suite
    │   ├── unit/
    │   ├── integration/
    │   └── mocks/
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

See the main traingate repository for contribution guidelines.

## Resources

- **TrainGate Documentation**: https://github.com/Osmosis-AI/traingate
- **Remote Rollout Design**: See `docs/remote_rollout_design.md` in the TrainGate repository
- **RolloutServer Implementation Guide**: See `docs/rollout_server.md` in the TrainGate repository
