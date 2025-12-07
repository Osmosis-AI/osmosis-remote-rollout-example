# Testing Guide

This guide explains the test suite structure, how to run tests, and how to test the RolloutServer independently.

## Test Suite Overview

This project has three types of tests organized by scope and execution requirements:

| Type | Location | Speed | External Dependencies |
|------|----------|-------|----------------------|
| **Unit Tests** | `tests/unit/` | Fast (~1s) | None |
| **Integration Tests** | `tests/integration/` | Fast (~2-3s) | None (in-process mocks) |
| **E2E Tests** | `tests/e2e/` | Slow (~5-10s) | Requires running servers |

## Quick Start

```bash
cd rollout_server

# Run all automated tests (unit + integration)
uv run pytest tests/unit tests/integration -v

# Run with coverage
uv run pytest tests/unit tests/integration --cov=rollout_server --cov-report=term-missing

# Run specific test file
uv run pytest tests/unit/test_session.py -v
```

## Test Directory Structure

```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_schemas.py      # Schema validation tests
│   └── test_session.py      # RolloutSession tests
├── integration/             # Integration tests (in-process mocks)
│   ├── test_rollout_api.py  # Full /rollout endpoint tests
│   └── test_response_mask.py # Response mask validation
├── e2e/                     # E2E tests (requires servers)
│   └── test_e2e_rollout.py  # Real HTTP communication tests
├── mocks/                   # Mock infrastructure
│   └── mock_trainer.py      # Standalone mock trainer server
├── conftest.py              # Shared pytest fixtures
└── README.md                # Quick reference (links here)
```

## Why Testing Needs a Mock Trainer

The `/rollout` endpoint implements the Remote Rollout Protocol, which requires:
1. RolloutServer receives a rollout request from the training cluster
2. RolloutServer **calls back** to the trainer's `/v1/chat/completions` endpoint for LLM generation
3. RolloutServer executes tools and loops until done
4. RolloutServer returns final messages to the training cluster

**The problem**: In standalone testing, there's no real training cluster to call back to!

**The solution**: Use a mock trainer that implements the `/v1/chat/completions` endpoint.

## Quick Start: One-Command Test Environment

The easiest way to test is using the provided startup scripts:

```bash
cd rollout_server

# Start both mock trainer and rollout server
./scripts/start_test_environment.sh

# The script will output:
# ==========================================
# Test Environment Ready!
# ==========================================
# Services:
#   Mock Trainer:   http://localhost:9001
#   Rollout Server: http://localhost:9000
#   API Docs:       http://localhost:9000/docs

# Run E2E tests
uv run python examples/e2e_test_with_servers.py

# Or test via API docs UI
# Open http://localhost:9000/docs in your browser

# When done, stop the environment
./scripts/stop_test_environment.sh
```

**What the script does:**
- Checks if ports 9000 and 9001 are available
- Starts mock trainer on port 9001
- Starts rollout server on port 9000
- Verifies both services are healthy
- Saves logs to `logs/` directory
- Creates PID file for easy cleanup

**Custom ports:**
```bash
MOCK_TRAINER_PORT=9002 ROLLOUT_SERVER_PORT=9100 ./scripts/start_test_environment.sh
```

## Manual Setup: Testing with Mock Trainer

### Step 1: Start the Mock Trainer

```bash
cd rollout_server
uv run python -m tests.mocks.mock_trainer
```

This starts a mock trainer on port 9001 that simulates LLM responses.

### Step 2: Start the RolloutServer

In a new terminal:

```bash
cd rollout_server
uv run python -m rollout_server.server
```

This starts the rollout server on port 9000.

### Step 3: Test the Endpoint

#### Option A: Run E2E Tests

```bash
cd rollout_server
uv run python examples/e2e_test_with_servers.py
```

#### Option B: Use FastAPI Docs UI

1. Open http://localhost:9000/docs
2. Navigate to POST /rollout
3. Click "Try it out"
4. Use this payload:

```json
{
  "rollout_id": "test-rollout-123",
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
  "tokenizer_revision": "main",
  "max_turns": 10,
  "max_tokens_total": 8192
}
```

**CRITICAL**: The `server_url` field MUST point to the mock trainer at `http://localhost:9001`.

#### Option C: Use curl

```bash
curl -X POST http://localhost:9000/rollout \
  -H "Content-Type: application/json" \
  -d '{
    "rollout_id": "test-rollout-123",
    "server_url": "http://localhost:9001",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful calculator assistant."
      },
      {
        "role": "user",
        "content": "Calculate 5 plus 3."
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

## Common Errors and Solutions

### Error: Connection Refused / Network Error

**Symptom**:
```json
{
  "status": "ERROR",
  "error_message": "Network error: ..."
}
```

**Cause**: The `server_url` in your request points to an endpoint that's not running.

**Solution**:
- Make sure the mock trainer is running on the correct port (default: 9001)
- Check that `server_url` in your payload is `http://localhost:9001`
- Verify with: `curl http://localhost:9001/health`

### Error: 422 Validation Error

**Symptom**: FastAPI returns validation errors about missing fields.

**Cause**: Required fields in the payload are missing or have wrong types.

**Solution**: Check that your payload includes:
- `rollout_id` (string)
- `server_url` (string, valid URL)
- `messages` (array of message objects)
- `sampling_params` (object with temperature, top_p, max_tokens)

### Error: 404 Not Found from Trainer

**Symptom**:
```json
{
  "status": "ERROR",
  "error_message": "Trainer HTTP error: 404"
}
```

**Cause**: The trainer endpoint doesn't have `/v1/chat/completions`.

**Solution**: Ensure the mock trainer is running and the URL is correct.

## Understanding the Test Flow

When you test with the mock trainer:

1. **You send** `/rollout` request to RolloutServer (port 9000)
2. **RolloutServer calls back** to mock trainer's `/v1/chat/completions` (port 9001)
3. **Mock trainer returns** a fake LLM response (possibly with tool calls)
4. **RolloutServer executes** the calculator tools
5. **RolloutServer calls back** again with tool results
6. **Mock trainer returns** a final response (no tool calls)
7. **RolloutServer returns** final conversation to you

## Example: Successful Response

```json
{
  "rollout_id": "test-rollout-123",
  "status": "COMPLETED",
  "finish_reason": "stop",
  "final_messages": [
    {
      "role": "system",
      "content": "You are a helpful calculator assistant."
    },
    {
      "role": "user",
      "content": "Calculate 5 plus 3."
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

## Customizing the Mock Trainer

To modify the mock trainer's behavior:

1. Edit `tests/mocks/mock_trainer.py`
2. Modify the `generate_mock_response()` function
3. Add custom logic for different user queries
4. Restart the mock trainer

## Integration with Real Training Cluster

In production, the real training cluster will:
- Send rollout requests to RolloutServer
- Provide a real `/v1/chat/completions` endpoint backed by vLLM/SGLang
- Use real LLM inference for generation

The mock trainer is ONLY for standalone testing and development.

## Test Types in Detail

### 1. Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation.

**Characteristics**:
- Tests single functions/classes
- No external dependencies
- Very fast execution (~1 second)
- High code coverage focus

**Run with**:
```bash
uv run pytest tests/unit/ -v
```

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Lightweight integration tests using in-process mocks.

**Characteristics**:
- Uses FastAPI `TestClient` - no external servers required
- Fast execution (2-3 seconds)
- Mocks trainer responses using pytest fixtures
- No network I/O - everything runs in-process
- **Suitable for CI/CD pipelines**

**Example**: `tests/integration/test_rollout_api.py`

**Run with**:
```bash
uv run pytest tests/integration/ -v
```

### 3. E2E Tests (`tests/e2e/`)

**Purpose**: End-to-end validation with real running servers.

**Characteristics**:
- Requires external servers on ports 9000 and 9001
- Tests actual HTTP communication
- Slower execution (network I/O)
- Automatically skipped if servers not running
- Marked with `@pytest.mark.requires_servers`

**Prerequisites**:
```bash
# Option 1: Use the test environment script
./scripts/start_test_environment.sh

# Run E2E tests
uv run pytest tests/e2e/ -v -m requires_servers

# Stop the environment
./scripts/stop_test_environment.sh

# Option 2: Start servers manually
# Terminal 1: Start mock trainer
uv run python -m tests.mocks.mock_trainer

# Terminal 2: Start rollout server
uv run python -m rollout_server.server

# Terminal 3: Run E2E tests
uv run pytest tests/e2e/ -v -m requires_servers
```

### 4. Interactive Demo Scripts (`examples/`)

**Purpose**: Interactive exploration and debugging (not automated tests).

**Characteristics**:
- Provides detailed logging output
- Intended for learning and debugging
- NOT part of the automated test suite
- Use `examples/e2e_test_with_servers.py` for interactive demos

## Test Strategy

### For Regular Development
```bash
# Run all automated tests (unit + integration)
uv run pytest tests/unit tests/integration -v

# Run with coverage
uv run pytest tests/unit tests/integration --cov=rollout_server --cov-report=term-missing
```

### For CI/CD Pipeline
```bash
# Only run automated tests (skip E2E)
uv run pytest tests/ -v -m "not requires_servers" --cov=rollout_server
```

### For Manual Validation
```bash
# Start servers first
./scripts/start_test_environment.sh

# Run E2E tests
uv run pytest tests/e2e/ -v -m requires_servers

# Or use the interactive demo script
uv run python examples/e2e_test_with_servers.py

# Stop servers
./scripts/stop_test_environment.sh
```

## Test Markers

The following pytest markers are available:

| Marker | Description |
|--------|-------------|
| `@pytest.mark.integration` | Integration tests requiring in-process mocks |
| `@pytest.mark.requires_servers` | Tests requiring external running servers (E2E) |
| `@pytest.mark.asyncio` | Async tests (automatically applied) |

## Mock Infrastructure

### In-Process Mocks (for integration tests)
- Located in `tests/conftest.py` as fixtures
- Use `TestClient` for FastAPI apps
- Monkey-patch httpx for HTTP calls
- Example: `mock_trainer_with_tracking` fixture

### External Mock Server (for E2E tests)
- `tests/mocks/mock_trainer.py`: Standalone mock trainer
- Implements `/v1/chat/completions` endpoint
- Runs on port 9001
- Provides realistic tool call responses

**Run standalone**:
```bash
uv run python -m tests.mocks.mock_trainer
```

## Best Practices

1. **Use unit and integration tests by default**
   - Fast, reliable, no external dependencies
   - Run these in CI/CD pipelines

2. **Use E2E tests sparingly**
   - Only for final validation or debugging
   - Tests are skipped if servers not running
   - Marked with `requires_servers` for easy filtering

3. **Keep tests isolated**
   - Each test should be independent
   - Use fixtures for common setup
   - Clean up resources after tests

4. **Test the contract, not implementation**
   - Focus on API behavior
   - Mock external dependencies
   - Verify request/response structure

## Common Issues

### "Connection refused" or tests skipped
- **Cause**: E2E tests require running servers
- **Solution**: Start servers with `./scripts/start_test_environment.sh` or use `-m "not requires_servers"`

### "Import could not be resolved" warnings
- **Cause**: IDE not using project virtual environment
- **Solution**: Configure IDE to use `.venv/bin/python`

### Tests timing out
- **Cause**: Network issues or slow external services
- **Solution**: Use integration tests instead of E2E tests
