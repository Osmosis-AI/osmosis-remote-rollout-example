# Testing Guide

This guide explains how to test the RolloutServer `/rollout` endpoint independently.

## Why Testing Needs a Mock Trainer

The `/rollout` endpoint implements the Remote Rollout Protocol, which requires:
1. RolloutServer receives a rollout request from the training cluster
2. RolloutServer **calls back** to the trainer's `/v1/completions` endpoint for LLM generation
3. RolloutServer executes tools and loops until done
4. RolloutServer returns final messages to the training cluster

**The problem**: In standalone testing, there's no real training cluster to call back to!

**The solution**: Use a mock trainer that implements the `/v1/completions` endpoint.

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
uv run pytest examples/e2e_test_with_servers.py -v

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
uv run pytest examples/e2e_test_with_servers.py -v
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

**Cause**: The trainer endpoint doesn't have `/v1/completions`.

**Solution**: Ensure the mock trainer is running and the URL is correct.

## Understanding the Test Flow

When you test with the mock trainer:

1. **You send** `/rollout` request to RolloutServer (port 9000)
2. **RolloutServer calls back** to mock trainer's `/v1/completions` (port 9001)
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

In production, the real training cluster (TrainGate) will:
- Send rollout requests to RolloutServer
- Provide a real `/v1/completions` endpoint backed by vLLM/SGLang
- Use real LLM inference for generation

The mock trainer is ONLY for standalone testing and development.
