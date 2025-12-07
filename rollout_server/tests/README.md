# Test Suite Documentation

This document explains the different types of tests in this project and when to use each.

## Test Types

### 1. Integration Tests (tests/integration/)

**Purpose**: Lightweight integration tests using in-process mocks

**Characteristics**:
- Uses FastAPI `TestClient` - no external servers required
- Fast execution (2-3 seconds)
- Mocks trainer responses using pytest fixtures
- No network I/O - everything runs in-process
- Suitable for CI/CD pipelines
- Tests the full API contract without external dependencies

**Example**: `tests/integration/test_rollout_api.py`

**Run with**:
```bash
pytest tests/integration/ -v
```

### 2. Unit Tests (tests/unit/)

**Purpose**: Test individual components in isolation

**Characteristics**:
- Tests single functions/classes
- No external dependencies
- Very fast execution
- High code coverage focus

**Run with**:
```bash
pytest tests/unit/ -v
```

### 3. E2E Tests (examples/e2e_test_with_servers.py)

**Purpose**: End-to-end validation with real running servers

**Characteristics**:
- Requires external servers on ports 9000 and 9001
- Tests actual HTTP communication
- Slower execution (network I/O)
- Used for manual testing and debugging
- **NOT suitable for regular pytest runs**
- Located in `examples/` because they're reference examples, not automated tests

**Prerequisites**:
```bash
# Terminal 1: Start mock trainer
python -m rollout_server.tests.mocks.mock_trainer

# Terminal 2: Start rollout server
python -m rollout_server.server

# Terminal 3: Run E2E tests
uv run python examples/e2e_test_with_servers.py
```

**Note**: If servers are not running, tests will be skipped (not fail).

## Test Strategy

### For Regular Development
```bash
# Run all automated tests (integration + unit)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rollout_server --cov-report=term-missing
```

### For CI/CD Pipeline
```bash
# Only run automated tests (no E2E)
pytest tests/ -v --cov=rollout_server
```

### For Manual Validation
```bash
# Start servers first, then run E2E tests
uv run python examples/e2e_test_with_servers.py
```

### For Specific Test Types
```bash
# Only integration tests
pytest tests/ -v -m integration

# Exclude tests requiring servers
pytest tests/ -v -m "not requires_servers"
```

## Test Markers

The following pytest markers are available:

- `@pytest.mark.integration`: Integration tests requiring in-process mocks
- `@pytest.mark.requires_servers`: Tests requiring external running servers (E2E)
- `@pytest.mark.asyncio`: Async tests (automatically applied)

## Mock Infrastructure

### In-Process Mocks (for integration tests)
- Located in test files as fixtures
- Use `TestClient` for FastAPI apps
- Monkey-patch httpx for HTTP calls
- Example: `test_rollout_api.py` mock_trainer fixtures

### External Mock Server (for E2E tests)
- `tests/mocks/mock_trainer.py`: Standalone mock trainer
- Implements `/v1/completions` endpoint
- Runs on port 9001
- Provides realistic tool call responses

## Best Practices

1. **Use integration tests by default**
   - Fast, reliable, no external dependencies
   - Located in `tests/integration/`

2. **Use E2E tests sparingly**
   - Only for final validation or debugging
   - Keep in `examples/` directory
   - Document server requirements clearly

3. **Keep tests isolated**
   - Each test should be independent
   - Use fixtures for common setup
   - Clean up resources after tests

4. **Test the contract, not implementation**
   - Focus on API behavior
   - Mock external dependencies
   - Verify request/response structure

## Common Issues

### "Connection refused" errors
- **Cause**: E2E tests running without servers
- **Solution**: Either start servers or run only `pytest tests/` (skips E2E)

### "Import could not be resolved" warnings
- **Cause**: IDE not using project virtual environment
- **Solution**: Configure IDE to use `.venv/bin/python`

### Tests timing out
- **Cause**: Network issues or slow external services
- **Solution**: Use integration tests instead of E2E tests
