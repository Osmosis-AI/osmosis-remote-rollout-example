# Test Suite

This directory contains tests for the Remote Rollout Server.

## Quick Reference

```bash
cd rollout_server

# Run all automated tests (unit + integration)
uv run pytest tests/unit tests/integration -v

# Run with coverage
uv run pytest tests/unit tests/integration --cov=rollout_server

# Run E2E tests (requires servers running)
./scripts/start_test_environment.sh
uv run pytest tests/e2e/ -v -m requires_servers
./scripts/stop_test_environment.sh
```

## Directory Structure

```
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # In-process tests with mocked trainer
├── e2e/            # E2E tests (requires running servers)
├── mocks/          # Mock infrastructure
└── conftest.py     # Shared pytest fixtures
```

## Documentation

For comprehensive testing documentation, including:
- Test types and when to use each
- Mock trainer setup
- CI/CD configuration
- Best practices and troubleshooting

**See [docs/TESTING.md](../docs/TESTING.md)**
