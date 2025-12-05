#!/bin/bash
# Start RolloutServer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting RolloutServer..."
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Start FastAPI server
uv run python -m rollout_server.server
