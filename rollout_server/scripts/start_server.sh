#!/bin/bash
# Start RolloutServer

# This script assumes `osmosis-ai[server]` is installed in the environment.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create logs directory if it doesn't exist
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/rollout_server_${TIMESTAMP}.log"

echo "Starting RolloutServer..."
echo "Project root: $PROJECT_ROOT"
echo "Log file: $LOG_FILE"
echo ""

cd "$PROJECT_ROOT"

# Start FastAPI server with output redirected to log file
# Use tee to both display output in terminal AND save to file
uv run python -m rollout_server.server 2>&1 | tee "$LOG_FILE"
