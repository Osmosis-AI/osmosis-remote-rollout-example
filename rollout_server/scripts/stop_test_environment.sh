#!/bin/bash

# Stop the test environment (mock trainer and rollout server)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOG_DIR/test_environment.pid"

echo "=========================================="
echo "Stopping Test Environment"
echo "=========================================="

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found at $PID_FILE"
    echo "Test environment may not be running or was started manually."
    echo ""
    echo "To manually stop services:"
    echo "  lsof -ti:9001 | xargs kill  # Stop mock trainer"
    echo "  lsof -ti:9000 | xargs kill  # Stop rollout server"
    exit 0
fi

# Read PIDs
PIDS=$(cat "$PID_FILE")
MOCK_TRAINER_PID=$(echo "$PIDS" | head -n 1)
ROLLOUT_SERVER_PID=$(echo "$PIDS" | tail -n 1)

# Stop mock trainer
if kill -0 $MOCK_TRAINER_PID 2>/dev/null; then
    echo "Stopping mock trainer (PID: $MOCK_TRAINER_PID)..."
    kill $MOCK_TRAINER_PID
    sleep 1
    if kill -0 $MOCK_TRAINER_PID 2>/dev/null; then
        echo "Force killing mock trainer..."
        kill -9 $MOCK_TRAINER_PID 2>/dev/null || true
    fi
    echo "✓ Mock trainer stopped"
else
    echo "✓ Mock trainer already stopped"
fi

# Stop rollout server
if kill -0 $ROLLOUT_SERVER_PID 2>/dev/null; then
    echo "Stopping rollout server (PID: $ROLLOUT_SERVER_PID)..."
    kill $ROLLOUT_SERVER_PID
    sleep 1
    if kill -0 $ROLLOUT_SERVER_PID 2>/dev/null; then
        echo "Force killing rollout server..."
        kill -9 $ROLLOUT_SERVER_PID 2>/dev/null || true
    fi
    echo "✓ Rollout server stopped"
else
    echo "✓ Rollout server already stopped"
fi

# Clean up PID file
rm -f "$PID_FILE"

echo ""
echo "=========================================="
echo "Test Environment Stopped"
echo "=========================================="
echo ""
echo "Logs are still available:"
echo "  Mock Trainer:   $LOG_DIR/mock_trainer.log"
echo "  Rollout Server: $LOG_DIR/rollout_server.log"
echo ""
