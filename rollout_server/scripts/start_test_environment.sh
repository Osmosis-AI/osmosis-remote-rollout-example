#!/bin/bash

# Start test environment with mock trainer and rollout server
# This script starts both services for standalone testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default ports
MOCK_TRAINER_PORT=${MOCK_TRAINER_PORT:-9001}
ROLLOUT_SERVER_PORT=${ROLLOUT_SERVER_PORT:-9000}

echo "=========================================="
echo "Starting Test Environment"
echo "=========================================="
echo "Mock Trainer Port: $MOCK_TRAINER_PORT"
echo "Rollout Server Port: $ROLLOUT_SERVER_PORT"
echo ""

# Check and clean up ports if already in use
if lsof -Pi :$MOCK_TRAINER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠ Port $MOCK_TRAINER_PORT is already in use, killing existing process..."
    PIDS=$(lsof -Pi :$MOCK_TRAINER_PORT -sTCP:LISTEN -t)
    for PID in $PIDS; do
        kill -9 $PID 2>/dev/null || true
        echo "  Killed process $PID"
    done
    sleep 1
fi

if lsof -Pi :$ROLLOUT_SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠ Port $ROLLOUT_SERVER_PORT is already in use, killing existing process..."
    PIDS=$(lsof -Pi :$ROLLOUT_SERVER_PORT -sTCP:LISTEN -t)
    for PID in $PIDS; do
        kill -9 $PID 2>/dev/null || true
        echo "  Killed process $PID"
    done
    sleep 1
fi

# Create log directory
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Start mock trainer
echo "Starting mock trainer on port $MOCK_TRAINER_PORT..."
cd "$PROJECT_DIR"
MOCK_TRAINER_PORT=$MOCK_TRAINER_PORT uv run python -m tests.mocks.mock_trainer \
    > "$LOG_DIR/mock_trainer.log" 2>&1 &
MOCK_TRAINER_PID=$!

# Wait for mock trainer to start
sleep 3
if ! kill -0 $MOCK_TRAINER_PID 2>/dev/null; then
    echo "Error: Mock trainer failed to start. Check logs:"
    echo "  cat $LOG_DIR/mock_trainer.log"
    exit 1
fi

# Verify mock trainer is responding
if ! curl -s http://localhost:$MOCK_TRAINER_PORT/health >/dev/null 2>&1; then
    echo "Error: Mock trainer is not responding. Check logs:"
    echo "  cat $LOG_DIR/mock_trainer.log"
    kill $MOCK_TRAINER_PID 2>/dev/null || true
    exit 1
fi

echo "✓ Mock trainer started (PID: $MOCK_TRAINER_PID)"
echo "  Health: http://localhost:$MOCK_TRAINER_PORT/health"
echo "  Logs: $LOG_DIR/mock_trainer.log"
echo ""

# Start rollout server
echo "Starting rollout server on port $ROLLOUT_SERVER_PORT..."
ROLLOUT_SERVER_PORT=$ROLLOUT_SERVER_PORT uv run python -m rollout_server.server \
    > "$LOG_DIR/rollout_server.log" 2>&1 &
ROLLOUT_SERVER_PID=$!

# Wait for rollout server to start
sleep 3
if ! kill -0 $ROLLOUT_SERVER_PID 2>/dev/null; then
    echo "Error: Rollout server failed to start. Check logs:"
    echo "  cat $LOG_DIR/rollout_server.log"
    kill $MOCK_TRAINER_PID 2>/dev/null || true
    exit 1
fi

# Verify rollout server is responding
if ! curl -s http://localhost:$ROLLOUT_SERVER_PORT/health >/dev/null 2>&1; then
    echo "Error: Rollout server is not responding. Check logs:"
    echo "  cat $LOG_DIR/rollout_server.log"
    kill $MOCK_TRAINER_PID 2>/dev/null || true
    kill $ROLLOUT_SERVER_PID 2>/dev/null || true
    exit 1
fi

echo "✓ Rollout server started (PID: $ROLLOUT_SERVER_PID)"
echo "  Health: http://localhost:$ROLLOUT_SERVER_PORT/health"
echo "  Docs: http://localhost:$ROLLOUT_SERVER_PORT/docs"
echo "  Logs: $LOG_DIR/rollout_server.log"
echo ""

# Save PIDs for cleanup
PID_FILE="$LOG_DIR/test_environment.pid"
echo "$MOCK_TRAINER_PID" > "$PID_FILE"
echo "$ROLLOUT_SERVER_PID" >> "$PID_FILE"

echo "=========================================="
echo "Test Environment Ready!"
echo "=========================================="
echo ""
echo "Services:"
echo "  Mock Trainer:   http://localhost:$MOCK_TRAINER_PORT"
echo "  Rollout Server: http://localhost:$ROLLOUT_SERVER_PORT"
echo "  API Docs:       http://localhost:$ROLLOUT_SERVER_PORT/docs"
echo ""
echo "Test the /rollout endpoint:"
echo "  uv run python -m tests.test_with_mock_trainer"
echo ""
echo "Or use curl:"
echo "  curl -X POST http://localhost:$ROLLOUT_SERVER_PORT/rollout \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{...payload...}'"
echo ""
echo "To stop the test environment:"
echo "  ./scripts/stop_test_environment.sh"
echo ""
echo "Logs:"
echo "  Mock Trainer:   tail -f $LOG_DIR/mock_trainer.log"
echo "  Rollout Server: tail -f $LOG_DIR/rollout_server.log"
echo ""
