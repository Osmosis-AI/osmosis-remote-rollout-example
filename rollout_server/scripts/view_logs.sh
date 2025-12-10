#!/bin/bash
# View RolloutServer logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

# Check if logs directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: Log directory does not exist: $LOG_DIR"
    exit 1
fi

# List available log files
echo "=========================================="
echo "Available log files:"
echo "=========================================="
ls -lth "$LOG_DIR"/rollout_server_*.log 2>/dev/null || echo "No log files found"
echo ""

# Get the most recent log file
LATEST_LOG=$(ls -t "$LOG_DIR"/rollout_server_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log files found"
    exit 0
fi

echo "Latest log file: $LATEST_LOG"
echo "=========================================="
echo ""

# If argument is provided, use it
if [ "$1" = "-f" ] || [ "$1" = "--follow" ]; then
    echo "Following log in real-time (Ctrl+C to exit)..."
    tail -f "$LATEST_LOG"
elif [ "$1" = "-a" ] || [ "$1" = "--all" ]; then
    echo "Displaying full log content:"
    cat "$LATEST_LOG"
elif [ "$1" = "-n" ] || [ "$1" = "--last" ]; then
    LINES=${2:-50}
    echo "Displaying last $LINES lines:"
    tail -n "$LINES" "$LATEST_LOG"
else
    echo "Usage:"
    echo "  $0 -f, --follow         Follow log in real-time"
    echo "  $0 -a, --all            Display full log"
    echo "  $0 -n, --last [lines]   Display last N lines (default 50)"
    echo ""
    echo "Examples:"
    echo "  $0 -f                   # Follow log in real-time"
    echo "  $0 -n 100               # Display last 100 lines"
    echo ""
    echo "Displaying last 50 lines:"
    tail -n 50 "$LATEST_LOG"
fi

