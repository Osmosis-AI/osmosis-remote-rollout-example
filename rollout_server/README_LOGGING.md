# Logging Guide

## Overview

RolloutServer now supports comprehensive logging functionality. All terminal output is automatically saved to log files, and each HTTP request is recorded with detailed timestamps.

## Logging Features

### 1. Automatic Log Recording
- **Created on startup**: A timestamped log file is automatically created each time the server starts
- **Log location**: `logs/rollout_server_YYYYMMDD_HHMMSS.log`
- **Dual output**: Using the `tee` command, logs are displayed in the terminal AND saved to file simultaneously

### 2. Timestamp Format
- **Application logs**: `2025-12-10 18:09:30,504 - rollout_server.server - INFO - Starting RolloutServer...`
- **Access logs**: `2025-12-10 18:09:30 - 127.0.0.1:44836 - "GET / HTTP/1.1" 404`

Each request includes:
- Full date and time (`YYYY-MM-DD HH:MM:SS`)
- Client IP address and port
- HTTP request method, path, and protocol
- Response status code

### 3. Log Levels
- **INFO**: Normal operation logs (startup, shutdown, request processing)
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **DEBUG**: Debug information (must be manually enabled)

## Usage

### Start Server (with logging)

```bash
# Start using the script (recommended)
./scripts/start_server.sh
```

On startup, you will see:
```
Starting RolloutServer...
Project root: /home/brian/Osmosis-AI/osmosis-remote-rollout-example/rollout_server
Log file: /home/brian/Osmosis-AI/osmosis-remote-rollout-example/rollout_server/logs/rollout_server_20251210_180930.log

[Server output...]
```

### Viewing Logs

A convenient log viewing script is provided:

```bash
# View last 50 lines (default)
./scripts/view_logs.sh

# Follow log in real-time (like tail -f)
./scripts/view_logs.sh -f
./scripts/view_logs.sh --follow

# Display full log
./scripts/view_logs.sh -a
./scripts/view_logs.sh --all

# Display last N lines
./scripts/view_logs.sh -n 100
./scripts/view_logs.sh --last 200
```

### View Log Files Directly

```bash
# List all log files
ls -lth logs/rollout_server_*.log

# View the latest log file
tail -f logs/rollout_server_*.log

# Search for specific content
grep "ERROR" logs/rollout_server_*.log
grep "POST /rollout" logs/rollout_server_*.log
```

## Log Examples

### Startup Logs
```
2025-12-10 18:09:30 - rollout_server.server - INFO - Starting RolloutServer...
2025-12-10 18:09:30 - rollout_server.executor - INFO - Initializing app state...
2025-12-10 18:09:30 - rollout_server.server - INFO - RolloutServer startup complete
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
```

### Request Logs
```
2025-12-10 18:09:35 - 127.0.0.1:44836 - "GET /health HTTP/1.1" 200
2025-12-10 18:09:40 - 107.192.105.96:54321 - "GET /docs HTTP/1.1" 200
2025-12-10 18:09:45 - 107.192.105.96:54321 - "GET /openapi.json HTTP/1.1" 200
2025-12-10 18:10:00 - 107.192.105.96:54322 - "POST /rollout HTTP/1.1" 200
```

## Log File Management

### Automatic Cleanup (Optional)

You can create a cleanup script to delete old logs:

```bash
# Delete logs older than 7 days
find logs/ -name "rollout_server_*.log" -mtime +7 -delete
```

### Log Rotation

For production environments, it's recommended to use `logrotate` or similar tools for log rotation.

Example logrotate configuration (`/etc/logrotate.d/rollout_server`):

```
/home/brian/Osmosis-AI/osmosis-remote-rollout-example/rollout_server/logs/rollout_server_*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    missingok
}
```

## Troubleshooting

### Log File Not Created

1. Check if the `logs/` directory has write permissions:
   ```bash
   ls -ld logs/
   ```

2. Manually create the directory:
   ```bash
   mkdir -p logs/
   chmod 755 logs/
   ```

### Incomplete Log Content

- Ensure you're using the `tee` command instead of simple redirection (`>`)
- Check disk space: `df -h`

### Incorrect Timestamps

- Check system time: `date`
- Synchronize system time: `sudo ntpdate -s time.nist.gov` (if using NTP)

## Configuration

### Modify Log Format

Edit the `log_config` in `src/rollout_server/server.py`:

```python
"formatters": {
    "default": {
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",  # Modify this to change time format
    },
}
```

### Modify Log Level

```python
uvicorn.run(
    ...
    log_level="debug",  # Change to "debug" for more detailed logs
)
```

## Integration with Test Environment

The `start_test_environment.sh` script already includes similar logging functionality:

```bash
# Start test environment (includes Mock Trainer and Rollout Server)
./scripts/start_test_environment.sh

# View logs
tail -f logs/rollout_server.log
tail -f logs/mock_trainer.log
```

## Performance Considerations

- Logging has minimal performance impact (typically < 1%)
- The `tee` command allows real-time log viewing while saving to file
- For high-traffic scenarios, consider async logging or dedicated log aggregation services (e.g., ELK Stack, Loki)

## Monitoring and Analysis

### Count Requests

```bash
# Count total requests
grep -c "HTTP/1.1" logs/rollout_server_*.log

# Count by status code
grep "HTTP/1.1" logs/rollout_server_*.log | awk '{print $NF}' | sort | uniq -c

# Count /rollout endpoint calls
grep "POST /rollout" logs/rollout_server_*.log | wc -l
```

### Find Errors

```bash
# Find all errors
grep -i "error" logs/rollout_server_*.log

# Find logs for a specific time period
grep "2025-12-10 18:" logs/rollout_server_*.log
```

### Analyze Response Times

You can combine application log timestamps to calculate request processing times.

## Summary

Your RolloutServer now has comprehensive logging capabilities:

✅ All terminal output automatically saved to files  
✅ Each HTTP request has detailed timestamps  
✅ Log files named by startup time for easy tracking  
✅ Convenient log viewing script provided  
✅ Supports both real-time viewing and file storage  

For more help, please refer to this documentation or view the script source code.

