# Production Deployment Guide

## Overview

This guide covers deploying the Remote Rollout Server in production environments.

## Prerequisites

- Python 3.11+
- Docker (recommended for production)
- Access to HuggingFace model hub (for tokenizer downloads)
- Network connectivity to training cluster

## Deployment Options

### Option 1: Docker (Recommended)

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --frozen --no-dev

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ROLLOUT_SERVER_PORT=9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${ROLLOUT_SERVER_PORT}/health || exit 1

# Run server
EXPOSE 9000
CMD ["uv", "run", "python", "-m", "rollout_server.server"]
```

#### Build and Run

```bash
# Generate lock file first (required for --frozen)
uv lock

# Build image
docker build -t rollout-server:latest .

# Run container
# Note: TOKENIZER_TRUST_REMOTE_CODE=true is required for Qwen3 models
docker run -d \
    --name rollout-server \
    -p 9000:9000 \
    -e TOKENIZER_CACHE_SIZE=10 \
    -e HTTP_CLIENT_TIMEOUT=600 \
    -e TOKENIZER_TRUST_REMOTE_CODE=true \
    rollout-server:latest
```

### Option 2: Kubernetes

#### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rollout-server
  labels:
    app: rollout-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rollout-server
  template:
    metadata:
      labels:
        app: rollout-server
    spec:
      containers:
      - name: rollout-server
        image: rollout-server:latest
        ports:
        - containerPort: 9000
        env:
        - name: ROLLOUT_SERVER_PORT
          value: "9000"
        - name: TOKENIZER_CACHE_SIZE
          value: "10"
        - name: HTTP_CLIENT_TIMEOUT
          value: "600"
        - name: TOKENIZER_TRUST_REMOTE_CODE
          value: "true"  # Required for Qwen3 models
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 9000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 9000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: rollout-server
spec:
  selector:
    app: rollout-server
  ports:
  - port: 9000
    targetPort: 9000
  type: ClusterIP
```

### Option 3: Systemd Service

```ini
# /etc/systemd/system/rollout-server.service
[Unit]
Description=Remote Rollout Server
After=network.target

[Service]
Type=simple
User=rollout
Group=rollout
WorkingDirectory=/opt/rollout-server
Environment=ROLLOUT_SERVER_PORT=9000
Environment=TOKENIZER_CACHE_SIZE=10
Environment=HTTP_CLIENT_TIMEOUT=600
Environment=TOKENIZER_TRUST_REMOTE_CODE=true
ExecStart=/opt/rollout-server/.venv/bin/python -m rollout_server.server
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Configuration

### Environment Variables

| Variable | Default | Production Recommendation | Description |
|----------|---------|--------------------------|-------------|
| `ROLLOUT_SERVER_PORT` | `9000` | `9000` | Server port |
| `TOKENIZER_CACHE_SIZE` | `5` | `10-20` | Increase for multi-model deployments |
| `HTTP_CLIENT_TIMEOUT` | `300.0` | `600.0` | Increase for long-running generations |
| `TOKENIZER_TRUST_REMOTE_CODE` | `true` | `true` | Required for Qwen3 and other models with custom tokenizer code |

### Resource Sizing

| Deployment Size | Memory | CPU | Tokenizer Cache | Concurrent Rollouts |
|-----------------|--------|-----|-----------------|---------------------|
| Small | 4GB | 2 cores | 5 | ~10 |
| Medium | 8GB | 4 cores | 10 | ~50 |
| Large | 16GB | 8 cores | 20 | ~200 |

**Note**: Each tokenizer uses ~1-2GB memory. Size the cache based on:
- Number of different models used
- Available memory
- Expected concurrent requests

## Network Configuration

### Firewall Rules

```bash
# Allow incoming requests from OsmosisAgentLoop
ufw allow from 10.0.0.0/8 to any port 9000 proto tcp

# Allow outgoing to trainer nodes (adjust IP range)
ufw allow out to 10.0.0.0/8 port 8080:8130 proto tcp
```

### Load Balancing

For high-availability deployments, use a load balancer:

```nginx
# nginx.conf
upstream rollout_servers {
    least_conn;
    server rollout-server-1:9000 weight=1;
    server rollout-server-2:9000 weight=1;
    server rollout-server-3:9000 weight=1;
}

server {
    listen 9000;
    
    location / {
        proxy_pass http://rollout_servers;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_connect_timeout 60s;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
    
    location /health {
        proxy_pass http://rollout_servers/health;
    }
}
```

## Monitoring

### Health Check Endpoint

```bash
curl http://localhost:9000/health
# Response: {"status": "healthy", "service": "rollout-server"}
```

### Metrics to Monitor

1. **Request latency** - `/rollout` endpoint response time
2. **Error rate** - Count of ERROR status responses
3. **Token throughput** - Tokens processed per second
4. **Memory usage** - Especially tokenizer cache size
5. **HTTP client pool** - Connection utilization

### Prometheus Metrics (Future Enhancement)

```python
# Example metrics to expose
rollout_requests_total{status="completed|error"}
rollout_duration_seconds{quantile="0.5|0.9|0.99"}
llm_calls_total
tool_calls_total
tokenizer_cache_size
tokenizer_cache_hits_total
tokenizer_cache_misses_total
```

### Logging

Configure structured JSON logging for production:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "rollout_id": getattr(record, "rollout_id", None),
        })
```

## Security Considerations

### 1. Network Security

- Deploy in private network (VPC)
- Use TLS for all external communications
- Restrict access to known trainer IPs

### 2. Callback Authentication

When the training cluster sends `callback_api_key` in `RolloutRequest`, the RolloutServer MUST include it as a Bearer token in all callbacks to `/v1/chat/completions`:

```python
# RolloutRequest includes callback_api_key
{
    "rollout_id": "...",
    "server_url": "http://trainer:8081",
    "callback_api_key": "secret-key-123",  # Optional
    ...
}

# RolloutServer adds Authorization header to callbacks
headers = {}
if callback_api_key:
    headers["Authorization"] = f"Bearer {callback_api_key}"

response = await httpx_client.post(
    f"{server_url}/v1/chat/completions",
    json=completions_request,
    headers=headers
)
```

**Trainer Configuration** (on training cluster):
```yaml
# Environment variable containing the callback API key
callback_api_key_env: "CALLBACK_API_KEY"
```

**Best Practices**:
- Use strong, randomly generated API keys
- Store keys in environment variables, not config files
- Rotate keys periodically
- Use different keys per environment (dev/staging/prod)

### 3. Tokenizer Security

```bash
# Default: Enable remote code for Qwen3 models
TOKENIZER_TRUST_REMOTE_CODE=true

# Only disable for standard tokenizers
# TOKENIZER_TRUST_REMOTE_CODE=false
```

**Note**: Qwen3 and other models with custom tokenizer code require `trust_remote_code=True`.

### 4. Input Validation

The server validates all inputs via Pydantic:
- `rollout_id` length limits (max 256 chars)
- `messages` non-empty requirement
- URL format validation (HTTP/HTTPS only)
- Numeric bounds checking

### 5. Error Handling

- Internal errors return generic messages
- No stack traces exposed to clients
- Detailed errors logged server-side only

## Troubleshooting

### Common Issues

#### 1. Tokenizer Download Timeout

**Symptom**: Server hangs on first request

**Solution**:
```bash
# Pre-download tokenizers before deployment
# Note: Qwen3 requires trust_remote_code=True for custom tokenizer code
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True)"

# Or use HuggingFace cache volume
docker run -v /path/to/hf-cache:/root/.cache/huggingface ...
```

#### 2. Connection Refused to Trainer

**Symptom**: `Network error communicating with trainer`

**Solution**:
- Verify trainer URL is reachable
- Check firewall rules
- Ensure trainer is running

#### 3. Out of Memory

**Symptom**: Server crashes or OOM killed

**Solution**:
- Reduce `TOKENIZER_CACHE_SIZE`
- Increase container memory limits
- Use smaller tokenizers if possible

#### 4. High Latency

**Symptom**: Rollouts take too long

**Solution**:
- Check network latency to trainer
- Increase `HTTP_CLIENT_TIMEOUT`
- Profile tool execution time

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG uv run python -m rollout_server.server

# Check tokenizer cache status
curl http://localhost:9000/health  # (extend to include cache stats)
```

## Scaling Strategies

### Horizontal Scaling

- Rollout servers are stateless
- Scale by adding more replicas
- Use load balancer for distribution

### Vertical Scaling

- Increase memory for more tokenizer cache
- More CPU cores for parallel tool execution

### Cache Warming

Pre-load frequently used tokenizers at startup:

```python
# Add to server startup
PRELOAD_TOKENIZERS = ["Qwen/Qwen3-8B", "meta-llama/Llama-3-8B"]

async def warm_cache():
    for tokenizer_name in PRELOAD_TOKENIZERS:
        await load_tokenizer_async(tokenizer_name)
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and data flow
- [RESPONSE_MASK_GUIDE.md](RESPONSE_MASK_GUIDE.md) - Critical response_mask documentation
- [TESTING.md](TESTING.md) - Test suite documentation

