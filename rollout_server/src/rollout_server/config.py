"""Centralized configuration for the Remote Rollout Server.

This module provides a single source of truth for all configuration values.
All settings can be overridden via environment variables.

Usage:
    from rollout_server.config import settings

    # Access configuration values
    cache_size = settings.tokenizer_cache_size
    timeout = settings.http_client_timeout

Environment Variables:
    ROLLOUT_SERVER_PORT: Server port (default: 9000)
    TOKENIZER_CACHE_SIZE: Max tokenizers in LRU cache (default: 5)
    HTTP_CLIENT_TIMEOUT: HTTP request timeout in seconds (default: 300.0)
    TOKENIZER_TRUST_REMOTE_CODE: Allow custom tokenizer code (default: true)
    MAX_CONCURRENT_ROLLOUTS: Rate limit concurrent rollouts (default: 100)
    ROLLOUT_TIMEOUT_SECONDS: Total rollout timeout (default: 600.0)
    MAX_ROLLOUT_RECORDS: Max rollout_id records to retain (default: 10000)
    ROLLOUT_RECORD_TTL_SECONDS: TTL for completed rollout_id records (default: 3600.0)
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional


def _get_bool_env(key: str, default: bool) -> bool:
    """Get boolean value from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value (case-insensitive "true"/"false")
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() == "true"


def _get_int_env(key: str, default: int) -> int:
    """Get integer value from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Integer value
    """
    value = os.getenv(key)
    if value is None:
        return default
    return int(value)


def _get_float_env(key: str, default: float) -> float:
    """Get float value from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Float value
    """
    value = os.getenv(key)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    """Application settings with environment variable support.

    This class is immutable (frozen=True) to prevent accidental modification
    of configuration values at runtime.

    Attributes:
        server_port: FastAPI server port
        tokenizer_cache_size: Max tokenizers in LRU cache (~1-2GB per tokenizer)
        http_client_timeout: HTTP client timeout in seconds
        tokenizer_trust_remote_code: Allow execution of custom tokenizer code
        max_concurrent_rollouts: Max concurrent rollouts (rate limiting)
        rollout_timeout_seconds: Total rollout timeout in seconds
        max_rollout_records: Max rollout_id records to retain in memory
        rollout_record_ttl_seconds: TTL for completed rollout_id records in memory
        default_temperature: Default sampling temperature
        default_top_p: Default nucleus sampling top_p
        default_max_tokens: Default max tokens per generation
    """

    # Server settings
    server_port: int = field(default_factory=lambda: _get_int_env("ROLLOUT_SERVER_PORT", 9000))

    # Tokenizer settings
    tokenizer_cache_size: int = field(default_factory=lambda: _get_int_env("TOKENIZER_CACHE_SIZE", 5))
    tokenizer_trust_remote_code: bool = field(
        default_factory=lambda: _get_bool_env("TOKENIZER_TRUST_REMOTE_CODE", True)
    )

    # HTTP client settings
    http_client_timeout: float = field(default_factory=lambda: _get_float_env("HTTP_CLIENT_TIMEOUT", 300.0))

    # Rate limiting and timeouts
    max_concurrent_rollouts: int = field(default_factory=lambda: _get_int_env("MAX_CONCURRENT_ROLLOUTS", 100))
    rollout_timeout_seconds: float = field(default_factory=lambda: _get_float_env("ROLLOUT_TIMEOUT_SECONDS", 600.0))

    # Rollout record retention (idempotency + memory safety)
    max_rollout_records: int = field(default_factory=lambda: _get_int_env("MAX_ROLLOUT_RECORDS", 10000))
    rollout_record_ttl_seconds: float = field(
        default_factory=lambda: _get_float_env("ROLLOUT_RECORD_TTL_SECONDS", 3600.0)
    )

    # Default sampling parameters
    default_temperature: float = 1.0
    default_top_p: float = 1.0
    default_max_tokens: int = 512


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance.

    This function is cached to ensure only one Settings instance is created.
    Environment variables are read once at first access.

    Returns:
        Cached Settings instance
    """
    return Settings()


# Convenient access to settings singleton
settings = get_settings()

