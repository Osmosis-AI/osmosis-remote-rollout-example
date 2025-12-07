"""Custom exceptions for the Remote Rollout Server.

This module defines all custom exception types used throughout the rollout server.
Centralizing exceptions improves maintainability and makes error handling consistent.

Exception Hierarchy:
    RolloutError (base)
    ├── TokenizerLoadError - Failed to load tokenizer from HuggingFace
    ├── ToolExecutionError - Tool execution failed
    ├── RateLimitExceededError - Too many concurrent rollouts
    └── RolloutTimeoutError - Rollout exceeded time limit
"""


class RolloutError(Exception):
    """Base exception for rollout errors.

    All custom exceptions in the rollout server should inherit from this class.
    This allows catching all rollout-specific errors with a single except clause.
    """
    pass


class TokenizerLoadError(RolloutError):
    """Error loading tokenizer from HuggingFace.

    Raised when:
    - Tokenizer name is invalid
    - Network error fetching tokenizer
    - Tokenizer files are corrupted
    """
    pass


class ToolExecutionError(RolloutError):
    """Error executing tools during rollout.

    Raised when:
    - Tool function raises an exception
    - Tool arguments are invalid
    - Tool execution times out
    """
    pass


class RateLimitExceededError(RolloutError):
    """Too many concurrent rollouts.

    Raised when:
    - Server is at MAX_CONCURRENT_ROLLOUTS capacity
    - Request cannot acquire semaphore
    """
    pass


class RolloutTimeoutError(RolloutError):
    """Rollout exceeded time limit.

    Raised when:
    - Total rollout duration exceeds ROLLOUT_TIMEOUT_SECONDS
    - Individual LLM call times out
    """
    pass

