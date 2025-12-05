"""Tool implementations for the Remote Rollout Server.

This package provides tool executors that can be called by the agent loop.

Public API:
- execute_calculator_calls: Execute calculator tool calls
"""

from rollout_server.tools.calculator import execute_calculator_calls

__all__ = [
    "execute_calculator_calls",
]

