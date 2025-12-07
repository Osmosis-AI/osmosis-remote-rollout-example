"""Constants used across schema modules.

This module defines shared constants to avoid circular imports.
"""

# Valid message roles for chat messages
VALID_MESSAGE_ROLES = {"system", "user", "assistant", "tool", "function"}

# Maximum tokens limit for sampling params
MAX_TOKENS_LIMIT = 32768

# Maximum response_mask length before warning
MAX_RESPONSE_MASK_LENGTH = 100000

