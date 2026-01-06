"""Reward functions for the example RolloutServer."""

import logging
import re
from typing import Optional

from osmosis_ai import osmosis_reward

logger = logging.getLogger(__name__)


def extract_solution(solution_str: str) -> Optional[str]:
    """Extract numeric solution from a string.

    Looks for the pattern '#### <number>' commonly used in math problems.

    Args:
        solution_str: The string to extract the solution from.

    Returns:
        The extracted numeric string, or None if not found.
    """
    solution = re.search(r"####\s*([-+]?\d*\.?\d+)", solution_str)
    if not solution or solution is None:
        return None
    return solution.group(1)


@osmosis_reward
def compute_reward(
    solution_str: str, ground_truth: str, extra_info: dict = None
) -> float:
    """Compute reward by comparing extracted solution to ground truth.

    This reward function extracts a numeric answer from the solution string
    (looking for '#### <number>' pattern) and compares it to the ground truth.

    Args:
        solution_str: The model's response containing the solution.
        ground_truth: The expected answer (numeric string).
        extra_info: Optional additional information (not used).

    Returns:
        1.0 if the extracted solution matches ground_truth, 0.0 otherwise.
    """
    extracted = extract_solution(solution_str)
    if extracted is None:
        logger.debug(f"Could not extract solution from: {solution_str[:100]}...")
        return 0.0

    try:
        sol_val = float(extracted)
    except (ValueError, TypeError):
        logger.debug(f"Failed to parse extracted solution as float: {extracted}")
        return 0.0

    try:
        gt_val = float(ground_truth)
    except (ValueError, TypeError):
        logger.warning(f"Failed to parse ground_truth as float: {ground_truth}")
        return 0.0

    if abs(gt_val - sol_val) < 1e-7:
        return 1.0
    return 0.0
