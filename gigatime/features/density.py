"""Compartment-normalised density helpers.

All densities are expressed as a fraction of the compartment area (number of
positive pixels / number of compartment pixels) rather than total slide area,
so that slides with different tissue content are comparable.
"""

from __future__ import annotations

import numpy as np


def density(channel: np.ndarray, compartment: np.ndarray) -> float:
    """Fraction of compartment pixels that are positive for a channel.

    Args:
        channel:     (H, W) float32 or bool binary map.
        compartment: (H, W) bool mask defining the region of interest.

    Returns:
        Scalar in [0, 1].  Returns 0.0 if the compartment is empty.
    """
    denom = compartment.sum()
    if denom == 0:
        return 0.0
    return float((channel.astype(bool) & compartment).sum() / denom)
