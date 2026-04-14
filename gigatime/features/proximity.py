"""Distance-transform-based spatial proximity scores.

Computes how close positive pixels of a *target* channel are to the nearest
positive pixel of a *reference* channel, in microns.

Useful for features where physical proximity matters (e.g. PD-1+ T cells near
PD-L1+ tumour cells) rather than same-cell co-expression.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

MPP = 0.5  # microns per pixel at the GigaTIME read level


def mean_distance_um(
    target: np.ndarray,
    reference: np.ndarray,
    mpp: float = MPP,
    max_distance_um: float = 200.0,
) -> float:
    """Mean distance (µm) from target-positive pixels to the nearest reference-positive pixel.

    Args:
        target:          (H, W) bool/float32 map — pixels whose distance is measured.
        reference:       (H, W) bool/float32 map — pixels that act as the reference set.
        mpp:             Microns per pixel. Defaults to 0.5 (GigaTIME read level).
        max_distance_um: Cap applied before averaging to avoid domination by isolated
                         pixels far from any reference signal.

    Returns:
        Mean distance in microns.  Returns ``max_distance_um`` if there are no
        target-positive pixels or no reference-positive pixels.
    """
    target_mask = target.astype(bool)
    ref_mask = reference.astype(bool)

    if not target_mask.any() or not ref_mask.any():
        return max_distance_um

    # EDT gives distance in pixels from every non-ref pixel to nearest ref pixel
    dist_px = distance_transform_edt(~ref_mask)
    dist_um = dist_px * mpp
    dist_um = np.clip(dist_um, 0.0, max_distance_um)

    return float(dist_um[target_mask].mean())


def proximity_score(
    target: np.ndarray,
    reference: np.ndarray,
    threshold_um: float = 50.0,
    mpp: float = MPP,
) -> float:
    """Fraction of target-positive pixels within ``threshold_um`` of any reference pixel.

    Args:
        target:       (H, W) bool/float32 map.
        reference:    (H, W) bool/float32 map.
        threshold_um: Distance threshold in microns. Default 50 µm ≈ 4–5 cell diameters.
        mpp:          Microns per pixel.

    Returns:
        Scalar in [0, 1].  Returns 0.0 if no target pixels exist.
    """
    target_mask = target.astype(bool)
    ref_mask = reference.astype(bool)

    if not target_mask.any() or not ref_mask.any():
        return 0.0

    dist_px = distance_transform_edt(~ref_mask)
    threshold_px = threshold_um / mpp

    return float((dist_px[target_mask] <= threshold_px).mean())
