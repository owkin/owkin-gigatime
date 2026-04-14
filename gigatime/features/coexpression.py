"""Co-expression helpers for pairs of channel maps.

Two strategies are provided — choose based on the expected cell size:

pixel_overlap
    Direct boolean AND of the two binary maps. Appropriate when both markers
    are expressed on large cells (e.g. CK+ tumour cells, ≥ 30 px diameter)
    where genuine pixel overlap is expected.

dilated_overlap
    Each map is dilated by a disk before AND-ing. Appropriate for small cells
    (lymphocytes, macrophages, ≈ 16–24 px diameter) where the prediction peak
    for two co-expressed markers may not land on exactly the same pixels.
    Default radius: 12 px ≈ 6 µm at 0.5 µm/px (half a lymphocyte diameter).
    Uses cv2.dilate which is ~4x faster than scipy.ndimage.binary_dilation.
"""

from __future__ import annotations

import cv2
import numpy as np


def _disk(radius: int) -> np.ndarray:
    """Return a filled circular structuring element of the given radius."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x**2 + y**2 <= radius**2).astype(np.uint8)


def pixel_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean AND of two binary maps at pixel level.

    Args:
        a: (H, W) float32 or bool binary map.
        b: (H, W) float32 or bool binary map.

    Returns:
        Boolean (H, W) co-expression mask.
    """
    return a.astype(bool) & b.astype(bool)


def dilated_overlap(a: np.ndarray, b: np.ndarray, radius_px: int = 12) -> np.ndarray:
    """Cell-footprint dilation co-expression mask.

    Dilates each map by a disk of ``radius_px`` before AND-ing, recovering
    same-cell co-expression for small cells where prediction peaks may not
    overlap exactly.

    Args:
        a:         (H, W) float32 or bool binary map.
        b:         (H, W) float32 or bool binary map.
        radius_px: Dilation radius in pixels. Default 12 px ≈ 6 µm at
                   0.5 µm/px (half a lymphocyte diameter).

    Returns:
        Boolean (H, W) co-expression mask.
    """
    disk = _disk(radius_px)
    a_u8 = a.astype(np.uint8)
    b_u8 = b.astype(np.uint8)
    return cv2.dilate(a_u8, disk).astype(bool) & cv2.dilate(b_u8, disk).astype(bool)
