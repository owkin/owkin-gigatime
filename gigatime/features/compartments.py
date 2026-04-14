"""Tissue compartment masks derived from channel maps.

Produces three boolean masks covering the slide:
    - tumor:      CK-positive cells (epithelial / tumour)
    - stroma:     tissue pixels outside the tumour compartment
    - background: non-tissue pixels (whitespace, glass)

All masks share the same (H, W) shape as the input channel maps.
"""

from __future__ import annotations

import numpy as np


def tissue_mask(dapi: np.ndarray) -> np.ndarray:
    """Boolean mask of all tissue pixels (non-background).

    Uses DAPI as a nuclear stain proxy: any pixel with DAPI signal is
    considered tissue.

    Args:
        dapi: (H, W) float32 binary map for the DAPI channel.

    Returns:
        Boolean (H, W) array; True = tissue.
    """
    return dapi.astype(bool)


def tumor_mask(ck: np.ndarray, dapi: np.ndarray) -> np.ndarray:
    """Boolean mask of tumour compartment pixels (CK+, DAPI+).

    Args:
        ck:   (H, W) float32 binary map for the CK channel.
        dapi: (H, W) float32 binary map for the DAPI channel.

    Returns:
        Boolean (H, W) array; True = tumour.
    """
    return ck.astype(bool) & dapi.astype(bool)


def stroma_mask(ck: np.ndarray, dapi: np.ndarray) -> np.ndarray:
    """Boolean mask of stromal compartment pixels (DAPI+, CK-).

    Args:
        ck:   (H, W) float32 binary map for the CK channel.
        dapi: (H, W) float32 binary map for the DAPI channel.

    Returns:
        Boolean (H, W) array; True = stroma.
    """
    return ~ck.astype(bool) & dapi.astype(bool)
