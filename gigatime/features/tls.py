"""Tertiary Lymphoid Structure (TLS) detection.

TLS are ectopic lymphoid aggregates that form in tumours and are associated
with better immunotherapy response in NSCLC.  GigaTIME does not predict CD21
(follicular dendritic cells, the canonical TLS marker), so TLS are approximated
as large spatial aggregates of B cells (CD20) co-localised with T cells (CD3).

Method
------
1. Build a combined lymphocyte mask: (CD20 | CD3) & DAPI.
2. Label connected components.
3. Filter components by minimum area (default ≥ π·r² with r = 50 µm ≈ 100 px).
4. Require each component to contain both CD20 and CD3 signal (B+T co-presence).
5. Return count and total area of qualifying clusters, normalised by tissue area.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

MPP = 0.5  # microns per pixel


def detect_tls(
    cd20: np.ndarray,
    cd3: np.ndarray,
    dapi: np.ndarray,
    min_radius_um: float = 50.0,
    mpp: float = MPP,
) -> dict[str, float]:
    """Detect TLS-like lymphocyte aggregates and return summary statistics.

    Args:
        cd20:          (H, W) float32 or bool map for CD20 (B cells).
        cd3:           (H, W) float32 or bool map for CD3 (T cells).
        dapi:          (H, W) float32 or bool map for DAPI (all nuclei / tissue).
        min_radius_um: Minimum equivalent radius for a cluster to qualify as TLS.
                       Default 50 µm — roughly the diameter of a small follicle.
        mpp:           Microns per pixel.

    Returns:
        Dict with keys:
            ``tls_count``        — number of qualifying clusters
            ``tls_area_fraction``— total TLS area / total tissue area
    """
    cd20_mask = cd20.astype(bool)
    cd3_mask = cd3.astype(bool)
    tissue_mask = dapi.astype(bool)

    lymphocyte_mask = (cd20_mask | cd3_mask) & tissue_mask

    labeled, n_components = label(lymphocyte_mask)
    if n_components == 0:
        return {"tls_count": 0.0, "tls_area_fraction": 0.0}

    min_area_px = np.pi * (min_radius_um / mpp) ** 2

    tls_count = 0
    tls_area_px = 0

    for component_id in range(1, n_components + 1):
        component = labeled == component_id
        area = int(component.sum())

        if area < min_area_px:
            continue

        # Require both B and T cell signal within the cluster
        has_b = (cd20_mask & component).any()
        has_t = (cd3_mask & component).any()
        if not (has_b and has_t):
            continue

        tls_count += 1
        tls_area_px += area

    tissue_area_px = int(tissue_mask.sum())
    tls_area_fraction = tls_area_px / tissue_area_px if tissue_area_px > 0 else 0.0

    return {
        "tls_count": float(tls_count),
        "tls_area_fraction": tls_area_fraction,
    }
