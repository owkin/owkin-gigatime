"""Top-level feature assembly for one slide.

Calls all sub-modules and returns a flat dict of scalar features keyed by
feature name.  Input is the prediction dict returned by
``gigatime.inference.predict`` after stitching, i.e. a mapping from channel
name to a (H, W) float32 array.
"""

from __future__ import annotations

import numpy as np

from .coexpression import dilated_overlap, pixel_overlap
from .compartments import stroma_mask, tissue_mask, tumor_mask
from .density import density
from .tls import detect_tls


def compute_features(maps: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute all LUAD spatial features for a single slide.

    Args:
        maps: Dict mapping channel name → (H, W) float32 binary prediction map.
              Expected keys: all 23 GigaTIME channels (see
              ``gigatime.inference.constants.CHANNEL_NAMES``).

    Returns:
        Flat dict of scalar feature values.
    """
    ck = maps["CK"]
    dapi = maps["DAPI"]
    cd8 = maps["CD8"]
    cd4 = maps["CD4"]
    cd3 = maps["CD3"]
    cd20 = maps["CD20"]
    cd68 = maps["CD68"]
    cd34 = maps["CD34"]
    pd1 = maps["PD-1"]
    pdl1 = maps["PD-L1"]
    ki67 = maps["Ki67"]
    casp3 = maps["Caspase3-D"]

    t_mask = tumor_mask(ck, dapi)
    s_mask = stroma_mask(ck, dapi)
    ti_mask = tissue_mask(dapi)

    features: dict[str, float] = {}

    # 1 — PD-L1 TPS (large cells → pixel overlap)
    ck_dapi = pixel_overlap(ck, dapi)
    pdl1_ck = pixel_overlap(pdl1, ck)
    denom = float(ck_dapi.sum())
    features["pdl1_tps"] = float(pdl1_ck.sum()) / denom if denom > 0 else 0.0

    # 2 — Intratumoral CD8 density
    features["cd8_intratumoral_density"] = density(cd8, t_mask)

    # 3 — Immune exclusion index
    cd8_stroma = density(cd8, s_mask)
    cd8_tumor = density(cd8, t_mask)
    features["immune_exclusion_index"] = cd8_stroma / (cd8_stroma + cd8_tumor + 1e-6)

    # 4 — CD8+PD-1 exhaustion fraction (small cells → dilation)
    cd8_pd1 = dilated_overlap(cd8, pd1)
    cd8_total = float(cd8.astype(bool).sum())
    features["cd8_pd1_exhaustion_fraction"] = (
        float(cd8_pd1.sum()) / cd8_total if cd8_total > 0 else 0.0
    )

    # 5 — Macrophage-to-T-cell ratio
    cd68_tissue = float((cd68.astype(bool) & ti_mask).sum())
    cd8_tissue = float((cd8.astype(bool) & ti_mask).sum())
    features["macrophage_to_tcell_ratio"] = cd68_tissue / (cd8_tissue + 1e-6)

    # 6 — Intratumoral CD4/CD8 ratio
    cd4_tumor = density(cd4, t_mask)
    cd8_tumor_d = density(cd8, t_mask)
    features["cd4_cd8_intratumoral_ratio"] = cd4_tumor / (cd8_tumor_d + 1e-6)

    # 7 — Tumour proliferation index (Ki67 TPI, large cells → pixel overlap)
    ki67_ck = pixel_overlap(ki67, ck)
    features["ki67_tpi"] = float(ki67_ck.sum()) / denom if denom > 0 else 0.0

    # 8 — Tumour apoptosis index (large cells → pixel overlap)
    casp3_ck = pixel_overlap(casp3, ck)
    features["apoptosis_index"] = float(casp3_ck.sum()) / denom if denom > 0 else 0.0

    # 9 — TLS score
    tls = detect_tls(cd20, cd3, dapi)
    features.update(tls)

    # 10 — Intratumoural vascular density
    features["vascular_density_intratumoral"] = density(cd34, t_mask)

    return features
