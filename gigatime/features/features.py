"""Top-level feature assembly for one slide.

Two entry points are provided:

compute_features_from_tiles  (recommended)
    Streams over ``(preds, tile)`` pairs produced during inference.
    All features are computed as running pixel-count sums — no tile
    predictions are kept in memory.

    Memory cost: O(1) beyond the model and one tile at a time.
    Compare to the naïve approach: N_tiles × 21 × 512² × 4 bytes — typically
    30–60 GB for a TCGA whole-slide image.

    TLS detection is not included in the streaming path: it requires
    connected-component analysis across the full slide canvas and should be
    added once a downsampled-canvas strategy is in place.

compute_features
    Accepts a pre-built ``maps`` dict (channel → H×W array).  Useful when
    the maps are already in memory for visualisation purposes.  Includes TLS.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..data.tiling import Tile, TileGrid
from .coexpression import dilated_overlap, pixel_overlap
from .compartments import stroma_mask, tissue_mask, tumor_mask
from .density import density
from .tls import detect_tls


def compute_features_from_tiles(
    tile_iter: Iterable[tuple[dict[str, np.ndarray], Tile]],
    grid: TileGrid,
) -> dict[str, float]:
    """Compute slide-level features by streaming over per-tile predictions.

    No tile prediction dicts are retained in memory.  All features are
    accumulated as integer pixel counts and resolved into scalars at the end.

    TLS detection is excluded from this path (requires a full slide canvas).
    Use :func:`compute_features` when maps are already available.

    Args:
        tile_iter: Iterable of ``(prediction_dict, tile)`` pairs, as produced
                   by pairing :func:`~gigatime.data.iter_tiles` with
                   :func:`~gigatime.inference.predict`.
        grid:      :class:`~gigatime.data.TileGrid` returned by
                   :func:`~gigatime.data.iter_tiles`.

    Returns:
        Flat ``dict[str, float]`` of scalar feature values.
    """
    H, W = grid.slide_height, grid.slide_width
    step = grid.tile_size - grid.overlap

    acc: dict[str, int] = {
        "ck_dapi": 0,  # tumour pixels (CK & DAPI) — shared denominator
        "pdl1_ck": 0,
        "ki67_ck": 0,
        "casp3_ck": 0,
        "tumor_px": 0,
        "stroma_px": 0,
        "cd8_tumor": 0,
        "cd8_stroma": 0,
        "cd8_total": 0,  # CD8 in tissue — exhaustion & macrophage ratio denom
        "cd8_pd1": 0,
        "cd68_tissue": 0,
        "cd4_tumor": 0,
        "cd34_tumor": 0,
    }

    for preds, tile in tile_iter:
        ts = grid.tile_size
        y0 = tile.row * step
        x0 = tile.col * step
        y1 = min(y0 + ts, H)
        x1 = min(x0 + ts, W)
        th, tw = y1 - y0, x1 - x0

        ck = preds["CK"][:th, :tw].astype(bool)
        dapi = preds["DAPI"][:th, :tw].astype(bool)
        cd8 = preds["CD8"][:th, :tw].astype(bool)
        cd4 = preds["CD4"][:th, :tw].astype(bool)
        cd68 = preds["CD68"][:th, :tw].astype(bool)
        cd34 = preds["CD34"][:th, :tw].astype(bool)
        pdl1 = preds["PD-L1"][:th, :tw].astype(bool)
        ki67 = preds["Ki67"][:th, :tw].astype(bool)
        casp3 = preds["Caspase3-D"][:th, :tw].astype(bool)

        t_mask = ck & dapi
        s_mask = ~ck & dapi

        acc["ck_dapi"] += int(t_mask.sum())
        acc["pdl1_ck"] += int((pdl1 & ck).sum())
        acc["ki67_ck"] += int((ki67 & ck).sum())
        acc["casp3_ck"] += int((casp3 & ck).sum())
        acc["tumor_px"] += int(t_mask.sum())
        acc["stroma_px"] += int(s_mask.sum())
        acc["cd8_tumor"] += int((cd8 & t_mask).sum())
        acc["cd8_stroma"] += int((cd8 & s_mask).sum())
        acc["cd8_total"] += int((cd8 & dapi).sum())
        acc["cd68_tissue"] += int((cd68 & dapi).sum())
        acc["cd4_tumor"] += int((cd4 & t_mask).sum())
        acc["cd34_tumor"] += int((cd34 & t_mask).sum())

        # Dilation co-expression is valid per tile: lymphocytes (~16–24 px)
        # never span tile boundaries (tiles are 512 px).
        acc["cd8_pd1"] += int(
            dilated_overlap(preds["CD8"][:th, :tw], preds["PD-1"][:th, :tw]).sum()
        )

    ck_dapi = acc["ck_dapi"]
    tumor_px = acc["tumor_px"]
    stroma_px = acc["stroma_px"]
    cd8_tumor = acc["cd8_tumor"]
    cd8_stroma = acc["cd8_stroma"]

    cd8_stroma_d = cd8_stroma / stroma_px if stroma_px > 0 else 0.0
    cd8_tumor_d = cd8_tumor / tumor_px if tumor_px > 0 else 0.0

    return {
        "pdl1_tps": acc["pdl1_ck"] / ck_dapi if ck_dapi > 0 else 0.0,
        "ki67_tpi": acc["ki67_ck"] / ck_dapi if ck_dapi > 0 else 0.0,
        "apoptosis_index": acc["casp3_ck"] / ck_dapi if ck_dapi > 0 else 0.0,
        "cd8_intratumoral_density": cd8_tumor_d,
        "immune_exclusion_index": cd8_stroma_d / (cd8_stroma_d + cd8_tumor_d + 1e-6),
        "cd8_pd1_exhaustion_fraction": (
            acc["cd8_pd1"] / acc["cd8_total"] if acc["cd8_total"] > 0 else 0.0
        ),
        "macrophage_to_tcell_ratio": acc["cd68_tissue"] / (acc["cd8_total"] + 1e-6),
        "cd4_cd8_intratumoral_ratio": (
            (acc["cd4_tumor"] / tumor_px) / (cd8_tumor_d + 1e-6) if tumor_px > 0 else 0.0
        ),
        "vascular_density_intratumoral": (acc["cd34_tumor"] / tumor_px if tumor_px > 0 else 0.0),
    }


def compute_features(maps: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute all LUAD spatial features from pre-built slide-level maps.

    Convenient when the maps are already in memory (e.g. for visualisation).
    For large-scale batch processing, prefer :func:`compute_features_from_tiles`
    which avoids materialising all channel maps.

    Args:
        maps: Dict mapping channel name → (H, W) float32 binary prediction map.

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

    ck_dapi = pixel_overlap(ck, dapi)
    denom = float(ck_dapi.sum())

    features["pdl1_tps"] = float(pixel_overlap(pdl1, ck).sum()) / denom if denom > 0 else 0.0
    features["ki67_tpi"] = float(pixel_overlap(ki67, ck).sum()) / denom if denom > 0 else 0.0
    features["apoptosis_index"] = (
        float(pixel_overlap(casp3, ck).sum()) / denom if denom > 0 else 0.0
    )
    features["cd8_intratumoral_density"] = density(cd8, t_mask)

    cd8_stroma_d = density(cd8, s_mask)
    cd8_tumor_d = density(cd8, t_mask)
    features["immune_exclusion_index"] = cd8_stroma_d / (cd8_stroma_d + cd8_tumor_d + 1e-6)

    cd8_pd1 = dilated_overlap(cd8, pd1)
    cd8_total = float(cd8.astype(bool).sum())
    features["cd8_pd1_exhaustion_fraction"] = (
        float(cd8_pd1.sum()) / cd8_total if cd8_total > 0 else 0.0
    )

    cd68_tissue = float((cd68.astype(bool) & ti_mask).sum())
    cd8_tissue = float((cd8.astype(bool) & ti_mask).sum())
    features["macrophage_to_tcell_ratio"] = cd68_tissue / (cd8_tissue + 1e-6)
    features["cd4_cd8_intratumoral_ratio"] = density(cd4, t_mask) / (cd8_tumor_d + 1e-6)
    features["vascular_density_intratumoral"] = density(cd34, t_mask)

    tls = detect_tls(cd20, cd3, dapi)
    features.update(tls)

    return features
