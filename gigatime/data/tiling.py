"""Tile extraction and result stitching for whole-slide inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np

from ..inference.constants import CHANNEL_NAMES, TILE_SIZE_PX
from .slide import SlideReader


@dataclass(frozen=True)
class Tile:
    """A single tile extracted from a slide.

    Attributes:
        array: uint8 RGB image of shape (tile_size, tile_size, 3), ready for
            :func:`inference.predict.predict`.
        row: Row index of this tile in the grid (0-based).
        col: Column index of this tile in the grid (0-based).
        x: Left edge of the tile in level-0 pixel coordinates.
        y: Top edge of the tile in level-0 pixel coordinates.
        tile_size: Spatial size of the tile in pixels (equal for H and W).
    """

    array: np.ndarray
    row: int
    col: int
    x: int
    y: int
    tile_size: int


@dataclass(frozen=True)
class TileGrid:
    """Describes the full tiling layout of a slide.

    All coordinates (slide_width, slide_height) are in target-MPP space
    (i.e. at the DZ level selected by :class:`~.slide.SlideReader`), which
    corresponds to ~0.5 µm/px (~20x) regardless of the native scan resolution.

    Attributes:
        n_rows: Number of tile rows.
        n_cols: Number of tile columns.
        tile_size: Tile size in pixels (height == width).
        overlap: Overlap between adjacent tiles in pixels.
        slide_width: Width of the slide at the selected DZ level in pixels.
        slide_height: Height of the slide at the selected DZ level in pixels.
    """

    n_rows: int
    n_cols: int
    tile_size: int
    overlap: int
    slide_width: int
    slide_height: int

    @property
    def n_tiles(self) -> int:
        return self.n_rows * self.n_cols


def iter_tiles(
    reader: SlideReader,
    tile_size: int = TILE_SIZE_PX,
    overlap: int = 0,
    min_tissue_fraction: float = 0.0,
    matter_mask: np.ndarray | None = None,
    matter_threshold: float = 0.6,
    progress: bool = False,
) -> Generator[tuple[Tile, TileGrid], None, None]:
    """Yield tiles from a slide together with the grid layout.

    Tiles are read via the slide's :class:`openslide.deepzoom.DeepZoomGenerator`
    at the DZ level preselected by :class:`~.slide.SlideReader` to match
    ~0.5 µm/px.  All tiles are exactly ``tile_size × tile_size`` pixels;
    partial tiles at the right and bottom edges are padded with white.

    The grid is the same object for every tile emitted in a single call —
    inspect it once from the first yielded value.

    Args:
        reader: Open :class:`~.slide.SlideReader`.
        tile_size: Output tile size in pixels. Should be 512 for GigaTIME.
        overlap: Overlap between adjacent tiles in pixels.
        min_tissue_fraction: Skip tiles where the fraction of non-white pixels
            is below this threshold.  Ignored when ``matter_mask`` is provided.
        matter_mask: Optional tissue probability mask from tilingtool's
            ``BRUNet()(slide)``, shape ``(W, H)`` at the detector's native
            resolution.  When provided, tiles are filtered by the mean
            probability in their region instead of the brightness-based
            ``_has_tissue`` heuristic.
        matter_threshold: Minimum mean probability in the mask region for a
            tile to be considered tissue.  Only used when ``matter_mask`` is
            provided.  Default 0.6 matches tilingtool's default.
        progress: If ``True``, display a tqdm progress bar.

    Yields:
        ``(tile, grid)`` tuples.
    """
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")
    if overlap != 0:
        raise NotImplementedError(
            "overlap > 0 is not supported with DeepZoomGenerator-based tiling"
        )

    dz = reader._dz
    dz_level = reader.dz_level

    slide_w, slide_h = dz.level_dimensions[dz_level]
    n_cols, n_rows = dz.level_tiles[dz_level]

    grid = TileGrid(
        n_rows=n_rows,
        n_cols=n_cols,
        tile_size=tile_size,
        overlap=0,
        slide_width=slide_w,
        slide_height=slide_h,
    )

    # Scale factor from DZ-level pixels to level-0 pixels, for Tile.x / Tile.y.
    native_w = reader.metadata.width
    l0_scale = native_w / slide_w

    # If a matter mask is provided, precompute the scale from tile grid coords
    # to mask coords.  The mask shape is (mask_w, mask_h) — note W×H order as
    # returned by tilingtool's MatterFilter.
    if matter_mask is not None:
        mask_w, mask_h = matter_mask.shape
        mask_scale_x = mask_w / slide_w
        mask_scale_y = mask_h / slide_h

    coords = ((row, col) for row in range(n_rows) for col in range(n_cols))
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as e:
            raise ImportError("tqdm is required for progress=True: pip install tqdm") from e
        coords = tqdm(coords, total=grid.n_tiles, unit="tile", desc="Tiling")

    for row, col in coords:
        # --- Tissue filtering (before reading the tile for speed) ---
        if matter_mask is not None:
            y0_m = int(row * tile_size * mask_scale_y)
            x0_m = int(col * tile_size * mask_scale_x)
            y1_m = max(y0_m + 1, int(min((row + 1) * tile_size, slide_h) * mask_scale_y))
            x1_m = max(x0_m + 1, int(min((col + 1) * tile_size, slide_w) * mask_scale_x))
            if matter_mask[x0_m:x1_m, y0_m:y1_m].mean() < matter_threshold:
                continue

        # DZG address is (col, row) — note the order
        img = dz.get_tile(dz_level, (col, row))
        region = np.array(img.convert("RGB"))

        # Edge tiles from DZG are smaller than tile_size — pad to tile_size × tile_size
        if region.shape[0] != tile_size or region.shape[1] != tile_size:
            padded = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            padded[: region.shape[0], : region.shape[1]] = region
            region = padded

        # Fall back to brightness heuristic when no matter mask is available.
        if matter_mask is None and min_tissue_fraction > 0.0:
            if not _has_tissue(region, min_tissue_fraction):
                continue

        x0 = round(col * tile_size * l0_scale)
        y0 = round(row * tile_size * l0_scale)
        yield Tile(array=region, row=row, col=col, x=x0, y=y0, tile_size=tile_size), grid


def stitch(
    results: list[tuple[dict[str, np.ndarray], Tile]],
    grid: TileGrid,
    channel: str,
    mode: str = "max",
) -> np.ndarray:
    """Assemble per-tile predictions into a slide-level map.

    Overlapping regions are resolved by the ``mode`` argument.

    Args:
        results: List of ``(prediction_dict, tile)`` pairs as returned by
            :func:`inference.predict.predict` paired with its :class:`Tile`.
        grid: The :class:`TileGrid` describing the layout (from :func:`iter_tiles`).
        channel: Name of the channel to stitch (e.g. ``"CD8"``).
            Use ``"probabilities"`` for the raw float map (averaged in overlaps).
        mode: How to resolve overlapping pixels.
            ``"max"`` keeps the highest prediction; ``"mean"`` averages them.

    Returns:
        Float32 array of shape ``(grid.slide_height, grid.slide_width)``.
    """
    if mode not in ("max", "mean"):
        raise ValueError(f"mode must be 'max' or 'mean', got {mode!r}")

    is_prob = channel == "probabilities"
    step = grid.tile_size - grid.overlap

    if is_prob:
        canvas = np.zeros(
            (grid.slide_height, grid.slide_width, len(CHANNEL_NAMES)), dtype=np.float32
        )
    else:
        canvas = np.zeros((grid.slide_height, grid.slide_width), dtype=np.float32)

    count = np.zeros((grid.slide_height, grid.slide_width), dtype=np.float32)

    for pred, tile in results:
        y0 = tile.row * step
        x0 = tile.col * step
        y1 = min(y0 + grid.tile_size, grid.slide_height)
        x1 = min(x0 + grid.tile_size, grid.slide_width)
        th = y1 - y0
        tw = x1 - x0

        patch = pred[channel][:th, :tw] if not is_prob else pred[channel][:th, :tw, :]

        if mode == "mean":
            canvas[y0:y1, x0:x1] += patch
            count[y0:y1, x0:x1] += 1
        else:
            canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], patch)
            count[y0:y1, x0:x1] = 1

    if mode == "mean":
        mask = count > 0
        if is_prob:
            canvas[mask] /= count[mask, np.newaxis]
        else:
            canvas[mask] /= count[mask]

    return canvas


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_tissue(tile: np.ndarray, min_fraction: float) -> bool:
    """Return True if the tile contains enough non-white pixels."""
    is_background = np.all(tile >= 230, axis=-1)
    tissue_fraction = 1.0 - is_background.mean()
    return tissue_fraction >= min_fraction
