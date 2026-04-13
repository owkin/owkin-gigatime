"""Tile extraction and result stitching for whole-slide inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np
from PIL import Image

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

    Attributes:
        n_rows: Number of tile rows.
        n_cols: Number of tile columns.
        tile_size: Tile size in pixels (height == width).
        overlap: Overlap between adjacent tiles in pixels.
        slide_width: Width of the slide at the read level in pixels.
        slide_height: Height of the slide at the read level in pixels.
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
    progress: bool = False,
) -> Generator[tuple[Tile, TileGrid], None, None]:
    """Yield tiles from a slide together with the grid layout.

    All tiles are exactly ``tile_size × tile_size`` pixels. Partial tiles at
    the right and bottom edges are padded with white (background).

    The grid is the same object for every tile emitted in a single call —
    inspect it once from the first yielded value.

    Args:
        reader: Open :class:`~inference.data.slide.SlideReader`.
        tile_size: Output tile size in pixels. Should be 512 for GigaTIME.
        overlap: Overlap between adjacent tiles in pixels.
        min_tissue_fraction: Skip tiles where the fraction of non-white pixels
            is below this threshold. Use 0.0 (default) to keep all tiles.
        progress: If ``True``, display a tqdm progress bar over all candidate
            tiles (before tissue filtering).

    Yields:
        ``(tile, grid)`` tuples.
    """
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")

    slide_w, slide_h = reader.dimensions_at_read_level
    step = tile_size - overlap

    n_cols = max(1, int(np.ceil((slide_w - overlap) / step)))
    n_rows = max(1, int(np.ceil((slide_h - overlap) / step)))

    grid = TileGrid(
        n_rows=n_rows,
        n_cols=n_cols,
        tile_size=tile_size,
        overlap=overlap,
        slide_width=slide_w,
        slide_height=slide_h,
    )

    level_ds = reader._slide.level_downsamples[reader.read_level]

    coords = ((row, col) for row in range(n_rows) for col in range(n_cols))
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as e:
            raise ImportError("tqdm is required for progress=True: pip install tqdm") from e
        coords = tqdm(coords, total=grid.n_tiles, unit="tile", desc="Tiling")

    for row, col in coords:
        # Coordinates in read-level pixels
        x_level = col * step
        y_level = row * step

        # Clamp to slide boundaries
        read_w = min(tile_size, slide_w - x_level)
        read_h = min(tile_size, slide_h - y_level)

        # Convert back to level-0 coordinates for OpenSlide
        x0 = int(x_level * level_ds)
        y0 = int(y_level * level_ds)

        region = reader.read_region(x0, y0, read_w, read_h)

        # Rescale if the read level doesn't exactly match target MPP
        if abs(reader.read_downsample - 1.0) > 1e-3:
            target_w = round(read_w / reader.read_downsample)
            target_h = round(read_h / reader.read_downsample)
            region = np.array(
                Image.fromarray(region).resize((target_w, target_h), Image.Resampling.LANCZOS)
            )

        # Crop oversized tiles (rounding in rescale can add 1-2 px) then pad partial tiles
        region = region[:tile_size, :tile_size]
        if region.shape[0] != tile_size or region.shape[1] != tile_size:
            padded = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            padded[: region.shape[0], : region.shape[1]] = region
            region = padded

        if min_tissue_fraction > 0.0 and not _has_tissue(region, min_tissue_fraction):
            continue

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
        Float32 array of shape ``(grid.slide_height, grid.slide_width)``
        (or ``(..., C)`` if ``channel == "probabilities"``).
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
        else:  # max
            if is_prob:
                canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], patch)
            else:
                canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], patch)
            count[y0:y1, x0:x1] = 1  # sentinel — not used for division

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
    # A pixel is considered background if all channels are >= 230
    is_background = np.all(tile >= 230, axis=-1)
    tissue_fraction = 1.0 - is_background.mean()
    return tissue_fraction >= min_fraction
