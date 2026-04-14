"""OpenSlide wrapper exposing metadata and region reading."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openslide

# Target microns-per-pixel the model was trained at (~20x)
TARGET_MPP = 0.5


@dataclass(frozen=True)
class SlideMetadata:
    path: Path
    width: int  # slide width at level 0 in pixels
    height: int  # slide height at level 0 in pixels
    mpp: float  # microns per pixel at level 0 (x-axis)
    level_count: int
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    vendor: str | None


class SlideReader:
    """Thin wrapper around an OpenSlide object.

    Automatically selects the best level to read from so that the effective
    resolution matches the model's training resolution (~0.5 µm/px, 20x).

    Args:
        path: Local path to the WSI file.
    """

    def __init__(self, path: str | Path) -> None:
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        path = str(path)
        if path.startswith("s3://"):
            from .s3 import download_slide

            self._tmp_dir = tempfile.TemporaryDirectory()
            path = str(download_slide(path, dest_dir=self._tmp_dir.name))
        self.path = Path(path)
        self._slide = openslide.OpenSlide(str(self.path))
        self.metadata = self._read_metadata()
        self.read_level, self.read_downsample = self._select_read_level()

    def __enter__(self) -> SlideReader:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._slide.close()
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimensions_at_read_level(self) -> tuple[int, int]:
        """(width, height) of the slide at the selected read level."""
        return self._slide.level_dimensions[self.read_level]

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Read a region at the selected level.

        Args:
            x: Left edge in level-0 pixel coordinates.
            y: Top edge in level-0 pixel coordinates.
            width: Region width in pixels at the read level.
            height: Region height in pixels at the read level.

        Returns:
            uint8 RGB array of shape (height, width, 3).
        """
        region = self._slide.read_region(
            location=(x, y),
            level=self.read_level,
            size=(width, height),
        )
        return np.array(region.convert("RGB"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_metadata(self) -> SlideMetadata:
        props = self._slide.properties
        raw_mpp = props.get(openslide.PROPERTY_NAME_MPP_X)
        mpp = float(raw_mpp) if raw_mpp is not None else float("nan")
        return SlideMetadata(
            path=self.path,
            width=self._slide.dimensions[0],
            height=self._slide.dimensions[1],
            mpp=mpp,
            level_count=self._slide.level_count,
            level_dimensions=tuple(self._slide.level_dimensions),
            level_downsamples=tuple(self._slide.level_downsamples),
            vendor=props.get(openslide.PROPERTY_NAME_VENDOR),
        )

    def _select_read_level(self) -> tuple[int, float]:
        """Return the (level_index, effective_downsample) closest to TARGET_MPP."""
        mpp = self.metadata.mpp
        if np.isnan(mpp):
            # No MPP metadata — fall back to level 0 and warn
            import warnings

            warnings.warn(
                f"No MPP metadata found in {self.path.name}. "
                "Falling back to level 0. Verify that the slide is at ~20x (0.5 µm/px).",
                UserWarning,
                stacklevel=3,
            )
            return 0, 1.0

        target_downsample = TARGET_MPP / mpp
        best_level = self._slide.get_best_level_for_downsample(target_downsample)
        level_ds = self._slide.level_downsamples[best_level]
        # Residual rescaling needed after reading at best_level
        effective_downsample = target_downsample / level_ds
        return best_level, effective_downsample
