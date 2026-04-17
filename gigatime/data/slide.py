"""OpenSlide wrapper exposing metadata and region reading."""

from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import openslide
from openslide.deepzoom import DeepZoomGenerator
from tilingtool.exceptions import MPPNotAvailableError
from tilingtool.utils.slide import (
    RobustOpenSlide,
    get_level_raw_mpp_mapping,
    get_slide_resolution,
    get_tiling_slide_level,
)

# Target microns-per-pixel the model was trained at (~20x)
TARGET_MPP = 0.5
# Warn (but don't crash) if the closest DZ level deviates from TARGET_MPP by
# more than this fraction.
_MPP_RTOL = 0.2


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

    Uses :func:`tilingtool.utils.slide.get_tiling_slide_level` with a
    :class:`openslide.deepzoom.DeepZoomGenerator` to select the zoom level
    whose effective resolution is closest to TARGET_MPP (0.5 µm/px, ~20x),
    regardless of the native pyramid layout.

    The underlying slide is opened via :class:`tilingtool.utils.slide.RobustOpenSlide`
    which retries reads at a lower level on ``OpenSlideError``, useful for old
    or partially-corrupt TCGA files.

    Args:
        path: Local path or S3 URI to the WSI file.
    """

    def __init__(self, path: str | Path) -> None:
        self._tmp_dir: tempfile.TemporaryDirectory | None = None
        path = str(path)
        if path.startswith("s3://"):
            from .s3 import download_slide

            self._tmp_dir = tempfile.TemporaryDirectory()
            path = str(download_slide(path, dest_dir=self._tmp_dir.name))
        self.path = Path(path)
        self._slide = RobustOpenSlide(str(self.path))
        self.metadata = self._read_metadata()
        self._dz = DeepZoomGenerator(self._slide, tile_size=512, overlap=0, limit_bounds=False)
        self.dz_level = self._select_dz_level()

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
        """(width, height) in pixels at the selected DZ level."""
        return self._dz.level_dimensions[self.dz_level]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_metadata(self) -> SlideMetadata:
        try:
            mpp = get_slide_resolution(self._slide)
        except KeyError:
            mpp = float("nan")
        return SlideMetadata(
            path=self.path,
            width=self._slide.dimensions[0],
            height=self._slide.dimensions[1],
            mpp=mpp,
            level_count=self._slide.level_count,
            level_dimensions=tuple(self._slide.level_dimensions),
            level_downsamples=tuple(self._slide.level_downsamples),
            vendor=self._slide.properties.get(openslide.PROPERTY_NAME_VENDOR),
        )

    def _select_dz_level(self) -> int:
        """Return the DZ level index whose MPP is closest to TARGET_MPP.

        Delegates to :func:`tilingtool.utils.slide.get_tiling_slide_level`.
        If no level falls within ``_MPP_RTOL`` of TARGET_MPP, emits a warning
        and falls back to the closest available level rather than crashing.
        """
        try:
            return get_tiling_slide_level(
                self._slide,
                self._dz,
                mpp=TARGET_MPP,
                default_mpp_max=0.25,
                mpp_rtol=_MPP_RTOL,
            )
        except MPPNotAvailableError as exc:
            warnings.warn(str(exc), UserWarning, stacklevel=3)
            # Tolerance exceeded — still pick the closest level rather than crashing.
            mapping = get_level_raw_mpp_mapping(self._slide, self._dz, default_mpp_max=0.25)
            return min(mapping, key=lambda lvl: abs(mapping[lvl] - TARGET_MPP))
