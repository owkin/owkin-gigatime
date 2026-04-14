"""Extract GigaTIME spatial features for all TCGA-LUAD slides.

Designed for long-running tmux sessions (20+ hours). Saves one JSON per slide
so that the script can be interrupted and restarted without reprocessing
completed slides.

Usage
-----
    python scripts/extract_features_tcga_luad.py [--output-dir OUTPUT_DIR] [--device DEVICE]

Output
------
    OUTPUT_DIR/
        <slide_stem>.json     — per-slide features + metadata
        extract_features.log  — full log with timestamps
        features.parquet      — merged table (written at the end, or on demand)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
from abstra.manifest import Manifest
from tqdm import tqdm

from gigatime.data import SlideReader, iter_tiles, list_slides
from gigatime.data.paths import TCGA_LUAD
from gigatime.features import compute_features_from_tiles
from gigatime.inference import load_model, predict

FEATURE_CHANNELS = [
    "CK", "DAPI", "CD8", "CD4", "CD68", "CD34",
    "PD-1", "PD-L1", "Ki67", "Caspase3-D",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------

def _process_slide(
    uri: str,
    model: torch.nn.Module,
    device: str,
) -> dict:
    """Run inference + feature extraction for a single slide.

    Returns a dict with ``features``, ``metadata``, and ``elapsed_s``.
    Raises on any error — the caller handles it.
    """
    reader = SlideReader(uri)
    w, h = reader.dimensions_at_read_level

    tile_stream: list = []
    grid = None
    future = None

    def _accumulate(preds: dict, tile) -> None:
        tile_stream.append(({ch: preds[ch] for ch in FEATURE_CHANNELS}, tile))

    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as executor, torch.inference_mode():
        for tile, grid in iter_tiles(reader, min_tissue_fraction=0.05):
            preds = predict(tile.array, model, device=device)
            if future is not None:
                future.result()
            future = executor.submit(_accumulate, preds, tile)
        if future is not None:
            future.result()

    reader.close()

    if grid is None:
        raise RuntimeError("No tissue tiles found — check slide content or min_tissue_fraction")

    n_tiles = len(tile_stream)
    features = compute_features_from_tiles(iter(tile_stream), grid)
    elapsed = time.perf_counter() - t0

    return {
        "features": features,
        "metadata": {
            "uri": uri,
            "slide_width_px": w,
            "slide_height_px": h,
            "n_tiles": n_tiles,
            "processed_at": datetime.utcnow().isoformat(),
        },
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GigaTIME features for TCGA-LUAD slides")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tcga_luad_features"),
        help="Directory where per-slide JSONs and the log are written (default: outputs/tcga_luad_features)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N slides (useful for dry-runs)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_logging(output_dir / "extract_features.log")
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("GigaTIME feature extraction — TCGA-LUAD")
    log.info(f"Output directory : {output_dir.resolve()}")
    log.info(f"Device           : {args.device}")

    # ---- Discover slides ----
    manifest = Manifest().load()
    dataset = manifest.datasets[TCGA_LUAD]
    all_slides = [s for s in list_slides(bucket=dataset.bucket, prefix=dataset.prefix) if "parafine" in s]
    if args.limit is not None:
        all_slides = all_slides[: args.limit]

    log.info(f"Total slides     : {len(all_slides)}")

    # ---- Determine which slides still need processing ----
    pending = []
    for uri in all_slides:
        stem = Path(uri).stem
        out_path = output_dir / f"{stem}.json"
        if out_path.exists():
            log.info(f"  SKIP (done)  {Path(uri).name}")
        else:
            pending.append(uri)

    log.info(f"Slides to process: {len(pending)}  (skipping {len(all_slides) - len(pending)} already done)")

    if not pending:
        log.info("Nothing to do — all slides already processed.")
        _merge_results(output_dir, log)
        return

    # ---- Load model once ----
    log.info("Loading model…")
    model = load_model(device=args.device)
    model.eval()
    log.info("Model ready.")

    # ---- Process ----
    errors: list[tuple[str, str]] = []
    slide_times: list[float] = []
    wall_start = time.perf_counter()

    for idx, uri in enumerate(tqdm(pending, desc="Slides", unit="slide", file=sys.stdout)):
        name = Path(uri).name
        stem = Path(uri).stem
        out_path = output_dir / f"{stem}.json"

        log.info(f"[{idx + 1}/{len(pending)}] {name}")

        try:
            result = _process_slide(uri, model, args.device)
        except Exception as exc:
            errors.append((name, str(exc)))
            log.error(f"  FAILED: {exc}")
            continue

        # Atomic write: write to .tmp then rename so a crash mid-write
        # doesn't leave a corrupt partial file that would be skipped next run.
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(result, indent=2))
        tmp_path.rename(out_path)

        elapsed = result["elapsed_s"]
        tiles_per_sec = result["metadata"]["n_tiles"] / elapsed if elapsed > 0 else float("nan")
        slide_times.append(elapsed)
        log.info(
            f"  OK — {result['metadata']['n_tiles']} tiles  "
            f"{elapsed:.1f}s  ({tiles_per_sec:.1f} tiles/s)  → {out_path.name}"
        )

    # ---- Summary ----
    total_wall = time.perf_counter() - wall_start
    log.info("=" * 60)
    log.info(
        f"Finished: {len(slide_times)} ok, {len(errors)} errors  "
        f"— wall time {total_wall / 60:.1f} min"
    )
    if slide_times:
        avg_s = sum(slide_times) / len(slide_times)
        log.info(f"Avg time/slide   : {avg_s:.1f}s  ({avg_s / 60:.1f} min)")
        remaining = len(all_slides) - (len(all_slides) - len(pending)) - len(slide_times) - len(errors)
        if remaining > 0:
            log.info(f"Estimated remaining: ~{remaining * avg_s / 3600:.1f} h for {remaining} slides")
    if errors:
        log.warning("Failed slides:")
        for name, msg in errors:
            log.warning(f"  {name}: {msg}")

    _merge_results(output_dir, log)


def _merge_results(output_dir: Path, log: logging.Logger) -> None:
    """Merge all per-slide JSONs into a single parquet file."""
    json_files = sorted(output_dir.glob("*.json"))
    if not json_files:
        log.info("No JSON files to merge.")
        return

    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not available — skipping parquet merge. Install with: pip install pandas pyarrow")
        return

    rows = []
    for p in json_files:
        try:
            data = json.loads(p.read_text())
            row = {"slide": p.stem}
            row.update(data.get("metadata", {}))
            row.update(data.get("features", {}))
            row["elapsed_s"] = data.get("elapsed_s")
            rows.append(row)
        except Exception as exc:
            log.warning(f"Could not parse {p.name}: {exc}")

    df = pd.DataFrame(rows)
    parquet_path = output_dir / "features.parquet"
    df.to_parquet(parquet_path, index=False)
    csv_path = output_dir / "features.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Merged {len(df)} slides → {parquet_path}  and  {csv_path}")


if __name__ == "__main__":
    main()
