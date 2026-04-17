"""Extract GigaTIME spatial features for all TCGA-LUAD slides.

Designed for long-running tmux sessions. Saves one JSON + per-channel PNG maps
per slide so that the script can be interrupted and restarted without
reprocessing completed slides.

Multi-GPU usage (recommended)
------------------------------
    python scripts/extract_features_tcga_luad.py

    Each available GPU gets its own worker process. Slides are distributed
    round-robin. Workers write to the same output directory using atomic renames
    so there are no conflicts.

Single-GPU / CPU
-----------------
    python scripts/extract_features_tcga_luad.py --num-workers 1 --device cuda:0

Options
-------
    --output-dir DIR      Where per-slide JSONs, PNG maps and logs go
                          (default: EFS outputs directory)
    --num-workers N       Worker processes to spawn (default: all available GPUs,
                          or 1 on CPU-only machines)
    --batch-size N        Tiles per forward pass per worker (default: 7)
    --prefetch N          Tile read-ahead buffer per worker (default: 48)
    --map-size N          Max dimension of per-channel PNG maps in pixels (default: 1024)
    --limit N             Process at most N slides total (dry-run helper)

Output
------
    OUTPUT_DIR/
        slide-level-features/<slide_stem>.json   — per-slide features + metadata
        maps/<slide_stem>/<ch>.png               — per-channel spatial map (mIF colors)
        worker_<rank>.log                        — per-worker log with timestamps
        extract_features.log                     — coordinator log
        gigatime_tcga_luad.parquet               — merged table, refreshed every 50 slides
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from abstra.manifest import Manifest
from PIL import Image
from tqdm import tqdm

from gigatime.data import SlideReader, iter_tiles, list_slides
from gigatime.data.paths import TCGA_LUAD
from gigatime.features import SlideFeatureAccumulator
from gigatime.inference import load_model, predict_batch
from gigatime.inference.constants import BACKGROUND_CHANNELS, CHANNEL_NAMES

ANALYSIS_CHANNELS: list[str] = [ch for ch in CHANNEL_NAMES if ch not in BACKGROUND_CHANNELS]

# Regenerate the merged parquet after every this many newly completed slides.
MERGE_EVERY = 50

# Per-channel mIF display colors (RGB). Black background, signal in channel color.
# Conventions follow standard multiplex immunofluorescence panel assignments.
CHANNEL_COLORS: dict[str, tuple[int, int, int]] = {
    "DAPI": (100, 149, 237),  # cornflower blue  — nuclei
    "CK": (255, 255, 0),  # yellow           — tumour cells
    "CD3": (50, 205, 50),  # lime green       — pan T-cells
    "CD8": (0, 255, 255),  # cyan             — cytotoxic T-cells
    "CD4": (144, 238, 144),  # light green      — helper T-cells
    "CD68": (255, 165, 0),  # orange           — macrophages
    "PD-L1": (255, 0, 0),  # red              — immune checkpoint (tumour)
    "PD-1": (255, 0, 255),  # magenta          — exhaustion / activation
    "CD20": (255, 215, 0),  # gold             — B-cells
    "Ki67": (255, 100, 0),  # orange-red       — proliferation
    "CD34": (220, 20, 60),  # crimson          — vasculature
    "CD138": (148, 0, 211),  # violet           — plasma cells
    "CD11c": (255, 200, 0),  # amber            — dendritic cells
    "CD14": (210, 105, 30),  # chocolate        — monocytes
    "CD16": (189, 183, 107),  # dark khaki       — NK cells / neutrophils
    "T-bet": (0, 191, 255),  # deep sky blue    — Th1 / CD8 transcription factor
    "Tryptase": (186, 85, 211),  # medium orchid    — mast cells
    "Actin-D": (169, 169, 169),  # dark grey        — smooth muscle / myofibroblasts
    "Caspase3-D": (127, 255, 0),  # chartreuse       — apoptosis
    "PHH3-B": (255, 255, 102),  # pale yellow      — mitosis
    "Transgelin": (119, 136, 153),  # light slate grey — smooth muscle actin
}

FEATURE_CHANNELS = [
    "CK",
    "DAPI",
    "CD8",
    "CD4",
    "CD68",
    "CD34",
    "PD-1",
    "PD-L1",
    "Ki67",
    "Caspase3-D",
]

# ---------------------------------------------------------------------------
# Tile prefetch helper
# ---------------------------------------------------------------------------


def _prefetch_tiles(reader, min_tissue_fraction: float, maxsize: int):
    """Yield (tile, grid) pairs — iter_tiles runs in a background thread."""
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    _DONE = object()

    def _run() -> None:
        try:
            for item in iter_tiles(reader, min_tissue_fraction=min_tissue_fraction):
                q.put(item)
        finally:
            q.put(_DONE)

    threading.Thread(target=_run, daemon=True).start()
    while True:
        item = q.get()
        if item is _DONE:
            break
        yield item


# ---------------------------------------------------------------------------
# Per-slide processing
# ---------------------------------------------------------------------------


def _process_slide(
    uri: str,
    model: torch.nn.Module,
    device: str,
    batch_size: int,
    prefetch: int,
    maps_dir: Path,
    map_size: int,
) -> dict:
    """Run inference + feature extraction for one slide.

    Saves per-channel PNG maps to ``maps_dir``.
    Returns a dict with ``features``, ``channels``, ``metadata``, ``elapsed_s``.
    """
    reader = SlideReader(uri)
    w, h = reader.dimensions_at_read_level

    grid = None
    acc: SlideFeatureAccumulator | None = None
    buf: list = []
    n_tiles = 0

    # Per-channel running pixel sums over DAPI+ tissue.
    ch_sums: dict[str, int] = {ch: 0 for ch in ANALYSIS_CHANNELS}
    tissue_px: int = 0

    canvas: dict[str, np.ndarray] = {}
    scale: float = 1.0
    he_thumbnail: np.ndarray | None = None

    t0 = time.perf_counter()

    def _update(preds: dict, tile) -> None:
        nonlocal tissue_px
        step = grid.tile_size - grid.overlap
        y0 = tile.row * step
        x0 = tile.col * step
        y1 = min(y0 + grid.tile_size, grid.slide_height)
        x1 = min(x0 + grid.tile_size, grid.slide_width)

        dapi_mask = preds["DAPI"].astype(bool)
        tissue_px += int(dapi_mask.sum())
        for ch in ANALYSIS_CHANNELS:
            ch_sums[ch] += int((preds[ch].astype(bool) & dapi_mask).sum())

        y0_c = int(y0 * scale)
        x0_c = int(x0 * scale)
        y1_c = int(y1 * scale)
        x1_c = int(x1 * scale)
        if y1_c > y0_c and x1_c > x0_c:
            probs_hwc = preds["probabilities"]  # (H, W, C) float32 sigmoid
            for ch in ANALYSIS_CHANNELS:
                prob_slice = probs_hwc[..., CHANNEL_NAMES.index(ch)]
                patch = np.array(
                    Image.fromarray((prob_slice * 255).astype(np.uint8)).resize(
                        (x1_c - x0_c, y1_c - y0_c), Image.NEAREST
                    )
                )
                canvas[ch][y0_c:y1_c, x0_c:x1_c] = np.maximum(
                    canvas[ch][y0_c:y1_c, x0_c:x1_c], patch
                )

    try:
        with torch.inference_mode():
            for tile, grid in _prefetch_tiles(reader, min_tissue_fraction=0.05, maxsize=prefetch):
                if acc is None:
                    acc = SlideFeatureAccumulator(grid)
                    slide_max = max(grid.slide_width, grid.slide_height)
                    scale = map_size / slide_max
                    canvas_h = max(1, int(grid.slide_height * scale))
                    canvas_w = max(1, int(grid.slide_width * scale))
                    he_thumbnail = np.array(
                        reader._slide.get_thumbnail((canvas_w, canvas_h))
                        .convert("RGB")
                        .resize((canvas_w, canvas_h), Image.LANCZOS)
                    )
                    canvas.update(
                        {
                            ch: np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                            for ch in ANALYSIS_CHANNELS
                        }
                    )
                buf.append(tile)
                n_tiles += 1
                if len(buf) < batch_size:
                    continue
                for t, preds in zip(
                    buf, predict_batch([t.array for t in buf], model, device=device, return_probabilities=True)
                ):
                    acc.update({ch: preds[ch] for ch in FEATURE_CHANNELS}, t)
                    _update(preds, t)
                buf.clear()

            if buf:
                for t, preds in zip(
                    buf, predict_batch([t.array for t in buf], model, device=device, return_probabilities=True)
                ):
                    acc.update({ch: preds[ch] for ch in FEATURE_CHANNELS}, t)
                    _update(preds, t)
    finally:
        reader.close()

    if grid is None or acc is None:
        raise RuntimeError("No tissue tiles found — check slide content or min_tissue_fraction")

    maps_dir.mkdir(parents=True, exist_ok=True)
    if he_thumbnail is not None:
        Image.fromarray(he_thumbnail).save(maps_dir / "HE.png")
    for ch in ANALYSIS_CHANNELS:
        arr = canvas[ch]
        p999 = np.percentile(arr[arr > 0], 99.9) if (arr > 0).any() else 1.0
        arr = np.clip(arr, 0, p999)
        arr = (arr * 255.0 / p999).astype(np.uint8)
        r, g, b = CHANNEL_COLORS.get(ch, (255, 255, 255))
        a = arr.astype(np.uint16)
        rgb = np.stack(
            [
                (a * r // 255).astype(np.uint8),
                (a * g // 255).astype(np.uint8),
                (a * b // 255).astype(np.uint8),
            ],
            axis=-1,
        )
        Image.fromarray(rgb, mode="RGB").save(maps_dir / f"{ch}.png")

    # Per-channel mean expression over tissue.
    channels = {ch: ch_sums[ch] / tissue_px if tissue_px > 0 else 0.0 for ch in ANALYSIS_CHANNELS}

    elapsed = time.perf_counter() - t0

    return {
        "features": acc.finalize(),
        "channels": channels,
        "metadata": {
            "uri": uri,
            "slide_width_px": w,
            "slide_height_px": h,
            "n_tiles": n_tiles,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        },
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Parquet merge
# ---------------------------------------------------------------------------


def _merge_results(output_dir: Path, log: logging.Logger) -> None:
    """Merge all per-slide JSONs into gigatime_tcga_luad.parquet (atomic write)."""
    features_dir = output_dir / "slide-level-features"
    json_files = sorted(features_dir.glob("*.json"))
    if not json_files:
        log.info("No JSON files to merge.")
        return

    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not available — skipping merge. pip install pandas pyarrow")
        return

    rows = []
    for p in json_files:
        try:
            data = json.loads(p.read_text())
            row = {"slide": p.stem}
            row.update(data.get("metadata", {}))
            row.update(data.get("features", {}))
            row.update(data.get("channels", {}))
            row["elapsed_s"] = data.get("elapsed_s")
            rows.append(row)
        except Exception as exc:
            log.warning(f"Could not parse {p.name}: {exc}")

    df = pd.DataFrame(rows)
    parquet_path = output_dir / "gigatime_tcga_luad.parquet"
    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(parquet_path)
    log.info(f"Merged {len(df)} slides → {parquet_path}")


# ---------------------------------------------------------------------------
# Worker (one per GPU)
# ---------------------------------------------------------------------------


def _worker(
    rank: int,
    pending: list[str],
    output_dir: Path,
    num_workers: int,
    batch_size: int,
    prefetch: int,
    map_size: int,
) -> None:
    """Process slides assigned to this worker (pending[rank::num_workers])."""
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    log_path = output_dir / f"worker_{rank}.log"
    fmt = f"%(asctime)s  [gpu{rank}]  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
        force=True,
    )
    log = logging.getLogger(__name__)

    log.info(f"Worker {rank} starting on {device}")
    model = load_model(device=device)
    model.eval()
    log.info("Model ready.")

    features_dir = output_dir / "slide-level-features"
    maps_root = output_dir / "maps"
    my_slides = pending[rank::num_workers]
    errors: list[tuple[str, str]] = []
    slide_times: list[float] = []

    for idx, uri in enumerate(tqdm(my_slides, desc=f"GPU{rank}", position=rank, leave=True)):
        name = Path(uri).name
        stem = Path(uri).stem
        out_path = features_dir / f"{stem}.json"

        if out_path.exists():
            log.info(f"  SKIP (done)  {name}")
            continue

        log.info(f"[{idx + 1}/{len(my_slides)}] {name}")

        try:
            result = _process_slide(
                uri,
                model,
                device,
                batch_size,
                prefetch,
                maps_dir=maps_root / stem,
                map_size=map_size,
            )
        except Exception as exc:
            errors.append((name, str(exc)))
            log.error(f"  FAILED: {exc}")
            continue

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

        # Regenerate the merged parquet every MERGE_EVERY completed slides
        # (counted across all workers via the total JSON count on disk).
        n_done = sum(1 for _ in features_dir.glob("*.json"))
        if n_done % MERGE_EVERY == 0:
            log.info(f"  {n_done} slides done — refreshing parquet…")
            _merge_results(output_dir, log)

    log.info(f"Worker {rank} done: {len(slide_times)} ok, {len(errors)} errors.")
    if errors:
        for name, msg in errors:
            log.warning(f"  {name}: {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    n_gpus = torch.cuda.device_count()
    default_workers = max(1, n_gpus)

    parser = argparse.ArgumentParser(description="Extract GigaTIME features for TCGA-LUAD slides")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/sagemaker-user/custom-file-systems/efs/fs-09913c1f7db79b6fd/gigatime_tcga_luad/outputs"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_workers,
        help=f"Worker processes (default: {default_workers} — one per available GPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=7,
        help="Tiles per forward pass per worker (default: 7)",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=48,
        help="Tile read-ahead buffer per worker (default: 48)",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=1024,
        help="Max dimension of per-channel PNG maps in pixels (default: 1024)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N slides total (dry-run helper)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slide-level-features").mkdir(exist_ok=True)

    # Keep downloaded slides and the HuggingFace model cache beside the output
    # directory. os.environ is used so spawned worker processes inherit both.
    tmp_dir = output_dir.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)

    hf_cache_dir = output_dir.parent / "hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache_dir)

    log_path = output_dir / "extract_features.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [coord]  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("GigaTIME feature extraction — TCGA-LUAD")
    log.info(f"Output directory : {output_dir.resolve()}")
    log.info(f"GPUs available   : {n_gpus}  |  Workers: {args.num_workers}")
    log.info(f"Batch size       : {args.batch_size}  |  Prefetch: {args.prefetch}")
    log.info(f"Map size         : {args.map_size}px  |  Channels: {len(ANALYSIS_CHANNELS)}")

    manifest = Manifest().load()
    dataset = manifest.datasets[TCGA_LUAD]
    all_slides = [
        s for s in list_slides(bucket=dataset.bucket, prefix=dataset.prefix) if "parafine" in s
    ]
    if args.limit is not None:
        all_slides = all_slides[: args.limit]

    log.info(f"Total slides     : {len(all_slides)}")

    features_dir = output_dir / "slide-level-features"
    pending = [uri for uri in all_slides if not (features_dir / f"{Path(uri).stem}.json").exists()]
    log.info(f"Slides to process: {len(pending)}  (skipping {len(all_slides) - len(pending)} done)")

    if not pending:
        log.info("Nothing to do — all slides already processed.")
        _merge_results(output_dir, log)
        return

    wall_start = time.perf_counter()

    worker_args = (
        pending,
        output_dir,
        args.num_workers,
        args.batch_size,
        args.prefetch,
        args.map_size,
    )

    if args.num_workers == 1:
        _worker(0, *worker_args)
    else:
        mp.start_processes(
            _worker,
            args=worker_args,
            nprocs=args.num_workers,
            start_method="spawn",
        )

    total_wall = time.perf_counter() - wall_start
    log.info(f"All workers finished — wall time {total_wall / 60:.1f} min")

    _merge_results(output_dir, log)


if __name__ == "__main__":
    main()
