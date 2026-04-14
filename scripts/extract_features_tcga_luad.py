"""Extract GigaTIME spatial features for all TCGA-LUAD slides.

Designed for long-running tmux sessions. Saves one JSON per slide so that the
script can be interrupted and restarted without reprocessing completed slides.

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
    --output-dir DIR      Where per-slide JSONs, logs and the merged parquet go
                          (default: outputs/tcga_luad_features)
    --num-workers N       Worker processes to spawn (default: all available GPUs,
                          or 1 on CPU-only machines)
    --batch-size N        Tiles per forward pass per worker (default: 16)
    --prefetch N          Tile read-ahead buffer per worker (default: 48)
    --limit N             Process at most N slides total (dry-run helper)

Output
------
    OUTPUT_DIR/
        <slide_stem>.json         — per-slide features + metadata
        worker_<rank>.log         — per-worker log with timestamps
        extract_features.log      — coordinator log
        features.parquet          — merged table (written after all workers finish)
        features.csv              — same, CSV copy
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.multiprocessing as mp
from abstra.manifest import Manifest
from tqdm import tqdm

from gigatime.data import SlideReader, iter_tiles, list_slides
from gigatime.data.paths import TCGA_LUAD
from gigatime.features import SlideFeatureAccumulator
from gigatime.inference import load_model, predict_batch

FEATURE_CHANNELS = [
    "CK", "DAPI", "CD8", "CD4", "CD68", "CD34",
    "PD-1", "PD-L1", "Ki67", "Caspase3-D",
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
) -> dict:
    """Run inference + feature extraction for one slide.

    Returns a dict with ``features``, ``metadata``, and ``elapsed_s``.
    Raises on any error — the caller handles it.
    """
    reader = SlideReader(uri)
    w, h = reader.dimensions_at_read_level

    grid = None
    acc: SlideFeatureAccumulator | None = None
    buf: list = []
    n_tiles = 0
    t0 = time.perf_counter()

    with torch.inference_mode():
        for tile, grid in _prefetch_tiles(reader, min_tissue_fraction=0.05, maxsize=prefetch):
            if acc is None:
                acc = SlideFeatureAccumulator(grid)
            buf.append(tile)
            n_tiles += 1
            if len(buf) < batch_size:
                continue
            for t, preds in zip(buf, predict_batch([t.array for t in buf], model, device=device)):
                acc.update({ch: preds[ch] for ch in FEATURE_CHANNELS}, t)
            buf.clear()

        if buf:
            for t, preds in zip(buf, predict_batch([t.array for t in buf], model, device=device)):
                acc.update({ch: preds[ch] for ch in FEATURE_CHANNELS}, t)

    reader.close()

    if grid is None or acc is None:
        raise RuntimeError("No tissue tiles found — check slide content or min_tissue_fraction")

    features = acc.finalize()
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
# Worker (one per GPU)
# ---------------------------------------------------------------------------

def _worker(
    rank: int,
    pending: list[str],
    output_dir: Path,
    num_workers: int,
    batch_size: int,
    prefetch: int,
) -> None:
    """Process slides assigned to this worker (pending[rank::num_workers])."""
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # Per-worker log file — avoids multiprocess write conflicts on a shared file.
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

    my_slides = pending[rank::num_workers]
    errors: list[tuple[str, str]] = []
    slide_times: list[float] = []

    for idx, uri in enumerate(tqdm(my_slides, desc=f"GPU{rank}", position=rank, leave=True)):
        name = Path(uri).name
        stem = Path(uri).stem
        out_path = output_dir / f"{stem}.json"

        # Double-check in case another worker raced us (shouldn't happen with
        # round-robin, but safe to guard against restarts with overlap).
        if out_path.exists():
            log.info(f"  SKIP (done)  {name}")
            continue

        log.info(f"[{idx + 1}/{len(my_slides)}] {name}")

        try:
            result = _process_slide(uri, model, device, batch_size, prefetch)
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
        default=Path("outputs/tcga_luad_features"),
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
        default=16,
        help="Tiles per forward pass per worker (default: 16)",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=48,
        help="Tile read-ahead buffer per worker (default: 48)",
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

    # Coordinator log (non-worker output).
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

    # Discover slides.
    manifest = Manifest().load()
    dataset = manifest.datasets[TCGA_LUAD]
    all_slides = [
        s for s in list_slides(bucket=dataset.bucket, prefix=dataset.prefix)
        if "parafine" in s
    ]
    if args.limit is not None:
        all_slides = all_slides[: args.limit]

    log.info(f"Total slides     : {len(all_slides)}")

    # Skip already-processed slides.
    pending = [
        uri for uri in all_slides
        if not (output_dir / f"{Path(uri).stem}.json").exists()
    ]
    log.info(f"Slides to process: {len(pending)}  (skipping {len(all_slides) - len(pending)} done)")

    if not pending:
        log.info("Nothing to do — all slides already processed.")
        _merge_results(output_dir, log)
        return

    wall_start = time.perf_counter()

    if args.num_workers == 1:
        # Single-worker path — no multiprocessing overhead.
        _worker(0, pending, output_dir, 1, args.batch_size, args.prefetch)
    else:
        # Spawn one process per worker. Each gets pending[rank::num_workers].
        # 'spawn' is required for CUDA — never use 'fork' with CUDA.
        mp.start_processes(
            _worker,
            args=(pending, output_dir, args.num_workers, args.batch_size, args.prefetch),
            nprocs=args.num_workers,
            start_method="spawn",
        )

    total_wall = time.perf_counter() - wall_start
    log.info(f"All workers finished — wall time {total_wall / 60:.1f} min")

    _merge_results(output_dir, log)


def _merge_results(output_dir: Path, log: logging.Logger) -> None:
    """Merge all per-slide JSONs into a single parquet + CSV."""
    json_files = sorted(output_dir.glob("*.json"))
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
            row["elapsed_s"] = data.get("elapsed_s")
            rows.append(row)
        except Exception as exc:
            log.warning(f"Could not parse {p.name}: {exc}")

    df = pd.DataFrame(rows)
    parquet_path = output_dir / "features.parquet"
    csv_path = output_dir / "features.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    log.info(f"Merged {len(df)} slides → {parquet_path}  and  {csv_path}")


if __name__ == "__main__":
    main()
