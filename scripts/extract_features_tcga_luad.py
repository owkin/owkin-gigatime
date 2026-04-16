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
                          (default: outputs/tcga_luad_features)
    --num-workers N       Worker processes to spawn (default: all available GPUs,
                          or 1 on CPU-only machines)
    --batch-size N        Tiles per forward pass per worker (default: 16)
    --prefetch N          Tile read-ahead buffer per worker (default: 48)
    --map-size N          Max dimension of per-channel PNG maps in pixels (default: 1024)
    --limit N             Process at most N slides total (dry-run helper)

Output
------
    OUTPUT_DIR/
        <slide_stem>.json             — per-slide features + per-channel means + metadata
        maps/<slide_stem>/<ch>.png    — per-channel spatial map (mIF colors, black background)
        worker_<rank>.log             — per-worker log with timestamps
        extract_features.log          — coordinator log
        features.parquet              — merged table (written after all workers finish)
        features.csv                  — same, CSV copy
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

import numpy as np
import torch
import torch.multiprocessing as mp
from abstra.manifest import Manifest
from PIL import Image, ImageFilter
from tqdm import tqdm

from gigatime.data import SlideReader, iter_tiles, list_slides
from gigatime.data.paths import TCGA_LUAD
from gigatime.features import SlideFeatureAccumulator
from gigatime.inference import load_model, predict_batch
from gigatime.inference.constants import BACKGROUND_CHANNELS, CHANNEL_NAMES

ANALYSIS_CHANNELS: list[str] = [ch for ch in CHANNEL_NAMES if ch not in BACKGROUND_CHANNELS]

# Per-channel mIF display colors (RGB). Black background, signal in channel color.
# Conventions follow standard multiplex immunofluorescence panel assignments.
CHANNEL_COLORS: dict[str, tuple[int, int, int]] = {
    "DAPI":        (100, 149, 237),  # cornflower blue  — nuclei
    "CK":          (255, 255,   0),  # yellow           — tumour cells
    "CD3":         ( 50, 205,  50),  # lime green       — pan T-cells
    "CD8":         (  0, 255, 255),  # cyan             — cytotoxic T-cells
    "CD4":         (144, 238, 144),  # light green      — helper T-cells
    "CD68":        (255, 165,   0),  # orange           — macrophages
    "PD-L1":       (255,   0,   0),  # red              — immune checkpoint (tumour)
    "PD-1":        (255,   0, 255),  # magenta          — exhaustion / activation
    "CD20":        (255, 215,   0),  # gold             — B-cells
    "Ki67":        (255, 100,   0),  # orange-red       — proliferation
    "CD34":        (220,  20,  60),  # crimson          — vasculature
    "CD138":       (148,   0, 211),  # violet           — plasma cells
    "CD11c":       (255, 200,   0),  # amber            — dendritic cells
    "CD14":        (210, 105,  30),  # chocolate        — monocytes
    "CD16":        (189, 183, 107),  # dark khaki       — NK cells / neutrophils
    "T-bet":       (  0, 191, 255),  # deep sky blue    — Th1 / CD8 transcription factor
    "Tryptase":    (186,  85, 211),  # medium orchid    — mast cells
    "Actin-D":     (169, 169, 169),  # dark grey        — smooth muscle / myofibroblasts
    "Caspase3-D":  (127, 255,   0),  # chartreuse       — apoptosis
    "PHH3-B":      (255, 255, 102),  # pale yellow      — mitosis
    "Transgelin":  (119, 136, 153),  # light slate grey — smooth muscle actin
}

FEATURE_CHANNELS = [
    "CK", "DAPI", "CD8", "CD4", "CD68", "CD34",
    "PD-1", "PD-L1", "Ki67", "Caspase3-D",
]

# Intermediate canvas resolution for artifact-free downsampling.
# Tiles are placed at this resolution; a single BOX downsample to the final
# map_size then averages across tile boundaries, eliminating the moiré that
# appears when each tile is resized independently to the small final canvas.
CANVAS_INTERMEDIATE_SIZE = 4096


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

    # Intermediate canvas for PNG maps — built at CANVAS_INTERMEDIATE_SIZE,
    # then BOX-downsampled once to map_size.  Building at a higher intermediate
    # resolution means the final global BOX downsample averages across many tile
    # widths, eliminating the moiré that arises when each tile is resized
    # independently to the small final canvas.
    canvas_int: dict[str, np.ndarray] = {}
    scale_int: float = 1.0
    canvas_h_int: int = 1
    canvas_w_int: int = 1

    t0 = time.perf_counter()

    def _update(preds: dict, tile) -> None:
        nonlocal tissue_px
        step = grid.tile_size - grid.overlap
        y0 = tile.row * step
        x0 = tile.col * step
        y1 = min(y0 + grid.tile_size, grid.slide_height)
        x1 = min(x0 + grid.tile_size, grid.slide_width)
        th = y1 - y0
        tw = x1 - x0

        # Running sums: positive pixels normalised by DAPI+ tissue.
        dapi_mask = preds["DAPI"][:th, :tw].astype(bool)
        tissue_px += int(dapi_mask.sum())
        for ch in ANALYSIS_CHANNELS:
            ch_sums[ch] += int((preds[ch][:th, :tw].astype(bool) & dapi_mask).sum())

        # Place tile onto intermediate canvas using round() for both endpoints
        # so adjacent tiles abut perfectly with no 1-pixel gaps.
        y0_c = round(y0 * scale_int)
        x0_c = round(x0 * scale_int)
        y1_c = round(y1 * scale_int)
        x1_c = round(x1 * scale_int)
        if y1_c > y0_c and x1_c > x0_c:
            target_h, target_w = y1_c - y0_c, x1_c - x0_c
            for ch in ANALYSIS_CHANNELS:
                patch = np.array(
                    Image.fromarray((preds[ch][:th, :tw] * 255).astype(np.uint8)).resize(
                        (target_w, target_h), Image.BOX
                    )
                ) / 255.0
                canvas_int[ch][y0_c:y1_c, x0_c:x1_c] = patch

    with torch.inference_mode():
        for tile, grid in _prefetch_tiles(reader, min_tissue_fraction=0.05, maxsize=prefetch):
            if acc is None:
                acc = SlideFeatureAccumulator(grid)
                # Intermediate scale — at least as large as map_size so the
                # final BOX downsample always reduces (never upscales).
                slide_max = max(grid.slide_width, grid.slide_height)
                scale_int = max(CANVAS_INTERMEDIATE_SIZE, map_size) / slide_max
                canvas_h_int = max(1, round(grid.slide_height * scale_int))
                canvas_w_int = max(1, round(grid.slide_width * scale_int))
                canvas_int.update({
                    ch: np.zeros((canvas_h_int, canvas_w_int), dtype=np.float32)
                    for ch in ANALYSIS_CHANNELS
                })
            buf.append(tile)
            n_tiles += 1
            if len(buf) < batch_size:
                continue
            for t, preds in zip(buf, predict_batch([t.array for t in buf], model, device=device)):
                acc.update({ch: preds[ch] for ch in FEATURE_CHANNELS}, t)
                _update(preds, t)
            buf.clear()

        if buf:
            for t, preds in zip(buf, predict_batch([t.array for t in buf], model, device=device)):
                acc.update({ch: preds[ch] for ch in FEATURE_CHANNELS}, t)
                _update(preds, t)

    reader.close()

    if grid is None or acc is None:
        raise RuntimeError("No tissue tiles found — check slide content or min_tissue_fraction")

    # Downsample intermediate canvas → final map_size using a global BOX filter,
    # which averages across tile boundaries and eliminates the moiré.
    # A small adaptive Gaussian blur is applied to the final image to smooth any
    # residual seam artifacts from the model's internal inference window (256px).
    slide_max = max(grid.slide_width, grid.slide_height)
    scale_final = map_size / slide_max
    canvas_h_final = max(1, round(grid.slide_height * scale_final))
    canvas_w_final = max(1, round(grid.slide_width * scale_final))
    # Blur radius proportional to tile footprint in the final canvas so seams
    # between inference windows (256px) are blended regardless of slide size.
    tile_px_in_canvas = max(1, round(512 * scale_final))
    blur_radius = max(2, tile_px_in_canvas // 2)

    maps_dir.mkdir(parents=True, exist_ok=True)
    for ch in ANALYSIS_CHANNELS:
        # Global BOX downsample: intermediate → final
        arr = np.array(
            Image.fromarray((canvas_int[ch] * 255).astype(np.uint8)).resize(
                (canvas_w_final, canvas_h_final), Image.BOX
            )
        ) / 255.0
        # Adaptive Gaussian blur to smooth residual tile-boundary seams
        arr_b = np.array(
            Image.fromarray((arr * 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            )
        ) / 255.0
        r, g, b = CHANNEL_COLORS.get(ch, (255, 255, 255))
        rgb = np.stack([
            (arr_b * r).astype(np.uint8),
            (arr_b * g).astype(np.uint8),
            (arr_b * b).astype(np.uint8),
        ], axis=-1)
        Image.fromarray(rgb, mode="RGB").save(maps_dir / f"{ch}.png")

    # Per-channel mean expression over tissue.
    channels = {
        ch: ch_sums[ch] / tissue_px if tissue_px > 0 else 0.0
        for ch in ANALYSIS_CHANNELS
    }

    elapsed = time.perf_counter() - t0

    return {
        "features": acc.finalize(),
        "channels": channels,
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

    maps_root = output_dir / "maps"
    my_slides = pending[rank::num_workers]
    errors: list[tuple[str, str]] = []
    slide_times: list[float] = []

    for idx, uri in enumerate(tqdm(my_slides, desc=f"GPU{rank}", position=rank, leave=True)):
        name = Path(uri).name
        stem = Path(uri).stem
        out_path = output_dir / f"{stem}.json"

        if out_path.exists():
            log.info(f"  SKIP (done)  {name}")
            continue

        log.info(f"[{idx + 1}/{len(my_slides)}] {name}")

        try:
            result = _process_slide(
                uri, model, device, batch_size, prefetch,
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
        s for s in list_slides(bucket=dataset.bucket, prefix=dataset.prefix)
        if "parafine" in s
    ]
    if args.limit is not None:
        all_slides = all_slides[: args.limit]

    log.info(f"Total slides     : {len(all_slides)}")

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

    worker_args = (pending, output_dir, args.num_workers, args.batch_size, args.prefetch, args.map_size)

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
            row.update(data.get("channels", {}))
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
