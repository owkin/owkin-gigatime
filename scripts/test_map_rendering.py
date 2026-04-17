"""Validate canvas rendering on 300 tissue tiles using the real model.

Runs inference on the first N_TILES tissue tiles, projects predictions onto a
slide-level canvas, and saves a colorized DAPI PNG for visual inspection.

Run with:
    python scripts/test_map_rendering.py
Output: test_dapi_canvas.png
"""

import sys
import torch
from pathlib import Path
from PIL import Image
import numpy as np

from abstra.manifest import Manifest
from gigatime.data import SlideReader, iter_tiles, list_slides
from gigatime.data.paths import TCGA_LUAD
from gigatime.inference import load_model, predict_batch

DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
N_TILES    = 300
BATCH_SIZE = 7
MAP_SIZE   = 512
DAPI_COLOR = (100, 149, 237)

# ---------------------------------------------------------------------------

manifest = Manifest().load()
dataset  = manifest.datasets[TCGA_LUAD]
slides   = [s for s in list_slides(bucket=dataset.bucket, prefix=dataset.prefix) if "parafine" in s]
if not slides:
    sys.exit("No slides found.")

uri = slides[0]
print(f"Slide : {Path(uri).name}")
print(f"Device: {DEVICE}")

print(f"Collecting first {N_TILES} tissue tiles...")
reader = SlideReader(uri)
print(f"DZ level : {reader.dz_level}  |  MPP metadata: {reader.metadata.mpp:.4f} µm/px")

tiles, grid = [], None
for tile, g in iter_tiles(reader, min_tissue_fraction=0.05):
    tiles.append(tile)
    grid = g
    if len(tiles) >= N_TILES:
        break
reader.close()

print(f"  {len(tiles)} tiles  |  slide {grid.slide_width}×{grid.slide_height} px (target-MPP space)")

# ---------------------------------------------------------------------------
# Build canvas — identical logic to _update() in extract_features_tcga_luad.py
# ---------------------------------------------------------------------------

slide_max = max(grid.slide_width, grid.slide_height)
scale     = MAP_SIZE / slide_max
canvas_h  = max(1, int(grid.slide_height * scale))
canvas_w  = max(1, int(grid.slide_width  * scale))
canvas    = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

print("Loading model...")
model = load_model(device=DEVICE)
model.eval()

step = grid.tile_size - grid.overlap

print("Running inference and projecting onto canvas...")
with torch.inference_mode():
    for i in range(0, len(tiles), BATCH_SIZE):
        batch = tiles[i : i + BATCH_SIZE]
        for t, preds in zip(batch, predict_batch([t.array for t in batch], model, device=DEVICE)):
            y0 = t.row * step
            x0 = t.col * step
            y1 = min(y0 + grid.tile_size, grid.slide_height)
            x1 = min(x0 + grid.tile_size, grid.slide_width)

            y0_c = int(y0 * scale)
            x0_c = int(x0 * scale)
            y1_c = int(y1 * scale)
            x1_c = int(x1 * scale)
            if y1_c > y0_c and x1_c > x0_c:
                patch = np.array(
                    Image.fromarray((preds["DAPI"] * 255).astype(np.uint8)).resize(
                        (x1_c - x0_c, y1_c - y0_c), Image.NEAREST
                    )
                )
                canvas[y0_c:y1_c, x0_c:x1_c] = np.maximum(
                    canvas[y0_c:y1_c, x0_c:x1_c], patch
                )

# ---------------------------------------------------------------------------
# Percentile-clip + colorize (same as save loop in extraction script)
# ---------------------------------------------------------------------------

arr  = canvas
p995 = np.percentile(arr[arr > 0], 99.5) if (arr > 0).any() else 1.0
arr  = np.clip(arr, 0, p995)
arr  = (arr * 255.0 / p995).astype(np.uint8)

r, g, b = DAPI_COLOR
a = arr.astype(np.uint16)
rgb = np.stack([
    (a * r // 255).astype(np.uint8),
    (a * g // 255).astype(np.uint8),
    (a * b // 255).astype(np.uint8),
], axis=-1)

out = "test_dapi_canvas.png"
Image.fromarray(rgb, mode="RGB").save(out)
print(f"\nSaved {out}  ({canvas_w}×{canvas_h} px canvas, {len(tiles)} tiles projected)")
