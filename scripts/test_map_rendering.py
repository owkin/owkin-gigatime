"""Validate model output at full resolution on a contiguous tissue patch.

Finds a tissue tile well inside the slide (skips the first N_SKIP tissue tiles
to avoid edge positions), reads PATCH_TILES × PATCH_TILES tiles around it as a
single region at the correct resolution, runs the model, and saves:

  test_fullres_patch.png  —  left: H&E  |  right: DAPI probability (blue)

Run with:
    python scripts/test_map_rendering.py
"""

import sys
import torch
from pathlib import Path
from PIL import Image
import numpy as np

from abstra.manifest import Manifest
from gigatime.data import SlideReader, iter_tiles, list_slides
from gigatime.data.paths import TCGA_LUAD
from gigatime.inference import load_model, predict

DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
PATCH_TILES = 10      # patch side in tiles (10 → 5120 px at 20x ≈ 2.5 mm)
TILE_SIZE   = 512
N_SKIP      = 50      # skip this many tissue tiles so we land inside the tissue
DAPI_COLOR  = (100, 149, 237)

# ---------------------------------------------------------------------------

manifest = Manifest().load()
dataset  = manifest.datasets[TCGA_LUAD]
slides   = [s for s in list_slides(bucket=dataset.bucket, prefix=dataset.prefix) if "parafine" in s]
if not slides:
    sys.exit("No slides found.")

uri = slides[0]
print(f"Slide : {Path(uri).name}")
print(f"Device: {DEVICE}")

reader   = SlideReader(uri)
slide_w, slide_h = reader.dimensions_at_read_level
level_ds = reader._slide.level_downsamples[reader.read_level]
rd       = reader.read_downsample          # residual rescale factor
print(f"Slide at read level : {slide_w} × {slide_h} px")
print(f"read_downsample     : {rd:.3f}")

# Find a tissue tile well inside the tissue by skipping the first N_SKIP
print(f"Searching for tissue tile (skipping first {N_SKIP})...")
anchor, count = None, 0
for tile, _ in iter_tiles(reader, min_tissue_fraction=0.1):
    if count >= N_SKIP:
        anchor = tile
        break
    count += 1

if anchor is None:
    # Fewer than N_SKIP tissue tiles — just use the first one
    for tile, _ in iter_tiles(reader, min_tissue_fraction=0.1):
        anchor = tile
        break

if anchor is None:
    reader.close()
    sys.exit("No tissue tiles found.")

print(f"Anchor tile: row={anchor.row}, col={anchor.col}")

# ---------------------------------------------------------------------------
# Build the read-level bounding box for PATCH_TILES × PATCH_TILES tiles.
#
# PATCH_TILES * TILE_SIZE is the desired size at *target* 20x resolution.
# At the read level we need rd times as many pixels before rescaling.
# We center the patch around the anchor tile so we don't fall off an edge.
# ---------------------------------------------------------------------------

target_px  = PATCH_TILES * TILE_SIZE          # desired pixels at 20x
read_px    = round(target_px * rd)            # pixels needed at the read level

# Anchor tile top-left in read-level pixels
ax = anchor.col * TILE_SIZE
ay = anchor.row * TILE_SIZE

# Center the patch around the anchor tile (anchor is at the top-left of the tile)
x_read = ax - read_px // 2 + TILE_SIZE // 2
y_read = ay - read_px // 2 + TILE_SIZE // 2

# Clamp so we stay within slide bounds
x_read = max(0, min(x_read, slide_w - read_px))
y_read = max(0, min(y_read, slide_h - read_px))

read_w = min(read_px, slide_w - x_read)
read_h = min(read_px, slide_h - y_read)

x0 = int(x_read * level_ds)
y0 = int(y_read * level_ds)

print(f"Reading {read_w}×{read_h} px at read level from level-0 ({x0}, {y0})...")
region = reader.read_region(x0, y0, read_w, read_h)
reader.close()

# Rescale the whole patch to target 20x resolution in one shot
if abs(rd - 1.0) > 1e-3:
    target_w = round(read_w / rd)
    target_h = round(read_h / rd)
    print(f"Rescaling {read_w}×{read_h} → {target_w}×{target_h} px")
    region = np.array(Image.fromarray(region).resize((target_w, target_h), Image.Resampling.LANCZOS))

# Pad to exact target_px × target_px (dimensions must be divisible by 16)
he = np.full((target_px, target_px, 3), 255, dtype=np.uint8)
rh, rw = min(region.shape[0], target_px), min(region.shape[1], target_px)
he[:rh, :rw] = region[:rh, :rw]

print(f"H&E patch: {he.shape[1]}×{he.shape[0]} px — loading model...")

# ---------------------------------------------------------------------------
# Run model on the whole patch at once (sliding window handles sub-tiles)
# ---------------------------------------------------------------------------

model = load_model(device=DEVICE)
model.eval()

with torch.inference_mode():
    preds = predict(he, model, device=DEVICE, return_probabilities=True)

dapi = preds["probabilities"][..., 0]   # float32 (H, W)
print(f"DAPI — min: {dapi.min():.3f}  mean: {dapi.mean():.3f}  max: {dapi.max():.3f}")

# ---------------------------------------------------------------------------
# Save side-by-side: H&E | DAPI
# ---------------------------------------------------------------------------

dapi_u8  = (dapi * 255).astype(np.uint8)
r, g, b  = DAPI_COLOR
dapi_rgb = np.stack([
    (dapi_u8.astype(np.uint16) * r // 255).astype(np.uint8),
    (dapi_u8.astype(np.uint16) * g // 255).astype(np.uint8),
    (dapi_u8.astype(np.uint16) * b // 255).astype(np.uint8),
], axis=-1)

combined = np.hstack([he, dapi_rgb])
out_path = "test_fullres_patch.png"
Image.fromarray(combined).save(out_path)
print(f"\nSaved {out_path}  ({combined.shape[1]}×{combined.shape[0]} px)")
print("Left = H&E  |  Right = DAPI probability (raw sigmoid, cornflower blue)")
