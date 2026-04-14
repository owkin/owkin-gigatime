"""Interactive python script to run inference of gigatime on TCGA HE slides."""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from abstra.manifest import Manifest
from PIL import Image

from gigatime.data import SlideReader, iter_tiles, list_slides, stitch
from gigatime.data.paths import TCGA_LUAD
from gigatime.inference import load_model, predict
from gigatime.inference.constants import CHANNEL_NAMES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %%
# Find the available images
manifest = Manifest().load()
dataset = manifest.datasets[TCGA_LUAD]
slides = list_slides(bucket=dataset.bucket, prefix=dataset.prefix)

print(f"Found {len(slides)} slides in TCGA-LUAD")
for i, uri in enumerate(slides[:10]):
    print(f"  [{i}] {Path(uri).name}")

# %%
# Load the model
model = load_model(device=DEVICE)
model.eval()
print(f"Loaded — {len(CHANNEL_NAMES)} output channels: {', '.join(CHANNEL_NAMES)}")

# %%
# Run the inference on one image
SLIDE_INDEX = 0
slide_uri = slides[SLIDE_INDEX]
print(f"Opening: {Path(slide_uri).name}")
reader = SlideReader(slide_uri)  # S3 URI — downloaded to a temp dir, cleaned up on close()
w, h = reader.dimensions_at_read_level
print(f"Slide at read level: {w} × {h} px  (mpp ≈ 0.5 µm/px)")

tile_results: list[tuple[dict[str, np.ndarray], object]] = []
grid = None

with torch.inference_mode():
    for tile, grid in iter_tiles(reader, min_tissue_fraction=0.05):
        preds = predict(tile.array, model, device=DEVICE)
        tile_results.append((preds, tile))
        if len(tile_results) % 10 == 0:
            print(f"  {len(tile_results)} tiles done…")

reader.close()
assert grid is not None, "No tissue tiles found — check min_tissue_fraction or slide content"
print(f"Done: {len(tile_results)} tissue tiles across a {grid.n_rows}×{grid.n_cols} grid")

# %%
# Display tile level inferences
# Pick a clinically relevant subset of channels for a quick overview
CHANNELS_TO_DISPLAY = ["CD8", "CD3", "PD-1", "PD-L1", "CK", "CD4"]

stitched = {ch: stitch(tile_results, grid, channel=ch, mode="max") for ch in CHANNELS_TO_DISPLAY}

# Build H&E thumbnail from the native pyramid (fast, no re-reading)
raw_slide = openslide.OpenSlide(str(reader.path))
thumb_w, thumb_h = 512, 512
thumbnail = np.array(raw_slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
raw_slide.close()

n = len(CHANNELS_TO_DISPLAY)
fig, axes = plt.subplots(2, (n // 2) + 1, figsize=(4 * ((n // 2) + 1), 8))
axes = axes.flatten()

# H&E panel
axes[0].imshow(thumbnail)
axes[0].set_title("H&E", fontsize=12, fontweight="bold")
axes[0].axis("off")

# Channel panels — downsample each stitched map to thumbnail size for display
for ax, channel in zip(axes[1:], CHANNELS_TO_DISPLAY):
    pred_map = stitched[channel]  # float32, shape (H, W), values in {0, 1}
    disp = np.array(
        Image.fromarray((pred_map * 255).astype(np.uint8)).resize((thumb_w, thumb_h), Image.NEAREST)
    )
    ax.imshow(disp, cmap="hot", vmin=0, vmax=255)
    ax.set_title(channel, fontsize=12)
    ax.axis("off")

# Hide any unused axes
for ax in axes[1 + n :]:
    ax.set_visible(False)

fig.suptitle(f"GigaTIME predictions — {reader.path.name}", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Positive-cell fraction per channel (tissue tiles only)
analysis_channels = [ch for ch in CHANNEL_NAMES if ch not in {"TRITC", "Cy5"}]
fractions = {}
for channel in analysis_channels:
    channel_map = stitch(tile_results, grid, channel=channel, mode="max")
    tissue_mask = channel_map >= 0  # all pixels covered by at least one tile
    fractions[channel] = float(channel_map[tissue_mask].mean())

sorted_fracs = sorted(fractions.items(), key=lambda x: x[1], reverse=True)
print(f"\nPositive-pixel fraction per channel ({reader.path.name}):")
for ch, frac in sorted_fracs:
    bar = "█" * int(frac * 40)
    print(f"  {ch:12s} {frac:.3f}  {bar}")

# %%
# Run inference on 10 slides and estimate throughput
import time

N_SLIDES = 10
slide_uris_batch = slides[:N_SLIDES]

print(f"Running inference on {N_SLIDES} slides…\n")

slide_times: list[float] = []
slide_tile_counts: list[int] = []
errors: list[tuple[str, str]] = []

batch_start = time.perf_counter()

for idx, uri in enumerate(slide_uris_batch):
    name = Path(uri).name
    print(f"[{idx + 1}/{N_SLIDES}] {name}")

    t0 = time.perf_counter()
    n_tiles = 0

    try:
        reader = SlideReader(uri)  # downloads to a temp dir, cleaned up on close()
        w, h = reader.dimensions_at_read_level
        print(f"         {w} × {h} px")

        with torch.inference_mode():
            for tile, grid in iter_tiles(reader, min_tissue_fraction=0.05):
                predict(tile.array, model, device=DEVICE)
                n_tiles += 1

        reader.close()
    except Exception as exc:
        errors.append((name, str(exc)))
        print(f"         ERROR: {exc}")
        continue

    elapsed = time.perf_counter() - t0
    slide_times.append(elapsed)
    slide_tile_counts.append(n_tiles)
    tiles_per_sec = n_tiles / elapsed if elapsed > 0 else float("nan")
    print(f"         {n_tiles} tiles  |  {elapsed:.1f}s  |  {tiles_per_sec:.1f} tiles/s\n")

total_elapsed = time.perf_counter() - batch_start

# ---- Summary ----
print("=" * 60)
print(f"Completed {len(slide_times)}/{N_SLIDES} slides successfully  ({len(errors)} errors)")
print(f"Total wall time : {total_elapsed / 60:.1f} min")
if slide_times:
    avg_time = sum(slide_times) / len(slide_times)
    avg_tiles = sum(slide_tile_counts) / len(slide_tile_counts)
    tiles_per_sec_overall = sum(slide_tile_counts) / sum(slide_times)
    print(f"Avg time/slide  : {avg_time:.1f}s  ({avg_time / 60:.1f} min)")
    print(f"Avg tiles/slide : {avg_tiles:.0f}")
    print(f"Throughput      : {tiles_per_sec_overall:.1f} tiles/s")
    print(f"\nExtrapolation to full dataset ({len(slides)} slides):")
    est_total_min = len(slides) * avg_time / 60
    print(f"  ~{est_total_min:.0f} min  ({est_total_min / 60:.1f} h)  on {DEVICE}")

if errors:
    print(f"\nFailed slides:")
    for name, msg in errors:
        print(f"  {name}: {msg}")
