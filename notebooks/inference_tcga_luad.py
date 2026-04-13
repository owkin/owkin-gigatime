"""Interactive python script to run inference of gigatime on TCGA HE slides."""

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from abstra.manifest import Manifest
from PIL import Image

from gigatime.data import SlideReader, download_slide, iter_tiles, list_slides, stitch
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
# Download one slide
SLIDE_INDEX = 0
slide_uri = slides[SLIDE_INDEX]
print(f"Downloading: {Path(slide_uri).name}")

local_dir = Path(tempfile.gettempdir()) / "gigatime_slides"
slide_path = download_slide(slide_uri, dest_dir=local_dir)
print(f"Saved to: {slide_path}")

# %%
# Load the model
model = load_model(device=DEVICE)
model.eval()
print(f"Loaded — {len(CHANNEL_NAMES)} output channels: {', '.join(CHANNEL_NAMES)}")

# %%
# Run the inference on one image
reader = SlideReader(slide_path)
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
raw_slide = openslide.OpenSlide(str(slide_path))
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

fig.suptitle(f"GigaTIME predictions — {Path(slide_path).name}", fontsize=13)
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
print(f"\nPositive-pixel fraction per channel ({Path(slide_path).name}):")
for ch, frac in sorted_fracs:
    bar = "█" * int(frac * 40)
    print(f"  {ch:12s} {frac:.3f}  {bar}")
