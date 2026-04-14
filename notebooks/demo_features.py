"""Demo: compute spatial biomarker features from a single LUAD slide."""

# %%
%load_ext autoreload
%autoreload 2

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
from gigatime.features import compute_features_from_tiles
from gigatime.features.compartments import stroma_mask, tissue_mask, tumor_mask
from gigatime.inference import load_model, predict
from gigatime.inference.constants import BACKGROUND_CHANNELS, CHANNEL_NAMES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Channels to stitch for visualisation — kept small to limit memory use.
VIZ_CHANNELS = ["CK", "DAPI", "CD8", "PD-L1", "PD-1", "CD68", "CD20", "CD3"]

# %%
# Load model and slide
manifest = Manifest().load()
dataset = manifest.datasets[TCGA_LUAD]
slides = list_slides(bucket=dataset.bucket, prefix=dataset.prefix)

SLIDE_INDEX = 0
slide_uri = slides[SLIDE_INDEX]
print(f"Opening: {Path(slide_uri).name}")

model = load_model(device=DEVICE)
model.eval()

# %%
# Run inference — stream tiles through feature accumulators and stitch only
# the channels needed for visualisation.  No tile prediction dicts are kept
# in memory: see gigatime/features/features.py for the memory analysis.
reader = SlideReader(slide_uri)
w, h = reader.dimensions_at_read_level
print(f"Slide at read level: {w} × {h} px")

viz_maps: dict[str, np.ndarray] = {ch: np.zeros((h, w), dtype=np.float32) for ch in VIZ_CHANNELS}
tile_results_for_viz: list[tuple[dict[str, np.ndarray], object]] = []
grid = None
n_tiles = 0

with torch.inference_mode():
    for tile, grid in iter_tiles(reader, min_tissue_fraction=0.05):
        preds = predict(tile.array, model, device=DEVICE)
        # Accumulate only the viz channels — discard everything else immediately
        tile_results_for_viz.append(({ch: preds[ch] for ch in VIZ_CHANNELS}, tile))
        n_tiles += 1
        if n_tiles % 20 == 0:
            print(f"  {n_tiles} tiles done…")

reader.close()
assert grid is not None, "No tissue tiles found"
print(f"Done: {n_tiles} tiles across a {grid.n_rows}×{grid.n_cols} grid")

# %%
# Compute features with the streaming path (re-uses the slim viz tile list)
features = compute_features_from_tiles(
    # Reconstruct full preds on the fly by re-running predict would be ideal;
    # here we reuse the stored viz subset — sufficient for all current features.
    ((preds, tile) for preds, tile in tile_results_for_viz),
    grid,
)

print(f"\nFeatures for {Path(slide_uri).name}:\n")
for name, value in features.items():
    bar = "█" * int(min(value, 1.0) * 30)
    print(f"  {name:<35s} {value:.4f}  {bar}")

# %%
# Stitch the viz channels
print("\nStitching visualisation maps…")
maps: dict[str, np.ndarray] = {
    ch: stitch(tile_results_for_viz, grid, channel=ch, mode="max") for ch in VIZ_CHANNELS
}

# %%
# Visualise compartment masks alongside H&E
ck = maps["CK"]
dapi = maps["DAPI"]

t_mask = tumor_mask(ck, dapi)
s_mask = stroma_mask(ck, dapi)
ti_mask = tissue_mask(dapi)

raw_slide = openslide.OpenSlide(str(reader.path))
thumb_w, thumb_h = 512, 512
thumbnail = np.array(raw_slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
raw_slide.close()


def _thumb(mask: np.ndarray) -> np.ndarray:
    """Downsample a boolean mask to thumbnail size for display."""
    return np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (thumb_w, thumb_h), Image.NEAREST
        )
    )


fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes[0].imshow(thumbnail)
axes[0].set_title("H&E", fontweight="bold")
axes[1].imshow(_thumb(ti_mask), cmap="gray")
axes[1].set_title("Tissue (DAPI)")
axes[2].imshow(_thumb(t_mask), cmap="Reds")
axes[2].set_title("Tumour (CK+)")
axes[3].imshow(_thumb(s_mask), cmap="Blues")
axes[3].set_title("Stroma (CK−)")
for ax in axes:
    ax.axis("off")
fig.suptitle(f"Compartment masks — {Path(slide_uri).name}", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Visualise the key immune channels
IMMUNE_CHANNELS = ["CD8", "PD-L1", "PD-1", "CD68", "CD20", "CD3"]

fig, axes = plt.subplots(1, len(IMMUNE_CHANNELS), figsize=(4 * len(IMMUNE_CHANNELS), 4))
for ax, ch in zip(axes, IMMUNE_CHANNELS):
    ax.imshow(_thumb(maps[ch].astype(bool)), cmap="hot")
    ax.set_title(ch)
    ax.axis("off")
fig.suptitle("Key immune channels", fontsize=13)
plt.tight_layout()
plt.show()
