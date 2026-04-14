"""Demo: compute and interpret spatial biomarker features for a single LUAD slide."""

# %%
%load_ext autoreload
%autoreload 2

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from abstra.manifest import Manifest
from tqdm import tqdm

from gigatime.data import SlideReader, iter_tiles, list_slides
from gigatime.data.paths import TCGA_LUAD
from gigatime.features import compute_features_from_tiles
from gigatime.inference import load_model, predict_batch

BATCH_SIZE = 8  # increase if GPU memory allows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Channels required by compute_features_from_tiles — nothing extra.
FEATURE_CHANNELS = [
    "CK", "DAPI", "CD8", "CD4", "CD68", "CD34",
    "PD-1", "PD-L1", "Ki67", "Caspase3-D",
]

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
# Run batched inference and compute features in a single pass
reader = SlideReader(slide_uri)
grid = None
tile_stream = []

tile_buf: list = []
grid_buf = None

with torch.inference_mode():
    for tile, grid in tqdm(iter_tiles(reader, min_tissue_fraction=0.05), desc="Inference"):
        tile_buf.append(tile)
        grid_buf = grid
        if len(tile_buf) < BATCH_SIZE:
            continue
        for preds, t in zip(
            predict_batch([t.array for t in tile_buf], model, device=DEVICE), tile_buf
        ):
            tile_stream.append(({ch: preds[ch] for ch in FEATURE_CHANNELS}, t))
        tile_buf = []

    # flush remaining tiles
    if tile_buf:
        for preds, t in zip(
            predict_batch([t.array for t in tile_buf], model, device=DEVICE), tile_buf
        ):
            tile_stream.append(({ch: preds[ch] for ch in FEATURE_CHANNELS}, t))

reader.close()
assert grid is not None, "No tissue tiles found"

features = compute_features_from_tiles(iter(tile_stream), grid)

# %%
# Display features
FEATURE_LABELS = {
    "pdl1_tps":                   "PD-L1 TPS",
    "cd8_intratumoral_density":   "CD8 intratumoral density",
    "immune_exclusion_index":     "Immune exclusion index",
    "cd8_pd1_exhaustion_fraction":"CD8+PD-1 exhaustion fraction",
    "macrophage_to_tcell_ratio":  "Macrophage / T-cell ratio",
    "cd4_cd8_intratumoral_ratio": "CD4 / CD8 intratumoral ratio",
    "ki67_tpi":                   "Ki67 tumour proliferation index",
    "apoptosis_index":            "Tumour apoptosis index",
    "vascular_density_intratumoral": "Intratumoural vascular density (CD34)",
}

labels = [FEATURE_LABELS.get(k, k) for k in features]
values = list(features.values())

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(labels, values, color="steelblue")
ax.bar_label(bars, fmt="{:.3f}", padding=4, fontsize=9)
ax.set_xlim(0, max(values) * 1.25)
ax.set_xlabel("Feature value")
ax.invert_yaxis()
ax.set_title(f"Spatial biomarker features\n{Path(slide_uri).name}", fontsize=11)
plt.tight_layout()
plt.show()

# %%
# Derive TME class from the three key axes
pdl1 = features["pdl1_tps"]
cd8 = features["cd8_intratumoral_density"]
excl = features["immune_exclusion_index"]

if cd8 > 0.05 and pdl1 > 0.01:
    tme_class = "Inflamed (hot)"
elif excl > 0.6:
    tme_class = "Excluded"
elif cd8 < 0.02:
    tme_class = "Desert (cold)"
else:
    tme_class = "Intermediate"

print(f"TME class : {tme_class}")
print(f"PD-L1 TPS : {pdl1:.3f}  ({'≥1%' if pdl1 >= 0.01 else '<1%'} pembrolizumab threshold)")
print(f"CD8 intra : {cd8:.3f}")
print(f"Exclusion : {excl:.3f}")
