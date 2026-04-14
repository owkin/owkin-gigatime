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

vis_canvas: dict[str, np.ndarray] = {}
all_canvas: dict[str, np.ndarray] = {}
sums: dict[str, float] = {ch: 0.0 for ch in CHANNEL_NAMES}
pixel_counts: dict[str, int] = {ch: 0 for ch in CHANNEL_NAMES}
sampled_tiles: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
grid = None
n_tiles = 0
vis_scale = 1.0
all_scale = 1.0

with torch.inference_mode():
    for tile, grid in iter_tiles(reader, min_tissue_fraction=0.05, progress=True):
        if not vis_canvas:
            step = grid.tile_size - grid.overlap
            vis_scale = CANVAS_MAX_DIM / max(grid.slide_width, grid.slide_height)
            all_scale = ALL_CANVAS_MAX_DIM / max(grid.slide_width, grid.slide_height)
            vis_canvas = {
                ch: np.zeros((max(1, int(grid.slide_height * vis_scale)), max(1, int(grid.slide_width * vis_scale))), dtype=np.float32)
                for ch in CHANNELS_TO_DISPLAY
            }
            all_canvas = {
                ch: np.zeros((max(1, int(grid.slide_height * all_scale)), max(1, int(grid.slide_width * all_scale))), dtype=np.float32)
                for ch in ANALYSIS_CHANNELS
            }
        preds = predict(tile.array, model, device=DEVICE)

        # Running sums for statistics (all channels, no spatial storage)
        for ch in CHANNEL_NAMES:
            sums[ch] += float(preds[ch].sum())
            pixel_counts[ch] += preds[ch].size

        # Update both downsampled canvases
        y0_full = tile.row * step
        x0_full = tile.col * step
        y1_full = min(y0_full + grid.tile_size, grid.slide_height)
        x1_full = min(x0_full + grid.tile_size, grid.slide_width)
        for canvas_dict, s in ((vis_canvas, vis_scale), (all_canvas, all_scale)):
            y0_c, x0_c = int(y0_full * s), int(x0_full * s)
            y1_c, x1_c = int(y1_full * s), int(x1_full * s)
            if y1_c > y0_c and x1_c > x0_c:
                for ch in canvas_dict:
                    patch = np.array(
                        Image.fromarray((preds[ch] * 255).astype(np.uint8)).resize(
                            (x1_c - x0_c, y1_c - y0_c), Image.NEAREST
                        )
                    ) / 255.0
                    canvas_dict[ch][y0_c:y1_c, x0_c:x1_c] = np.maximum(
                        canvas_dict[ch][y0_c:y1_c, x0_c:x1_c], patch
                    )

        # Reservoir sampling for full-res tile examples
        tile_data = (tile.array.copy(), {ch: preds[ch] for ch in CHANNELS_TO_DISPLAY})
        if n_tiles < N_SAMPLE_TILES:
            sampled_tiles.append(tile_data)
        else:
            j = np.random.randint(0, n_tiles + 1)
            if j < N_SAMPLE_TILES:
                sampled_tiles[j] = tile_data
        n_tiles += 1

reader.close()
assert grid is not None, "No tissue tiles found — check min_tissue_fraction or slide content"
print(f"Done: {n_tiles} tissue tiles across a {grid.n_rows}×{grid.n_cols} grid")

# %%
# Display tile level inferences
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

# Channel panels — vis_canvas is already downsampled, resize only to match thumbnail
for ax, channel in zip(axes[1:], CHANNELS_TO_DISPLAY):
    disp = np.array(
        Image.fromarray((vis_canvas[channel] * 255).astype(np.uint8)).resize(
            (thumb_w, thumb_h), Image.NEAREST
        )
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
fractions = {
    ch: sums[ch] / pixel_counts[ch] if pixel_counts[ch] > 0 else 0.0
    for ch in ANALYSIS_CHANNELS
}

sorted_fracs = sorted(fractions.items(), key=lambda x: x[1], reverse=True)
print(f"\nPositive-pixel fraction per channel ({reader.path.name}):")
for ch, frac in sorted_fracs:
    bar = "█" * int(frac * 40)
    print(f"  {ch:12s} {frac:.3f}  {bar}")

# %%
# Full-resolution randomly sampled tiles with model predictions
n_cols = 1 + len(CHANNELS_TO_DISPLAY)
fig, axes = plt.subplots(N_SAMPLE_TILES, n_cols, figsize=(3 * n_cols, 3 * N_SAMPLE_TILES))
for row, (he_array, tile_preds) in enumerate(sampled_tiles):
    axes[row, 0].imshow(he_array)
    axes[row, 0].axis("off")
    if row == 0:
        axes[row, 0].set_title("H&E", fontsize=10, fontweight="bold")
    for col, ch in enumerate(CHANNELS_TO_DISPLAY, start=1):
        axes[row, col].imshow(tile_preds[ch], cmap="hot", vmin=0, vmax=1)
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(ch, fontsize=10)
fig.suptitle("Randomly sampled tissue tiles — full resolution", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Slide-level thumbnail — all analysis channels
n_ch = len(ANALYSIS_CHANNELS)
n_cols_grid = 7
n_rows_grid = (n_ch + n_cols_grid - 1) // n_cols_grid
fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(3.5 * n_cols_grid, 3.5 * n_rows_grid))
axes = axes.flatten()
for i, ch in enumerate(ANALYSIS_CHANNELS):
    disp = np.array(
        Image.fromarray((all_canvas[ch] * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
    )
    axes[i].imshow(disp, cmap="hot", vmin=0, vmax=255)
    axes[i].set_title(ch, fontsize=10)
    axes[i].axis("off")
for ax in axes[n_ch:]:
    ax.set_visible(False)
fig.suptitle(f"Slide-level predictions — all channels\n{Path(slide_path).name}", fontsize=12)
plt.tight_layout()
plt.show()

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
