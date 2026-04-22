"""Interactive python script to run inference of gigatime on HE slides."""

# %%
%load_ext autoreload
%autoreload 2

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from abstra.manifest import Manifest
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from tqdm import tqdm

from tilingtool.filters.matter_detection import BRUNet

from gigatime.data import SlideReader, iter_tiles, list_slides, stitch
from gigatime.data.paths import TCGA_LUAD, MOSAIC_NSCLC_UKER
from gigatime.inference import load_model, predict, predict_batch
from gigatime.inference.constants import BACKGROUND_CHANNELS, CHANNEL_NAMES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Traditional mIF fluorophore colors (black → channel color colormaps)
MIF_COLORS: dict[str, tuple[int, int, int]] = {
    "DAPI":       (100, 149, 237),  # cornflower blue
    "TRITC":      (255, 100,   0),  # orange (background channel)
    "Cy5":        (200,   0, 200),  # magenta (background channel)
    "PD-1":       (255,  80,   0),  # orange-red
    "CD14":       (255, 200,   0),  # yellow
    "CD4":        (  0, 200,  80),  # green
    "T-bet":      (180,   0, 255),  # violet
    "CD34":       (255, 150, 200),  # pink
    "CD68":       (255,  30,  30),  # red
    "CD16":       (255, 140,   0),  # dark orange
    "CD11c":      (150, 255,   0),  # lime
    "CD138":      (255,   0, 180),  # magenta-pink
    "CD20":       (  0, 200, 255),  # sky blue
    "CD3":        (255, 230,   0),  # yellow
    "CD8":        (  0, 255, 200),  # turquoise
    "PD-L1":      (220,   0,   0),  # red
    "CK":         (255, 255,   0),  # yellow (cytokeratin)
    "Ki67":       (255,   0, 150),  # hot pink
    "Tryptase":   (128,   0, 255),  # purple
    "Actin-D":    (255, 180, 100),  # peach
    "Caspase3-D": (255,  50,  50),  # light red
    "PHH3-B":     ( 80,  80, 255),  # blue-violet
    "Transgelin": (  0, 200, 180),  # teal
}


def _mif_cmap(channel: str) -> LinearSegmentedColormap:
    """Black-to-color colormap for a given mIF channel."""
    r, g, b = MIF_COLORS.get(channel, (255, 255, 255))
    return LinearSegmentedColormap.from_list(channel, ["black", (r / 255, g / 255, b / 255)])

BATCH_SIZE = 16  # increase if GPU memory allows
CHANNELS_TO_DISPLAY = ["DAPI", "CD8", "CD3", "PD-1", "PD-L1", "CK", "CD4"]
ANALYSIS_CHANNELS = [ch for ch in CHANNEL_NAMES if ch not in BACKGROUND_CHANNELS]
N_SAMPLE_TILES = 3
CANVAS_MAX_DIM = 2048   # downsampled canvas for display
ALL_CANVAS_MAX_DIM = 1024

# %%
# --- Choose dataset ---
AVAILABLE_DATASETS = [TCGA_LUAD, MOSAIC_NSCLC_UKER]
DATASET = AVAILABLE_DATASETS[1]  # change key to switch datasets

manifest = Manifest().load()
dataset = manifest.datasets[DATASET]
slides = list_slides(bucket=dataset.bucket, prefix=dataset.prefix)
if DATASET == TCGA_LUAD:
    slides = [s for s in slides if "parafine" in s]

print(f"Found {len(slides)} slides in {dataset.name}")
for i, uri in enumerate(slides[:10]):
    print(f"  [{i}] {Path(uri).name}")

# %%
# Load the model and matter detector
model = load_model(device=DEVICE)
model.eval()
matter_detector = BRUNet(gpu=0 if DEVICE.startswith("cuda") else -1)
print(f"Loaded — {len(CHANNEL_NAMES)} output channels: {', '.join(CHANNEL_NAMES)}")

# %%
# Run the inference on one image
SLIDE_INDEX = 2
slide_uri = slides[SLIDE_INDEX]
print(f"Opening: {Path(slide_uri).name}")
reader = SlideReader(slide_uri)  # S3 URI — downloaded to a temp dir, cleaned up on close()
w, h = reader.dimensions_at_read_level
print(f"Slide at read level: {w} × {h} px  (mpp ≈ 0.5 µm/px)")
matter_mask = matter_detector(reader._slide)
print(f"Matter mask: {matter_mask.shape} (at 4 µm/px)")

vis_canvas: dict[str, np.ndarray] = {}
all_canvas: dict[str, np.ndarray] = {}
sums: dict[str, float] = {ch: 0.0 for ch in CHANNEL_NAMES}
pixel_counts: dict[str, int] = {ch: 0 for ch in CHANNEL_NAMES}
sampled_tiles: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
grid = None
n_tiles = 0
vis_scale = 1.0
all_scale = 1.0

def _process_tile(tile, preds, grid, step):
    """Update canvases, stats and reservoir sample for one tile.

    preds must have been produced with return_probabilities=True so that
    preds["probabilities"] is the raw (H, W, C) sigmoid array.  The per-channel
    binary masks (preds[ch]) are still used for positive-fraction stats.
    """
    probs_hwc = preds["probabilities"]  # (H, W, C) float32 in [0, 1]

    for ch in CHANNEL_NAMES:
        sums[ch] += float(preds[ch].sum())
        pixel_counts[ch] += preds[ch].size

    y0_full = tile.row * step
    x0_full = tile.col * step
    y1_full = min(y0_full + grid.tile_size, grid.slide_height)
    x1_full = min(x0_full + grid.tile_size, grid.slide_width)
    for canvas_dict, s in ((vis_canvas, vis_scale), (all_canvas, all_scale)):
        y0_c, x0_c = int(y0_full * s), int(x0_full * s)
        y1_c, x1_c = int(y1_full * s), int(x1_full * s)
        if y1_c > y0_c and x1_c > x0_c:
            for ch in canvas_dict:
                ch_idx = CHANNEL_NAMES.index(ch)
                prob_slice = probs_hwc[..., ch_idx]  # (H, W), continuous [0, 1]
                patch = (np.array(
                    Image.fromarray((prob_slice * 255).astype(np.uint8)).resize(
                        (x1_c - x0_c, y1_c - y0_c), Image.NEAREST
                    )
                ) / 255.0).astype(np.float16)
                canvas_dict[ch][y0_c:y1_c, x0_c:x1_c] = np.maximum(
                    canvas_dict[ch][y0_c:y1_c, x0_c:x1_c], patch
                )

    global n_tiles
    # Store probabilities (not binary) for full-res tile display
    tile_data = (tile.array.copy(), {ch: probs_hwc[..., CHANNEL_NAMES.index(ch)] for ch in CHANNELS_TO_DISPLAY})
    if n_tiles < N_SAMPLE_TILES:
        sampled_tiles.append(tile_data)
    else:
        j = np.random.randint(0, n_tiles + 1)
        if j < N_SAMPLE_TILES:
            sampled_tiles[j] = tile_data
    n_tiles += 1


tile_buf: list = []

_w, _h = reader.dimensions_at_read_level
_step = 512  # default tile_size with no overlap
_n_grid_tiles = max(1, -(-_w // _step)) * max(1, -(-_h // _step))  # ceil division

with torch.inference_mode():
    for tile, grid in tqdm(iter_tiles(reader, matter_mask=matter_mask), desc="Inference", total=_n_grid_tiles):
        if not vis_canvas:
            step = grid.tile_size - grid.overlap
            vis_scale = CANVAS_MAX_DIM / max(grid.slide_width, grid.slide_height)
            all_scale = ALL_CANVAS_MAX_DIM / max(grid.slide_width, grid.slide_height)
            vis_canvas.update({
                ch: np.zeros((max(1, int(grid.slide_height * vis_scale)), max(1, int(grid.slide_width * vis_scale))), dtype=np.float16)
                for ch in CHANNELS_TO_DISPLAY
            })
            all_canvas.update({
                ch: np.zeros((max(1, int(grid.slide_height * all_scale)), max(1, int(grid.slide_width * all_scale))), dtype=np.float16)
                for ch in ANALYSIS_CHANNELS
            })

        tile_buf.append(tile)
        if len(tile_buf) < BATCH_SIZE:
            continue

        for t, preds in zip(tile_buf, predict_batch([t.array for t in tile_buf], model, device=DEVICE, return_probabilities=True)):
            _process_tile(t, preds, grid, step)
        tile_buf = []

    if tile_buf:
        for t, preds in zip(tile_buf, predict_batch([t.array for t in tile_buf], model, device=DEVICE, return_probabilities=True)):
            _process_tile(t, preds, grid, step)

assert grid is not None, "No tissue tiles found — check min_tissue_fraction or slide content"
print(f"Done: {n_tiles} tissue tiles across a {grid.n_rows}×{grid.n_cols} grid")

# %%
# Display tile level inferences
# Build H&E thumbnail from the native pyramid (fast, no re-reading)
thumb_w, thumb_h = 512, 512
thumbnail = np.array(reader._slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
reader.close()

n = len(CHANNELS_TO_DISPLAY)
fig, axes = plt.subplots(2, (n // 2) + 1, figsize=(4 * ((n // 2) + 1), 8))
axes = axes.flatten()

# H&E panel
axes[0].imshow(thumbnail)
axes[0].set_title("H&E", fontsize=12, fontweight="bold")
axes[0].axis("off")

# Channel panels — vis_canvas is already downsampled, resize only to match thumbnail
for ax, channel in zip(axes[1:], CHANNELS_TO_DISPLAY):
    arr = vis_canvas[channel]  # raw float32, already in [0, 1]
    disp = np.array(Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (thumb_w, thumb_h), Image.NEAREST
    ))
    ax.imshow(disp, cmap=_mif_cmap(channel), vmin=0, vmax=255)
    nz = arr[arr > 0]
    ax.set_title(
        f"{channel}\nmed={np.median(nz):.2f} p99={np.percentile(nz, 99):.2f}" if nz.size else channel,
        fontsize=9,
    )
    ax.axis("off")

# Hide any unused axes
for ax in axes[1 + n :]:
    ax.set_visible(False)

fig.suptitle(f"GigaTIME predictions — {reader.path.name}", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Per-channel histograms for CHANNELS_TO_DISPLAY — raw probabilities, no clipping
fig, axes_h = plt.subplots(1, len(CHANNELS_TO_DISPLAY), figsize=(4 * len(CHANNELS_TO_DISPLAY), 3))
fig.patch.set_facecolor("#111111")
for ax, channel in zip(axes_h, CHANNELS_TO_DISPLAY):
    nz = vis_canvas[channel]
    nz = nz[nz > 0].ravel()
    r, g, b = MIF_COLORS.get(channel, (255, 255, 255))
    ax.hist(nz, bins=80, color=(r / 255, g / 255, b / 255), edgecolor="none", density=True)
    for pct, ls in ((50, ":"), (99, "--"), (99.9, "-")):
        v = np.percentile(nz, pct) if nz.size else 0
        ax.axvline(v, color="white", linestyle=ls, linewidth=1.0, alpha=0.8)
        ax.text(v, ax.get_ylim()[1] * 0.95, f"p{pct}\n{v:.2f}", color="white",
                fontsize=6, ha="center", va="top")
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=7)
    ax.spines[:].set_color("#444444")
    ax.set_title(channel, color="white", fontsize=10)
    ax.set_xlabel("prob", color="white", fontsize=8)
fig.suptitle("Raw probability distributions — tissue pixels only", color="white", fontsize=11)
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
        axes[row, col].imshow(tile_preds[ch], cmap=_mif_cmap(ch), vmin=0, vmax=1)
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
    arr = all_canvas[ch]  # raw float32, no clipping
    disp = np.array(Image.fromarray((arr * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST))
    axes[i].imshow(disp, cmap=_mif_cmap(ch), vmin=0, vmax=255)
    axes[i].set_title(ch, fontsize=10)
    axes[i].axis("off")
for ax in axes[n_ch:]:
    ax.set_visible(False)
fig.suptitle(f"Slide-level predictions — all channels\n{Path(slide_uri).name}", fontsize=12)
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
        mm = matter_detector(reader._slide)

        with torch.inference_mode():
            for tile, grid in iter_tiles(reader, matter_mask=mm):
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
