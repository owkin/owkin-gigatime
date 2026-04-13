# GigaTIME — Owkin Fork

This is Owkin's fork of the [original GigaTIME repository](https://github.com/prov-gigatime/GigaTIME) (Microsoft Research / Providence). The original code is preserved under `legacy_gigatime_repo/`. This fork repackages the model as a clean Python library aimed at running inference on Owkin's platforms.

> **Paper:** [GigaTIME: Multimodal AI generates virtual population for tumor microenvironment modeling](https://aka.ms/gigatime-paper) — *Cell*
> **Model weights:** [prov-gigatime/GigaTIME on HuggingFace](https://huggingface.co/prov-gigatime/GigaTIME) (access requires accepting the terms of use)

---

## What GigaTIME does

GigaTIME takes a standard H&E tile and predicts 23 virtual multiplexed immunofluorescence (mIF) channels — including CD3, CD8, CD4, CD68, PD-L1, Ki67, and CK — using a UNet++ architecture trained on the HE-COMET dataset.

Input requirements: **512×512 px RGB tiles at ~20x magnification (0.5 µm/px)**.

---

## Setup

Requires [uv](https://github.com/astral-sh/uv).

```bash
uv sync            # inference only
uv sync --extra train      # + training dependencies
uv sync --extra notebooks  # + Jupyter
```

Set your HuggingFace token to download the model weights:

```bash
export HF_TOKEN=<your-huggingface-read-only-token>
```

---

## Usage

### As a library

```python
from gigatime import load_model, predict
from gigatime.data import SlideReader, iter_tiles, stitch

# Load model (downloads weights from HuggingFace if no path given)
model = load_model(device="cuda")

# Open a WSI — resolution is handled automatically
results = []
with SlideReader("path/to/slide.svs") as reader:
    for tile, grid in iter_tiles(reader, tile_size=512, overlap=32, min_tissue_fraction=0.1):
        pred = predict(tile.array, model, device="cuda")
        results.append((pred, tile))

# Reassemble a slide-level prediction map
cd8_map = stitch(results, grid, channel="CD8", mode="mean")  # (H, W) float32
```

### From S3

```python
from gigatime.data import list_slides, download_slide

uris = list_slides("my-bucket", prefix="cohort/slides/")
local_path = download_slide(uris[0], dest_dir="/tmp/slides")
```

### CLI

```bash
uv run gigatime-infer \
  --input tiles/ \
  --output_dir ./results \
  --device cuda \
  --overlap 32
```

---

## Repository structure

```
gigatime/               # main Python package
├── model.py            # GigaTIME architecture + weight loading
├── predict.py          # inference logic (sliding window)
├── constants.py        # channel names, tile size, normalisation
├── cli.py              # command-line interface
└── data/
    ├── s3.py           # S3 slide discovery and download
    ├── slide.py        # OpenSlide wrapper (auto level selection)
    └── tiling.py       # tile iterator and slide-level stitching
legacy_gigatime_repo/   # original Microsoft / Providence code (unmodified)
pyproject.toml          # uv-managed dependencies
```

---

## Output channels

| Index | Channel | Notes |
|---|---|---|
| 0 | DAPI | Nuclear stain |
| 1 | TRITC | Background — excluded from analysis |
| 2 | Cy5 | Background — excluded from analysis |
| 3–22 | PD-1, CD14, CD4, T-bet, CD34, CD68, CD16, CD11c, CD138, CD20, CD3, CD8, PD-L1, CK, Ki67, Tryptase, Actin-D, Caspase3-D, PHH3-B, Transgelin | |

---

## License

Model weights and original code are subject to the [original research-only license](legacy_gigatime_repo/LICENSE). This fork inherits the same terms.
