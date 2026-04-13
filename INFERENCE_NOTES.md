# GigaTIME — Internal Inference Notes

## What it does
Takes a standard H&E patch and predicts **23 virtual mIF channels** (CD3, CD8, CD4, CD68, PD-L1, Ki67, CK, etc.) via a UNet++ architecture. Intended for tumor microenvironment analysis — research use only.

## Input requirements

| Property | Value |
|---|---|
| Format | RGB PNG patch (no OpenSlide/WSI support built-in) |
| Tile size | 556×556 px (resized to 512×512 before inference) |
| Resolution | ~0.5 µm/px (20x magnification) — not enforced in code but implied by HE-COMET training data |
| Normalization | ImageNet mean/std via albumentations |

For WSIs: extract 556×556 tiles at 20x yourself (e.g. with OpenSlide), then pass them through the model.
If your scanner is 40x (0.25 µm/px), downsample tiles by 2x before inference.

## Loading the model

```python
from huggingface_hub import snapshot_download
import torch, os, archs  # archs.py is in scripts/

local_dir = snapshot_download(repo_id="prov-gigatime/GigaTIME")  # requires HF_TOKEN env var
state_dict = torch.load(os.path.join(local_dir, "model.pth"), map_location="cpu")
model = archs.gigatime(num_classes=23, input_channels=3)
model.load_state_dict(state_dict)
model.eval()
```

## Preprocessing

```python
import albumentations as A
from albumentations.augmentations import transforms
import numpy as np
from PIL import Image

transform = A.Compose([A.Resize(512, 512), transforms.Normalize()])

img = np.array(Image.open("patch.png").convert("RGB"))
x = transform(image=img)["image"].astype("float32").transpose(2, 0, 1)
x = torch.from_numpy(x).unsqueeze(0)  # (1, 3, 512, 512)
```

## Inference

Feed **512×512** tiles (the model's training resolution). `do_inference` splits each tile into **256×256 sub-windows** internally for memory efficiency, then stitches results back — this is transparent to the caller:

```python
def do_inference(x, model, window_size=256):
    b, c, h, w = x.shape
    logits = torch.zeros(b, 23, h, w)
    with torch.no_grad():
        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                logits[:, :, i:i+window_size, j:j+window_size] = model(x[:, :, i:i+window_size, j:j+window_size])
    return logits

probs = torch.sigmoid(do_inference(x, model))  # (1, 23, 512, 512), values in [0, 1]
preds = (probs > 0.5).float()                  # binary masks
```

## Output channels (index → marker)

`0:DAPI, 1:TRITC*, 2:Cy5*, 3:PD-1, 4:CD14, 5:CD4, 6:T-bet, 7:CD34, 8:CD68, 9:CD16, 10:CD11c, 11:CD138, 12:CD20, 13:CD3, 14:CD8, 15:PD-L1, 16:CK, 17:Ki67, 18:Tryptase, 19:Actin-D, 20:Caspase3-D, 21:PHH3-B, 22:Transgelin`

*TRITC and Cy5 are background channels, excluded from evaluations.

## Passing a full image

The architecture is fully convolutional (conv + pool + upsample only, no dense layers), so it accepts **any input size divisible by 16** (4 pooling layers). `do_inference` already tiles arbitrarily large inputs via its sliding window.

However, each 256px window is processed **independently with no cross-border context**, which can cause visible seams at tile boundaries. For large images, use an **overlapping sliding window with blended/averaged overlap regions** rather than hard cuts.

## Key files

| File | Role |
|---|---|
| [scripts/archs.py](scripts/archs.py) | UNet++ model definition |
| [scripts/prov_data.py](scripts/prov_data.py) | Data loading, transforms, tile format |
| [scripts/gigatime_testing.ipynb](scripts/gigatime_testing.ipynb) | End-to-end worked example |
| [scripts/db_train.py](scripts/db_train.py) / [db_test.py](scripts/db_test.py) | Training/evaluation scripts |
| [data/sample_metadata.csv](data/sample_metadata.csv) | Expected metadata format |
