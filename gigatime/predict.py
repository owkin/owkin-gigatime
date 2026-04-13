"""Core inference logic for GigaTIME."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .constants import (
    CHANNEL_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INFERENCE_WINDOW_SIZE,
)


def predict(
    image: np.ndarray,
    model: nn.Module,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
    window_size: int = INFERENCE_WINDOW_SIZE,
    overlap: int = 0,
) -> dict[str, np.ndarray]:
    """Run GigaTIME on a single H&E image array.

    Args:
        image: RGB image as a uint8 numpy array of shape (H, W, 3).
            Spatial dimensions must be divisible by 16. The recommended size
            is 512×512 px at 20x magnification (~0.5 µm/px).
        model: GigaTIME model returned by ``load_model()``.
        device: Torch device to run inference on.
        threshold: Probability threshold for binary predictions.
        window_size: Sliding window size (pixels). Must divide both H and W.
        overlap: Overlap between adjacent windows in pixels. Overlapping
            regions are averaged to reduce boundary artefacts. Must be
            strictly less than ``window_size``.

    Returns:
        Dictionary mapping channel name → binary prediction mask (H, W) as
        a float32 numpy array with values in {0, 1}.
        Also includes a ``"probabilities"`` key with the raw (H, W, C) float32
        probability array before thresholding.
    """
    if overlap >= window_size:
        raise ValueError(f"overlap ({overlap}) must be less than window_size ({window_size})")

    tensor = _preprocess(image, device)
    probs = _sliding_window_inference(tensor, model, window_size, overlap)  # (1, C, H, W)
    probs_np = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)

    result: dict[str, np.ndarray] = {"probabilities": probs_np}
    for i, name in enumerate(CHANNEL_NAMES):
        result[name] = (probs_np[..., i] >= threshold).astype(np.float32)

    return result


def predict_from_path(
    image_path: str | Path,
    model: nn.Module,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
    window_size: int = INFERENCE_WINDOW_SIZE,
    overlap: int = 0,
) -> dict[str, np.ndarray]:
    """Convenience wrapper around :func:`predict` that accepts a file path."""
    from PIL import Image

    image = np.array(Image.open(image_path).convert("RGB"))
    return predict(
        image, model, device=device, threshold=threshold, window_size=window_size, overlap=overlap
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _preprocess(image: np.ndarray, device: str | torch.device) -> torch.Tensor:
    """Normalise an HWC uint8 RGB array and return a (1, 3, H, W) float tensor."""
    h, w = image.shape[:2]
    if h % 16 != 0 or w % 16 != 0:
        raise ValueError(
            f"Image dimensions ({h}×{w}) must be divisible by 16. "
            f"Consider resizing to {round(h / 16) * 16}×{round(w / 16) * 16}."
        )

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)

    x_np = image.astype(np.float32) / 255.0
    x_np = (x_np - mean) / std
    x = torch.from_numpy(x_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return x.to(device)


def _sliding_window_inference(
    x: torch.Tensor,
    model: nn.Module,
    window_size: int,
    overlap: int,
) -> torch.Tensor:
    """Return sigmoid probabilities (1, C, H, W) using a sliding window."""
    _, _, h, w = x.shape
    num_classes = len(CHANNEL_NAMES)
    step = window_size - overlap

    accumulator = torch.zeros(1, num_classes, h, w, device=x.device)
    count_map = torch.zeros(1, 1, h, w, device=x.device)

    with torch.no_grad():
        for i in range(0, h, step):
            for j in range(0, w, step):
                i_end = min(i + window_size, h)
                j_end = min(j + window_size, w)
                i_start = i_end - window_size
                j_start = j_end - window_size

                window = x[:, :, i_start:i_end, j_start:j_end]
                logits = model(window)
                accumulator[:, :, i_start:i_end, j_start:j_end] += torch.sigmoid(logits)
                count_map[:, :, i_start:i_end, j_start:j_end] += 1

    return accumulator / count_map
