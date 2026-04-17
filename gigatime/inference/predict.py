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

_N_CLASSES = len(CHANNEL_NAMES)


def predict(
    image: np.ndarray,
    model: nn.Module,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
    window_size: int = INFERENCE_WINDOW_SIZE,
    overlap: int = 0,
    return_probabilities: bool = False,
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
        return_probabilities: If ``True``, include a ``"probabilities"`` key
            with the raw (H, W, C) float32 array before thresholding.
            Defaults to ``False`` to save memory.

    Returns:
        Dictionary mapping channel name → binary prediction mask (H, W) as
        a float32 numpy array with values in {0, 1}.
        If ``return_probabilities=True``, also includes a ``"probabilities"``
        key with the raw (H, W, C) float32 probability array.
    """
    if overlap >= window_size:
        raise ValueError(f"overlap ({overlap}) must be less than window_size ({window_size})")

    tensor = _preprocess(image, device)
    probs = _sliding_window_inference(tensor, model, window_size, overlap)  # (1, C, H, W)
    probs_np = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)

    result: dict[str, np.ndarray] = {}
    if return_probabilities:
        result["probabilities"] = probs_np
    for i, name in enumerate(CHANNEL_NAMES):
        result[name] = (probs_np[..., i] >= threshold).astype(np.float32)

    return result


def predict_batch(
    images: list[np.ndarray],
    model: nn.Module,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
    window_size: int = INFERENCE_WINDOW_SIZE,
    overlap: int = 0,
    return_probabilities: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Run GigaTIME on a batch of H&E tiles in a single forward pass.

    All sliding-window crops from every tile in the batch are stacked and
    processed together, maximising GPU utilisation compared to calling
    :func:`predict` repeatedly.

    Args:
        images:               List of B uint8 RGB arrays, each of shape (H, W, 3).
                              All images must have the same spatial dimensions.
        model:                GigaTIME model returned by ``load_model()``.
        device:               Torch device to run inference on.
        threshold:            Probability threshold for binary predictions.
        window_size:          Sliding window size in pixels.
        overlap:              Overlap between adjacent windows in pixels.
        return_probabilities: If ``True``, each result dict also contains a
                              ``"probabilities"`` key with the raw (H, W, C)
                              float32 sigmoid array before thresholding.

    Returns:
        List of B dicts, each mapping channel name → (H, W) float32 mask.
        If ``return_probabilities=True``, each dict also has ``"probabilities"``.
    """
    if overlap >= window_size:
        raise ValueError(f"overlap ({overlap}) must be less than window_size ({window_size})")

    B = len(images)
    # (B, 3, H, W)
    batch = torch.cat([_preprocess(img, device) for img in images])
    probs = _sliding_window_inference_batch(batch, model, window_size, overlap)  # (B, C, H, W)
    probs_np = probs.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)

    results = []
    for b in range(B):
        d = {
            name: (probs_np[b, ..., i] >= threshold).astype(np.float32)
            for i, name in enumerate(CHANNEL_NAMES)
        }
        if return_probabilities:
            d["probabilities"] = probs_np[b]  # (H, W, C)
        results.append(d)
    return results


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
    return _sliding_window_inference_batch(x, model, window_size, overlap)


def _sliding_window_inference_batch(
    x: torch.Tensor,
    model: nn.Module,
    window_size: int,
    overlap: int,
) -> torch.Tensor:
    """Return sigmoid probabilities (B, C, H, W) for a batch of tiles.

    All sliding-window crops across all B tiles are stacked into a single
    tensor and processed in one model call, maximising GPU utilisation.
    """
    B, _, h, w = x.shape
    step = window_size - overlap

    # Collect all (i_start, i_end, j_start, j_end) positions once
    positions: list[tuple[int, int, int, int]] = []
    for i in range(0, h, step):
        for j in range(0, w, step):
            i_end = min(i + window_size, h)
            j_end = min(j + window_size, w)
            positions.append((i_end - window_size, i_end, j_end - window_size, j_end))

    n_windows = len(positions)

    # Stack all crops: (n_windows * B, 3, window_size, window_size)
    crops = torch.stack([x[:, :, r0:r1, c0:c1] for r0, r1, c0, c1 in positions], dim=1).reshape(
        n_windows * B, x.shape[1], window_size, window_size
    )

    with torch.no_grad():
        logits = model(crops)  # (n_windows * B, C, window_size, window_size)

    # crops was built (B, n_windows, ...) then reshaped to (B*n_windows, ...), so
    # logits are in B-major order: [t0w0, t0w1, ..., t0w(n-1), t1w0, ...].
    # Reshape back to (B, n_windows, ...) — NOT (n_windows, B, ...).
    probs = torch.sigmoid(logits).reshape(B, n_windows, _N_CLASSES, window_size, window_size)

    accumulator = torch.zeros(B, _N_CLASSES, h, w, device=x.device)
    count_map = torch.zeros(B, 1, h, w, device=x.device)

    for k, (r0, r1, c0, c1) in enumerate(positions):
        accumulator[:, :, r0:r1, c0:c1] += probs[:, k]  # (B, C, tile_h, tile_w)
        count_map[:, :, r0:r1, c0:c1] += 1

    return accumulator / count_map
