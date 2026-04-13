"""Command-line interface for GigaTIME inference.

Usage examples
--------------
# Single tile, weights downloaded from HuggingFace:
python -m inference.cli --input tile.png --output_dir ./results

# Batch of tiles, local weights, GPU:
python -m inference.cli --input tiles/ --output_dir ./results \\
    --weights model.pth --device cuda --overlap 32

# Save raw probabilities instead of binary masks:
python -m inference.cli --input tile.png --output_dir ./results --probabilities
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import BACKGROUND_CHANNELS, CHANNEL_NAMES, INFERENCE_WINDOW_SIZE, TILE_SIZE_PX
from .model import load_model
from .predict import predict

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gigatime-infer",
        description="Run GigaTIME inference on H&E image tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a single image file or a directory of images.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where prediction outputs will be written.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=Path,
        help="Path to model.pth. Downloads from HuggingFace if omitted (requires HF_TOKEN).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (e.g. 'cpu', 'cuda', 'cuda:1', 'mps').",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help=f"Resize input images to {TILE_SIZE_PX}×{TILE_SIZE_PX} before inference.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=INFERENCE_WINDOW_SIZE,
        help="Sliding window size in pixels.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Overlap between adjacent windows in pixels. Reduces boundary artefacts.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary masks.",
    )
    parser.add_argument(
        "--probabilities",
        action="store_true",
        help="Save raw probability maps (float32 .npy) instead of binary masks.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        choices=[c for c in CHANNEL_NAMES if c not in BACKGROUND_CHANNELS],
        help="Subset of channels to save. Saves all non-background channels by default.",
    )
    return parser.parse_args()


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not files:
            print(f"No supported image files found in {path}", file=sys.stderr)
            sys.exit(1)
        return files
    print(f"Input path does not exist: {path}", file=sys.stderr)
    sys.exit(1)


def load_image(path: Path, resize: bool) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if resize:
        image = image.resize((TILE_SIZE_PX, TILE_SIZE_PX), Image.Resampling.LANCZOS)
    return np.array(image)


def save_predictions(
    predictions: dict[str, np.ndarray],
    output_dir: Path,
    stem: str,
    channels: list[str],
    save_probabilities: bool,
) -> None:
    if save_probabilities:
        probs = predictions["probabilities"]  # (H, W, C)
        channel_indices = [CHANNEL_NAMES.index(c) for c in channels]
        np.save(output_dir / f"{stem}_probabilities.npy", probs[..., channel_indices])
    else:
        for channel in channels:
            mask = (predictions[channel] * 255).astype(np.uint8)
            Image.fromarray(mask).save(output_dir / f"{stem}_{channel}.png")


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    channels = args.channels or [c for c in CHANNEL_NAMES if c not in BACKGROUND_CHANNELS]

    print("Loading model...", flush=True)
    model = load_model(weights_path=args.weights, device=args.device)

    inputs = collect_inputs(args.input)
    print(f"Running inference on {len(inputs)} image(s)...", flush=True)

    for i, path in enumerate(inputs, 1):
        print(f"  [{i}/{len(inputs)}] {path.name}", flush=True)
        image = load_image(path, resize=args.resize)
        predictions = predict(
            image,
            model,
            device=args.device,
            threshold=args.threshold,
            window_size=args.window_size,
            overlap=args.overlap,
        )
        save_predictions(predictions, args.output_dir, path.stem, channels, args.probabilities)

    print(f"Done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
