"""Model loading utilities."""

from pathlib import Path

import torch
import torch.nn as nn

from .constants import CHANNEL_NAMES, HF_REPO_ID

# ---------------------------------------------------------------------------
# Architecture (inlined from scripts/archs.py to keep inference self-contained)
# ---------------------------------------------------------------------------


class _VGGBlock(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GigaTIME(nn.Module):
    """UNet++ predicting 23 virtual mIF channels from an H&E RGB image.

    Accepts any input whose spatial dimensions are divisible by 16.
    """

    CHANNEL_NAMES = CHANNEL_NAMES

    def __init__(self, num_classes: int = 23, input_channels: int = 3):
        super().__init__()
        f = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = _VGGBlock(input_channels, f[0], f[0])
        self.conv1_0 = _VGGBlock(f[0], f[1], f[1])
        self.conv2_0 = _VGGBlock(f[1], f[2], f[2])
        self.conv3_0 = _VGGBlock(f[2], f[3], f[3])
        self.conv4_0 = _VGGBlock(f[3], f[4], f[4])

        self.conv0_1 = _VGGBlock(f[0] + f[1], f[0], f[0])
        self.conv1_1 = _VGGBlock(f[1] + f[2], f[1], f[1])
        self.conv2_1 = _VGGBlock(f[2] + f[3], f[2], f[2])
        self.conv3_1 = _VGGBlock(f[3] + f[4], f[3], f[3])

        self.conv0_2 = _VGGBlock(f[0] * 2 + f[1], f[0], f[0])
        self.conv1_2 = _VGGBlock(f[1] * 2 + f[2], f[1], f[1])
        self.conv2_2 = _VGGBlock(f[2] * 2 + f[3], f[2], f[2])

        self.conv0_3 = _VGGBlock(f[0] * 3 + f[1], f[0], f[0])
        self.conv1_3 = _VGGBlock(f[1] * 3 + f[2], f[1], f[1])

        self.conv0_4 = _VGGBlock(f[0] * 4 + f[1], f[0], f[0])

        self.final = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        return self.final(x0_4)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_model(
    weights_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> GigaTIME:
    """Load GigaTIME weights and return the model in eval mode.

    Args:
        weights_path: Path to a local ``model.pth`` file. If ``None``, the
            weights are downloaded from HuggingFace (requires ``HF_TOKEN``).
        device: Torch device to load the model onto.

    Returns:
        GigaTIME model in eval mode.
    """
    if weights_path is None:
        weights_path = _download_weights()

    state_dict = torch.load(weights_path, map_location=device)
    model = GigaTIME()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _download_weights() -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download weights automatically. "
            "Install it with: pip install huggingface-hub"
        ) from e

    local_dir = snapshot_download(repo_id=HF_REPO_ID)
    return Path(local_dir) / "model.pth"
