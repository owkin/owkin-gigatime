HF_REPO_ID = "prov-gigatime/GigaTIME"

# Channels in output order. TRITC and Cy5 are background channels excluded from analysis.
CHANNEL_NAMES = [
    "DAPI",
    "TRITC",  # background — excluded from analysis
    "Cy5",  # background — excluded from analysis
    "PD-1",
    "CD14",
    "CD4",
    "T-bet",
    "CD34",
    "CD68",
    "CD16",
    "CD11c",
    "CD138",
    "CD20",
    "CD3",
    "CD8",
    "PD-L1",
    "CK",
    "Ki67",
    "Tryptase",
    "Actin-D",
    "Caspase3-D",
    "PHH3-B",
    "Transgelin",
]

BACKGROUND_CHANNELS = {"TRITC", "Cy5"}

# ImageNet normalisation (used during training)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Resolution the model was trained at
TILE_SIZE_PX = 512  # pixels fed to the model
INFERENCE_WINDOW_SIZE = 256  # internal sliding window size (must divide TILE_SIZE_PX evenly)
