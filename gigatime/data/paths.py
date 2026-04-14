"""Centralized path handling."""

import os
import tempfile

from abstra.manifest import Manifest
from sagemaker_toolbox.utils import (
    ABSTRA_ROOT_PATH,
    SAGEMAKER_ROOT_PATH,
    Path,
    am_i_on_abstra,
    am_i_on_sagemaker,
    am_i_on_sagemaker_training_job,
)
from sagemaker_toolbox.utils.env_utils import SAGEMAKER_JOB_ROOT_PATH

# Dataset hash from Abstra
TCGA_LUAD = "4d2063d1-0d25-43c3-83c3-b006355a3f68"
MOSAIC_OV = "ff2679f4-71dc-4b49-823f-ba90b1244f8b"
MOSAIC_BLCA = "c1521eca-c504-4752-8597-bd375b4deaa1"

ROOT_PATH = str(
    SAGEMAKER_JOB_ROOT_PATH
    if am_i_on_sagemaker_training_job()
    else SAGEMAKER_ROOT_PATH
    if am_i_on_sagemaker()
    else ABSTRA_ROOT_PATH
    if am_i_on_abstra
    else Path.home()
)

if am_i_on_sagemaker():
    _TMP = f"{ROOT_PATH}/tmp"
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = _TMP
    tempfile.tempdir = _TMP
    os.makedirs(_TMP, exist_ok=True)


def get_absolute_dataset_path(dataset_hash: str, relative_path: Path) -> Path:
    """Get the full S3 path for a given dataset and relative path.

    Parameters
    ----------
    dataset_hash : str
        The hash of the dataset on Abstra.
    relative_path : Path
        The relative path to the file within the dataset.

    Returns
    -------
    Path
        The full S3 path to the file.
    """
    manifest = Manifest().load()
    dataset = manifest.datasets[dataset_hash]
    return Path(f"s3://{dataset.bucket}") / dataset.prefix / relative_path


def get_absolute_project_path(relative_path: Path) -> Path:
    """Get the full S3 path given a path to a file stored in the project bucket.

    Parameters
    ----------
    relative_path : Path
        The relative path to the file within the project.

    Returns
    -------
    Path
        The full S3 path to the file.
    """
    manifest = Manifest().load()
    return (
        Path(f"s3://{manifest.project.storage.bucket}")
        / manifest.project.storage.prefix
        / relative_path
    )
