"""S3 utilities: slide discovery and download."""

import threading
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# File extensions recognised as whole-slide images by OpenSlide
WSI_EXTENSIONS = {".svs", ".ndpi", ".tiff", ".tif", ".mrxs", ".scn", ".vms", ".vmu", ".bif"}


def list_slides(
    bucket: str,
    prefix: str = "",
    extensions: set[str] | None = None,
) -> list[str]:
    """Return S3 URIs for all WSI files under ``s3://bucket/prefix``.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix to filter by (e.g. ``"cohort/slides/"``).
        extensions: Set of lowercase file extensions to match.
            Defaults to :data:`WSI_EXTENSIONS`.

    Returns:
        Sorted list of ``s3://bucket/key`` URIs.
    """
    exts = extensions if extensions is not None else WSI_EXTENSIONS
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    uris: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if Path(key).suffix.lower() in exts:
                uris.append(f"s3://{bucket}/{key}")

    return sorted(uris)


def download_slide(
    uri: str,
    dest_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Download a slide from S3 to a local directory.

    Args:
        uri: S3 URI in the form ``s3://bucket/key``.
        dest_dir: Local directory to write the file into.
        overwrite: If ``False`` (default) and the file already exists, skip
            the download and return the existing path.

    Returns:
        Local path of the downloaded file.
    """
    bucket, key = _parse_uri(uri)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(key).name

    if dest_path.exists() and not overwrite:
        return dest_path

    s3 = boto3.client("s3")
    try:
        size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except ClientError as e:
        raise FileNotFoundError(f"Could not find {uri}: {e}") from e

    _download_with_progress(s3, bucket, key, dest_path, size)
    return dest_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {uri!r}")
    parts = uri[5:].split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Malformed S3 URI: {uri!r}")
    return parts[0], parts[1]


def _download_with_progress(
    s3_client,
    bucket: str,
    key: str,
    dest: Path,
    total_bytes: int,
) -> None:
    """Download with a simple stderr progress counter."""
    downloaded = 0
    lock = threading.Lock()

    def _callback(chunk: int) -> None:
        nonlocal downloaded
        with lock:
            downloaded += chunk
            pct = downloaded / total_bytes * 100
            print(f"\r  {dest.name}: {pct:.1f}%", end="", flush=True)

    s3_client.download_file(bucket, key, str(dest), Callback=_callback)
    print()  # newline after progress line
