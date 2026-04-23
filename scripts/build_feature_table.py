"""Merge per-slide JSON files into a single feature table.

Usage
-----
    # TCGA (derives patient_id from TCGA barcodes):
    python scripts/build_feature_table.py --input-dir outputs/tcga_luad_features --dataset tcga-luad

    # MOSAIC (patient_id is pseudonymized using owkin-data-loader's mapping):
    python scripts/build_feature_table.py --input-dir outputs/gigatime_mosaic_nsclc_uker --dataset mosaic-nsclc-uker

Output
------
    INPUT_DIR/features.parquet
    INPUT_DIR/features.csv

For MOSAIC datasets the script also rewrites everything under INPUT_DIR to
replace raw patient identifiers with the pseudonyms produced by
owkin-data-loader:
    INPUT_DIR/slide-level-features/<pseudo_stem>.json  (renamed + uri rewritten)
    INPUT_DIR/maps/<pseudo_stem>/                       (renamed)

One row per slide. Columns:
    slide                       — slide identifier (pseudonymized for MOSAIC)
    patient_id                  — patient identifier (pseudonymized for MOSAIC)
    uri                         — source slide URI (raw IDs replaced for MOSAIC)
    slide_width_px, ...         — metadata
    pdl1_tps, cd8_intratumoral_density, ...  — spatial features
    DAPI, PD-1, CD8, ...        — 21 per-channel mean expressions over tissue
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_TCGA_BARCODE_RE = re.compile(r"^TCGA-\w{2}-\w{4}")
# MOSAIC patient IDs look like ER_L_01, GR_C_05, CH_B_001 (center_TA_number).
_MOSAIC_PATIENT_RE = re.compile(r"^([A-Z]{2}_[A-Z]_\d{2,3})")


def _is_tcga_dataset(dataset: str | None) -> bool:
    """Return True when the dataset is a TCGA preset."""
    return dataset is not None and dataset.lower().startswith("tcga")


def _is_mosaic_dataset(dataset: str | None) -> bool:
    """Return True when the dataset is a MOSAIC preset."""
    return dataset is not None and dataset.lower().startswith("mosaic")


def _extract_mosaic_patient_id(stem: str) -> str | None:
    """Extract a MOSAIC patient_id (e.g. ``CH_B_001``) from a slide stem."""
    m = _MOSAIC_PATIENT_RE.match(stem)
    return m.group(1) if m else None


def _apply_mapping(value, mapping: dict[str, str]):
    """Replace any raw patient_id occurrence in ``value`` with its pseudonym."""
    if not isinstance(value, str):
        return value
    for raw, pseudo in mapping.items():
        if raw in value:
            value = value.replace(raw, pseudo)
    return value


def _pseudonymize_mosaic(df) -> tuple:
    """Pseudonymize patient_id using the same mapping as owkin-data-loader.

    Delegates to ``data_loader.patient_level_data.tools.pseudonymisation_tools
    .pseudonymise_dataframe``, which hashes via ``PseudoAnonymizationService``
    and reuses the existing S3 mapping so that IDs stay consistent with other
    MOSAIC products. Then applies the raw→pseudo mapping to any embedded
    occurrence in ``slide`` and ``uri``.

    Returns the pseudonymized DataFrame and the raw→pseudo mapping, so callers
    can apply the same renaming to files/folders on disk.
    """
    from data_loader.patient_level_data.tools.pseudonymisation_tools import (
        pseudonymise_dataframe,
    )

    raw_ids = df["patient_id"].tolist()
    df = pseudonymise_dataframe(df)
    pseudo_ids = df["patient_id"].tolist()
    mapping = {r: p for r, p in zip(raw_ids, pseudo_ids) if r and p and r != p}

    for col in ("slide", "uri"):
        if col in df.columns:
            df[col] = df[col].map(lambda v: _apply_mapping(v, mapping))
    return df, mapping


def _pseudonymize_outputs_dir(outputs_dir: Path, mapping: dict[str, str]) -> None:
    """Rename per-slide JSONs and map folders on disk using ``mapping``.

    Also rewrites the ``metadata.uri`` field inside each JSON so the file
    no longer references the raw patient identifier. Safe to re-run: entries
    already pseudonymized are skipped.
    """
    if not mapping:
        return

    features_dir = outputs_dir / "slide-level-features"
    if features_dir.is_dir():
        renamed = 0
        for p in sorted(features_dir.glob("*.json")):
            new_stem = _apply_mapping(p.stem, mapping)
            if new_stem == p.stem:
                continue
            new_path = features_dir / f"{new_stem}.json"
            if new_path.exists():
                print(f"  SKIP {p.name} — {new_path.name} already exists", file=sys.stderr)
                continue
            data = json.loads(p.read_text())
            metadata = data.get("metadata") or {}
            if "uri" in metadata:
                metadata["uri"] = _apply_mapping(metadata["uri"], mapping)
            tmp_path = new_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(data, indent=2))
            tmp_path.rename(new_path)
            p.unlink()
            renamed += 1
        print(f"  Renamed {renamed} JSON files in {features_dir}")

    maps_dir = outputs_dir / "maps"
    if maps_dir.is_dir():
        renamed = 0
        for p in sorted(maps_dir.iterdir()):
            if not p.is_dir():
                continue
            new_name = _apply_mapping(p.name, mapping)
            if new_name == p.name:
                continue
            new_path = maps_dir / new_name
            if new_path.exists():
                print(f"  SKIP maps/{p.name} — maps/{new_name} already exists", file=sys.stderr)
                continue
            p.rename(new_path)
            renamed += 1
        print(f"  Renamed {renamed} map folders in {maps_dir}")


def build_table(input_dir: Path, dataset: str | None = None) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required: pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    # Look for JSONs directly in input_dir, or in the slide-level-features/
    # subdirectory produced by extract_features.py.
    json_dir = input_dir
    subdir = input_dir / "slide-level-features"
    if subdir.is_dir() and any(subdir.glob("*.json")):
        json_dir = subdir

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}", file=sys.stderr)
        sys.exit(1)

    tcga = _is_tcga_dataset(dataset)
    mosaic = _is_mosaic_dataset(dataset)
    print(f"Reading {len(json_files)} JSON files… (dataset={dataset or 'auto'})")

    rows = []
    skipped = 0
    for p in json_files:
        try:
            data = json.loads(p.read_text())
        except Exception as exc:
            print(f"  WARNING: could not parse {p.name}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        features = data.get("features", {})
        metadata = data.get("metadata", {})

        if not features:
            print(f"  WARNING: {p.name} has no features — skipped", file=sys.stderr)
            skipped += 1
            continue

        row: dict = {"slide": p.stem}

        if tcga or _TCGA_BARCODE_RE.match(p.stem):
            row["patient_id"] = "-".join(p.stem.split("-")[:3])
        elif mosaic:
            pid = _extract_mosaic_patient_id(p.stem)
            if pid is None:
                print(
                    f"  WARNING: {p.name} does not match MOSAIC pattern — skipped",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            row["patient_id"] = pid

        uri = metadata.get("uri")
        if uri is not None:
            row["uri"] = uri

        row["slide_width_px"] = metadata.get("slide_width_px")
        row["slide_height_px"] = metadata.get("slide_height_px")
        row["n_tiles"] = metadata.get("n_tiles")
        row["processed_at"] = metadata.get("processed_at", "")
        row["elapsed_s"] = data.get("elapsed_s")

        row.update(features)
        row.update(data.get("channels", {}))

        rows.append(row)

    if not rows:
        print("No valid records to write.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    if mosaic and "patient_id" in df.columns:
        print("Pseudonymizing patient IDs via owkin-data-loader…")
        df, mapping = _pseudonymize_mosaic(df)
        _pseudonymize_outputs_dir(input_dir, mapping)

    parquet_path = input_dir / "features.parquet"
    csv_path = input_dir / "features.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"\nWrote {len(df)} slides  ({skipped} skipped)")
    print(f"  → {parquet_path}")
    print(f"  → {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-slide JSONs into a feature table")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/tcga_luad_features"),
        help="Directory containing per-slide JSON files or a slide-level-features/ subdirectory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g. tcga-luad, mosaic-nsclc-uker). "
        "Used to determine patient_id derivation strategy. "
        "TCGA barcodes are auto-detected when omitted.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    build_table(args.input_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
