"""Merge per-slide JSON files into a single feature table.

Usage
-----
    python scripts/build_feature_table.py --input-dir outputs/tcga_luad_features

Output
------
    INPUT_DIR/features.parquet
    INPUT_DIR/features.csv

One row per slide. Columns:
    patient_id, slide           — TCGA patient barcode + full slide stem
    slide_width_px, ...         — metadata
    pdl1_tps, cd8_intratumoral_density, ...  — 9 spatial features
    DAPI, PD-1, CD8, ...        — 21 per-channel mean expressions over tissue
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_table(input_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required: pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {len(json_files)} JSON files…")

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

        required = {
            "pdl1_tps", "cd8_intratumoral_density", "immune_exclusion_index",
            "cd8_pd1_exhaustion_fraction", "macrophage_to_tcell_ratio",
            "ki67_tpi", "apoptosis_index", "vascular_density_intratumoral",
        }
        missing = required - features.keys()
        if missing:
            print(f"  WARNING: {p.name} missing keys {missing} — skipped", file=sys.stderr)
            skipped += 1
            continue

        patient_id = "-".join(p.stem.split("-")[:3])
        row: dict = {"patient_id": patient_id, "slide": p.stem}

        row["slide_width_px"]  = metadata.get("slide_width_px")
        row["slide_height_px"] = metadata.get("slide_height_px")
        row["n_tiles"]         = metadata.get("n_tiles")
        row["processed_at"]    = metadata.get("processed_at", "")
        row["elapsed_s"]       = data.get("elapsed_s")

        row.update(features)
        row.update(data.get("channels", {}))

        rows.append(row)

    if not rows:
        print("No valid records to write.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    parquet_path = input_dir / "features.parquet"
    csv_path     = input_dir / "features.csv"
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
        help="Directory containing per-slide JSON files (default: outputs/tcga_luad_features)",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    build_table(args.input_dir)


if __name__ == "__main__":
    main()
