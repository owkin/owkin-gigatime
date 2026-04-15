"""Merge per-slide JSON files into a single feature table with categorical variables.

Usage
-----
    python scripts/build_feature_table.py --input-dir outputs/tcga_luad_features

Output
------
    INPUT_DIR/features.parquet
    INPUT_DIR/features.csv

Categorical variables added
---------------------------
    pdl1_status          negative / low / high
                             <1% | 1–49% | ≥50%  (FDA pembrolizumab TPS thresholds)

    tme_class            Inflamed | Excluded | Desert | Intermediate
                             Inflamed  : CD8 intra >5%  AND PD-L1 >1%
                             Excluded  : exclusion index >60%
                             Desert    : CD8 intra <2%
                             Intermediate: everything else

    cd8_infiltration     low | medium | high
                             <2% | 2–5% | >5%  (approximate tertile for LUAD)

    exhaustion_level     low | high
                             CD8+PD-1 fraction <20% | ≥20%

    myeloid_bias         low | high
                             macrophage/T-cell ratio <2 | ≥2

    proliferation        low | high
                             Ki67 TPI <5% | ≥5%

    apoptosis            low | high
                             apoptosis index <2% | ≥2%

    vascular_density     low | high
                             intratumoural CD34 density <1% | ≥1%
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _pdl1_status(tps: float) -> str:
    if tps >= 0.50:
        return "high"
    if tps >= 0.01:
        return "low"
    return "negative"


def _tme_class(cd8: float, pdl1: float, excl: float) -> str:
    if cd8 > 0.05 and pdl1 > 0.01:
        return "Inflamed"
    if excl > 0.60:
        return "Excluded"
    if cd8 < 0.02:
        return "Desert"
    return "Intermediate"


def _cd8_infiltration(density: float) -> str:
    if density > 0.05:
        return "high"
    if density >= 0.02:
        return "medium"
    return "low"


def _binary(value: float, threshold: float, labels: tuple[str, str] = ("low", "high")) -> str:
    return labels[1] if value >= threshold else labels[0]


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

        # Skip incomplete records.
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

        pdl1   = features["pdl1_tps"]
        cd8    = features["cd8_intratumoral_density"]
        excl   = features["immune_exclusion_index"]
        exh    = features["cd8_pd1_exhaustion_fraction"]
        macro  = features["macrophage_to_tcell_ratio"]
        ki67   = features["ki67_tpi"]
        casp3  = features["apoptosis_index"]
        cd34   = features["vascular_density_intratumoral"]
        cd4cd8 = features.get("cd4_cd8_intratumoral_ratio", float("nan"))

        row: dict = {"slide": p.stem}

        # Metadata
        row["uri"]              = metadata.get("uri", "")
        row["slide_width_px"]   = metadata.get("slide_width_px")
        row["slide_height_px"]  = metadata.get("slide_height_px")
        row["n_tiles"]          = metadata.get("n_tiles")
        row["processed_at"]     = metadata.get("processed_at", "")
        row["elapsed_s"]        = data.get("elapsed_s")

        # Continuous features
        row["pdl1_tps"]                      = pdl1
        row["cd8_intratumoral_density"]      = cd8
        row["immune_exclusion_index"]        = excl
        row["cd8_pd1_exhaustion_fraction"]   = exh
        row["macrophage_to_tcell_ratio"]     = macro
        row["cd4_cd8_intratumoral_ratio"]    = cd4cd8
        row["ki67_tpi"]                      = ki67
        row["apoptosis_index"]               = casp3
        row["vascular_density_intratumoral"] = cd34

        # Categorical derived variables
        row["pdl1_status"]     = _pdl1_status(pdl1)
        row["tme_class"]       = _tme_class(cd8, pdl1, excl)
        row["cd8_infiltration"]= _cd8_infiltration(cd8)
        row["exhaustion_level"]= _binary(exh,   0.20)
        row["myeloid_bias"]    = _binary(macro,  2.00)
        row["proliferation"]   = _binary(ki67,   0.05)
        row["apoptosis"]       = _binary(casp3,  0.02)
        row["vascular_density"]= _binary(cd34,   0.01)

        rows.append(row)

    if not rows:
        print("No valid records to write.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Cast categoricals for memory efficiency and downstream groupby speed.
    cat_cols = [
        "pdl1_status", "tme_class", "cd8_infiltration",
        "exhaustion_level", "myeloid_bias", "proliferation",
        "apoptosis", "vascular_density",
    ]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col])

    parquet_path = input_dir / "features.parquet"
    csv_path     = input_dir / "features.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"\nWrote {len(df)} slides  ({skipped} skipped)")
    print(f"  → {parquet_path}")
    print(f"  → {csv_path}")
    print()

    # Quick summary of categorical distributions.
    print("─" * 40)
    for col in cat_cols:
        counts = df[col].value_counts().sort_index()
        print(f"\n{col}")
        for label, n in counts.items():
            pct = 100 * n / len(df)
            bar = "█" * int(pct / 2)
            print(f"  {label:20s} {n:4d}  ({pct:4.1f}%)  {bar}")
    print("─" * 40)


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
