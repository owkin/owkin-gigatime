# GigaTIME — Spatial Feature Extraction

This package computes slide-level biomarker features from the stitched prediction maps
produced by GigaTIME. Each feature is a scalar derived from one or more `(H, W)` float32
channel maps (values in {0, 1}).

All spatial features are computed on TCGA-LUAD first; the same pipeline generalises to
other cohorts (MOSAIC-BLCA, etc.) by changing the input maps.

---

## A note on computation methodology

### Pixel-level AND vs. cell-footprint dilation

A naive co-expression score — `sum(channel_A & channel_B) / sum(channel_A)` — is
appropriate only when both markers are expected to occupy overlapping pixel area on the
same cell. Whether this holds depends on cell size and staining pattern:

**Use raw pixel overlap for:**
- Any feature involving CK as one operand (`CK & PD-L1`, `CK & Ki67`, `CK & Caspase3-D`).
  Tumour cells are large (15–25 µm diameter ≈ 30–50 px at 0.5 µm/px), so genuine
  pixel-level overlap is expected when both markers are expressed on the same cell.

**Use cell-footprint dilation for:**
- Co-expression between lymphocytes or macrophages (`CD8 & PD-1`, `CD68 & CD11c`,
  `CD20 & CD3`). These cells are small (8–12 µm ≈ 16–24 px), and the binary prediction
  maps for different channels can peak at slightly different sub-cellular locations.
  Dilating each mask by a disk of radius ~12 px (6 µm, half a lymphocyte) before AND-ing
  recovers same-cell co-expression without instance segmentation.

Both variants are implemented in `coexpression.py`.

### Density normalisation

Raw positive-pixel fractions conflate tissue content with marker expression (a slide with
10% tissue and a slide with 80% tissue are not comparable). All density features are
normalised by the relevant compartment area (in pixels) rather than total slide area. The
CK + DAPI mask defines the tumour compartment; its complement within tissue defines the
stroma.

---

## Feature catalogue

### 1 — PD-L1 Tumour Proportion Score (TPS)

| | |
|---|---|
| **Inputs** | `PD-L1`, `CK`, `DAPI` |
| **Computation** | `sum(PD-L1 & CK) / sum(CK & DAPI)` — pixel overlap, large cells |
| **Output** | Scalar in [0, 1] |
| **Biomedical question** | *Does GigaTIME-derived TPS replicate pathologist IHC scoring, and does it separate OS/PFS curves at the clinical cut-offs (1% and 50%) used for pembrolizumab eligibility?* |

---

### 2 — Intratumoral CD8 Density

| | |
|---|---|
| **Inputs** | `CD8`, `CK`, `DAPI` |
| **Computation** | `sum(CD8 & tumor_mask) / sum(tumor_mask)` |
| **Output** | Scalar in [0, 1] |
| **Biomedical question** | *Does intratumoral CD8 density separate survival curves in TCGA-LUAD independently of stage and PD-L1 TPS?* |

---

### 3 — Immune Exclusion Index

| | |
|---|---|
| **Inputs** | `CD8`, `CK`, `Actin-D` |
| **Computation** | `density(CD8 in stroma) / (density(CD8 in stroma) + density(CD8 in tumour) + ε)` |
| **Output** | Scalar in [0, 1]; high = excluded phenotype |
| **Biomedical question** | *Does the exclusion index correlate with STK11 / KEAP1 mutation status, and does it identify patients unlikely to benefit from pembrolizumab monotherapy?* |

---

### 4 — CD8+PD-1 Exhaustion Fraction

| | |
|---|---|
| **Inputs** | `CD8`, `PD-1` |
| **Computation** | `sum(dilated(CD8) & dilated(PD-1)) / sum(CD8)` — cell-footprint dilation |
| **Output** | Scalar in [0, 1] |
| **Biomedical question** | *In CD8-high tumours, does a high exhaustion fraction predict differential benefit from anti-PD-1 vs. anti-PD-L1 therapy?* |

---

### 5 — Macrophage-to-T-cell Ratio

| | |
|---|---|
| **Inputs** | `CD68`, `CD8`, `DAPI` |
| **Computation** | `sum(CD68 & DAPI) / (sum(CD8 & DAPI) + ε)` |
| **Output** | Scalar ≥ 0 |
| **Biomedical question** | *Does a high CD68/CD8 ratio identify the STK11-like cold tumour subtype, and does it correlate with poor prognosis in immunotherapy-naïve patients?* |

---

### 6 — CD4/CD8 T-helper Ratio (intratumoural)

| | |
|---|---|
| **Inputs** | `CD4`, `CD8`, `CK`, `DAPI` |
| **Computation** | `sum(CD4 & tumor_mask) / (sum(CD8 & tumor_mask) + ε)` |
| **Output** | Scalar ≥ 0 |
| **Biomedical question** | *Does a cytotoxic-skewed (low CD4/CD8) intratumoural T-cell composition independently predict OS in LUAD?* |

---

### 7 — Tumour Proliferation Index (Ki67 TPI)

| | |
|---|---|
| **Inputs** | `Ki67`, `CK`, `DAPI` |
| **Computation** | `sum(Ki67 & CK) / sum(CK & DAPI)` — pixel overlap, large cells |
| **Output** | Scalar in [0, 1] |
| **Biomedical question** | *Does Ki67 TPI correlate with histological grade and KRAS mutation status, and does it interact with immune features to predict response to chemo-IO combinations?* |

---

### 8 — Tumour Apoptosis Index

| | |
|---|---|
| **Inputs** | `Caspase3-D`, `CK`, `DAPI` |
| **Computation** | `sum(Caspase3-D & CK) / sum(CK & DAPI)` — pixel overlap, large cells |
| **Output** | Scalar in [0, 1] |
| **Biomedical question** | *Is active tumour apoptosis elevated in CD8-high tumours (immune-mediated killing), and does it mark tumours with a better response to IO independently of PD-L1 TPS?* |

---

### 9 — Tertiary Lymphoid Structure (TLS) Score

| | |
|---|---|
| **Inputs** | `CD20`, `CD3`, `DAPI` |
| **Computation** | Connected-component analysis on `(CD20 \| CD3) & DAPI`; count components with area ≥ threshold (≈ 100 px radius, ~50 µm cluster), normalised by tissue area |
| **Output** | TLS count per mm² of tissue |
| **Biomedical question** | *Does TLS presence predict benefit from chemo-immunotherapy combinations in EGFR-WT LUAD, and is it enriched in STK11-WT tumours?* |
| **Note** | GigaTIME has no CD21/FDC marker; B+T aggregation is an approximation. |

---

### 10 — Intratumoural Vascular Density

| | |
|---|---|
| **Inputs** | `CD34`, `CK`, `DAPI` |
| **Computation** | `sum(CD34 & tumor_mask) / sum(tumor_mask)` |
| **Output** | Scalar in [0, 1] |
| **Biomedical question** | *Does high intratumoural vascular density co-occur with immune exclusion (VEGF-driven exclusion mechanism), and does the combination identify patients who might benefit from anti-VEGF + IO combinations?* |

---

## Module structure

```
features/
├── compartments.py   # tumour / stroma / background mask derivation from CK + DAPI + Actin-D
├── coexpression.py   # pixel-level AND and cell-footprint dilation co-expression helpers
├── density.py        # per-compartment density normalisation
├── proximity.py      # distance-transform-based spatial proximity scores
├── tls.py            # connected-component TLS detection
└── features.py       # assembles all features into a flat dict for one slide
```
