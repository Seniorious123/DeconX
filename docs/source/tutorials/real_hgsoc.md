# Tutorial: real HGSOC bulk

This tutorial walks through running DeconX on real high-grade serous ovarian
cancer (HGSOC) bulk RNA-seq, with a single-cell HGSOC reference that is
deliberately incomplete (Plasma cells held out; adipocytes naturally absent
because the reference uses single-nucleus data).

## 1. Preprocess

```bash
python scripts/01_preprocess_hgsoc.py \
    --sc_input data/raw/hgsoc_sc.h5ad \
    --bulk_input data/raw/hgsoc_bulk.tsv \
    --output_dir data/processed/
```

This step:

- subsets to highly-variable genes (default 6 000),
- removes a noise blacklist (`MALAT1`, `MT-CO1`, `NEAT1`),
- aligns gene panels between bulk and single-cell,
- writes `hgsoc_sc_processed.csv` and `hgsoc_bulk_processed.csv`.

## 2. Build the reference

The HGSOC pipeline uses two signature matrices:

- `known_signatures_hgsoc.csv` — sc cell types except plasma (held out).
- `candidate_signatures_gse176171_reduced_no_plasma.csv` — adipocytes,
  mesothelium, macrophage from GSE176171.

```bash
python scripts/02_build_signature_library.py \
    --sc_data data/processed/hgsoc_sc_processed.csv \
    --external_sig data/external/gse176171_subset.csv \
    --output_dir data/reference/
```

## 3. Run DeconX

```bash
python scripts/run_distillation.py \
    --sc_data data/processed/hgsoc_sc_processed.csv \
    --test_bulk data/processed/hgsoc_bulk_processed.csv \
    --known_celltypes "B cells,T cells,Macrophages,Epithelial cells,Endothelial cells,Fibroblasts,NK cells,Monocytes,DC" \
    --target_celltypes "Plasma cells,adipocyte" \
    --known_sig_path data/reference/known_signatures_hgsoc.csv \
    --candidate_sig_path data/reference/candidate_signatures_gse176171_reduced_no_plasma.csv \
    --output_path outputs/hgsoc_discovery
```

Expected wall clock: about 29 min on a 5 GB MIG slice (initial alignment 4 min;
rounds 0–3 between 4 and 8 min each).

## 4. Inspect results

```bash
cat outputs/hgsoc_discovery/final_discovery_summary.json
```

DeconX typically commits 4 rounds:

- Round 0: Plasma cells (highest residual similarity)
- Round 1: adipocyte
- Round 2: macrophage (an additional hit)
- Round 3: mesothelium (an additional hit)

Whether macrophage and mesothelium are committed depends on the exact
candidate library and t1/t2 thresholds; both are well-known stromal
populations in HGSOC and can serve as positive controls.

## 5. Validate

```bash
python scripts/validate_discovery.py --output_dir outputs/hgsoc_discovery
```

Look at:

- `validation_results/round_*_signature_comparison.csv` — committed cell
  types' signature similarity to the candidate library.
- `validation_results/residual_reduction.png` — the residual norm should
  decrease monotonically across committed rounds.
