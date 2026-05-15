# Tutorial: real HGSOC bulk

This tutorial walks through running DeconX on real high-grade serous ovarian
cancer (HGSOC) bulk RNA-seq, with a single-cell HGSOC reference that is
deliberately incomplete (Plasma cells held out; adipocytes naturally absent
because the reference uses single-nucleus data).

## 1. Preprocess

```bash
python scripts/01_preprocess_hgsoc.py \
    --raw_dir data/raw/ \
    --processed_dir data/processed/
```

`data/raw/` is the directory layout used by GEO GSE217517: `GSM*_single_cell_matrix_*.mtx` triplets with paired barcodes / features TSVs, `GSM*_bulk_chunk_ribo_*.tsv` bulk files, and `cell_labels/*_labels.txt` cell-type annotations. To preprocess **your own** (non-HGSOC) data, use `scripts/preprocess_user_data.py` instead.

This step:

- subsets to highly-variable genes (default `--n_top_genes 6000`),
- retains the marker panel from `configs/hgsoc_markers.txt` (17 adipocyte markers) even if not highly variable,
- removes the noise panel from `configs/hgsoc_noise_genes.txt` (`MALAT1`, `MT-CO1`, `MT-CO3`, `NEAT1`),
- aligns gene panels between bulk and single-cell,
- writes `hgsoc_sc_processed.csv` and `hgsoc_bulk_processed.csv`.

Both gene panels are plain text (one gene per line, `#` comments allowed); edit them in place to customise without changing the script. Override with `--marker_genes_file` / `--noise_genes_file` if you need a different panel for a sensitivity test.

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
