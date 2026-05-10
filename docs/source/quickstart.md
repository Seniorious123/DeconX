# Quickstart

This walkthrough runs the full DeconX discovery pipeline on a small simulated
PBMC pseudobulk dataset. It should finish in under 30 minutes on a single MIG
slice of an A100.

## 1. Generate a simulated bulk

We hold out two cell types (DCs, Plasmablast) from the single-cell reference
to play the role of "missing":

```bash
python scripts/run_simulation.py \
    --experiment-name DCs_Plasmablast_quickstart \
    --known-celltypes B_cell CD4 CD8 CD14 \
    --unknown-celltypes DCs Plasmablast \
    --num-samples 1000 \
    --seed 42
```

This writes `bulk.csv` (1000 samples × ~6000 genes) and `frac.csv`
(ground-truth proportions, used only for validation) under
`outputs/simulated_data/DCs_Plasmablast_quickstart/`.

## 2. Run iterative discovery

```bash
python scripts/run_distillation.py \
    --sc_data data/raw/preprocessed_pbmc_sc.h5ad \
    --test_bulk outputs/simulated_data/DCs_Plasmablast_quickstart/bulk.csv \
    --test_frac outputs/simulated_data/DCs_Plasmablast_quickstart/frac.csv \
    --known_celltypes "B_cell,CD4,CD8,CD14" \
    --target_celltypes "DCs,Plasmablast" \
    --known_sig_path data/reference/AllsigOfCNS_neurous.csv \
    --output_path outputs/quickstart_run \
    --sample 1000
```

DeconX will:

1. train an autoencoder on the four known cell types (initial alignment);
2. compute the residual (bulk − reconstructed-known) and rank candidate
   signatures by similarity;
3. commit the highest-similarity candidate that also passes the t1 / t2
   thresholds, retrain, and repeat;
4. stop when no candidate passes, or when the residual reduction falls below
   the marker-uniqueness threshold.

## 3. Inspect the discovery output

After the run, the output directory contains:

- `final_discovery_summary.json` — the cell types the model committed to.
- `round_<k>_similarity_ranking.csv` — candidates ranked at round *k*.
- `validation_inputs/round_<k>_gene_level_scale_analysis.csv` — per-gene
  reference vs. virtual-cell expression for the discovered type.
- `final_complete_model.pth` — the full trained model.

For a clean run with the held-out PBMC simulation, expect to see
`Plasmablast` and `DCs` listed in `final_discovery_summary.json`.

## 4. Validate

```bash
python scripts/validate_discovery.py --output_dir outputs/quickstart_run
```

This produces residual-PCA plots, signature similarity heatmaps, and a residual
reduction trend chart.

## Next steps

- Use your own data: see [Input format](user_guide/input_format.md).
- Customise the candidate signature library:
  [Building signatures](user_guide/building_signatures.md).
- Adjust thresholds: [Hyperparameters](user_guide/hyperparameters.md).
