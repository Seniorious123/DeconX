# Tutorial: simulated PBMC benchmark

This tutorial reproduces the held-out PBMC benchmark used in the DeconX paper.
We simulate pseudobulk samples from a PBMC single-cell reference, hold out
two cell types (DCs, Plasmablast), and check that DeconX recovers them.

## 1. Prepare the single-cell reference

Download a preprocessed PBMC `.h5ad` file (any source — we used the COVID-19
PBMC atlas in our experiments). Place it under `data/raw/`.

```bash
ls data/raw/preprocessed_covid-19_sc.h5ad
```

## 2. Generate a pseudobulk

```bash
python scripts/run_simulation.py \
    --experiment-name DCs_Plasmablast_n3000 \
    --known-celltypes B_cell CD4 CD8 CD14 \
    --unknown-celltypes DCs Plasmablast \
    --alphas 2 2 \
    --alpha-known 2 \
    --num-samples 3000 \
    --seed 42 \
    --cpu 4
```

This produces `bulk.csv` and `frac.csv` under
`outputs/simulated_data/DCs_Plasmablast_n3000/`. The Dirichlet `--alphas` and
`--alpha-known` parameters control how unbalanced the proportions are; the
defaults (2, 2) give a moderate spread.

## 3. Build the reference

```bash
python scripts/02_build_signature_library.py \
    --sc_data data/raw/preprocessed_covid-19_sc.h5ad \
    --output_dir data/reference/
```

This writes `data/reference/AllsigOfCNS_neurous.csv` (the combined PBMC
signature matrix used here as both known and candidate library; in production
use you would split them).

## 4. Run DeconX

```bash
python scripts/run_distillation.py \
    --sc_data data/raw/preprocessed_covid-19_sc.h5ad \
    --test_bulk outputs/simulated_data/DCs_Plasmablast_n3000/bulk.csv \
    --test_frac outputs/simulated_data/DCs_Plasmablast_n3000/frac.csv \
    --known_celltypes "B_cell,CD4,CD8,CD14" \
    --target_celltypes "DCs,Plasmablast" \
    --known_sig_path data/reference/AllsigOfCNS_neurous.csv \
    --output_path outputs/sim_DCs_Plasmablast_n3000 \
    --sample 3000
```

Expected wall clock: about 78 min on a single 5 GB MIG slice.

## 5. Verify the discovery

```bash
cat outputs/sim_DCs_Plasmablast_n3000/final_discovery_summary.json
```

Should show:

```json
{
  "total_discovered": 2,
  "discovered_types": ["Plasmablast", "DCs"],
  "final_cell_types": ["B_cell", "CD4", "CD8", "CD14", "Plasmablast", "DCs"],
  "initial_known_types": 4
}
```

## 6. Generate validation figures

```bash
python scripts/validate_discovery.py \
    --output_dir outputs/sim_DCs_Plasmablast_n3000
```

Outputs:

- residual-PCA plots per round
- per-gene `real_mean` vs `virtual_mean` for the discovered types
- characteristic-gene heatmap showing the marker programs DeconX learned

## What to expect

- The Plasmablast canonical markers (IGHG1, IGKC, MZB1, …) should appear with
  a 200–300 × increase in `virtual_mean` over `real_mean`, confirming that the
  residual carries the held-out signal.
- A control panel of CD4 markers (which were *not* held out) should show a
  ratio close to 1 — confirming that DeconX does not "rediscover" cell types
  already in the reference.
