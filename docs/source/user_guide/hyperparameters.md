# Hyperparameters

DeconX exposes a small number of parameters that influence the discovery
behaviour and runtime. This page lists the defaults, their meanings, and
guidance on when to change them.

## Training

| Parameter        | Default | Where it lives                | Effect |
|------------------|:-------:|--------------------------------|--------|
| `EPOCHS`         | 200     | `scripts/run_distillation.py` | Initial alignment epochs. Reduce for faster smoke tests; increase if `round_0_total_loss.png` has not plateaued. |
| `BATCH_SIZE`     | 256     | `scripts/run_distillation.py` | Pseudobulk batch size. Lower if you hit GPU OOM. |
| `LEARNING_RATE`  | 1e-3    | `scripts/run_distillation.py` | Adam learning rate. |
| `frac_lambda`    | 1000    | `src/distiller/decon/trainer.py` | Weight on the fraction-prediction loss. The dominant term in the composite loss. |
| `sig_lambda`     | 1.0     | `src/distiller/decon/trainer.py` | Weight on the signature-preservation loss (keeps known-cell-type signatures stable across iterations). |

## Residual analysis

| Parameter        | Default | Effect |
|------------------|:-------:|--------|
| `TOP_PCA_FRAC`   | 0.10    | Fraction of high-PCA-loading genes used to build the residual signature. Increasing to 0.20–0.30 can help when the residual is weak; decreasing to 0.05 sharpens specificity but risks missing genuine signals. |
| Number of model genes | 6 030 | Set by `01_preprocess_hgsoc.py` (HVG selection, default `--n_top_genes 6000`, marker panel from `configs/hgsoc_markers.txt`, noise panel from `configs/hgsoc_noise_genes.txt`). Edit the config files to change the gene panels without touching the script. |

We have validated DeconX with `TOP_PCA_FRAC` ∈ {0.05, 0.10, 0.15, 0.20, 0.30}
on the simulated PBMC benchmark (n = 3 000); all values recovered the held-out
{DCs, Plasmablast} pair, demonstrating robustness in this range.

## Stopping criteria

See the dedicated page on [Stopping criteria](stopping_criteria.md). The two
thresholds are:

| Parameter | Default       | Meaning |
|-----------|:-------------:|---------|
| **t1**    | 0.5 – 0.7     | Minimum cosine similarity between the residual signature and the best-matching candidate. |
| **t2**    | 0.4           | Minimum marker-uniqueness score (the candidate must contribute genes that are *not* already explained by the existing model). |

A round is committed only if **both** t1 and t2 are satisfied; otherwise the
round is aborted and DeconX exits.

## Hardware

| Resource          | Tested value                                 |
|-------------------|----------------------------------------------|
| GPU               | NVIDIA A100-SXM4-40GB MIG 1 g.5 gb (5 GB)    |
| CPU cores         | 4                                            |
| RAM               | 150 GB                                       |
| Peak GPU usage    | 215 – 574 MB across n = 100 … 5 000 samples |

## Runtime expectations

For pseudobulk simulations forced to two rounds:

| n samples | Wall clock |
|----------:|-----------:|
| 100       | 24 min     |
| 1 000     | 49 min     |
| 3 000     | 78 min     |
| 5 000     | 110 min    |

For real HGSOC bulk (24 samples), DeconX commits four rounds and runs in
**~29 min** total (initial training 4 min, rounds 0–3 take 4–8 min each).
