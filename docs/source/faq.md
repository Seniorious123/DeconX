# FAQ

## Do I need to specify the target cell types in advance?

**Not strictly.** `--target_celltypes` is required for *validation* — it tells
the script which cell types to compare ground-truth proportions against in a
held-out simulation. In a pure discovery run on real bulk, you can pass any
placeholder string here (the labels are not used by the algorithm itself); the
candidates that DeconX chooses come from `--candidate_sig_path` and are
selected automatically by the t1 / t2 stopping criteria.

We recommend listing your top biological hypotheses in `--target_celltypes` so
that the validation report compares against them, even on real data.

## Where does the candidate signature library come from?

The candidate matrix can be:

- a subset of the single-cell reference itself (when you want DeconX to
  confirm cell types you already know about); or
- an external scRNA-seq signature set (for example GSE176171 for adipose
  populations, when the internal reference is single-nucleus and lacks fat
  cells).

See [Building a signature library](user_guide/building_signatures.md). The
script that generates the candidate matrix lives in
`scripts/02_build_signature_library.py`.

## How do I pick t1 and t2?

For first-time use we recommend the defaults:

- **t1** ∈ [0.5, 0.7] (cosine similarity)
- **t2** = 0.4 (marker uniqueness)

Tune t1 if DeconX stops too early (lower it toward 0.5) or commits noisy
candidates (raise it toward 0.7). t2 rarely needs adjustment. See
[Stopping criteria](user_guide/stopping_criteria.md) for the diagnostics that
help you choose.

## How do I pick the loss weights?

Defaults `frac_lambda = 1000` and `sig_lambda = 1.0` (in
`src/distiller/decon/trainer.py`). They have been left untouched across all
benchmarks reported in the paper. The fraction-loss weight is large because
the fraction labels are bounded in [0, 1] and would otherwise be dominated by
the unbounded reconstruction loss.

A commented-out top-level `LAMDA = 10.0` in some run scripts is a legacy
constant that is **not** consumed by the pipeline; it is safe to ignore.

## How long does a typical run take?

On a single 5 GB MIG slice of an A100:

- 100 simulated samples: 24 min
- 1 000 simulated samples: 49 min
- 3 000 simulated samples: 78 min
- 5 000 simulated samples: 110 min
- Real HGSOC, 24 samples, 4 rounds committed: 29 min

GPU memory peaks well below 1 GB even at 5 000 samples, so any consumer-grade
NVIDIA GPU is sufficient.

## Why does my residual look noisy after the second round?

DeconX subtracts the contribution of every committed cell type before
extracting the residual. After two or three high-quality commits, what remains
is genuinely close to noise — that is the *intended* signal that t1/t2 should
catch and stop. If `round_<k>_similarity_ranking.csv` has no candidate above
t1, DeconX is correctly stopping.

## Can I run DeconX without a GPU?

Yes — set `CUDA_VISIBLE_DEVICES=""` or `DEVICE='cpu'` in the run script. CPU
runs are 5–10× slower but otherwise produce identical results.

## How does DeconX compare to NMF?

NMF (and residual-NMF as in Ivich et al. 2025) requires the user to specify
the number of missing cell types in advance and uses ground-truth proportions
or correlation-based component matching. DeconX iterates without knowing the
number a priori, uses signature-similarity matching that works on real bulk
where ground-truth proportions are unavailable, and uses a deep autoencoder
rather than linear factorisation. In our HGSOC benchmark, residual-NMF found
zero of two held-out cell types (Plasma cells cosine 0.14, adipocyte
component degenerate); DeconX found both.
