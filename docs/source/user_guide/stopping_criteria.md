# Stopping criteria

A central design choice in DeconX is **automatic termination**: the algorithm
stops adding cell types when neither of two complementary conditions about the
residual is met. This page explains both, and how to inspect them.

## The two thresholds

| Threshold | Meaning | Default |
|-----------|---------|:-------:|
| **t1**    | Cosine similarity between the residual signature (top-PCA-loading genes) and the best-matching candidate signature. | 0.5 – 0.7 |
| **t2**    | Marker-uniqueness score: the fraction of high-loading genes that are *not* already explained by any committed cell type. | 0.4 |

Each round, DeconX:

1. computes the residual after subtracting all currently-committed cell types;
2. extracts the top `TOP_PCA_FRAC` (default 10%) high-loading genes;
3. ranks every candidate signature by cosine similarity to that gene set;
4. picks the top candidate, computes t1 and t2;
5. **commits** the candidate if both pass; **aborts** the round otherwise.

If the round aborts because of t2 in particular, that means the residual,
while similar to a candidate, mostly overlaps with cell types DeconX has
already explained — adding it would be redundant.

## Where to inspect t1 and t2

- `round_<k>_similarity_ranking.csv` — every candidate's t1 (cosine similarity)
  for round *k*.
- `round_<k>_internal_model_similarity.csv` — pairwise similarity among
  committed cell-type signatures (used by t2).
- `final_discovery_summary.json` — the full chain of committed rounds and the
  reason any subsequent round aborted (`round_<k>_aborted_t1` or
  `round_<k>_aborted_t2`).

## Recommended ranges

We have empirically observed:

- t1 of **0.3–0.5** is too permissive; almost any reference signature passes.
- t1 of **0.5–0.7** balances sensitivity and specificity on PBMC and HGSOC
  benchmarks.
- t1 of **>0.8** is very strict; DeconX may stop after round 0 even when a
  genuine missing cell type is present, because the residual signature is
  noisy.

For t2, the default of 0.4 has not needed adjustment in our benchmarks.

## Manually overriding the stop

If you want DeconX to keep iterating regardless (for example, to inspect the
residual structure beyond what the thresholds permit), pass
`--no-residual-restriction` to `run_distillation.py`. Use this for diagnostics
only — committed rounds beyond the auto-stop are not guaranteed to correspond
to real cell types.
