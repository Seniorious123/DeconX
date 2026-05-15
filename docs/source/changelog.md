# Changelog

## Unreleased

- Externalised the HGSOC preprocessing marker / noise panels to
  `configs/hgsoc_markers.txt` and `configs/hgsoc_noise_genes.txt`.
  `scripts/01_preprocess_hgsoc.py` no longer hard-codes the panels; CLI
  flags `--marker_genes_file`, `--noise_genes_file`, `--n_top_genes`,
  `--min_cells_per_celltype` let users override defaults without editing
  the script.

## 0.1.0 — initial public release (2026)

- Iterative residual-based cell-type discovery.
- Simulated PBMC and real HGSOC benchmarks.
- Auto-stopping criteria t1 (signature similarity) and t2 (marker uniqueness).
- ReadTheDocs documentation site.
