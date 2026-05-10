# Building a signature library

DeconX needs two signature matrices:

- **Known signatures** — one row per cell type that DeconX should subtract
  during initial alignment.
- **Candidate signatures** — one row per cell type that DeconX may discover
  in the residual.

The script `scripts/02_build_signature_library.py` constructs both from a
single-cell reference plus, optionally, an external signature dataset (for
example, GSE176171 for adipose populations).

```{note}
Earlier preprints referred to this script as `02_build_signature_library.py`.
The file is named `02_build_signature_library.py` in the current repository
and behaves identically.
```

## Inputs

- A processed single-cell `.h5ad` or CSV that includes `celltype` labels.
- (Optional) An external CSV with extra cell-type signatures, used to populate
  the candidate library for cell types that are absent from the internal
  single-cell reference (for example, adipocytes from external adipose tissue
  scRNA-seq).

## Outputs

- `data/reference/known_signatures_<dataset>.csv` — known cell types only.
- `data/reference/candidate_signatures_<dataset>.csv` — candidate library.
- `data/reference/unified_signature_matrix.csv` — the union, used only for
  diagnostic comparisons after a discovery run.

## Typical run

```bash
python scripts/02_build_signature_library.py \
    --sc_data data/processed/hgsoc_sc_processed.csv \
    --external_sig data/external/gse176171_adipose_signatures.csv \
    --output_dir data/reference/
```

The script groups the single-cell reference by `celltype`, computes the mean
expression per gene per cell type, and writes the result. External signatures
are concatenated as additional rows in the candidate matrix (they are *not*
added to the known matrix unless their cell types are also present in the
single-cell reference).

## Choosing what is "known" vs "candidate"

A cell type belongs in the **known** matrix if you are confident the
single-cell reference covers it well: it has more than ~50 cells and produces
a stable mean signature.

A cell type belongs in the **candidate** matrix if either:

- it is technically missing from the single-cell reference (e.g., adipocytes
  in single-nucleus data), or
- it is biologically rare and you want DeconX to confirm it as a discovered
  type rather than train it as a known type.

The `--known_celltypes` argument to `run_distillation.py` decides at runtime
which subset of the known matrix to use; you can therefore reuse one signature
library across multiple held-out experiments.
