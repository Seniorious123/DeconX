# Input format

DeconX consumes three required inputs (single-cell reference, bulk matrix,
known-signature matrix) and one optional one (candidate-signature matrix
for missing-cell discovery; an additional held-out proportion matrix is
used only for validation runs). This page specifies the exact shape,
column conventions and value semantics expected by `run_distillation.py`,
so that you can either prepare your data manually or generate the inputs
automatically with [`preprocess_user_data.py`](#preprocessing-your-own-data).

## File-format conventions

These conventions apply to every CSV consumed by DeconX:

- **Header row**: gene symbols (or cell-type labels for signature matrices /
  proportion files).
- **Index column**: cell barcode (single-cell), sample id (bulk and
  proportion), or cell-type label (signature). The index lives in the first
  column and is read with `pd.read_csv(..., index_col=0)`.
- **Gene symbols**: HGNC-style symbols in **upper case**
  (e.g. `IGHG1`, `MZB1`, `CD38`). If your file uses Ensembl IDs
  (`ENSG…`), convert them first; DeconX matches by string equality and
  does not perform symbol-to-Ensembl mapping internally.
- **Decimal separator**: `.` (English locale). Do not use `,` as decimal.
- **Encoding**: UTF-8.
- **Missing values**: not allowed — fill with 0 before saving.
- **Negative values**: not allowed — clip to 0 before saving.

## 1. Single-cell reference (`--sc_data`)

CSV or `.h5ad`. Used to build the autoencoder training signal and to derive
mean-expression signatures.

**CSV format**

| index (cell barcode) | gene_1 | gene_2 | … | gene_G | celltype |
|----------------------|:------:|:------:|:-:|:------:|:--------:|
| AAACATG-1            |  0.0   |  3.7   | … |  0.0   | CD4      |
| AAACCAA-1            |  0.0   |  0.0   | … |  2.4   | B_cell   |

- Rows = cells (barcodes as the index column).
- Columns = genes (HGNC symbols, upper case).
- A single extra column `celltype` (**singular, lower case**) is required
  and must list one cell-type label per cell (string). The CSV loader
  looks for this exact name; `preprocess_user_data.py` accepts
  `--celltype_col` if your source file uses a different name and will
  rename it to `celltype` in the output.
- Counts can be raw, CPM, or log-normalised; DeconX log-transforms
  internally for HVG selection but feeds raw values into the autoencoder.

**H5AD format**

- `adata.X` of shape (n_cells × n_genes), raw or normalised counts.
  Sparse (`csr_matrix`) or dense layouts are both accepted.
- `adata.var_names` = gene symbols (HGNC, upper case).
- `adata.obs["celltypes"]` (**plural**, historical naming in this code path)
  = cell-type labels (string column). The loader renames the column to
  `celltype` (singular) once the matrix has been densified, so all
  downstream code and the CSV format below use the singular form.

## 2. Bulk RNA-seq (`--test_bulk`)

A CSV file. Holds the expression matrix to be deconvolved and analysed
for missing cell types.

| index (sample id) | gene_1 | gene_2 | … | gene_G |
|-------------------|:------:|:------:|:-:|:------:|
| TCGA-04-1331-01A  |  4.2   |  0.0   | … |  9.8   |
| TCGA-09-1666-01A  |  3.1   |  0.0   | … |  6.5   |

- Rows = samples (any unique sample id).
- Columns = genes; **only the intersection with the single-cell
  reference is retained**, so it is safe to ship the bulk with a larger
  gene panel.
- Use TPM, FPKM, or log-normalised counts — what matters is that bulk
  and single-cell are on **comparable scales** (e.g., both log-CPM, or
  both raw counts). DeconX applies a global 99.95 % cap to align the
  dynamic range during training; nothing further is required from you.

## 3. Reference signature matrices

Two CSV files: `--known_sig_path` (required) and `--candidate_sig_path`
(optional but recommended for missing-cell discovery).

| index (cell type) | gene_1 | gene_2 | … | gene_G |
|-------------------|:------:|:------:|:-:|:------:|
| B_cell            |  0.05  |  3.20  | … |  0.10  |
| CD4               |  0.02  |  0.40  | … |  1.20  |

- Rows = cell types (one signature per row).
- Columns = genes (must overlap with bulk + single-cell gene panels).
- Values = mean expression for that cell type — typically computed by
  grouping the single-cell reference by `celltype` and taking the mean
  (raw or log-transformed, but kept consistent with the matrix passed to
  `--sc_data`).

The **known** signature matrix carries the cell types that DeconX should
treat as background; their row labels must match the values passed to
`--known_celltypes`. The **candidate** matrix is the library DeconX
searches when scoring residuals at each round, so include every cell
type you suspect might be missing (rare lineages, technically-lost
populations, tissue-specific stromal cells, etc.).

## 4. (Optional) Cell-type proportion file

When running a held-out simulation you can pass `--test_frac` to enable
ground-truth comparison plots:

| index (sample id) | celltype_1 | celltype_2 | … | celltype_K |
|-------------------|:----------:|:----------:|:-:|:----------:|
| 0                 |   0.099    |   0.104    | … |   0.193    |

Rows = samples (matching bulk index), columns = cell types, values =
proportions (rows should sum to 1). Omit this argument for real-bulk
runs where ground truth is unavailable.

(preprocessing-your-own-data)=
## Preprocessing your own data

If you are bringing your own dataset (any tissue, organism or missing
cell type), use `scripts/preprocess_user_data.py` to generate the
processed CSVs that `run_distillation.py` expects:

```bash
python scripts/preprocess_user_data.py \
    --sc_data        data/raw/my_sc.h5ad \
    --bulk_data      data/raw/my_bulk.csv \
    --output_dir     data/processed/ \
    --output_prefix  my_study \
    --marker_genes   ADIPOQ LEP FABP4 \
    --n_top_genes    6000 \
    --celltype_col   celltype
```

This produces `my_study_sc_processed.csv` and `my_study_bulk_processed.csv`,
both ready to feed to `run_distillation.py`. The `--marker_genes` argument
keeps a user-specified panel for the held-out missing cell type even if
those markers are not highly variable; pass an empty list to rely on HVGs
alone.

The HGSOC paper pipeline lives in `scripts/01_preprocess_hgsoc.py` and is
tied to the GSE217517 raw layout — it is a worked example for paper
reproduction, **not** the entry point for your own data. Its marker and
noise gene panels live in `configs/hgsoc_markers.txt` and
`configs/hgsoc_noise_genes.txt`; both are plain text (one gene per line,
`#` comments allowed) and can be overridden with `--marker_genes_file` /
`--noise_genes_file` without editing the script.

## End-to-end checklist

Before launching `run_distillation.py`, verify:

- [ ] Bulk and single-cell share at least 5 000 common genes
      (`bulk_df.columns.intersection(adata.var_names)`).
- [ ] All gene symbols are upper case HGNC; no Ensembl IDs, no mixed case.
- [ ] `--known_celltypes` is a comma-separated string with the **exact**
      labels appearing in `adata.obs["celltype"]` (case- and
      whitespace-sensitive).
- [ ] The known signature matrix's row index uses the same labels.
- [ ] Every entry in the candidate signature matrix is a real cell type
      with a biological hypothesis — DeconX does not learn signatures
      from scratch, so a row of noise will not become a discovery.
- [ ] Output directory does not already contain a partial run (DeconX
      writes `round_<k>_*.csv` files cumulatively).

## Troubleshooting

**"Single-cell file is missing obs column 'celltype'" (CSV) /
"'celltypes'" (h5ad)**
: The CSV loader expects the singular `celltype` column; the h5ad loader
  expects `adata.obs["celltypes"]` (plural). If your column name differs,
  either rename it or run the file through `preprocess_user_data.py`
  with `--celltype_col <your_column_name>` and use the produced CSV
  directly.

**`KeyError` on a gene during signature building**
: Gene symbol convention mismatch. Convert Ensembl IDs (`ENSG…`) to HGNC
  symbols, or homogenise upper / lower case. The single-cell reference,
  bulk matrix and signature matrices must share the same conventions.

**Fewer than 1 000 common genes between bulk and single-cell**
: `preprocess_user_data.py` warns at this threshold. Usually caused by
  one file using Ensembl IDs and the other using HGNC symbols. Map them
  to a common space (e.g., `pyensembl`, `mygene`) before preprocessing.

**`run_distillation.py` exits at round 0 without committing anything**
: The candidate signature matrix is either empty or contains only cell
  types that strongly overlap the known reference. Add at least one
  hypothesised missing cell type to `--candidate_sig_path`.

**Memory error during HVG selection**
: Reduce `--n_top_genes` (default 6 000) to 3 000, or pre-filter the
  single-cell reference to fewer cells per type. The autoencoder needs
  the gene panel; the cell count is less critical.

## See also

- [Quickstart](../quickstart.md) — full end-to-end run on the bundled
  example.
- [Building signatures](building_signatures.md) — turning a single-cell
  reference into the known / candidate matrices.
- [Hyperparameters](hyperparameters.md) — choosing `--n_top_genes`,
  `--known_celltypes`, `t1`/`t2`, λ, and the PCA top fraction.
- [FAQ](../faq.md) — answers to the four most common usability
  questions reviewers have raised.
