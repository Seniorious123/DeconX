# Input format

DeconX consumes three kinds of inputs. This page documents the expected shape
and column conventions so you can prepare your own data.

## 1. Single-cell reference

Either a CSV or an `.h5ad` file. Pass via `--sc_data`.

**CSV format**

| index (cell barcode) | gene_1 | gene_2 | … | gene_G | celltype |
|----------------------|:------:|:------:|:-:|:------:|:--------:|
| AAACATG-1            |  0.0   |  3.7   | … |  0.0   | CD4      |
| AAACCAA-1            |  0.0   |  0.0   | … |  2.4   | B_cell   |

- Rows = cells (barcodes as the index column)
- Columns = genes (gene symbols as the header)
- A single extra column `celltype` is required and must list one cell-type
  label per cell (string).
- Counts can be raw or normalised; DeconX log-transforms internally.

**H5AD format**

- `adata.X` of shape (n_cells × n_genes), raw or normalised counts
- `adata.var_names` = gene symbols
- `adata.obs["celltype"]` = cell-type labels (string column)

## 2. Bulk RNA-seq

A CSV file. Pass via `--test_bulk`.

| index (sample id) | gene_1 | gene_2 | … | gene_G |
|-------------------|:------:|:------:|:-:|:------:|
| TCGA-04-1331-01A  |  4.2   |  0.0   | … |  9.8   |
| TCGA-09-1666-01A  |  3.1   |  0.0   | … |  6.5   |

- Rows = samples (any unique sample id)
- Columns = genes (must include the same gene symbols you want to use during
  training; only the **intersection** with the single-cell reference is kept)
- Use TPM, FPKM, or log-normalised counts — what matters is that bulk and
  single-cell are on **comparable scales**. DeconX applies a global 99.95%
  cap to align the dynamic range.

## 3. Reference signature matrices

Two CSV files. Pass via `--known_sig_path` (required) and
`--candidate_sig_path` (optional).

| index (cell type) | gene_1 | gene_2 | … | gene_G |
|-------------------|:------:|:------:|:-:|:------:|
| B_cell            |  0.05  |  3.20  | … |  0.10  |
| CD4               |  0.02  |  0.40  | … |  1.20  |

- Rows = cell types (one signature per row)
- Columns = genes (must overlap with bulk + single-cell gene panels)
- Values = mean expression for that cell type — typically computed by grouping
  the single-cell reference by `celltype` and taking the mean.

The known signature matrix is used during initial alignment; the candidate
signature matrix is used at each iteration to label the residual signal.

## 4. (Optional) Cell-type proportion file for validation

When running a held-out simulation you can pass `--test_frac` to enable
ground-truth comparison plots:

| index (sample id) | celltype_1 | celltype_2 | … | celltype_K |
|-------------------|:----------:|:----------:|:-:|:----------:|
| 0                 |   0.099    |   0.104    | … |   0.193    |

Rows = samples (matching bulk index), columns = cell types, values = proportions
(rows should sum to 1).

## End-to-end checklist

Before launching `run_distillation.py`, verify:

- [ ] Bulk and single-cell share at least 5 000 common genes (`pd.Index.intersection`).
- [ ] `--known_celltypes` is a comma-separated string with the **exact**
      labels appearing in `adata.obs["celltype"]`.
- [ ] The known signature matrix's row index uses the same labels.
- [ ] Output directory does not already contain a partial run (DeconX writes
      `round_<k>_*.csv` files cumulatively).
