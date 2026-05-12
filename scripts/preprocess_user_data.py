"""Generic preprocessing for users bringing their own bulk + sc data.

When to use this script
-----------------------
Use this script if you have **your own bulk and single-cell datasets** (any
tissue, any organism, any missing cell type) and you want the harmonised
inputs that ``run_distillation.py`` expects.

If you instead want to reproduce the HGSOC results in the manuscript, use
``01_preprocess_hgsoc.py`` (which is tied to the GSE217517 directory layout).

What it does
------------
1. Loads the single-cell reference (``.h5ad`` or ``.csv``) and bulk matrix
   (``.csv``). Both formats are documented in
   ``docs/source/user_guide/input_format.md``.
2. Intersects genes between the two, so downstream training operates on a
   common gene panel.
3. Optionally retains a user-supplied marker panel for the held-out missing
   cell type, even if those markers are not highly variable.
4. Selects top-N highly variable genes (HVG, scanpy ``seurat`` flavour) on
   the single-cell side after CPM + log1p normalisation; the union of HVGs
   and marker genes is kept.
5. Writes ``<prefix>_sc_processed.csv`` and ``<prefix>_bulk_processed.csv``
   to the output directory, ready to feed to ``run_distillation.py``.

Example
-------
::

    python scripts/preprocess_user_data.py \\
        --sc_data        data/raw/my_sc.h5ad \\
        --bulk_data      data/raw/my_bulk.csv \\
        --output_dir     data/processed/ \\
        --output_prefix  my_study \\
        --marker_genes   ADIPOQ LEP FABP4 \\
        --n_top_genes    6000 \\
        --celltype_col   celltype
"""

import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def load_sc(path, celltype_col):
    if path.endswith(".h5ad"):
        adata = ad.read_h5ad(path)
        if celltype_col not in adata.obs.columns:
            sys.exit(
                f"Single-cell file is missing obs column '{celltype_col}'. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        return adata
    if path.endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
        if celltype_col not in df.columns:
            sys.exit(
                f"Single-cell CSV is missing column '{celltype_col}'. "
                f"Last columns: {list(df.columns)[-5:]}"
            )
        ct = df[celltype_col].astype(str).values
        expr = df.drop(columns=[celltype_col])
        adata = ad.AnnData(
            X=expr.values.astype(np.float32),
            obs=pd.DataFrame({celltype_col: ct}, index=df.index),
            var=pd.DataFrame(index=expr.columns),
        )
        return adata
    sys.exit(f"Unsupported single-cell extension: {path}")


def load_bulk(path):
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] < 100:
        print(
            f"[warn] bulk has only {df.shape[1]} columns; expecting samples as "
            "rows and genes as columns. If that is reversed, transpose the file."
        )
    return df


def harmonise(adata, bulk_df, n_top_genes, marker_genes, celltype_col):
    common = bulk_df.columns.intersection(adata.var_names)
    print(f"--- Common genes between sc and bulk: {len(common)}")
    if len(common) < 1000:
        print(
            "[warn] fewer than 1000 common genes; check that both files use the "
            "same gene-symbol convention (HGNC vs Ensembl)."
        )
    adata = adata[:, common].copy()
    bulk_df = bulk_df[common].copy()

    print(f"--- Selecting top {n_top_genes} HVGs (sc, seurat) ---")
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    sc.pp.highly_variable_genes(
        adata_norm, n_top_genes=n_top_genes, flavor="seurat", subset=False
    )
    hvg = set(adata_norm.var_names[adata_norm.var["highly_variable"]])

    marker_kept = [g for g in marker_genes if g in adata.var_names]
    marker_missing = sorted(set(marker_genes) - set(marker_kept))
    if marker_missing:
        print(f"[warn] {len(marker_missing)} markers not found: {marker_missing}")
    print(f"--- Retaining marker panel ({len(marker_kept)}/{len(marker_genes)} found)")

    keep = sorted(hvg | set(marker_kept))
    print(f"--- Final gene panel: {len(keep)} (HVG ∪ markers)")
    adata = adata[:, keep].copy()
    bulk_df = bulk_df[keep].copy()
    return adata, bulk_df


def save_processed(adata, bulk_df, out_dir, prefix, celltype_col):
    os.makedirs(out_dir, exist_ok=True)
    sc_df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names, columns=adata.var_names,
    )
    # Downstream CSV loader (read_sc_input in data_manager.py) always looks
    # for the singular column name "celltype"; rename here so any user-side
    # column ("celltypes", "cell_type", ...) lands as the expected name.
    sc_df["celltype"] = adata.obs[celltype_col].values
    sc_path = os.path.join(out_dir, f"{prefix}_sc_processed.csv")
    bulk_path = os.path.join(out_dir, f"{prefix}_bulk_processed.csv")
    sc_df.to_csv(sc_path)
    bulk_df.to_csv(bulk_path)
    print(f"--- Wrote {sc_path}")
    print(f"--- Wrote {bulk_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generic preprocessing for users bringing their own single-cell "
            "reference and bulk matrices. Produces the CSVs that "
            "run_distillation.py expects."
        ),
    )
    parser.add_argument("--sc_data", required=True,
                        help="Single-cell reference (.h5ad or .csv).")
    parser.add_argument("--bulk_data", required=True,
                        help="Bulk matrix CSV (samples × genes).")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write the processed CSVs.")
    parser.add_argument("--output_prefix", default="user",
                        help="Filename stem for the two outputs.")
    parser.add_argument("--marker_genes", nargs="*", default=[],
                        help="Optional marker symbols to retain for the held-out "
                             "missing cell type, even if not highly variable.")
    parser.add_argument("--n_top_genes", type=int, default=6000,
                        help="Number of HVGs to keep (default 6000).")
    parser.add_argument("--celltype_col", default="celltype",
                        help="Name of the cell-type column in the sc reference "
                             "(default 'celltype').")
    args = parser.parse_args()

    print(f"=== preprocess_user_data ===")
    print(f"sc_data       : {args.sc_data}")
    print(f"bulk_data     : {args.bulk_data}")
    print(f"output_dir    : {args.output_dir}")
    print(f"output_prefix : {args.output_prefix}")
    print(f"marker_genes  : {args.marker_genes if args.marker_genes else '(none)'}")
    print(f"n_top_genes   : {args.n_top_genes}")
    print(f"celltype_col  : {args.celltype_col}")

    adata = load_sc(args.sc_data, args.celltype_col)
    bulk_df = load_bulk(args.bulk_data)
    print(f"--- sc : {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"--- bulk: {bulk_df.shape[0]} samples × {bulk_df.shape[1]} genes")

    adata_f, bulk_f = harmonise(
        adata, bulk_df,
        n_top_genes=args.n_top_genes,
        marker_genes=args.marker_genes,
        celltype_col=args.celltype_col,
    )
    save_processed(adata_f, bulk_f, args.output_dir, args.output_prefix,
                   args.celltype_col)
    print("--- Done.")


if __name__ == "__main__":
    main()
