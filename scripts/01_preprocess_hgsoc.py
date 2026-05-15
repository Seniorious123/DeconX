"""HGSOC-specific preprocessing for reproducing the manuscript results.

When to use this script
-----------------------
Use this script **only to reproduce the HGSOC analysis in the DeconX
manuscript** (Hippen et al., Genome Biology 2023; GEO GSE217517). It is tied
to that dataset's file-layout conventions: ``GSM*_single_cell_matrix_*.mtx``
triplets with paired ``barcodes`` / ``features`` TSVs, ``GSM*_bulk_chunk_ribo_*.tsv``
bulk files, and ``cell_labels/*_labels.txt`` cell-type annotation files
(including the special ``pooled_labels.txt`` format).

If you want to preprocess **your own** bulk + scRNA-seq dataset (any tissue,
any organism, any missing cell type), use ``scripts/preprocess_user_data.py``
instead, which is dataset-agnostic.

What it does
------------
1. Reads cell-type labels from ``--label_dir`` (one ``<sample>_labels.txt``
   per sample, plus an optional ``pooled_labels.txt`` for multi-sample
   pools).
2. Loads single-cell matrices and bulk profiles from ``--raw_dir``.
3. Aligns genes between single-cell and bulk on Ensembl IDs, then maps to
   gene symbols.
4. Selects HVGs on the single-cell side and unions them with the marker
   panel loaded from ``--marker_genes_file`` so that missing-cell-type
   markers are kept even when not highly variable.
5. Removes noise genes loaded from ``--noise_genes_file`` (mitochondrial
   transcripts, nuclear ncRNAs).
6. Filters rare cell types below ``--min_cells_per_celltype``.
7. Compresses extreme values (99.9th percentile clip) and writes the
   harmonised ``<prefix>_sc_processed.csv`` / ``<prefix>_bulk_processed.csv``
   to ``--processed_dir``.

Configuration files
-------------------
- ``configs/hgsoc_markers.txt`` (default for ``--marker_genes_file``):
  17 adipocyte markers retained beyond HVG selection. Adipocytes are
  technically missing from the HGSOC scRNA-seq reference and the markers
  preserve the recoverable signal in the residual.
- ``configs/hgsoc_noise_genes.txt`` (default for ``--noise_genes_file``):
  mitochondrial and nuclear-retained ncRNA genes that distort HVG
  selection on the HGSOC samples.

Both files are plain text, one gene symbol per line, ``#``-prefixed lines and
blank lines ignored. Edit them in place to alter the panels without changing
this script.
"""

import argparse
import glob
import os
import re

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
from scipy.sparse import csr_matrix


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MARKER_FILE = os.path.join(REPO_ROOT, "configs", "hgsoc_markers.txt")
DEFAULT_NOISE_FILE = os.path.join(REPO_ROOT, "configs", "hgsoc_noise_genes.txt")


def load_gene_list(path):
    """Load a one-gene-per-line file, ignoring blank lines and ``#`` comments."""
    genes = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            genes.append(line)
    return genes


def filter_genes(adata, bulk_df, n_top_genes, marker_genes, noise_genes):
    print(f"--- Filtering genes: top {n_top_genes} HVGs + up to {len(marker_genes)} markers ---")

    # Keep original Ensembl IDs for mapping; gene_symbols are used for filtering.
    common_genes = bulk_df.columns.intersection(adata.var_names)
    adata = adata[:, common_genes].copy()
    bulk_df = bulk_df[common_genes].copy()

    # Save raw counts before any normalization.
    adata.layers['counts'] = adata.X.copy()

    print("    Normalizing data for HVG calculation (raw counts preserved)...")
    adata_temp = adata.copy()
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)

    sc.pp.highly_variable_genes(adata_temp, n_top_genes=n_top_genes, subset=False, flavor='seurat')
    hvg_genes = adata_temp.var_names[adata_temp.var['highly_variable']].tolist()

    print("\n    Processing marker genes...")

    if 'gene_symbols' in adata.var.columns:
        # Build mapping: gene_symbol -> Ensembl_ID (both original case and uppercase).
        symbol_to_id = {}
        for ensembl_id, gene_symbol in zip(adata.var_names, adata.var['gene_symbols']):
            if pd.notna(gene_symbol) and gene_symbol.strip():
                symbol_to_id[gene_symbol.strip()] = ensembl_id
                symbol_to_id[gene_symbol.strip().upper()] = ensembl_id

        print(f"    Built mapping dictionary with {len(symbol_to_id)} entries")
        print(f"\n    Sample mappings from dictionary (first 10 entries):")
        for i, (symbol, ensembl) in enumerate(list(symbol_to_id.items())[:10]):
            print(f"      '{symbol}' -> '{ensembl}'")

        print(f"\n    Checking for marker genes in dictionary:")
        for marker in marker_genes[:5]:
            found = False
            for key in symbol_to_id.keys():
                if marker.upper() in key.upper():
                    print(f"      {marker}: Found as '{key}' -> {symbol_to_id[key]}")
                    found = True
                    break
            if not found:
                print(f"      {marker}: Not found")

        mapped_markers = []
        found_markers = []
        missing_markers = []
        already_in_hvg = []

        for marker in marker_genes:
            mapped_id = None
            if marker in symbol_to_id:
                mapped_id = symbol_to_id[marker]
            elif marker.upper() in symbol_to_id:
                mapped_id = symbol_to_id[marker.upper()]
            elif marker in adata.var_names:
                mapped_id = marker

            if mapped_id:
                mapped_markers.append(mapped_id)
                gene_sym = adata.var.loc[mapped_id, 'gene_symbols'] if mapped_id in adata.var_names else marker
                if mapped_id in hvg_genes:
                    already_in_hvg.append(f"{marker} → {gene_sym} ({mapped_id})")
                else:
                    found_markers.append(f"{marker} → {gene_sym} ({mapped_id})")
            else:
                missing_markers.append(marker)

        print(f"    Marker gene mapping results:")
        print(f"      Total requested: {len(marker_genes)}")
        print(f"      Successfully mapped: {len(mapped_markers)}")
        print(f"      Already in HVG list: {len(already_in_hvg)}")
        print(f"      New markers added: {len(found_markers)}")
        print(f"      Not found in data: {len(missing_markers)}")

        if already_in_hvg:
            print(f"\n    Markers already in HVG list (no duplication):")
            for marker in already_in_hvg:
                print(f"      ✓ {marker}")

        if found_markers:
            print(f"\n    New markers successfully added:")
            for marker in found_markers:
                print(f"      + {marker}")

        if missing_markers:
            print(f"\n    Markers NOT found in dataset:")
            for marker in missing_markers:
                print(f"      ✗ {marker}")

        marker_genes_ids = mapped_markers
    else:
        marker_genes_ids = [g for g in marker_genes if g in adata.var_names]
        missing = set(marker_genes) - set(marker_genes_ids)
        print(f"     Warning: No gene symbol mapping available")
        if missing:
            print(f"     Missing markers: {missing}")

    # Identify and remove noise genes loaded from the noise-genes config file.
    print(f"\n      Identifying and removing {len(noise_genes)} specified noise genes...")
    noise_genes_ids_set = set()
    found_noise_details = []
    missing_noise_symbols = []

    if 'symbol_to_id' in locals() and isinstance(symbol_to_id, dict):
        for noise_symbol in noise_genes:
            mapped_id = None
            if noise_symbol in symbol_to_id:
                mapped_id = symbol_to_id[noise_symbol]
            elif noise_symbol.upper() in symbol_to_id:
                mapped_id = symbol_to_id[noise_symbol.upper()]
            elif noise_symbol in adata.var_names:
                mapped_id = noise_symbol

            if mapped_id and mapped_id in adata.var_names:
                noise_genes_ids_set.add(mapped_id)
                found_noise_details.append(f"{noise_symbol} -> {mapped_id}")
            else:
                missing_noise_symbols.append(noise_symbol)
    else:
        print("      Warning: symbol_to_id mapping not built; assuming noise genes are Ensembl IDs.")
        for noise_id in noise_genes:
            if noise_id in adata.var_names:
                noise_genes_ids_set.add(noise_id)
                found_noise_details.append(f"{noise_id} (as ID)")
            else:
                missing_noise_symbols.append(noise_id)

    print(f"      Identified {len(noise_genes_ids_set)} noise gene IDs to remove:")
    if found_noise_details:
        for detail in found_noise_details:
            print(f"        - {detail}")
    if missing_noise_symbols:
        print(f"      Warning: Could not find/map the following noise genes: {missing_noise_symbols}")

    intermediate_gene_set = set(hvg_genes).union(set(marker_genes_ids))
    print(f"\n      Gene set size before noise removal: {len(intermediate_gene_set)}")

    final_gene_set_ids = sorted(list(intermediate_gene_set - noise_genes_ids_set))
    removed_count = len(intermediate_gene_set) - len(final_gene_set_ids)
    print(f"      Removed {removed_count} noise gene(s). Final gene set size (Ensembl IDs): {len(final_gene_set_ids)}")

    print(f"\n      Filtering AnnData object to {len(final_gene_set_ids)} genes (using Ensembl IDs)...")
    adata_filtered = adata[:, final_gene_set_ids].copy()

    print(f"      Filtering Bulk DataFrame to {len(final_gene_set_ids)} genes (using Ensembl IDs)...")
    missing_in_bulk = list(set(final_gene_set_ids) - set(bulk_df.columns))
    ids_to_use_for_bulk = final_gene_set_ids
    if missing_in_bulk:
        print(f"      \033[93mWarning:\033[0m {len(missing_in_bulk)} gene IDs not in bulk_df columns (e.g., {missing_in_bulk[:5]}).")
        ids_to_use_for_bulk = [gid for gid in final_gene_set_ids if gid in bulk_df.columns]
        print(f"               Filtering bulk with {len(ids_to_use_for_bulk)} available genes.")
        if len(ids_to_use_for_bulk) < len(final_gene_set_ids):
            print(f"               Adjusting AnnData filter to match bulk's {len(ids_to_use_for_bulk)} genes...")
            adata_filtered = adata_filtered[:, ids_to_use_for_bulk].copy()
            final_gene_set_ids = ids_to_use_for_bulk

    bulk_df_filtered = bulk_df[final_gene_set_ids].copy()

    print(f"\n      Preparing gene symbols for renaming {len(final_gene_set_ids)} genes...")
    final_symbols = adata_filtered.var['gene_symbols'].tolist()

    if len(final_symbols) != len(set(final_symbols)):
        print("      Warning: Duplicate gene symbols found in the final set. Applying 'make_unique'.")
        from anndata.utils import make_index_unique
        unique_final_symbols_index = make_index_unique(pd.Index(final_symbols))
        unique_final_symbols = unique_final_symbols_index.tolist()
        print(f"         Example unique names: {unique_final_symbols[:5]}...")
    else:
        unique_final_symbols = final_symbols

    print(f"      Applying final unique symbols...")
    adata_filtered.var_names = unique_final_symbols
    if 'gene_symbols' in adata_filtered.var.columns:
        adata_filtered.var['gene_symbols'] = unique_final_symbols

    bulk_df_filtered.columns = unique_final_symbols
    print(f"      Renaming complete.")

    print(f"\n      Gene filtering summary:")
    print(f"        Original common gene count (adata aligned with bulk): {adata.n_vars}")
    print(f"        Initial HVG count (before noise removal): {len(hvg_genes)}")
    print(f"        Final gene count (after all filters): {adata_filtered.n_vars}")

    hvg_after_noise = set(hvg_genes) - noise_genes_ids_set
    final_gene_set_after_noise = set(final_gene_set_ids)
    genes_added_count = len(final_gene_set_after_noise - hvg_after_noise)

    print(f"        Genes added beyond HVGs (approx, after noise removal): {genes_added_count}")
    print(f"        Filtered AnnData shape: {adata_filtered.shape}")
    print(f"        Filtered bulk data shape: {bulk_df_filtered.shape}")

    return adata_filtered, bulk_df_filtered


def run_data_processing(args):
    os.makedirs(args.processed_dir, exist_ok=True)

    marker_genes = load_gene_list(args.marker_genes_file)
    noise_genes = load_gene_list(args.noise_genes_file)
    print(f"Loaded {len(marker_genes)} marker genes from {args.marker_genes_file}")
    print(f"Loaded {len(noise_genes)} noise genes from {args.noise_genes_file}")

    print("\n--- 1. Processing and combining single-cell data ---")

    print("    Reading cell type labels from 'cell_labels' directory...")
    cell_labels = {}
    label_files = glob.glob(os.path.join(args.label_dir, '*_labels.txt'))
    if not label_files:
        print(f"Error: No cell label files found under {args.label_dir}. Please check your raw data path.")
        return

    for f in label_files:
        # Special handling for the pooled labels file (multiple samples).
        if 'pooled_labels.txt' in os.path.basename(f):
            try:
                print(f"    Parsing pooled labels file: {os.path.basename(f)}")
                # Force-read Pool column as string so '01132022' is not coerced to 1132022.
                pool_df = pd.read_csv(f, sep='\t', dtype={'Pool': str})

                if {'Pool', 'Barcode', 'cellType'}.issubset(pool_df.columns):
                    for pool_id, group in pool_df.groupby('Pool'):
                        # Pad to 8 chars so trimmed leading zeros are restored ('1132022' -> '01132022').
                        sid = str(pool_id).strip().zfill(8)
                        cell_labels[sid] = dict(zip(group['Barcode'], group['cellType']))
                        print(f"    Loaded labels from pool for sample ID: {sid}")
                else:
                    print(f"    Warning: pooled_labels.txt missing required columns.")
            except Exception as e:
                print(f"    Warning: Failed to load pooled labels: {e}")
            continue

        # Standard individual label file (e.g., 2497_labels.txt).
        try:
            sample_id = os.path.basename(f).split('_')[0]
            labels_df = pd.read_csv(f, sep='\t', index_col=0, header=0)
            if labels_df.shape[1] > 0:
                labels_df.columns = ['celltype']
                cell_labels[sample_id] = labels_df['celltype'].to_dict()
                print(f"    Loaded labels for sample ID: {sample_id}")
        except Exception as e:
            print(f"Warning: Failed to load labels from {f}: {e}")

    sc_matrix_files = sorted(glob.glob(os.path.join(args.raw_dir, 'GSM*_single_cell_matrix_*.mtx')))
    if not sc_matrix_files:
        print(f"Error: No single-cell matrix files found under {args.raw_dir}. Please check your raw data path.")
        return

    adata_sc_list = []
    for mtx_file in sc_matrix_files:
        print(f"\n    Processing matrix file: {os.path.basename(mtx_file)}")

        match = re.search(r'(GSM\d+)_(?:pooled_)?single_cell_matrix_(\d+)\.mtx$', mtx_file)
        if not match:
            print(f"    Warning: Could not parse filename, skipping")
            continue

        prefix_name = match.group(1)
        sample_id = match.group(2)
        print(f"    Parsed prefix: {prefix_name}, sample_id: {sample_id}")

        barcodes_patterns = [
            os.path.join(args.raw_dir, f'{prefix_name}_pooled_single_cell_barcodes_{sample_id}.tsv'),
            os.path.join(args.raw_dir, f'{prefix_name}_single_cell_barcodes_{sample_id}.tsv'),
        ]
        features_patterns = [
            os.path.join(args.raw_dir, f'{prefix_name}_pooled_single_cell_features_{sample_id}.tsv'),
            os.path.join(args.raw_dir, f'{prefix_name}_single_cell_features_{sample_id}.tsv'),
        ]

        barcodes_path = next((p for p in barcodes_patterns if os.path.exists(p)), None)
        features_path = next((p for p in features_patterns if os.path.exists(p)), None)

        if barcodes_path is None:
            print(f"    Warning: Barcodes file not found for sample {sample_id}, skipping")
            continue
        if features_path is None:
            print(f"    Warning: Features file not found for sample {sample_id}, skipping")
            continue

        print(f"    Found barcodes at: {os.path.basename(barcodes_path)}")
        print(f"    Found features at: {os.path.basename(features_path)}")

        matrix = scipy.io.mmread(mtx_file).T.tocsr()
        barcodes = pd.read_csv(barcodes_path, header=None, sep='\t')[0].values
        features_df = pd.read_csv(features_path, header=None, sep='\t')

        print(f"    Features file has {features_df.shape[1]} column(s)")
        if features_df.shape[1] == 1:
            gene_ids = features_df[0].values
            gene_symbols = features_df[0].values
        elif features_df.shape[1] >= 2:
            gene_ids = features_df[0].values
            gene_symbols = features_df[1].values
        else:
            print(f"    Error: Features file format unexpected, skipping")
            continue

        if sample_id in cell_labels:
            sample_labels = pd.Series(cell_labels[sample_id])
            celltype_series = sample_labels.reindex(barcodes)

            valid_cells_mask = celltype_series.notna() & ~celltype_series.isin(['Unknown1', 'Unknown2'])

            if valid_cells_mask.sum() == 0:
                print(f"    Warning: All cells for sample {sample_id} were filtered out. Skipping sample.")
                continue

            valid_cells_mask_np = valid_cells_mask.values

            matrix = matrix[valid_cells_mask_np, :]
            barcodes = barcodes[valid_cells_mask_np]
            celltype_series = celltype_series[valid_cells_mask]

            print(f"    Successfully filtered and annotated {valid_cells_mask.sum()} cells for sample {sample_id}")

            adata = ad.AnnData(
                X=matrix,
                obs={'celltype': pd.Categorical(celltype_series)},
                var=pd.DataFrame(index=gene_ids, data={'gene_symbols': gene_symbols}),
            )
            adata.obs.index = barcodes

        else:
            print(f"    Warning: No cell type labels found for sample ID {sample_id}, skipping.")
            continue

        adata.uns['gene_id_to_symbol'] = dict(zip(gene_ids, gene_symbols))
        adata_sc_list.append(adata)
        print(f"    Successfully loaded sample {sample_id} with shape {adata.shape}")

    if not adata_sc_list:
        print("\n--- No single-cell data with valid labels was processed. Exiting. ---")
        return

    adata_sc_combined = ad.concat(adata_sc_list, join='outer', merge='same')
    adata_sc_combined.obs_names_make_unique()

    if 'gene_symbols' not in adata_sc_combined.var.columns or adata_sc_combined.var['gene_symbols'].equals(adata_sc_combined.var_names):
        print("    Warning: gene_symbols column lost during concatenation, reconstructing from saved mapping...")
        if 'gene_id_to_symbol' in adata_sc_list[0].uns:
            gene_mapping = adata_sc_list[0].uns['gene_id_to_symbol']
            adata_sc_combined.var['gene_symbols'] = [gene_mapping.get(gid, gid) for gid in adata_sc_combined.var_names]
            print(f"    Successfully reconstructed gene_symbols for {len(adata_sc_combined.var_names)} genes")
        else:
            print("    Error: Could not find saved gene mapping, using Ensembl IDs as fallback")
            adata_sc_combined.var['gene_symbols'] = adata_sc_combined.var_names

    print(f"    Combined single-cell data shape: {adata_sc_combined.shape}")

    print("\n--- 2. Processing and combining bulk data ---")
    bulk_files_classic = sorted(glob.glob(os.path.join(args.raw_dir, 'GSM*_bulk_chunk_ribo_*.tsv')))
    if not bulk_files_classic:
        print("Error: No classic bulk files found. Skipping bulk data processing.")
        return

    bulk_dfs = [pd.read_csv(f, sep='\t', index_col=0) for f in bulk_files_classic]
    bulk_df_classic = pd.concat(bulk_dfs, axis=1)
    bulk_df_classic = bulk_df_classic.T  # 'samples x genes'

    print(f"    Combined classic bulk data shape: {bulk_df_classic.shape}")

    print("\n--- 3. Aligning genes and filtering ---")

    common_genes = adata_sc_combined.var_names.intersection(bulk_df_classic.columns)
    adata_sc_combined = adata_sc_combined[:, common_genes].copy()
    bulk_df_classic = bulk_df_classic.loc[:, common_genes].copy()

    adata_sc_filtered, bulk_df_filtered_classic = filter_genes(
        adata_sc_combined,
        bulk_df_classic,
        args.n_top_genes,
        marker_genes,
        noise_genes,
    )

    print("\n--- 4. Filtering rare cell types ---")

    celltype_counts = adata_sc_filtered.obs['celltype'].value_counts()
    print(f"Cell type distribution before filtering:")
    for ct, count in celltype_counts.items():
        print(f"  {ct}: {count} cells")

    valid_celltypes = celltype_counts[celltype_counts >= args.min_cells_per_celltype].index.tolist()
    removed_celltypes = celltype_counts[celltype_counts < args.min_cells_per_celltype].index.tolist()

    print(f"\nFiltering threshold: {args.min_cells_per_celltype} cells")
    print(f"Keeping {len(valid_celltypes)} cell types:")
    for ct in valid_celltypes:
        print(f"  ✓ {ct}: {celltype_counts[ct]} cells")

    if removed_celltypes:
        print(f"\nRemoving {len(removed_celltypes)} cell types with insufficient samples:")
        for ct in removed_celltypes:
            print(f"  ✗ {ct}: {celltype_counts[ct]} cells")

    cells_to_keep = adata_sc_filtered.obs['celltype'].isin(valid_celltypes)
    adata_sc_filtered = adata_sc_filtered[cells_to_keep, :].copy()

    print(f"\nAfter filtering: {adata_sc_filtered.shape[0]} cells retained")
    final_counts = adata_sc_filtered.obs['celltype'].value_counts()
    for ct, count in final_counts.items():
        percentage = (count / len(adata_sc_filtered)) * 100
        print(f"  {ct}: {count} cells ({percentage:.2f}%)")

    print("\n--- 5. Saving processed data to disk ---")
    sc_df = adata_sc_filtered.to_df()
    sc_df['celltype'] = adata_sc_filtered.obs['celltype'].values

    print("\n" + "=" * 80)
    print("Compressing single-cell expression matrix...")
    print("=" * 80)

    celltype_col = sc_df['celltype']
    sc_df_expr = sc_df.drop(columns='celltype')

    print(f"Matrix shape: {sc_df_expr.shape[0]} cells × {sc_df_expr.shape[1]} genes")
    threshold = np.percentile(sc_df_expr.values.flatten(), 99.9)
    print(f"Calculated 99.9th percentile threshold: {threshold:.2f}")

    compressed_count = (sc_df_expr.values > threshold).sum()
    total_count = sc_df_expr.size
    compressed_percentage = (compressed_count / total_count) * 100
    print(f"Number of values to be compressed: {compressed_count} ({compressed_percentage:.4f}% of total)")

    sum_before = sc_df_expr.values.sum()
    top5_before = np.sort(sc_df_expr.values.flatten())[-5:]

    sc_df_expr_clipped = sc_df_expr.clip(upper=threshold)

    sum_after = sc_df_expr_clipped.values.sum()
    sum_reduction = ((sum_before - sum_after) / sum_before) * 100
    print(f"Total expression sum reduced from {sum_before:.2f} to {sum_after:.2f} (decreased by {sum_reduction:.2f}%)")

    top5_after = np.sort(sc_df_expr_clipped.values.flatten())[-5:]
    print(f"Top 5 values before compression: {top5_before}")
    print(f"Top 5 values after compression: {top5_after}")

    sc_df_clipped = sc_df_expr_clipped.copy()
    sc_df_clipped['celltype'] = celltype_col

    print("=" * 80)
    print("Single-cell compression completed")
    print("=" * 80 + "\n")

    sc_df_clipped.to_csv(os.path.join(args.processed_dir, f'{args.output_prefix}_sc_processed.csv'))

    print("\n" + "=" * 80)
    print("Compressing bulk expression matrix...")
    print("=" * 80)

    print(f"Matrix shape: {bulk_df_filtered_classic.shape[0]} samples × {bulk_df_filtered_classic.shape[1]} genes")
    threshold_bulk = np.percentile(bulk_df_filtered_classic.values.flatten(), 99.9)
    print(f"Calculated 99.9th percentile threshold: {threshold_bulk:.2f}")

    compressed_count_bulk = (bulk_df_filtered_classic.values > threshold_bulk).sum()
    total_count_bulk = bulk_df_filtered_classic.size
    compressed_percentage_bulk = (compressed_count_bulk / total_count_bulk) * 100
    print(f"Number of values to be compressed: {compressed_count_bulk} ({compressed_percentage_bulk:.4f}% of total)")

    sum_before_bulk = bulk_df_filtered_classic.values.sum()
    top5_before_bulk = np.sort(bulk_df_filtered_classic.values.flatten())[-5:]

    bulk_df_filtered_classic_clipped = bulk_df_filtered_classic.clip(upper=threshold_bulk)

    sum_after_bulk = bulk_df_filtered_classic_clipped.values.sum()
    sum_reduction_bulk = ((sum_before_bulk - sum_after_bulk) / sum_before_bulk) * 100
    print(f"Total expression sum reduced from {sum_before_bulk:.2f} to {sum_after_bulk:.2f} (decreased by {sum_reduction_bulk:.2f}%)")

    top5_after_bulk = np.sort(bulk_df_filtered_classic_clipped.values.flatten())[-5:]
    print(f"Top 5 values before compression: {top5_before_bulk}")
    print(f"Top 5 values after compression: {top5_after_bulk}")

    print("=" * 80)
    print("Bulk compression completed")
    print("=" * 80 + "\n")

    bulk_df_filtered_classic_clipped.to_csv(os.path.join(args.processed_dir, f'{args.output_prefix}_bulk_processed.csv'))

    print(f"\n--- Preprocessing complete! Files saved in '{args.processed_dir}' ---")


def main():
    parser = argparse.ArgumentParser(
        description="HGSOC-specific preprocessing for reproducing the DeconX manuscript results. "
                    "For preprocessing your own dataset, use scripts/preprocess_user_data.py instead.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--raw_dir', default='data/raw/',
                        help='Directory containing GSM* matrix / barcodes / features files and bulk TSVs.')
    parser.add_argument('--label_dir', default=None,
                        help='Directory containing *_labels.txt cell-type annotation files. '
                             'Default: <raw_dir>/cell_labels.')
    parser.add_argument('--processed_dir', default='data/processed/',
                        help='Output directory for harmonised <prefix>_sc_processed.csv / <prefix>_bulk_processed.csv.')
    parser.add_argument('--output_prefix', default='hgsoc',
                        help='Filename prefix for the processed outputs.')
    parser.add_argument('--marker_genes_file', default=DEFAULT_MARKER_FILE,
                        help='Plain-text file (one gene per line, # comments allowed) of marker genes '
                             'to retain beyond HVG selection. Default panel: adipocyte markers for the '
                             'HGSOC missing-cell-type analysis.')
    parser.add_argument('--noise_genes_file', default=DEFAULT_NOISE_FILE,
                        help='Plain-text file of noise genes to drop after HVG selection.')
    parser.add_argument('--n_top_genes', type=int, default=6000,
                        help='Number of highly variable genes selected on the single-cell side.')
    parser.add_argument('--min_cells_per_celltype', type=int, default=80,
                        help='Discard cell types with fewer than this many cells.')

    args = parser.parse_args()
    if args.label_dir is None:
        args.label_dir = os.path.join(args.raw_dir, 'cell_labels')

    run_data_processing(args)


if __name__ == "__main__":
    main()
