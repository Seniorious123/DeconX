import os
import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import re
import glob
import scipy.io
from scipy.sparse import csr_matrix

# ==============================================================================
# HGSOC End-to-End Distillation Experiment Runner for REAL Data
# This script orchestrates the entire workflow on a real-world dataset.
# ==============================================================================

# --- Global Configurations ---
RAW_DIR = 'data/raw/'
PROCESSED_DIR = 'data/processed/'
LABEL_DIR = os.path.join(RAW_DIR, 'cell_labels')
N_TOP_GENES = 6000
ADIPOCYTE_MARKERS = [
    'ADIPOQ', 'LEP', 'CEBPA', 'FABP4',  # 原有的经典标记
    'PPARG', 'PLIN1', 'PLIN2', 'LPL',   # 核心脂肪细胞功能
    'CIDEC', 'CD36', 'RETN', 'FASN',    # 脂质代谢和储存
    'ACACA', 'SCD', 'PCK1', 'LIPE', 'GPD1'  # 代谢酶和脂解
]

def filter_genes(adata, bulk_df, n_top_genes, marker_genes):
    print(f"--- Filtering genes: top {n_top_genes} HVGs + up to {len(marker_genes)} markers ---")
    
    # IMPORTANT: Don't change var_names yet, keep original Ensembl IDs for mapping
    # We'll use gene_symbols for filtering but keep the mapping intact
    
    # Pre-filter genes that are not in the bulk data to avoid errors later
    common_genes = bulk_df.columns.intersection(adata.var_names)
    adata = adata[:, common_genes].copy()
    bulk_df = bulk_df[common_genes].copy()
    
    # CRITICAL: Save raw counts before any normalization
    adata.layers['counts'] = adata.X.copy()
    
    # Create a temporary working copy for normalization (only used for HVG calculation)
    print("    Normalizing data for HVG calculation (raw counts will be preserved)...")
    adata_temp = adata.copy()
    sc.pp.normalize_total(adata_temp, target_sum=1e4)
    sc.pp.log1p(adata_temp)
    
    # Calculate highly variable genes on the normalized temporary copy
    sc.pp.highly_variable_genes(adata_temp, n_top_genes=n_top_genes, subset=False, flavor='seurat')
    
    # Extract the list of highly variable genes (these are Ensembl IDs at this point)
    hvg_genes = adata_temp.var_names[adata_temp.var['highly_variable']].tolist()
    
    # Now handle marker genes - they are in gene symbol format but data uses Ensembl IDs
    print("\n    Processing marker genes...")
    
    # Create a mapping from gene symbols to Ensembl IDs
    # At this point, var_names still contains Ensembl IDs
    if 'gene_symbols' in adata.var.columns:
        # Build the mapping: gene_symbol -> Ensembl_ID
        symbol_to_id = {}
        for ensembl_id, gene_symbol in zip(adata.var_names, adata.var['gene_symbols']):
            # Store both uppercase and original case versions to handle case-insensitive matching
            if pd.notna(gene_symbol) and gene_symbol.strip():
                symbol_to_id[gene_symbol.strip()] = ensembl_id
                symbol_to_id[gene_symbol.strip().upper()] = ensembl_id
        
        print(f"    Built mapping dictionary with {len(symbol_to_id)} entries")
        # Debug: print some actual mappings to see the format
        print(f"\n    Sample mappings from dictionary (first 10 entries):")
        for i, (symbol, ensembl) in enumerate(list(symbol_to_id.items())[:10]):
            print(f"      '{symbol}' -> '{ensembl}'")
        
        # Debug: check if our marker genes exist in any form
        print(f"\n    Checking for marker genes in dictionary:")
        for marker in marker_genes[:5]:  # Just check first 5 to avoid too much output
            found = False
            for key in symbol_to_id.keys():
                if marker.upper() in key.upper():
                    print(f"      {marker}: Found as '{key}' -> {symbol_to_id[key]}")
                    found = True
                    break
            if not found:
                print(f"      {marker}: Not found")       


        # Try to map each marker gene
        mapped_markers = []
        found_markers = []
        missing_markers = []
        already_in_hvg = []
        
        for marker in marker_genes:
            mapped_id = None
            
            # First try exact match
            if marker in symbol_to_id:
                mapped_id = symbol_to_id[marker]
            # Then try uppercase match
            elif marker.upper() in symbol_to_id:
                mapped_id = symbol_to_id[marker.upper()]
            # Finally, check if marker is already an Ensembl ID
            elif marker in adata.var_names:
                mapped_id = marker
            
            if mapped_id:
                mapped_markers.append(mapped_id)
                # Check if this marker was already selected as HVG
                if mapped_id in hvg_genes:
                    # Get the gene symbol for display
                    gene_sym = adata.var.loc[mapped_id, 'gene_symbols'] if mapped_id in adata.var_names else marker
                    already_in_hvg.append(f"{marker} → {gene_sym} ({mapped_id})")
                else:
                    gene_sym = adata.var.loc[mapped_id, 'gene_symbols'] if mapped_id in adata.var_names else marker
                    found_markers.append(f"{marker} → {gene_sym} ({mapped_id})")
            else:
                missing_markers.append(marker)
        
        # Print detailed report about marker genes
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
        # No gene symbols available, just use markers that exist in data as-is
        marker_genes_ids = [g for g in marker_genes if g in adata.var_names]
        missing = set(marker_genes) - set(marker_genes_ids)
        print(f"     Warning: No gene symbol mapping available")
        if missing:
            print(f"     Missing markers: {missing}")

    # --- START: Inserted code to identify and remove noise genes ---
    print("\n      Identifying and removing specified noise genes...")
    # Define the list of noise gene symbols to remove
    noise_genes_to_remove = ['MALAT1', 'MT-CO1', 'MT-CO3', 'NEAT1']
    noise_genes_ids_set = set()
    found_noise_details = []
    missing_noise_symbols = []

    # Use the existing symbol_to_id mapping (created earlier in your function) if available
    if 'symbol_to_id' in locals() and isinstance(symbol_to_id, dict):
        for noise_symbol in noise_genes_to_remove:
            mapped_id = None
            # Try exact symbol match, then uppercase match, then check if the name itself is an Ensembl ID
            if noise_symbol in symbol_to_id:
                mapped_id = symbol_to_id[noise_symbol]
            elif noise_symbol.upper() in symbol_to_id:
                mapped_id = symbol_to_id[noise_symbol.upper()]
            elif noise_symbol in adata.var_names: # Check if the noise symbol might actually be an Ensembl ID already
                 mapped_id = noise_symbol

            # Ensure the mapped ID actually exists in the current adata.var_names (Ensembl IDs)
            if mapped_id and mapped_id in adata.var_names:
                noise_genes_ids_set.add(mapped_id)
                found_noise_details.append(f"{noise_symbol} -> {mapped_id}")
            else:
                missing_noise_symbols.append(noise_symbol)
    else:
        # Fallback if no symbol mapping exists (less likely based on your provided code)
        print("      Warning: symbol_to_id dictionary not found or not a dictionary. Assuming noise genes are Ensembl IDs.")
        for noise_id in noise_genes_to_remove:
            if noise_id in adata.var_names:
                noise_genes_ids_set.add(noise_id)
                found_noise_details.append(f"{noise_id} (as ID)")
            else:
                missing_noise_symbols.append(noise_id)

    print(f"      Identified {len(noise_genes_ids_set)} noise gene IDs to remove:")
    if found_noise_details:
        for detail in found_noise_details: print(f"        - {detail}")
    if missing_noise_symbols:
        print(f"      Warning: Could not find/map the following noise genes: {missing_noise_symbols}")

    # Combine HVGs and Marker genes (both should be sets of Ensembl IDs here)
    intermediate_gene_set = set(hvg_genes).union(set(marker_genes_ids))
    print(f"\n      Gene set size before noise removal: {len(intermediate_gene_set)}")

    # Remove the identified noise gene IDs from the combined set
    # The result 'final_gene_set_ids' now contains only desired Ensembl IDs
    final_gene_set_ids = sorted(list(intermediate_gene_set - noise_genes_ids_set))
    removed_count = len(intermediate_gene_set) - len(final_gene_set_ids)
    print(f"      Removed {removed_count} noise gene(s). Final gene set size (Ensembl IDs): {len(final_gene_set_ids)}")
    # --- END: Inserted code ---

    # Step 1: Filter adata using the cleaned Ensembl ID list ('final_gene_set_ids')
    print(f"\n      Filtering AnnData object to {len(final_gene_set_ids)} genes (using Ensembl IDs)...")
    adata_filtered = adata[:, final_gene_set_ids].copy()

    # Step 2: Filter bulk_df using the SAME cleaned Ensembl ID list
    print(f"      Filtering Bulk DataFrame to {len(final_gene_set_ids)} genes (using Ensembl IDs)...")
    # Check for missing genes in bulk *before* filtering
    missing_in_bulk = list(set(final_gene_set_ids) - set(bulk_df.columns))
    ids_to_use_for_bulk = final_gene_set_ids # Assume all are present initially
    if missing_in_bulk:
        print(f"      \033[93mWarning:\033[0m {len(missing_in_bulk)} gene IDs not in bulk_df columns (e.g., {missing_in_bulk[:5]}).")
        # Use only the IDs that ARE present in bulk for filtering bulk
        ids_to_use_for_bulk = [gid for gid in final_gene_set_ids if gid in bulk_df.columns]
        print(f"               Filtering bulk with {len(ids_to_use_for_bulk)} available genes.")
        # IMPORTANT: We still keep adata_filtered with the slightly larger set for now,
        #            but renaming needs careful handling later. Or adjust adata here too?
        # Let's adjust adata here for guaranteed consistency, minimal change from previous attempt:
        if len(ids_to_use_for_bulk) < len(final_gene_set_ids):
             print(f"               Adjusting AnnData filter to match bulk's {len(ids_to_use_for_bulk)} genes...")
             adata_filtered = adata_filtered[:, ids_to_use_for_bulk].copy()
             # Update final_gene_set_ids itself to reflect the actual final list
             final_gene_set_ids = ids_to_use_for_bulk

    # Now filter bulk_df with the consistent list
    bulk_df_filtered = bulk_df[final_gene_set_ids].copy() # CORRECTED: Use final_gene_set_ids

    # Step 3: Prepare for Renaming - Get final symbols based on filtered adata
    print(f"\n      Preparing gene symbols for renaming {len(final_gene_set_ids)} genes...")
    # Get symbols corresponding *only* to the genes remaining in adata_filtered
    final_symbols = adata_filtered.var['gene_symbols'].tolist()

    # Step 4: Handle potential duplicate symbols
    if len(final_symbols) != len(set(final_symbols)):
        print("      Warning: Duplicate gene symbols found in the final set. Applying 'make_unique'.")
        from anndata.utils import make_index_unique
        unique_final_symbols_index = make_index_unique(pd.Index(final_symbols))
        unique_final_symbols = unique_final_symbols_index.tolist()
        print(f"         Example unique names: {unique_final_symbols[:5]}...")
    else:
        unique_final_symbols = final_symbols

    # Step 5: Apply final unique symbols to BOTH objects
    print(f"      Applying final unique symbols...")
    adata_filtered.var_names = unique_final_symbols
    # Update the 'gene_symbols' column if needed
    if 'gene_symbols' in adata_filtered.var.columns:
        adata_filtered.var['gene_symbols'] = unique_final_symbols

    bulk_df_filtered.columns = unique_final_symbols
    print(f"      Renaming complete.")

    # --- End Corrected Filtering and Renaming Block ---

    # --- Summary Print (using corrected variables) ---
    print(f"\n      Gene filtering summary:")
    # Original common count before any HVG/Marker/Noise filtering
    print(f"        Original common gene count (adata aligned with bulk): {adata.n_vars}")
    # HVG count before noise removal
    print(f"        Initial HVG count (before noise removal): {len(hvg_genes)}")
    # Final count from filtered object
    print(f"        Final gene count (after all filters): {adata_filtered.n_vars}")

    # Calculate added markers based on final set vs HVGs (both cleaned)
    if 'noise_genes_ids_set' not in locals(): noise_genes_ids_set = set() # Safety
    hvg_after_noise = set(hvg_genes) - noise_genes_ids_set
    final_gene_set_after_noise = set(final_gene_set_ids) # Use the final consistent ID list
    genes_added_count = len(final_gene_set_after_noise - hvg_after_noise)

    print(f"        Genes added beyond HVGs (approx, after noise removal): {genes_added_count}")
    print(f"        Filtered AnnData shape: {adata_filtered.shape}")
    print(f"        Filtered bulk data shape: {bulk_df_filtered.shape}")

    # Return the correctly filtered and named objects
    return adata_filtered, bulk_df_filtered

def run_data_processing():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("\n--- 1. Processing and combining single-cell data ---")
    
    # --- New logic to load cell type labels ---
    print("    Reading cell type labels from 'cell_labels' directory...")
    cell_labels = {}
    label_files = glob.glob(os.path.join(LABEL_DIR, '*_labels.txt'))
    if not label_files:
        print("Error: No cell label files found. Please check your raw data path.")
        return
        
    for f in label_files:
        # Case 1: Special handling for the pooled labels file which contains multiple samples
        if 'pooled_labels.txt' in os.path.basename(f):
            try:
                print(f"    Parsing pooled labels file: {os.path.basename(f)}")
                # Read the 3-column format: Pool, Barcode, cellType
                # FIX 1: 增加 dtype={'Pool': str}，强制把 Pool 列当作字符串读，防止 '01132022' 变成数字 1132022
                pool_df = pd.read_csv(f, sep='\t', dtype={'Pool': str})
                
                # Check if required columns exist to ensure it's the correct format
                if {'Pool', 'Barcode', 'cellType'}.issubset(pool_df.columns):
                    # Iterate through each sample group (e.g., 12162021, 01132022) within the file
                    for pool_id, group in pool_df.groupby('Pool'):
                        # FIX 2: 增加 .zfill(8)，确保如果前面的0丢了，能强制补齐成8位 (例如 '1132022' -> '01132022')
                        sid = str(pool_id).strip().zfill(8)
                        
                        # Create dictionary mapping: Barcode -> cellType
                        cell_labels[sid] = dict(zip(group['Barcode'], group['cellType']))
                        print(f"    Loaded labels from pool for sample ID: {sid}")
                else:
                    print(f"    Warning: pooled_labels.txt missing required columns.")
            except Exception as e:
                print(f"    Warning: Failed to load pooled labels: {e}")
            continue # Skip standard processing for this file

        # Case 2: Standard handling for individual label files (e.g., 2497_labels.txt)
        try:
            # Parse the sample ID from the filename
            sample_id = os.path.basename(f).split('_')[0]
            # Load the labels file (Standard 2-column format)
            labels_df = pd.read_csv(f, sep='\t', index_col=0, header=0)
            # Handle potential empty or malformed files
            if labels_df.shape[1] > 0:
                labels_df.columns = ['celltype']
                cell_labels[sample_id] = labels_df['celltype'].to_dict()
                print(f"    Loaded labels for sample ID: {sample_id}")
        except Exception as e:
            print(f"Warning: Failed to load labels from {f}: {e}")

    # 1.1 Find all single-cell matrix files
    sc_matrix_files = sorted(glob.glob(os.path.join(RAW_DIR, 'GSM*_single_cell_matrix_*.mtx')))
    if not sc_matrix_files:
        print("Error: No single-cell matrix files found. Please check your raw data path.")
        return

    adata_sc_list = []
    for mtx_file in sc_matrix_files:
        print(f"\n    Processing matrix file: {os.path.basename(mtx_file)}")
        
        # 1.2 Parse the filename to extract prefix and sample_id
        # Handle both pooled and non-pooled samples
        match = re.search(r'(GSM\d+)_(?:pooled_)?single_cell_matrix_(\d+)\.mtx$', mtx_file)
        if not match: 
            print(f"    Warning: Could not parse filename, skipping")
            continue
        
        prefix_name = match.group(1)
        sample_id = match.group(2)
        print(f"    Parsed prefix: {prefix_name}, sample_id: {sample_id}")
        
        # 1.3 Construct paths for barcodes and features files
        # Try both pooled and non-pooled naming patterns
        barcodes_patterns = [
            os.path.join(RAW_DIR, f'{prefix_name}_pooled_single_cell_barcodes_{sample_id}.tsv'),
            os.path.join(RAW_DIR, f'{prefix_name}_single_cell_barcodes_{sample_id}.tsv')
        ]
        features_patterns = [
            os.path.join(RAW_DIR, f'{prefix_name}_pooled_single_cell_features_{sample_id}.tsv'),
            os.path.join(RAW_DIR, f'{prefix_name}_single_cell_features_{sample_id}.tsv')
        ]
        
        # Find the actual file paths that exist
        barcodes_path = None
        for path in barcodes_patterns:
            if os.path.exists(path):
                barcodes_path = path
                break
        
        features_path = None
        for path in features_patterns:
            if os.path.exists(path):
                features_path = path
                break
        
        # Check if both required files exist
        if barcodes_path is None:
            print(f"    Warning: Barcodes file not found for sample {sample_id}, skipping")
            continue
        if features_path is None:
            print(f"    Warning: Features file not found for sample {sample_id}, skipping")
            continue
        
        print(f"    Found barcodes at: {os.path.basename(barcodes_path)}")
        print(f"    Found features at: {os.path.basename(features_path)}")
        
        # 1.4 Read the files with explicit parameters
        matrix = scipy.io.mmread(mtx_file).T.tocsr()
        barcodes = pd.read_csv(barcodes_path, header=None, sep='\t')[0].values
        features_df = pd.read_csv(features_path, header=None, sep='\t')
        
        # Check how many columns the features file has and handle accordingly
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
        
        # --- NEW LOGIC: Look up cell types using the loaded labels ---
        if sample_id in cell_labels:
            sample_labels = pd.Series(cell_labels[sample_id])
            # Use the cell barcodes to look up their labels.
            # `reindex` is robust; it will handle cases where barcodes might not match.
            celltype_series = sample_labels.reindex(barcodes)
            
            # --- FILTERING `Unknown` CELLS ---
            # Remove any cells that were not found in the labels file (NaN) or are labeled as 'Unknown'.
            # The name 'unknown' is based on the README you found. You can add more if needed.
            # For pooled samples, it may also include patient IDs.
            valid_cells_mask = celltype_series.notna() & ~celltype_series.isin(['Unknown1', 'Unknown2'])
            
            # Now, filter the matrix, barcodes, and celltype_series
            if valid_cells_mask.sum() == 0:
                print(f"    Warning: All cells for sample {sample_id} were filtered out. Skipping sample.")
                continue

            # Convert the pandas Series mask to a pure numpy boolean array before use.
            # This is a robust way to ensure that scipy.sparse can index the matrix without error.
            valid_cells_mask_np = valid_cells_mask.values

            matrix = matrix[valid_cells_mask_np, :]
            barcodes = barcodes[valid_cells_mask_np]
            celltype_series = celltype_series[valid_cells_mask]
            
            print(f"    Successfully filtered and annotated {valid_cells_mask.sum()} cells for sample {sample_id}")
            
            # 1.5 Create an AnnData object manually
            adata = ad.AnnData(
                X=matrix, 
                obs={'celltype': pd.Categorical(celltype_series)},
                var=pd.DataFrame(index=gene_ids, data={'gene_symbols': gene_symbols})
            )
            adata.obs.index = barcodes
            
        else:
            print(f"    Warning: No cell type labels found for sample ID {sample_id}, skipping.")
            continue
        
        # Store gene_symbols in uns to preserve through concatenation
        adata.uns['gene_id_to_symbol'] = dict(zip(gene_ids, gene_symbols))
        
        adata_sc_list.append(adata)
        print(f"    Successfully loaded sample {sample_id} with shape {adata.shape}")

    if not adata_sc_list:
        print("\n--- No single-cell data with valid labels was processed. Exiting. ---")
        return

    adata_sc_combined = ad.concat(adata_sc_list, join='outer', merge='same')
    
    # Make cell barcodes unique across samples to avoid warnings
    adata_sc_combined.obs_names_make_unique()
    
    # Ensure gene_symbols column exists after concatenation
    if 'gene_symbols' not in adata_sc_combined.var.columns or adata_sc_combined.var['gene_symbols'].equals(adata_sc_combined.var_names):
        print("    Warning: gene_symbols column lost during concatenation, reconstructing from saved mapping...")
        # Reconstruct from the saved mapping in the first sample
        if 'gene_id_to_symbol' in adata_sc_list[0].uns:
            gene_mapping = adata_sc_list[0].uns['gene_id_to_symbol']
            adata_sc_combined.var['gene_symbols'] = [gene_mapping.get(gid, gid) for gid in adata_sc_combined.var_names]
            print(f"    Successfully reconstructed gene_symbols for {len(adata_sc_combined.var_names)} genes")
        else:
            print("    Error: Could not find saved gene mapping, using Ensembl IDs as fallback")
            adata_sc_combined.var['gene_symbols'] = adata_sc_combined.var_names
    
    print(f"    Combined single-cell data shape: {adata_sc_combined.shape}")
    
    # --- 2. Processing and combining bulk data ---
    print("\n--- 2. Processing and combining bulk data ---")
    bulk_files_classic = sorted(glob.glob(os.path.join(RAW_DIR, 'GSM*_bulk_chunk_ribo_*.tsv')))
    if not bulk_files_classic:
        print("Error: No classic bulk files found. Skipping bulk data processing.")
        return
        
    bulk_dfs = [pd.read_csv(f, sep='\t', index_col=0) for f in bulk_files_classic]
    bulk_df_classic = pd.concat(bulk_dfs, axis=1) # Genes as index, Sample IDs as columns
    bulk_df_classic = bulk_df_classic.T # Transpose to 'samples x genes' for filtering
    
    print(f"    Combined classic bulk data shape: {bulk_df_classic.shape}")
    
    # --- 3. Aligning and filtering genes ---
    print("\n--- 3. Aligning genes and filtering ---")
    
    common_genes = adata_sc_combined.var_names.intersection(bulk_df_classic.columns)
    adata_sc_combined = adata_sc_combined[:, common_genes].copy()
    bulk_df_classic = bulk_df_classic.loc[:, common_genes].copy()
    
    # Filter genes based on HVGs and markers
    adata_sc_filtered, bulk_df_filtered_classic = filter_genes(
        adata_sc_combined, 
        bulk_df_classic,
        N_TOP_GENES,
        ADIPOCYTE_MARKERS
    )


    # --- Filter out rare cell types with insufficient samples ---
    print("\n--- Filtering rare cell types ---")
    MIN_CELLS_PER_TYPE = 80

    celltype_counts = adata_sc_filtered.obs['celltype'].value_counts()
    print(f"Cell type distribution before filtering:")
    for ct, count in celltype_counts.items():
        print(f"  {ct}: {count} cells")

    valid_celltypes = celltype_counts[celltype_counts >= MIN_CELLS_PER_TYPE].index.tolist()
    removed_celltypes = celltype_counts[celltype_counts < MIN_CELLS_PER_TYPE].index.tolist()

    print(f"\nFiltering threshold: {MIN_CELLS_PER_TYPE} cells")
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
    
    # --- 4. Saving processed data ---
    print("\n--- 5. Saving processed data to disk ---")
    sc_df = adata_sc_filtered.to_df()
    sc_df['celltype'] = adata_sc_filtered.obs['celltype'].values
    
    # Compress single-cell expression data before saving
    print("\n" + "=" * 80)
    print("Compressing single-cell expression matrix...")
    print("=" * 80)
    
    # Separate celltype column for compression
    celltype_col = sc_df['celltype']
    sc_df_expr = sc_df.drop(columns='celltype')
    
    # Calculate compression statistics before clipping
    print(f"Matrix shape: {sc_df_expr.shape[0]} cells × {sc_df_expr.shape[1]} genes")
    threshold = np.percentile(sc_df_expr.values.flatten(), 99.9)
    print(f"Calculated 99.9th percentile threshold: {threshold:.2f}")
    
    # Count values that will be compressed
    compressed_count = (sc_df_expr.values > threshold).sum()
    total_count = sc_df_expr.size
    compressed_percentage = (compressed_count / total_count) * 100
    print(f"Number of values to be compressed: {compressed_count} ({compressed_percentage:.4f}% of total)")
    
    # Calculate sum before compression
    sum_before = sc_df_expr.values.sum()
    
    # Get top 5 values before compression
    top5_before = np.sort(sc_df_expr.values.flatten())[-5:]
    
    # Perform clipping
    sc_df_expr_clipped = sc_df_expr.clip(upper=threshold)
    
    # Calculate sum after compression
    sum_after = sc_df_expr_clipped.values.sum()
    sum_reduction = ((sum_before - sum_after) / sum_before) * 100
    print(f"Total expression sum reduced from {sum_before:.2f} to {sum_after:.2f} (decreased by {sum_reduction:.2f}%)")
    
    # Get top 5 values after compression
    top5_after = np.sort(sc_df_expr_clipped.values.flatten())[-5:]
    print(f"Top 5 values before compression: {top5_before}")
    print(f"Top 5 values after compression: {top5_after}")
    
    # Reconstruct DataFrame with celltype column
    sc_df_clipped = sc_df_expr_clipped.copy()
    sc_df_clipped['celltype'] = celltype_col
    
    print("=" * 80)
    print("Single-cell compression completed")
    print("=" * 80 + "\n")
    
    # Save the compressed data
    sc_df_clipped.to_csv(os.path.join(PROCESSED_DIR, 'hgsoc_sc_processed.csv'))

    # Compress bulk expression data before saving
    print("\n" + "=" * 80)
    print("Compressing bulk expression matrix...")
    print("=" * 80)
    
    # Calculate compression statistics
    print(f"Matrix shape: {bulk_df_filtered_classic.shape[0]} samples × {bulk_df_filtered_classic.shape[1]} genes")
    threshold_bulk = np.percentile(bulk_df_filtered_classic.values.flatten(), 99.9)
    print(f"Calculated 99.9th percentile threshold: {threshold_bulk:.2f}")
    
    # Count values that will be compressed
    compressed_count_bulk = (bulk_df_filtered_classic.values > threshold_bulk).sum()
    total_count_bulk = bulk_df_filtered_classic.size
    compressed_percentage_bulk = (compressed_count_bulk / total_count_bulk) * 100
    print(f"Number of values to be compressed: {compressed_count_bulk} ({compressed_percentage_bulk:.4f}% of total)")
    
    # Calculate sum before compression
    sum_before_bulk = bulk_df_filtered_classic.values.sum()
    
    # Get top 5 values before compression
    top5_before_bulk = np.sort(bulk_df_filtered_classic.values.flatten())[-5:]
    
    # Perform clipping
    bulk_df_filtered_classic_clipped = bulk_df_filtered_classic.clip(upper=threshold_bulk)
    
    # Calculate sum after compression
    sum_after_bulk = bulk_df_filtered_classic_clipped.values.sum()
    sum_reduction_bulk = ((sum_before_bulk - sum_after_bulk) / sum_before_bulk) * 100
    print(f"Total expression sum reduced from {sum_before_bulk:.2f} to {sum_after_bulk:.2f} (decreased by {sum_reduction_bulk:.2f}%)")
    
    # Get top 5 values after compression
    top5_after_bulk = np.sort(bulk_df_filtered_classic_clipped.values.flatten())[-5:]
    print(f"Top 5 values before compression: {top5_before_bulk}")
    print(f"Top 5 values after compression: {top5_after_bulk}")
    
    print("=" * 80)
    print("Bulk compression completed")
    print("=" * 80 + "\n")
    
    # Save the compressed data
    bulk_df_filtered_classic_clipped.to_csv(os.path.join(PROCESSED_DIR, 'hgsoc_bulk_processed.csv'))
    
    print("\n--- Preprocessing complete! Files saved in 'data/processed/' ---")

if __name__ == "__main__":
    run_data_processing()


# sbatch --job-name=preprocess_hgsoc --output="%x_%j.out" --error="%x_%j.err" --time=2:00:00 --ntasks=1 --cpus-per-task=1 --mem=64G --account=ctb-liyue --wrap="source /home/yiminfan/projects/ctb-liyue/yiminfan/project_yue/tape/bin/activate && cd /home/yiminfan/projects/ctb-liyue/yiminfan/project_yixuan/Distillation_Plasma_preprocess && python scripts/01_preprocess_hgsoc.py"