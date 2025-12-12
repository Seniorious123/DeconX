#!/usr/bin/env python
"""
Builds a highly specific signature library to improve deconvolution discovery rates.

Method:
1. Extracts known type signatures from HGSOC data.
2. Extracts candidate type signatures from GSE176171 data.
3. Performs differential gene expression analysis (DEG) on candidate types
   to select the most discriminatory genes.
4. Uses these specific genes to build the signature matrix.
5. Combines and saves all signatures.
"""

import os
import pandas as pd
import scanpy as sc
import numpy as np
import scipy.io
import traceback
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
GSE_DIR = 'data/external'
HGSOC_FILE = 'data/processed/hgsoc_sc_processed.csv'
REFERENCE_DIR = 'data/reference'

# --- Output files ---
KNOWN_SIG_FILE = os.path.join(REFERENCE_DIR, 'known_signatures_hgsoc.csv')
CANDIDATE_SIG_FILE = os.path.join(REFERENCE_DIR, 'candidate_signatures_gse176171.csv')
COMBINED_SIG_FILE = os.path.join(REFERENCE_DIR, 'combined_signatures.csv')
CANDIDATE_SIG_REDUCED_FILE = os.path.join(REFERENCE_DIR, 'candidate_signatures_gse176171_reduced.csv')
FEATURE_GENES_FILE = os.path.join(REFERENCE_DIR, 'selected_feature_genes.txt')

# --- Candidate type selection rules ---
MIN_CELLS = 100
MAX_CANDIDATES = 5
NUM_DEG = 10  # Number of discriminatory genes to select per type

print("=" * 80)
print("BUILDING HIGHLY SPECIFIC SIGNATURE LIBRARY")
print("=" * 80)

# ========== PART 1: Extract Known Signatures from HGSOC ==========
print("\n[PART 1] Extracting KNOWN cell type signatures from HGSOC...")

hgsoc_data = pd.read_csv(HGSOC_FILE, index_col=0)
target_genes = hgsoc_data.drop(columns='celltype').columns.tolist()
print(f"  Gene set: {len(target_genes)} genes")


# --- Define cell type lists ---
KNOWN_CELLTYPES_HGSOC = [
    'B cells', 'T cells', 'Macrophages', 'Epithelial cells', 
    'Endothelial cells', 'Fibroblasts',
    'NK cells', 'Monocytes', 'DC'
]
CANDIDATE_CELLTYPES_HGSOC = [
    'Plasma cells'
]
print(f"  Defined {len(KNOWN_CELLTYPES_HGSOC)} known types: {KNOWN_CELLTYPES_HGSOC}")
print(f"  Defined {len(CANDIDATE_CELLTYPES_HGSOC)} HGSOC-derived candidate types: {CANDIDATE_CELLTYPES_HGSOC}")
# --- End of definition ---



known_signatures = {}
for celltype in KNOWN_CELLTYPES_HGSOC:
    if celltype not in hgsoc_data['celltype'].unique():
        print(f"  ! Warning: Known type '{celltype}' not found in {HGSOC_FILE}. Skipping.")
        continue
    celltype_data = hgsoc_data[hgsoc_data['celltype'] == celltype].drop(columns='celltype')
    mean_expr = celltype_data.mean(axis=0)
    known_signatures[celltype] = mean_expr
    print(f"  ✓ {celltype}: {len(celltype_data)} cells")

known_sig_df = pd.DataFrame(known_signatures).T

# --- ADDED [PART 1b] ---
print("\n[PART 1b] Extracting HGSOC-derived CANDIDATE signatures...")
hgsoc_candidate_signatures = {}
for celltype in CANDIDATE_CELLTYPES_HGSOC:
    if celltype not in hgsoc_data['celltype'].unique():
        print(f"  ! Warning: HGSOC candidate '{celltype}' not found in {HGSOC_FILE}. Skipping.")
        continue
    celltype_data = hgsoc_data[hgsoc_data['celltype'] == celltype].drop(columns='celltype')
    mean_expr = celltype_data.mean(axis=0)
    hgsoc_candidate_signatures[celltype] = mean_expr
    print(f"  ✓ {celltype}: {len(celltype_data)} cells")

hgsoc_candidate_sig_df = pd.DataFrame(hgsoc_candidate_signatures).T
print(f"\n  Total HGSOC-derived CANDIDATE signatures: {len(hgsoc_candidate_sig_df)}")
# --- END ADDED SECTION ---

print(f"\n  Total KNOWN signatures: {len(known_sig_df)}")

# ========== PART 2: Extract and Filter Candidate Signatures from GSE176171 ==========
print("\n[PART 2] Extracting CANDIDATE cell type signatures from GSE176171...")

try:
    # Load metadata
    metadata_file = os.path.join(GSE_DIR, 'GSE176171_cell_metadata.tsv')
    print(f"  Loading metadata: {metadata_file}")
    metadata = pd.read_csv(metadata_file, sep='\t', index_col=0)
    
    # Identify and select candidate types
    celltype_counts = metadata['cell_type__custom'].value_counts()
    print(f"\n  Available cell types (showing all):")
    for ct, count in celltype_counts.items():
        print(f"    {ct}: {count} cells")
    

    print(f"\n  Selecting candidates (min {MIN_CELLS} cells, max {MAX_CANDIDATES} types):")
    candidate_types_priority = ['adipocyte', 'mesothelium', 'macrophage']
    selected_candidates = []
    for ct in candidate_types_priority:
        if ct in celltype_counts and celltype_counts[ct] >= MIN_CELLS:
            selected_candidates.append(ct)

    if not selected_candidates:
        raise ValueError("No valid candidates found from predefined list")

    # Limit to MAX_CANDIDATES if the list is too long
    candidate_types = selected_candidates[:MAX_CANDIDATES]

    for ct in candidate_types:
        print(f"  ✓ {ct} ({celltype_counts[ct]} cells)")

    # Load expression data and create AnnData object
    print("\n  Loading expression data...")
    matrix = scipy.io.mmread(os.path.join(GSE_DIR, 'GSE176171_Hs10X.counts.mtx')).T.tocsr()
    barcodes = pd.read_csv(os.path.join(GSE_DIR, 'GSE176171_Hs10X.counts.barcodes.tsv'), 
                        header=None, sep='\t')[0].values
    features = pd.read_csv(os.path.join(GSE_DIR, 'GSE176171_Hs10X.counts.features.tsv'), 
                        header=None, sep='\t')
    gene_names = features[1].values if features.shape[1] >= 2 else features[0].values

    adata = sc.AnnData(X=matrix, 
                    obs=pd.DataFrame(index=barcodes),
                    var=pd.DataFrame(index=gene_names))
    print(f"  ✓ Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # Filter AnnData to only include candidate cells for DEG analysis
    print(f"  Filtering AnnData to only include candidate cells for DEG analysis...")
    mask = metadata['cell_type__custom'].isin(candidate_types)
    adata_filtered = adata[adata.obs_names.isin(metadata[mask].index)].copy()
    adata_filtered.obs['cell_type__custom'] = metadata.loc[adata_filtered.obs_names, 'cell_type__custom'].astype('category')
    print(f"  ✓ Filtered AnnData shape: {adata_filtered.shape}")

    # Normalize and log-transform for DEG analysis
    print("\n  Performing differential gene expression analysis for discriminatory gene selection...")
    sc.pp.normalize_total(adata_filtered, target_sum=1e4)
    sc.pp.log1p(adata_filtered)

    # Rank genes for each cell type to find the most specific ones
    sc.tl.rank_genes_groups(adata_filtered, 'cell_type__custom', method='t-test')

    # Get a union of the top N genes for all candidates
    deg_genes = set()
    for ct in candidate_types:
        try:
            top_genes = adata_filtered.uns['rank_genes_groups']['names'][ct][:NUM_DEG].tolist()
            deg_genes.update(top_genes)
        except KeyError as e:
            print(f"  Warning: No rank_genes_groups data for group '{ct}'. Skipping.")

    deg_genes = list(deg_genes)
    print(f"  ✓ Found a total of {len(deg_genes)} discriminatory genes.")

    with open(FEATURE_GENES_FILE, 'w') as f:
        for gene in deg_genes:
            f.write(f"{gene}\n")
    print(f"  ✓ Feature genes list saved to: {FEATURE_GENES_FILE}")

    
 

    # Build signatures: Full-dimensional for alignment, reduced for focused comparison
    print("\n  Building candidate signatures...")
    candidate_signatures = {}
    candidate_signatures_reduced = {}

    # Keep the full adata object for building complete signatures
    # We'll build TWO versions:
    # 1. Full signature with all genes (for HGSOC alignment)
    # 2. Reduced signature with only DEG genes (for focused comparison)

    for celltype in candidate_types:
        mask = metadata['cell_type__custom'] == celltype
        matched_cells = metadata[mask].index
        
        # Build FULL signature using ALL genes from original data
        celltype_adata_full = adata[adata.obs_names.isin(matched_cells)]
        
        # Calculate mean expression for FULL signature
        if hasattr(celltype_adata_full.X, 'toarray'):
            mean_expr_full = celltype_adata_full.X.toarray().mean(axis=0).flatten()
        else:
            mean_expr_full = np.array(celltype_adata_full.X.mean(axis=0)).flatten()
        
        full_signature = pd.Series(mean_expr_full, index=celltype_adata_full.var_names)
        
        # Align full signature to HGSOC gene set
        common_genes = full_signature.index.intersection(target_genes)
        aligned_sig = pd.Series(0.0, index=target_genes)
        aligned_sig[common_genes] = full_signature[common_genes]
        candidate_signatures[celltype] = aligned_sig
        
        # Build REDUCED signature (only DEG genes) for focused comparison
        deg_genes_available = [g for g in deg_genes if g in celltype_adata_full.var_names]
        if deg_genes_available:
            celltype_adata_deg = celltype_adata_full[:, deg_genes_available]
            if hasattr(celltype_adata_deg.X, 'toarray'):
                mean_expr_deg = celltype_adata_deg.X.toarray().mean(axis=0).flatten()
            else:
                mean_expr_deg = np.array(celltype_adata_deg.X.mean(axis=0)).flatten()
            reduced_signature = pd.Series(mean_expr_deg, index=deg_genes_available)
            candidate_signatures_reduced[celltype] = reduced_signature
        
        print(f"  ✓ {celltype}: full signature aligned to {len(common_genes)} genes, reduced signature has {len(deg_genes_available)} DEG genes")
    
    candidate_sig_df = pd.DataFrame(candidate_signatures).T
    candidate_sig_df_reduced = pd.DataFrame(candidate_signatures_reduced).T
    print(f"\n  Total CANDIDATE signatures: {len(candidate_sig_df)}")
    print(f"  Reduced signature dimensions: {candidate_sig_df_reduced.shape[1]} genes")

except Exception as e:
    print(f"\n  ERROR: {e}")
    traceback.print_exc()
    candidate_sig_df = pd.DataFrame()

# ========== PART 3: Save All Signature Files ==========
print("\n[PART 3] Saving signature files...")
os.makedirs(REFERENCE_DIR, exist_ok=True)

# Compress known signatures before saving
print("\n" + "=" * 80)
print("Compressing known cell type signature matrix...")
print("=" * 80)

print(f"Matrix shape: {known_sig_df.shape[0]} cell types × {known_sig_df.shape[1]} genes")
threshold_known = np.percentile(known_sig_df.values.flatten(), 99.9)
print(f"Calculated 99.9th percentile threshold: {threshold_known:.2f}")

compressed_count_known = (known_sig_df.values > threshold_known).sum()
total_count_known = known_sig_df.size
compressed_percentage_known = (compressed_count_known / total_count_known) * 100
print(f"Number of values to be compressed: {compressed_count_known} ({compressed_percentage_known:.4f}% of total)")

sum_before_known = known_sig_df.values.sum()
top5_before_known = np.sort(known_sig_df.values.flatten())[-5:]

known_sig_df_clipped = known_sig_df.clip(upper=threshold_known)

sum_after_known = known_sig_df_clipped.values.sum()
sum_reduction_known = ((sum_before_known - sum_after_known) / sum_before_known) * 100
print(f"Total expression sum reduced from {sum_before_known:.2f} to {sum_after_known:.2f} (decreased by {sum_reduction_known:.2f}%)")

top5_after_known = np.sort(known_sig_df_clipped.values.flatten())[-5:]
print(f"Top 5 values before compression: {top5_before_known}")
print(f"Top 5 values after compression: {top5_after_known}")

print("=" * 80)
print("Known signatures compression completed")
print("=" * 80 + "\n")

# Update the DataFrame to use compressed version
known_sig_df = known_sig_df_clipped

known_sig_df.to_csv(KNOWN_SIG_FILE)
print(f"  ✓ Known signatures ({len(known_sig_df)} types) -> {KNOWN_SIG_FILE}")

if not candidate_sig_df.empty:

    # --- ADDED: Check for HGSOC candidates and merge ---
    if 'hgsoc_candidate_sig_df' in locals() and not hgsoc_candidate_sig_df.empty:
        print("\n  Merging GSE176171 candidates with HGSOC candidates (Plasma cells)...")
        candidate_sig_df = pd.concat([candidate_sig_df, hgsoc_candidate_sig_df], axis=0)
        print(f"  Final candidate list: {list(candidate_sig_df.index)}")
    else:
        print("\n  Warning: HGSOC candidate signatures (Plasma cells) not found, proceeding with GSE176171 only.")
    # --- END ADDED SECTION ---

    # Compress candidate signatures before saving
    print("\n" + "=" * 80)
    print("Compressing candidate cell type signature matrix (full)...")
    print("=" * 80)
    
    print(f"Matrix shape: {candidate_sig_df.shape[0]} cell types × {candidate_sig_df.shape[1]} genes")
    threshold_candidate = np.percentile(candidate_sig_df.values.flatten(), 99.9)
    print(f"Calculated 99.9th percentile threshold: {threshold_candidate:.2f}")
    
    compressed_count_candidate = (candidate_sig_df.values > threshold_candidate).sum()
    total_count_candidate = candidate_sig_df.size
    compressed_percentage_candidate = (compressed_count_candidate / total_count_candidate) * 100
    print(f"Number of values to be compressed: {compressed_count_candidate} ({compressed_percentage_candidate:.4f}% of total)")
    
    sum_before_candidate = candidate_sig_df.values.sum()
    top5_before_candidate = np.sort(candidate_sig_df.values.flatten())[-5:]
    
    candidate_sig_df_clipped = candidate_sig_df.clip(upper=threshold_candidate)
    
    sum_after_candidate = candidate_sig_df_clipped.values.sum()
    sum_reduction_candidate = ((sum_before_candidate - sum_after_candidate) / sum_before_candidate) * 100
    print(f"Total expression sum reduced from {sum_before_candidate:.2f} to {sum_after_candidate:.2f} (decreased by {sum_reduction_candidate:.2f}%)")
    
    top5_after_candidate = np.sort(candidate_sig_df_clipped.values.flatten())[-5:]
    print(f"Top 5 values before compression: {top5_before_candidate}")
    print(f"Top 5 values after compression: {top5_after_candidate}")
    
    print("=" * 80)
    print("Candidate signatures compression completed")
    print("=" * 80 + "\n")
    
    # Update to use compressed version
    candidate_sig_df = candidate_sig_df_clipped
    
    candidate_sig_df.to_csv(CANDIDATE_SIG_FILE)
    print(f"  ✓ Candidate signatures ({len(candidate_sig_df)} types) -> {CANDIDATE_SIG_FILE}")
    
    # Compress reduced candidate signatures before saving
    print("\n" + "=" * 80)
    print("Compressing candidate cell type signature matrix (reduced)...")
    print("=" * 80)
    
    print(f"Matrix shape: {candidate_sig_df_reduced.shape[0]} cell types × {candidate_sig_df_reduced.shape[1]} genes")
    threshold_reduced = np.percentile(candidate_sig_df_reduced.values.flatten(), 99.9)
    print(f"Calculated 99.9th percentile threshold: {threshold_reduced:.2f}")
    
    compressed_count_reduced = (candidate_sig_df_reduced.values > threshold_reduced).sum()
    total_count_reduced = candidate_sig_df_reduced.size
    compressed_percentage_reduced = (compressed_count_reduced / total_count_reduced) * 100
    print(f"Number of values to be compressed: {compressed_count_reduced} ({compressed_percentage_reduced:.4f}% of total)")
    
    sum_before_reduced = candidate_sig_df_reduced.values.sum()
    top5_before_reduced = np.sort(candidate_sig_df_reduced.values.flatten())[-5:]
    
    candidate_sig_df_reduced_clipped = candidate_sig_df_reduced.clip(upper=threshold_reduced)
    
    sum_after_reduced = candidate_sig_df_reduced_clipped.values.sum()
    sum_reduction_reduced = ((sum_before_reduced - sum_after_reduced) / sum_before_reduced) * 100
    print(f"Total expression sum reduced from {sum_before_reduced:.2f} to {sum_after_reduced:.2f} (decreased by {sum_reduction_reduced:.2f}%)")
    
    top5_after_reduced = np.sort(candidate_sig_df_reduced_clipped.values.flatten())[-5:]
    print(f"Top 5 values before compression: {top5_before_reduced}")
    print(f"Top 5 values after compression: {top5_after_reduced}")
    
    print("=" * 80)
    print("Reduced candidate signatures compression completed")
    print("=" * 80 + "\n")
    
    # Update to use compressed version
    candidate_sig_df_reduced = candidate_sig_df_reduced_clipped
    
    candidate_sig_df_reduced.to_csv(CANDIDATE_SIG_REDUCED_FILE)
    print(f"  ✓ Reduced candidate signatures ({len(candidate_sig_df_reduced)} types, {candidate_sig_df_reduced.shape[1]} genes) -> {CANDIDATE_SIG_REDUCED_FILE}")
    
    combined_df = pd.concat([known_sig_df, candidate_sig_df], axis=0)

    # Compress combined signatures before saving (for safety)
    print("\n" + "=" * 80)
    print("Compressing combined signature matrix...")
    print("=" * 80)
    
    print(f"Matrix shape: {combined_df.shape[0]} cell types × {combined_df.shape[1]} genes")
    threshold_combined = np.percentile(combined_df.values.flatten(), 99.9)
    print(f"Calculated 99.9th percentile threshold: {threshold_combined:.2f}")
    
    compressed_count_combined = (combined_df.values > threshold_combined).sum()
    total_count_combined = combined_df.size
    compressed_percentage_combined = (compressed_count_combined / total_count_combined) * 100
    print(f"Number of values to be compressed: {compressed_count_combined} ({compressed_percentage_combined:.4f}% of total)")
    
    sum_before_combined = combined_df.values.sum()
    top5_before_combined = np.sort(combined_df.values.flatten())[-5:]
    
    combined_df_clipped = combined_df.clip(upper=threshold_combined)
    
    sum_after_combined = combined_df_clipped.values.sum()
    sum_reduction_combined = ((sum_before_combined - sum_after_combined) / sum_before_combined) * 100
    print(f"Total expression sum reduced from {sum_before_combined:.2f} to {sum_after_combined:.2f} (decreased by {sum_reduction_combined:.2f}%)")
    
    top5_after_combined = np.sort(combined_df_clipped.values.flatten())[-5:]
    print(f"Top 5 values before compression: {top5_before_combined}")
    print(f"Top 5 values after compression: {top5_after_combined}")
    
    print("=" * 80)
    print("Combined signatures compression completed")
    print("=" * 80 + "\n")
    
    # Update to use compressed version
    combined_df = combined_df_clipped

    combined_df.to_csv(COMBINED_SIG_FILE)
    print(f"  ✓ Combined signatures ({len(combined_df)} types) -> {COMBINED_SIG_FILE}")
else:
    print(f"  ⚠️  No candidate signatures to save")

print("\n" + "=" * 80)
print("SIGNATURE LIBRARY SUMMARY")
print("=" * 80)
print(f"Known types ({len(known_sig_df)}):")
for ct in known_sig_df.index:
    print(f"  - {ct}")

if not candidate_sig_df.empty:
    print(f"\nCandidate types ({len(candidate_sig_df)}):")
    # We need to access the HGSOC candidate list defined in PART 1
    if 'CANDIDATE_CELLTYPES_HGSOC' not in locals():
        CANDIDATE_CELLTYPES_HGSOC = [] 

    for ct in candidate_sig_df.index:
        # Add a flag to show its origin
        if ct in CANDIDATE_CELLTYPES_HGSOC:
            print(f"  - {ct} (from HGSOC)")
        else:
            print(f"  - {ct} (from GSE176171)")

print("\n" + "=" * 80)
print("✓ SUCCESS! Signature library built")
print("=" * 80)


# sbatch --job-name=build_signatures --output="%x_%j.out" --error="%x_%j.err" --time=1:30:00 --ntasks=1 --cpus-per-task=4 --mem=48G --account=ctb-liyue --wrap="source /home/yiminfan/projects/ctb-liyue/yiminfan/project_yue/tape/bin/activate && cd /home/yiminfan/projects/ctb-liyue/yiminfan/project_yixuan/Distillation_Plasma_preprocess && python scripts/03_build_signature_library.py"