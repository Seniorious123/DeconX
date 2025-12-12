#!/usr/bin/env python
"""
Discovery Validation Script for HGSOC Distillation Results

This script generates three core validation figures to demonstrate:
1. The biological identity of discovered features (adipocyte markers enrichment)
2. The effectiveness of model improvement (residual reduction)
3. The quality of learned signatures (similarity scores)

Usage:
    python scripts/validate_discovery.py --output_dir outputs/hgso_Plasma_results

Author: Validation Pipeline
Date: 2025
"""

import os
import sys

# ============================================================================
# CRITICAL: Add project root to Python path for module imports
# ============================================================================
# When torch.load tries to unpickle saved models, it needs to import the 
# custom model classes from src.distiller.decon.models. To make this work,
# the project root directory must be in Python's module search path.

# Step 1: Get the absolute path of this script file
# __file__ gives us the path to this script, which is:
# /home/yiminfan/.../Distillation_Plasma/scripts/validate_discovery.py
script_path = os.path.abspath(__file__)

# Step 2: Get the directory containing this script (the scripts/ directory)
# This gives us: /home/yiminfan/.../Distillation_Plasma/scripts
script_dir = os.path.dirname(script_path)

# Step 3: Go up one level to get the project root directory
# This gives us: /home/yiminfan/.../Distillation_Plasma
project_root = os.path.dirname(script_dir)

# Step 4: Add the project root to the beginning of Python's module search path
# Now Python can find 'src' when it tries to import src.distiller.decon.models
sys.path.insert(0, project_root)

print(f"Added to Python path: {project_root}")

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.spatial.distance import cosine
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5

def load_characteristic_genes(diagnostics_dir):
    """
    Load the list of characteristic genes identified from virtual cells.
    
    Args:
        diagnostics_dir: Path to round_0_diagnostics directory
        
    Returns:
        List of gene names
    """
    gene_file = os.path.join(diagnostics_dir, 'round_0_characteristic_genes.txt')

    if not os.path.exists(gene_file):
        raise FileNotFoundError(f"Characteristic genes file not found: {gene_file}")
    
    with open(gene_file, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(genes)} characteristic genes from discovery")
    return genes

def load_virtual_cell_expression(diagnostics_dir):
    """
    Load virtual cell expression data from gene-level scale analysis.
    
    Args:
        diagnostics_dir: Path to round_0_diagnostics directory
        
    Returns:
        DataFrame with gene as index and virtual_mean as values
    """
    analysis_file = os.path.join(diagnostics_dir, 'round_0_gene_level_scale_analysis.csv')
    if not os.path.exists(analysis_file):
        raise FileNotFoundError(f"Gene analysis file not found: {analysis_file}")
    
    df = pd.read_csv(analysis_file)
    # Set gene as index and extract virtual_mean column
    virtual_expr = df.set_index('gene')['virtual_mean']
    
    print(f"Loaded virtual cell expression for {len(virtual_expr)} genes")
    return virtual_expr

def load_reference_signatures(reference_dir):
    """
    Load candidate and known cell type signatures.
    
    Args:
        reference_dir: Path to data/reference directory
        
    Returns:
        Tuple of (candidate_signatures, known_signatures) DataFrames
    """
    candidate_file = os.path.join(reference_dir, 'candidate_signatures_gse176171.csv')
    known_file = os.path.join(reference_dir, 'known_signatures_hgsoc.csv')
    
    if not os.path.exists(candidate_file):
        raise FileNotFoundError(f"Candidate signatures not found: {candidate_file}")
    if not os.path.exists(known_file):
        raise FileNotFoundError(f"Known signatures not found: {known_file}")
    
    candidate_sigs = pd.read_csv(candidate_file, index_col=0)
    known_sigs = pd.read_csv(known_file, index_col=0)
    
    print(f"Loaded candidate signatures: {candidate_sigs.shape}")
    print(f"  Cell types: {list(candidate_sigs.index)}")
    print(f"Loaded known signatures: {known_sigs.shape}")
    print(f"  Cell types: {list(known_sigs.index)}")
    
    return candidate_sigs, known_sigs

def load_single_cell_data(sc_file):
    """
    Load single-cell reference data.
    
    Args:
        sc_file: Path to hgsoc_sc_processed.csv
        
    Returns:
        DataFrame with cells as rows, genes as columns (excluding celltype)
    """
    if not os.path.exists(sc_file):
        raise FileNotFoundError(f"Single-cell data not found: {sc_file}")
    
    sc_data = pd.read_csv(sc_file, index_col=0)
    print(f"Loaded single-cell data: {sc_data.shape}")
    print(f"  Cell types: {sc_data['celltype'].unique()}")
    
    return sc_data

def calculate_residuals_from_model(model, bulk_data, device='cpu'):
    """
    Calculate residuals by comparing bulk data with model predictions.
    
    Args:
        model: Trained PyTorch model
        bulk_data: DataFrame with samples as rows, genes as columns
        device: Device to run computation on
        
    Returns:
        DataFrame of residuals with same shape as bulk_data
    """
    model.eval()
    model = model.to(device)
    
    # Convert bulk data to tensor
    bulk_tensor = torch.from_numpy(bulk_data.values).float().to(device)
    
    # Get model predictions
    with torch.no_grad():
        bulk_recon, _, _ = model(bulk_tensor)
    
    # Calculate residuals
    residuals = bulk_data.values - bulk_recon.cpu().numpy()
    # Set negative residuals to zero (biological constraint)
    residuals = np.maximum(residuals, 0)
    
    # Convert back to DataFrame with original indices
    residuals_df = pd.DataFrame(
        residuals,
        index=bulk_data.index,
        columns=bulk_data.columns
    )
    
    print(f"Calculated residuals with shape: {residuals_df.shape}")
    print(f"  Mean residual intensity: {residuals.sum(axis=1).mean():.2f}")
    
    return residuals_df

# def extract_learned_signature_from_model(model, cell_type_index):
#     """
#     Extract learned cell type signature from model's decoder weights.
    
#     Args:
#         model: Trained PyTorch model
#         cell_type_index: Index of the cell type to extract (0-based)
        
#     Returns:
#         numpy array of gene expression values for this cell type
#     """
#     model.eval()
    
#     # The decoder's first linear layer contains cell type signatures
#     # Shape is (n_genes, n_celltypes), so we need to transpose
#     decoder_weights = model.decoder[0].weight.data.cpu().numpy()
    
#     # Extract the signature for the specified cell type
#     signature = decoder_weights[:, cell_type_index]
    
#     print(f"Extracted learned signature for cell type index {cell_type_index}")
#     print(f"  Signature shape: {signature.shape}")
#     print(f"  Mean expression: {signature.mean():.4f}")
    
#     return signature

def extract_learned_signature_from_model(model, cell_type_index):
    """
    Extract learned cell type signature from model using the sigmatrix method.
    
    The AutoEncoderPlus model has a special method called sigmatrix() that 
    properly computes and returns the learned gene expression signatures.
    This method correctly handles the model's internal architecture and 
    returns signatures in the right format (n_celltypes, n_genes).
    
    We cannot simply extract decoder weights because the decoder may have
    multiple layers with different dimensions. The sigmatrix() method
    performs the correct forward computation to generate gene-level signatures.
    
    Args:
        model: Trained PyTorch model (AutoEncoderPlus)
        cell_type_index: Index of the cell type to extract (0-based)
        
    Returns:
        numpy array of gene expression values for this cell type (length = n_genes)
    """
    model.eval()
    
    # Use the model's built-in sigmatrix() method to extract signature matrix
    # This method is specifically designed to compute cell type signatures
    # It returns a tensor of shape (n_celltypes, n_genes)
    signature_matrix = model.sigmatrix().detach().cpu().numpy()
    
    print(f"Extracted signature matrix from model:")
    print(f"  Shape: {signature_matrix.shape}")
    print(f"  Number of cell types: {signature_matrix.shape[0]}")
    print(f"  Number of genes: {signature_matrix.shape[1]}")
    
    # Validate that the requested cell type index is within range
    if cell_type_index >= signature_matrix.shape[0]:
        raise ValueError(
            f"Requested cell type index {cell_type_index} is out of range. "
            f"Model only has {signature_matrix.shape[0]} cell types (indices 0-{signature_matrix.shape[0]-1})."
        )
    
    # Extract the signature for the specified cell type
    # Each row in the signature matrix represents one cell type's gene expression pattern
    signature = signature_matrix[cell_type_index, :]
    
    print(f"\nExtracted signature for cell type at index {cell_type_index}:")
    print(f"  Signature length: {signature.shape[0]} genes")
    print(f"  Mean expression: {signature.mean():.4f}")
    print(f"  Max expression: {signature.max():.4f}")
    print(f"  Non-zero genes: {(signature > 0).sum()} ({(signature > 0).sum() / len(signature) * 100:.1f}%)")
    
    return signature

def calculate_cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector (numpy array or pandas Series)
        vec2: Second vector (numpy array or pandas Series)
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    # Convert to numpy arrays if needed
    if isinstance(vec1, pd.Series):
        vec1 = vec1.values
    if isinstance(vec2, pd.Series):
        vec2 = vec2.values
    
    # Calculate similarity (1 - cosine distance)
    similarity = 1 - cosine(vec1, vec2)
    
    return similarity

def create_figure1_feature_heatmap(char_genes, virtual_expr, candidate_sigs, 
                                   known_sigs, sc_data, output_dir):
    """
    Create Figure 1: Heatmap showing expression of characteristic genes
    across virtual cells, reference adipocytes, and known cell types.
    
    This figure demonstrates that the automatically selected features
    are enriched for adipocyte markers and show appropriate expression patterns.
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 1: Feature Gene Expression Pattern Heatmap")
    print("="*80)
    
    # Known adipocyte markers for annotation
    adipocyte_markers = ['ADIPOQ', 'PLIN1', 'CIDEC', 'LEP', 'GPD1', 'CIDEA', 'LPL', 'FASN']
    
    # Build expression matrix for heatmap
    # Columns: Virtual Cell | Reference Adipocyte | Known Cell Types (mean)
    heatmap_data = pd.DataFrame(index=char_genes)
    
    # Column 1: Virtual cell expression
    heatmap_data['Virtual Cell'] = virtual_expr.reindex(char_genes, fill_value=0)
    
    # Column 2: Reference adipocyte expression
    if 'adipocyte' in candidate_sigs.index:
        heatmap_data['Ref Adipocyte'] = candidate_sigs.loc['adipocyte', char_genes].values
    else:
        print("Warning: adipocyte not found in candidate signatures")
        heatmap_data['Ref Adipocyte'] = 0
    
    # Column 3-15: Mean expression in each known cell type
    for celltype in known_sigs.index:
        celltype_expr = known_sigs.loc[celltype, char_genes].values
        heatmap_data[celltype] = celltype_expr
    
    # Log-transform for better visualization (add pseudocount to avoid log(0))
    heatmap_data_log = np.log1p(heatmap_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(heatmap_data_log, cmap='YlOrRd', cbar_kws={'label': 'Log(Expression + 1)'},
                linewidths=0.5, linecolor='lightgray', ax=ax)
    
    # Add asterisks to known adipocyte markers
    for i, gene in enumerate(char_genes):
        if gene in adipocyte_markers:
            ax.text(-0.5, i + 0.5, '★', fontsize=12, color='red', 
                   ha='right', va='center', weight='bold')
    
    ax.set_xlabel('Cell Types', fontsize=12, weight='bold')
    ax.set_ylabel('Characteristic Genes', fontsize=12, weight='bold')
    ax.set_title('Expression Pattern of Discovered Feature Genes\n(★ = Known Adipocyte Markers)', 
                fontsize=14, weight='bold', pad=20)
    
    # Rotate x-axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'Figure1_Feature_Expression_Heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 1 saved to: {output_file}")
    print(f"  Featured {len(char_genes)} genes across {len(heatmap_data.columns)} cell types")
    print(f"  Annotated {sum(g in adipocyte_markers for g in char_genes)} known markers")

# ============================================================================
# MODIFIED FIGURE 2: NNLS-Purified Residual Reduction Plot
# ============================================================================
# Make sure pandas, os, numpy, matplotlib.pyplot are imported at the top
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def create_figure2_residual_reduction(output_dir, bulk_data_columns): # <-- MODIFICATION 1: Added bulk_data_columns parameter
    """
    MODIFIED: Create Figure 2: Bar plot showing NNLS-purified residual reduction
    for adipocyte marker genes after discoveries. Loads data from saved CSVs.
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 2 (MODIFIED): NNLS-Purified Residual Reduction")
    print("="*80)

    # Core adipocyte markers to analyze
    # marker_genes = ['FASN', 'PLIN1', 'LPL', 'ADIPOQ', 'GPD1', 'FABP4', 'CIDEC']
    marker_genes = ['FASN', 'PLIN1', 'LPL', 'IGLV3-21', 'IGLV1-44', 'IGLV2-23', 'DSP']
    # Filter to genes that exist in the bulk data/model columns provided
    marker_genes = [g for g in marker_genes if g in bulk_data_columns] # <-- MODIFICATION 2: Use passed gene list for filtering

    if not marker_genes:
        print("[ERROR] No specified marker genes found in the provided gene list. Aborting Figure 2.")
        return

    print(f"Analyzing {len(marker_genes)} adipocyte marker genes: {marker_genes}")

    mean_residuals = {}
    global_mean_residuals = {}
    # --- MODIFICATION 3: Path construction logic changed ---
    # Get the main output dir (parent of validation_results)
    base_output_dir = os.path.dirname(output_dir)
    # CHANGE: Point to validation_inputs where your CSVs actually are
    residual_record_dir = os.path.join(base_output_dir, 'validation_inputs')

    # --- MODIFICATION 4: Define paths to specific NNLS residual CSVs ---
    stages_to_load = {
        'Round 0 (Initial)': os.path.join(residual_record_dir, 'round_0_01_all_residuals_raw.csv'),
        'Round 1 ': os.path.join(residual_record_dir, 'round_1_01_all_residuals_raw.csv'),
        'Round 2 ': os.path.join(residual_record_dir, 'round_2_01_all_residuals_raw.csv')
    }

    available_stages = []


    print("\n=== Debugging Round 2 Missing Issue ===")
    print(f"Base output dir: {base_output_dir}")
    print(f"Residual record dir: {residual_record_dir}")
    print("\nChecking files:")
    for stage_name, file_path in stages_to_load.items():
        exists = os.path.exists(file_path)
        print(f"  {stage_name}: {'EXISTS' if exists else 'NOT FOUND'}")
        print(f"    -> {file_path}")



    # --- MODIFICATION 5: Load data from CSVs instead of calculating from models ---
    print("--- Loading NNLS-purified residuals from saved CSV files ---")
    for stage_name, file_path in stages_to_load.items():
        if os.path.exists(file_path):
            print(f"   Loading NNLS residuals for '{stage_name}' from: {file_path}")
            try:
                # Assuming the first column in the CSV is the sample index
                residuals_df = pd.read_csv(file_path, index_col=0)

                residuals_df = residuals_df.clip(lower=0)

                # --- START: ADDED CODE TO CALCULATE GLOBAL MEAN ---
                # Calculate global mean for all genes in this stage
                global_mean = residuals_df.mean().mean()
                global_mean_residuals[stage_name] = global_mean
                # --- END: ADDED CODE ---

                # Check if marker genes exist in the columns of the loaded CSV
                missing_markers = [g for g in marker_genes if g not in residuals_df.columns]
                if missing_markers:
                    print(f"   [WARNING] Marker genes not found in {os.path.basename(file_path)}: {missing_markers}. Skipping them for this stage.")

                present_markers = [g for g in marker_genes if g in residuals_df.columns]
                if present_markers:
                    # Calculate mean across samples (axis=0) for present marker genes
                    mean_res = residuals_df[present_markers].mean(axis=0)
                    # Store means, ensuring index matches marker_genes order and uses NaN for missing
                    mean_residuals[stage_name] = mean_res.reindex(marker_genes, fill_value=np.nan)
                    available_stages.append(stage_name)
                else:
                    print(f"   [ERROR] No target marker genes found in {os.path.basename(file_path)}. Cannot calculate means for this stage.")
                    # Still create an entry with NaNs to maintain structure if needed later
                    mean_residuals[stage_name] = pd.Series([np.nan] * len(marker_genes), index=marker_genes)

            except Exception as e:
                print(f"   [ERROR] Failed to load or process {file_path}: {e}")
                # Fill with NaN on error
                mean_residuals[stage_name] = pd.Series([np.nan] * len(marker_genes), index=marker_genes)
        else:
            print(f"   [WARNING] Residual file not found for '{stage_name}': {file_path}")
            # Continue trying to load other stages, allow plotting with fewer bars


    print(f"\n=== Loading Summary ===")
    print(f"Attempted to load: {list(stages_to_load.keys())}")
    print(f"Successfully loaded: {available_stages}")
    print(f"Missing: {set(stages_to_load.keys()) - set(available_stages)}")

    if not available_stages:
        print("[ERROR] No NNLS residual data could be loaded. Aborting Figure 2.")
        return

    # Prepare data for plotting, ensure columns follow the order in available_stages
    plot_data = pd.DataFrame(mean_residuals).reindex(columns=available_stages)

    # --- START: ADDED CODE TO COMBINE DATA ---
    # Convert the global mean dictionary to a Series
    global_mean_series = pd.Series(global_mean_residuals, name='Avg. Global Residual')

    # Add this Series as a new row to the plot_data DataFrame
    plot_data = pd.concat([plot_data, global_mean_series.to_frame().T])
    # --- END: ADDED CODE ---

    print("\nMean NNLS-Purified Residuals to plot:")
    print(plot_data)

    # Calculate reduction percentage (Round 0 vs Round 1) if both exist
    reduction_pct = pd.Series(index=marker_genes, dtype=float)
    if 'Round 0 (Initial)' in plot_data.columns and 'Round 1 (After Adipo)' in plot_data.columns:
        res_before = plot_data['Round 0 (Initial)']
        res_after = plot_data['Round 1 (After Adipo)']
        # Avoid division by zero or NaN using a boolean mask
        mask = (res_before.notna() & res_after.notna() & (res_before != 0))
        reduction_pct[mask] = ((res_before[mask] - res_after[mask]) / res_before[mask]) * 100
        # For display purposes, set NaN where calculation wasn't possible
        reduction_pct.fillna(np.nan, inplace=True)


    # --- Create the plot (plotting logic remains similar) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- START: MODIFIED CODE FOR LABELS ---
    all_labels = plot_data.index.tolist() # Get all labels (FASN, PLIN1, ..., DSP, Avg. Global Residual)
    n_groups = len(all_labels) # Use the count of all groups
    n_stages_plot = len(available_stages) # Number of stages actually loaded
    x = np.arange(n_groups) # X-axis ticks based on number of groups
    # --- END: MODIFIED CODE ---

    # n_genes = len(marker_genes)
    # n_stages_plot = len(available_stages) # Number of stages actually loaded
    # x = np.arange(n_genes)
    width = 0.8 / n_stages_plot # Adjust width based on available stages

    # Define distinct colors
    colors = ['#3498db', '#e74c3c', '#2ecc71'] # Blue, Red, Green

    bars = []
    for i, stage_name in enumerate(available_stages):
        offset = (i - (n_stages_plot - 1) / 2) * width
        # Fill NaN with 0 *only for plotting*, keep NaN in plot_data
        means_to_plot = plot_data[stage_name].fillna(0).values
        bar = ax.bar(x + offset, means_to_plot, width, label=stage_name,
                     color=colors[i % len(colors)], edgecolor='black', linewidth=1.5)
        bars.append(bar)

    # Add percentage reduction labels (Round 0 vs Round 1) on top
    # Use plot_data which retains NaNs to find appropriate max height
    max_heights = plot_data.max(axis=1).fillna(0) # Get max height per gene across available stages
    # for i, pct in enumerate(reduction_pct):
    #     gene_name = marker_genes[i] # Get gene name by index
    #     if pd.notna(pct): # Only add label if reduction was validly calculated
    #         height = max_heights.get(gene_name, 0) # Get max height for this specific gene
    #         # Make sure height is sensible before plotting text
    #         if np.isfinite(height):
    #             ax.text(i, height * 1.05, f'{pct:.1f}% Reduction', ha='center', va='bottom',
    #                     fontsize=9, weight='bold', color='green' if pct > 0 else 'red')
    #         else:
    #             print(f"[WARNING] Invalid max height for gene {gene_name}, skipping reduction label.")


    # --- START: MODIFIED CODE FOR TEXT LABELS ---
    # Loop through the reduction_pct Series (which has gene names as its index)
    for gene_name, pct in reduction_pct.items():
        if pd.notna(pct): # Only add label if reduction was validly calculated

            # Find the x-axis position (index) for this gene name
            if gene_name in all_labels:
                i = all_labels.index(gene_name) # Get the correct x-tick index
                height = max_heights.get(gene_name, 0) # Get max height for this specific gene

                # Make sure height is sensible before plotting text
                if np.isfinite(height):
                    ax.text(i, height * 1.05, f'{pct:.1f}% Reduction', ha='center', va='bottom',
                        fontsize=9, weight='bold', color='green' if pct > 0 else 'red')
                else:
                    print(f"[WARNING] Invalid max height for gene {gene_name}, skipping reduction label.")


    # # --- Add labels, title, and legend ---
    # ax.set_xlabel('Adipocyte Marker Genes', fontsize=12, weight='bold')
    # # --- MODIFICATION 6: Update Y axis label ---
    # ax.set_ylabel('Mean Raw Residual Intensity (Clipped)', fontsize=12, weight='bold')
    # # --- MODIFICATION 7: Update Title ---
    # ax.set_title('Raw Residual Reduction for Adipocyte Markers',
    #              fontsize=14, weight='bold', pad=20)
    # ax.set_xticks(x)
    # ax.set_xticklabels(marker_genes, rotation=45, ha='right')


    # --- Add labels, title, and legend ---
    ax.set_xlabel('Genes and Control Groups', fontsize=12, weight='bold')
    # --- MODIFICATION 6: Update Y axis label ---
    ax.set_ylabel('Mean Raw Residual Intensity (Clipped)', fontsize=12, weight='bold')
    # --- MODIFICATION 7: Update Title ---
    ax.set_title('Specific Residual Reduction Across Discovery Rounds', 
                fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right') 

    ax.legend(frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Adjust y-axis limit to give space for labels
    # Force lower limit to 0 since we clipped the data
    y_lower = 0 
    # Add 20% padding at the top for percentage labels
    y_upper = max(1, np.nanmax(plot_data.fillna(0).values) * 1.20) if pd.notna(np.nanmax(plot_data.fillna(0).values)) else 1 
    ax.set_ylim(y_lower, y_upper)


    plt.tight_layout()

    # --- Save figure ---
    # --- MODIFICATION 8: Change output filename ---
    output_file = os.path.join(output_dir, 'Figure2_Raw_Residual_Reduction.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Figure 2 (Modified - Raw Residuals Clipped) saved to: {output_file}")

    # --- Save numerical results ---
    # --- MODIFICATION 9: Change output filename and save structure ---
    results_file = os.path.join(output_dir, 'raw_residual_reduction_stats.csv')
    # Create DataFrame with genes as index and stages as columns
    plot_data_save = pd.DataFrame(mean_residuals).reindex(index=marker_genes, columns=stages_to_load.keys()) # Use original keys for full structure
    # Add the calculated reduction percentage as a new column
    plot_data_save['Reduction_Pct_R0_vs_R1'] = reduction_pct
    plot_data_save.to_csv(results_file)
    print(f"    Detailed raw residual statistics saved to: {results_file}")

# ============================================================================

def create_figure3_quality_scorecard(learned_sig, reference_sig, initial_model, 
                                     final_model, known_sigs, gene_list, output_dir):
    """
    Create Figure 3: Quality scorecard showing three key metrics:
    1. Similarity between learned and reference adipocyte signatures
    2. Overall signature quality score
    3. Stability of known cell type signatures
    
    This figure provides a comprehensive quality assessment of the discovery.
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 3: Discovery Quality Scorecard")
    print("="*80)
    
    # Metric 1: Learned vs Reference Adipocyte Signature Similarity
    print("  Computing learned signature similarity...")
    learned_ref_similarity = calculate_cosine_similarity(learned_sig, reference_sig)
    
    # Metric 2: Virtual Cell vs Reference Signature Similarity (already computed during discovery)
    # We'll use the learned signature similarity as the primary metric
    overall_quality = learned_ref_similarity
    
    # # Metric 3: Stability of Known Cell Types
    # print("  Assessing known cell type stability...")
    # initial_sigs = initial_model.decoder[0].weight.data.cpu().numpy()
    # final_sigs = final_model.decoder[0].weight.data.cpu().numpy()
    
    # # Calculate similarity for each known cell type (first 13 dimensions)
    # n_known = min(13, initial_sigs.shape[1], final_sigs.shape[1])
    # stabilities = []
    # for i in range(n_known):
    #     sim = calculate_cosine_similarity(initial_sigs[:, i], final_sigs[:, i])
    #     stabilities.append(sim)
    
    # mean_stability = np.mean(stabilities)

    # Metric 3: Stability of Known Cell Types
    print("  Assessing known cell type stability...")
    
    # Extract complete signature matrices using the model's sigmatrix() method
    # For the initial model (13 known cell types), this returns shape (13, n_genes)
    # For the final model 8 cell types including discovered), this returns shape (8, n_genes)
    initial_sigs = initial_model.sigmatrix().detach().cpu().numpy()

    # n_cells_per_bulk = 3000.0 # Or read from config if available
    # initial_sigs = initial_sigs / n_cells_per_bulk # Scale down initial model signatures

    final_sigs = final_model.sigmatrix().detach().cpu().numpy()

    # n_cells_per_bulk = 3000.0 # Ensure consistency
    # final_sigs = final_sigs / n_cells_per_bulk # Scale down final model signatures
    
    print(f"  Initial model signature matrix: {initial_sigs.shape}")
    print(f"  Final model signature matrix: {final_sigs.shape}")
    
    # Verify that the final model has at least as many cell types as the initial model
    n_known = initial_sigs.shape[0]
    if final_sigs.shape[0] < n_known:
        raise ValueError(
            f"Unexpected model dimensions: final model has {final_sigs.shape[0]} cell types "
            f"but initial model has {n_known} cell types. Final model should have more."
        )
    
    # Calculate cosine similarity for each known cell type
    # We compare the first n_known signatures between initial and final models
    # These represent the same known cell types before and after adding the discovered type
    stabilities = []
    for i in range(n_known):
        # Extract the i-th cell type signature from both models
        initial_sig = initial_sigs[i, :]
        final_sig = final_sigs[i, :]
        
        # Calculate how similar this cell type's signature remained after training
        sim = calculate_cosine_similarity(initial_sig, final_sig)
        stabilities.append(sim)
    
    # Calculate the mean stability across all known cell types
    mean_stability = np.mean(stabilities)
    
    print(f"  Individual cell type stabilities: min={min(stabilities):.4f}, max={max(stabilities):.4f}")
    print(f"  Mean stability across {n_known} known types: {mean_stability:.4f}")

    print(f"  Learned Signature Similarity: {learned_ref_similarity:.4f}")
    print(f"  Mean Known Type Stability: {mean_stability:.4f}")
    
    # Create scorecard figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ('Learned Signature\nSimilarity', learned_ref_similarity),
        ('Overall Discovery\nQuality', overall_quality),
        ('Known Types\nStability', mean_stability)
    ]
    
    for ax, (label, score) in zip(axes, metrics):
        # Determine color based on score
        if score >= 0.7:
            color = '#2ecc71'  # Green for excellent
            quality = 'Excellent'
        elif score >= 0.5:
            color = '#f39c12'  # Orange for good
            quality = 'Good'
        else:
            color = '#e74c3c'  # Red for needs improvement
            quality = 'Fair'
        
        # Create horizontal bar
        ax.barh([0], [score], color=color, edgecolor='black', linewidth=2, height=0.5)
        ax.barh([0], [1-score], left=[score], color='lightgray', edgecolor='black', 
               linewidth=2, height=0.5, alpha=0.3)
        
        # Add score text
        ax.text(score/2, 0, f'{score:.3f}', ha='center', va='center',
               fontsize=16, weight='bold', color='white')
        
        # Add quality label
        ax.text(0.5, -0.8, quality, ha='center', va='center',
               fontsize=12, weight='bold', color=color)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Score', fontsize=10, weight='bold')
        ax.set_title(label, fontsize=12, weight='bold', pad=15)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    plt.suptitle('Discovery Quality Assessment', fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'Figure3_Quality_Scorecard.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 3 saved to: {output_file}")
    
    # Save metrics to file
    metrics_df = pd.DataFrame({
        'Metric': [m[0].replace('\n', ' ') for m in metrics],
        'Score': [m[1] for m in metrics],
        'Quality': ['Excellent' if m[1] >= 0.7 else 'Good' if m[1] >= 0.5 else 'Fair' 
                   for m in metrics]
    })
    metrics_file = os.path.join(output_dir, 'quality_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  Metrics saved to: {metrics_file}")

def create_figure4_signature_similarity_heatmap(final_model, candidate_sigs, known_sigs, 
                                                model_genes, discovered_celltype_name,
                                                output_dir):
    """
    Create Figure 4: Comprehensive signature similarity heatmap showing the similarity
    between learned signatures and reference signatures for all 8 cell types
    (13 known + 1 discovered).
    
    This figure provides a complete quality assessment by showing:
    - How well the model learned each cell type signature (diagonal values)
    - How well the model distinguishes between different cell types (off-diagonal values)
    - Whether any cell type pairs are being confused by the model
    
    Args:
        final_model: The trained model after discovery (contains 8 cell types)
        candidate_sigs: Candidate signatures DataFrame (for the discovered type)
        known_sigs: Known signatures DataFrame (for the 13 known types)
        model_genes: List of genes used by the model
        discovered_celltype_name: Name of the discovered cell type (e.g., 'adipocyte')
        output_dir: Directory to save the output figure
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 4: Complete Signature Similarity Heatmap")
    print("="*80)
    
    # Step 1: Extract learned signatures from the model
    # The model has 8 cell types: 13 known types + 1 discovered type
    print("  Step 1: Extracting learned signatures from model...")
    learned_sig_matrix = final_model.sigmatrix().detach().cpu().numpy()

    # n_cells_per_bulk = 3000.0 # Or read from config
    # learned_sig_matrix = learned_sig_matrix / n_cells_per_bulk # Scale down the learned matrix
    # print(f"\n[INFO] Scaled learned signature matrix (Figure 4) by dividing by {n_cells_per_bulk}")
    
    print(f"    Learned signature matrix shape: {learned_sig_matrix.shape}")
    print(f"    Number of cell types: {learned_sig_matrix.shape[0]}")
    print(f"    Number of genes: {learned_sig_matrix.shape[1]}")
    
    # Step 2: Build the complete reference signature matrix
    # We need to combine known signatures with the discovered type's reference signature
    print("\n  Step 2: Building complete reference signature matrix...")
    
    # CRITICAL FIX: The model was trained with a specific cell type order that differs
    # from the order in the known_signatures CSV file. We must use the training order
    # to ensure correct alignment between learned and reference signatures.
    # 
    # Training order (hardcoded based on the main distillation script):
    training_order_known_types = [
        'B cells', 
        'T cells', 
        'Macrophages', 
        'Epithelial cells', 
        'Endothelial cells', 
        'Fibroblasts'
    ]

    # The complete model includes the discovered type(s) at the end
    if isinstance(discovered_celltype_name, list):
        all_celltype_names = training_order_known_types + discovered_celltype_name
    else:
        # Fallback for old behavior
        all_celltype_names = training_order_known_types + [discovered_celltype_name]

    print(f"    Using training-time cell type order:")
    for i, ct in enumerate(all_celltype_names):
        print(f"      Position {i}: {ct}")
    
    print(f"\n  Step 2b: Reordering reference signatures to match training order...")
    print(f"    Original known_sigs order: {known_sigs.index.tolist()}")

    # Reindex the DataFrame to the training order
    known_sigs_reordered = known_sigs.reindex(index=training_order_known_types)

    print(f"    Reordered known_sigs order: {known_sigs_reordered.index.tolist()}")

    # Critical validation: ensure no cell types were lost during reindexing
    if known_sigs_reordered.isnull().any().any():
        missing_types = known_sigs_reordered.index[known_sigs_reordered.isnull().any(axis=1)].tolist()
        raise ValueError(
            f"CRITICAL ERROR: Cell types not found in known_sigs: {missing_types}. "
            f"Available types: {known_sigs.index.tolist()}"
        )

    # Create a combined reference signature matrix
    # Start with known signatures, then add the discovered type's reference
    reference_sig_matrix = known_sigs_reordered.copy()

    # Extract the discovered cell type signature as a DataFrame (NOT as .values array)
    # Using the list of discovered names ensures we get a DataFrame
    discovered_ref_sig_df = candidate_sigs.loc[discovered_celltype_name]

    # Concatenate known signatures with discovered signature
    # pandas.concat will automatically align columns by gene names
    reference_sig_matrix = pd.concat([reference_sig_matrix, discovered_ref_sig_df])

    # Reindex to ensure final order matches training order
    reference_sig_matrix = reference_sig_matrix.reindex(index=all_celltype_names)

    print(f"    Combined reference signature shape: {reference_sig_matrix.shape}")
    print(f"    Final reference matrix rows: {reference_sig_matrix.index.tolist()}")
    
    # Step 3: Align reference signatures to model genes
    # The reference signatures may have different genes than the model
    # We need to reindex to match the model's gene list
    print("\n  Step 3: Aligning reference signatures to model genes...")
    
    reference_sig_matrix_aligned = reference_sig_matrix.reindex(
        columns=model_genes,
        fill_value=0
    )
    
    print(f"    Reference signatures aligned to {len(model_genes)} genes")
    print(f"    Genes matched: {(reference_sig_matrix_aligned != 0).sum().sum()} non-zero values")
    
    # Convert to numpy array for similarity calculation
    reference_sig_array = reference_sig_matrix_aligned.values
    
    # Step 4: Calculate cosine similarity matrix
    # This will be a 8x8 matrix where element (i,j) is the similarity
    # between learned signature i and reference signature j
    print("\n  Step 4: Computing cosine similarity matrix...")
    
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    
    # Calculate similarity: learned (rows) vs reference (columns)
    similarity_matrix = sklearn_cosine_similarity(learned_sig_matrix, reference_sig_array)
    
    print(f"    Similarity matrix shape: {similarity_matrix.shape}")
    print(f"    Diagonal values (should be high):")
    for i, celltype in enumerate(all_celltype_names):
        diagonal_value = similarity_matrix[i, i]
        print(f"      {celltype}: {diagonal_value:.4f}")
    
    # Step 5: Create the heatmap visualization
    print("\n  Step 5: Creating heatmap visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create the heatmap with annotations
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".3f",
        cmap='coolwarm',  
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Cosine Similarity'},
        xticklabels=all_celltype_names,
        yticklabels=all_celltype_names,
        linewidths=0.5,
        linecolor='gray',
        square=True,  # Make cells square-shaped
        ax=ax
    )
    
    # Customize the plot
    ax.set_xlabel('Reference Signatures', fontsize=14, weight='bold')
    ax.set_ylabel('Learned Signatures', fontsize=14, weight='bold')
    ax.set_title(
        'Comprehensive Signature Similarity: Learned vs Reference\n' +
        '(13 Known Cell Types + 1 Discovered Adipocyte)',
        fontsize=16, weight='bold', pad=20
    )
    
    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    # Add a box around the discovered cell type (last row and column)
    # This highlights the newly discovered adipocyte
    from matplotlib.patches import Rectangle
    
    # Highlight the discovered cell type rows and columns
    from matplotlib.patches import Rectangle
    num_known = len(training_order_known_types)
    num_discovered = len(all_celltype_names) - num_known
    num_total = len(all_celltype_names)

    # Highlight the last N rows (discovered types as learned signatures)
    rect_row = Rectangle((0, num_known), num_total, num_discovered, fill=False, edgecolor='blue', linewidth=3)
    ax.add_patch(rect_row)

    # Highlight the last N columns (discovered types as reference signatures)
    rect_col = Rectangle((num_known, 0), num_discovered, num_total, fill=False, edgecolor='blue', linewidth=3)
    ax.add_patch(rect_col)
    
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(output_dir, 'Figure4_Complete_Signature_Similarity_Heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure 4 saved to: {output_file}")
    
    # Step 6: Generate diagnostic statistics
    print("\n  Step 6: Computing diagnostic statistics...")
    
    # Calculate average diagonal similarity (identity preservation)
    diagonal_similarities = np.diag(similarity_matrix)
    avg_diagonal = np.mean(diagonal_similarities)
    min_diagonal = np.min(diagonal_similarities)
    
    # Calculate average off-diagonal similarity (cross-type confusion)
    num_total_types = len(all_celltype_names)
    off_diagonal_mask = ~np.eye(num_total_types, dtype=bool)
    off_diagonal_values = similarity_matrix[off_diagonal_mask]
    avg_off_diagonal = np.mean(off_diagonal_values)
    max_off_diagonal = np.max(off_diagonal_values)
    
    # Find the most confused cell type pair (highest off-diagonal value)
    off_diag_matrix = similarity_matrix.copy()
    np.fill_diagonal(off_diag_matrix, 0)
    max_confusion_idx = np.unravel_index(off_diag_matrix.argmax(), off_diag_matrix.shape)
    most_confused_pair = (
        all_celltype_names[max_confusion_idx[0]],
        all_celltype_names[max_confusion_idx[1]],
        off_diag_matrix[max_confusion_idx]
    )
    
    print(f"    Average diagonal similarity: {avg_diagonal:.4f} (higher is better)")
    print(f"    Minimum diagonal similarity: {min_diagonal:.4f}")
    print(f"    Average off-diagonal similarity: {avg_off_diagonal:.4f} (lower is better)")
    print(f"    Maximum off-diagonal similarity: {max_off_diagonal:.4f}")
    print(f"    Most confused pair: {most_confused_pair[0]} <-> {most_confused_pair[1]} "
          f"(similarity: {most_confused_pair[2]:.4f})")
    
    # Save detailed statistics to CSV
    stats_df = pd.DataFrame({
        'Cell_Type': all_celltype_names,
        'Diagonal_Similarity': diagonal_similarities,
        'Avg_OffDiag_Similarity': [
            np.mean(similarity_matrix[i, [j for j in range(num_total_types) if j != i]])
            for i in range(num_total_types)
        ]
    })
    
    stats_file = os.path.join(output_dir, 'signature_similarity_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"    Detailed statistics saved to: {stats_file}")
    
    # Save the complete similarity matrix
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=all_celltype_names,
        columns=all_celltype_names
    )
    matrix_file = os.path.join(output_dir, 'complete_similarity_matrix.csv')
    similarity_df.to_csv(matrix_file)
    print(f"    Complete similarity matrix saved to: {matrix_file}")
    
    print(f"\n✓ Figure 4 generation completed successfully")
    
    return similarity_matrix, all_celltype_names


# ============================================================================
# FIGURE 5: Ratio Reduction Plot
# ============================================================================
def create_figure5_ratio_reduction(output_dir):
    """
    Create Figure 5: Bar plot showing the reduction of the Ratio
    (Virtual Cell vs Real SC) for the top 3 genes identified in Round 1,
    across multiple discovery rounds.

    This figure demonstrates how the relative distinctiveness of the
    initial top residual signals diminishes as the model learns.
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 5: Top Gene Ratio Reduction Across Rounds")
    print("="*80)

    all_ratio_data = {}
    top_genes_round0 = None
    # Attempt to load data for up to 3 rounds (diagnostics for stages 0, 1, 2)
    max_rounds_to_load = 3

    # --- Load Ratio data from each round's diagnostics ---
    print("--- Loading Ratio data from diagnostic files ---")
    # Note: output_dir is validation_results. Need parent dir for round_X_diagnostics.
    # Note: output_dir is validation_results. Need parent dir.
    base_output_dir = os.path.dirname(output_dir)
    # CHANGE: Point to validation_inputs
    validation_inputs_dir = os.path.join(base_output_dir, 'validation_inputs')
    
    for i in range(max_rounds_to_load):
        # CHANGE: No sub-folders, look directly in inputs with prefixed filename
        ratio_file = os.path.join(validation_inputs_dir, f'round_{i}_gene_level_scale_analysis.csv')

        if os.path.exists(ratio_file):
            print(f"   Loading ratio data for Stage {i} from: {ratio_file}")
            try:
                df = pd.read_csv(ratio_file)
                # Use gene name as index for easy lookup
                all_ratio_data[f'Stage {i}'] = df.set_index('gene')['ratio']

                # Identify Top 3 genes from the first round (Stage 0)
                if i == 0:
                    top_genes_round0_df = df.sort_values('ratio', ascending=False).head(3)
                    top_genes_round0 = top_genes_round0_df['gene'].tolist()
                    print(f"   Identified Top 3 genes from Stage 0: {top_genes_round0}")
                    print(top_genes_round0_df[['gene', 'ratio']]) # Print top 3 ratios for verification

            except Exception as e:
                print(f"   [ERROR] Failed to load or process {ratio_file}: {e}")
                # If any file fails, we might not be able to proceed depending on which one
                if i == 0:
                     print("   [CRITICAL] Cannot proceed without Stage 0 ratio data.")
                     return
        else:
            print(f"   [WARNING] Ratio file not found for Stage {i}: {ratio_file}. Plot will only show available stages.")
            # Break if we can't find the first round's data
            if i == 0:
                print("   [CRITICAL] Cannot proceed without Stage 0 ratio data.")
                return
            # Allow plotting with fewer stages if later files are missing
            break # Stop loading if a round's file is missing

    if top_genes_round0 is None:
        print("[ERROR] Could not determine Top 3 genes from Stage 0. Aborting Figure 5.")
        return

    available_stages = list(all_ratio_data.keys())
    if len(available_stages) < 2:
        print("[INFO] Need data from at least two stages to show reduction. Aborting Figure 5.")
        return

    print(f"--- Preparing data for Top 3 genes: {top_genes_round0} ---")
    plot_data = pd.DataFrame(index=top_genes_round0)
    for stage_name, ratio_series in all_ratio_data.items():
        # Extract ratios for the top 3 genes, fill with 0 if a gene wasn't found (shouldn't happen with full gene list)
        plot_data[stage_name] = ratio_series.reindex(top_genes_round0, fill_value=0)

    print("Ratio values to plot:")
    print(plot_data)

    # --- Create the plot ---
    print("--- Generating plot ---")
    fig, ax = plt.subplots(figsize=(10, 7))

    n_genes = len(top_genes_round0)
    n_stages = len(available_stages)
    x = np.arange(n_genes) # the label locations
    width = 0.8 / n_stages # the width of the bars

    # Define distinct colors for potentially up to 3 stages
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = []
    for i, stage_name in enumerate(available_stages):
        offset = (i - (n_stages - 1) / 2) * width
        # Add a small epsilon for log calculation, especially if ratio can be 0
        ratios_to_plot = np.log10(plot_data[stage_name].values + 1e-9) # Use log10 scale
        bar = ax.bar(x + offset, ratios_to_plot, width, label=stage_name,
                     color=colors[i % len(colors)], edgecolor='black', linewidth=1)
        bars.append(bar)

        # Add text labels on top of bars (showing original ratio value)
        for j, rect in enumerate(bar):
            height = rect.get_height()
            original_ratio = plot_data[stage_name].values[j]
            # Format the label based on the magnitude of the original ratio
            if original_ratio >= 10:
                label_text = f'{original_ratio:.0f}x' # Integer for large ratios
            elif original_ratio >= 0.1:
                label_text = f'{original_ratio:.1f}x' # One decimal for medium ratios
            else:
                label_text = f'{original_ratio:.2f}x' # Two decimals for small ratios

            ax.text(rect.get_x() + rect.get_width() / 2., height,
                    label_text,
                    ha='center', va='bottom', fontsize=8, rotation=0)

    # --- Add labels, title, and legend ---
    ax.set_xlabel('Top 3 Genes (Ranked by Initial Ratio)', fontsize=12, weight='bold')
    ax.set_ylabel('Log10 (Ratio [Virtual Cell / Real SC] + ε)', fontsize=12, weight='bold') # Indicate log scale
    ax.set_title('Ratio Reduction for Top Genes Across Discovery Stages',
                 fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(top_genes_round0, rotation=0, ha='center') # No rotation needed for 3 genes
    ax.legend(frameon=True, shadow=True, title="Discovery Stage")
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Adjust y-axis limits if needed (log scale might need adjustment)
    # Ensure y-axis starts slightly below 0 if there are negative log values, or at 0 otherwise
    min_log_ratio = np.min(np.log10(plot_data.replace(0, 1e-9).values))
    max_log_ratio = np.max(np.log10(plot_data.values + 1e-9))
    y_min = min(0, min_log_ratio * 1.1) # Start slightly below 0 if negative logs exist
    y_max = max_log_ratio * 1.15 if max_log_ratio > 0 else 1 # Ensure some space at top
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    # --- Save figure ---
    output_file = os.path.join(output_dir, 'Figure5_Ratio_Reduction.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure object

    print(f"✓ Figure 5 saved to: {output_file}")

    # --- Save numerical results ---
    results_file = os.path.join(output_dir, 'ratio_reduction_stats.csv')
    plot_data.to_csv(results_file)
    print(f"   Plotted data saved to: {results_file}")

# ============================================================================





def perform_detailed_similarity_diagnostics(final_model, candidate_sigs, known_sigs, 
                                           model_genes, discovered_celltype_name, 
                                           output_dir):
    """
    Perform comprehensive diagnostic analysis of similarity contributions for all cell types.
    
    This is a standalone diagnostic function that analyzes how each gene contributes to
    the cosine similarity between learned and reference signatures. It helps understand:
    - Which genes dominate the similarity calculation
    - Whether marker genes contribute as expected
    - How concentrated the expression is in top genes
    
    This function is completely independent and can be called separately from figure generation.
    
    Args:
        final_model: The trained model containing learned signatures
        candidate_sigs: Candidate signatures DataFrame
        known_sigs: Known signatures DataFrame
        model_genes: List of genes used by the model
        discovered_celltype_name: Name of discovered cell type
        output_dir: Directory to save diagnostic outputs
    """
    print("\n" + "="*80)
    print("PERFORMING DETAILED SIMILARITY CONTRIBUTION DIAGNOSTICS")
    print("="*80)
    print("This analysis will decompose cosine similarities to understand gene-level contributions")
    
    # Define marker genes for each cell type
    marker_gene_dict = {
        'B cells': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'CD22'],
        'T cells': ['CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD4', 'CD2'],
        'Macrophages': ['CD68', 'CD14', 'FCGR3A', 'CSF1R', 'CD163'],
        'Epithelial cells': ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'CDH1'],
        'Endothelial cells': ['PECAM1', 'VWF', 'CDH5', 'CD34', 'TEK'],
        'Fibroblasts': ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'VIM'],
        'Plasma cells': ['SDC1', 'CD38', 'XBP1', 'JCHAIN', 'MZB1'],
        'adipocyte': ['ADIPOQ', 'PLIN1', 'CIDEC', 'LEP', 'LPL', 'FASN', 'GPD1', 'CIDEA']
    }
    
    # Get training order for cell types
    training_order_known_types = [
        'B cells', 'T cells', 'Macrophages', 'Epithelial cells', 
        'Endothelial cells', 'Fibroblasts'
    ]
    # The complete model includes the discovered type(s) at the end
    if isinstance(discovered_celltype_name, list):
        all_celltype_names = training_order_known_types + discovered_celltype_name
    else:
        # Fallback for old behavior
        all_celltype_names = training_order_known_types + [discovered_celltype_name]

    # Extract learned signatures
    learned_sig_matrix = final_model.sigmatrix().detach().cpu().numpy()

    # n_cells_per_bulk = 3000.0 # Or read from config
    # learned_sig_matrix = learned_sig_matrix / n_cells_per_bulk # Scale down the learned matrix
    # print(f"\n[INFO] Scaled learned signature matrix (Diagnostics) by dividing by {n_cells_per_bulk}")

    # Build reference signature matrix with correct order
    known_sigs_reordered = known_sigs.reindex(index=training_order_known_types)
    # Use the list of discovered names
    discovered_ref_sig = candidate_sigs.loc[discovered_celltype_name]
    # No need for .to_frame().T, discovered_ref_sig is already a DataFrame
    reference_sig_matrix = pd.concat([known_sigs_reordered, discovered_ref_sig])
    reference_sig_matrix_aligned = reference_sig_matrix.reindex(columns=model_genes, fill_value=0)
    reference_sig_array = reference_sig_matrix_aligned.values
    
    # Storage for all analyses
    all_contribution_analyses = {}
    
    # Analyze each cell type
    for i, celltype in enumerate(all_celltype_names):
        print(f"\n{'='*80}")
        print(f"ANALYZING CELL TYPE {i+1}/{len(all_celltype_names)}: {celltype}")
        print(f"{'='*80}")
        
        learned_sig = learned_sig_matrix[i, :]
        reference_sig = reference_sig_array[i, :]
        markers = marker_gene_dict.get(celltype, None)
        
        # Calculate gene-level contributions
        gene_contributions = learned_sig * reference_sig
        total_dot_product = np.sum(gene_contributions)
        contribution_percentages = (gene_contributions / total_dot_product) * 100
        
        # Build contribution dataframe
        contribution_df = pd.DataFrame({
            'gene': model_genes,
            'learned_expr': learned_sig,
            'reference_expr': reference_sig,
            'contribution': gene_contributions,
            'contribution_pct': contribution_percentages
        })
        contribution_df['abs_contribution'] = np.abs(contribution_df['contribution'])
        contribution_df = contribution_df.sort_values('abs_contribution', ascending=False)
        
        # Print top contributing genes
        print(f"\nTop 20 genes contributing to cosine similarity:")
        print(f"{'Rank':<6} {'Gene':<15} {'Learned':<12} {'Reference':<12} {'Contribution':<15} {'% of Total':<12}")
        print("-" * 85)
        
        for rank, (idx, row) in enumerate(contribution_df.head(20).iterrows(), 1):
            marker_flag = " ★" if markers and row['gene'] in markers else ""
            print(f"{rank:<6} {row['gene']:<15} {row['learned_expr']:>11.2f} {row['reference_expr']:>11.2f} "
                  f"{row['contribution']:>14.2f} {row['contribution_pct']:>11.2f}%{marker_flag}")
        
        # Calculate cumulative contributions
        top_5_pct = contribution_df.head(5)['contribution_pct'].sum()
        top_10_pct = contribution_df.head(10)['contribution_pct'].sum()
        top_20_pct = contribution_df.head(20)['contribution_pct'].sum()
        top_50_pct = contribution_df.head(50)['contribution_pct'].sum()
        
        print(f"\nCumulative Contribution Analysis:")
        print(f"  Top 5 genes:   {top_5_pct:>6.2f}% of total similarity")
        print(f"  Top 10 genes:  {top_10_pct:>6.2f}% of total similarity")
        print(f"  Top 20 genes:  {top_20_pct:>6.2f}% of total similarity")
        print(f"  Top 50 genes:  {top_50_pct:>6.2f}% of total similarity")
        
        # Analyze marker genes if available
        marker_contribution_pct = 0
        if markers:
            valid_markers = [g for g in markers if g in model_genes]
            marker_df = contribution_df[contribution_df['gene'].isin(valid_markers)]
            
            if len(marker_df) > 0:
                marker_contribution_pct = marker_df['contribution_pct'].sum()
                print(f"\nMarker Gene Analysis:")
                print(f"  Found {len(marker_df)}/{len(markers)} marker genes")
                print(f"  Marker genes contribute: {marker_contribution_pct:.2f}% of total similarity")
                print(f"\n  {'Gene':<15} {'Learned':<12} {'Reference':<12} {'% Contribution':<15}")
                print("  " + "-" * 60)
                for idx, row in marker_df.iterrows():
                    print(f"  {row['gene']:<15} {row['learned_expr']:>11.2f} {row['reference_expr']:>11.2f} "
                          f"{row['contribution_pct']:>14.2f}%")
        
        # Analyze expression concentration
        total_learned = np.sum(learned_sig)
        total_reference = np.sum(reference_sig)
        
        learned_sorted = contribution_df.sort_values('learned_expr', ascending=False)
        learned_top10_pct = (learned_sorted.head(10)['learned_expr'].sum() / total_learned) * 100
        learned_top50_pct = (learned_sorted.head(50)['learned_expr'].sum() / total_learned) * 100
        
        ref_sorted = contribution_df.sort_values('reference_expr', ascending=False)
        ref_top10_pct = (ref_sorted.head(10)['reference_expr'].sum() / total_reference) * 100
        ref_top50_pct = (ref_sorted.head(50)['reference_expr'].sum() / total_reference) * 100
        
        print(f"\nExpression Concentration:")
        print(f"  Learned signature - Top 10 genes: {learned_top10_pct:.2f}% of expression")
        print(f"  Learned signature - Top 50 genes: {learned_top50_pct:.2f}% of expression")
        print(f"  Reference signature - Top 10 genes: {ref_top10_pct:.2f}% of expression")
        print(f"  Reference signature - Top 50 genes: {ref_top50_pct:.2f}% of expression")
        
        # Calculate and verify cosine similarity
        learned_norm = np.linalg.norm(learned_sig)
        reference_norm = np.linalg.norm(reference_sig)
        cosine_sim = total_dot_product / (learned_norm * reference_norm)
        
        print(f"\nVerification:")
        print(f"  Cosine similarity: {cosine_sim:.6f}")
        print(f"  Dot product: {total_dot_product:.2f}")
        print(f"  Learned L2 norm: {learned_norm:.2f}")
        print(f"  Reference L2 norm: {reference_norm:.2f}")
        
        # Store results
        all_contribution_analyses[celltype] = {
            'cosine_similarity': cosine_sim,
            'top_5_pct': top_5_pct,
            'top_10_pct': top_10_pct,
            'top_20_pct': top_20_pct,
            'top_50_pct': top_50_pct,
            'marker_contribution_pct': marker_contribution_pct,
            'learned_top10_conc': learned_top10_pct,
            'learned_top50_conc': learned_top50_pct,
            'ref_top10_conc': ref_top10_pct,
            'ref_top50_conc': ref_top50_pct,
            'contribution_df': contribution_df
        }
    
    # Save summary statistics
    print(f"\n{'='*80}")
    print("SAVING DIAGNOSTIC RESULTS")
    print(f"{'='*80}")
    
    summary_data = []
    for celltype, analysis in all_contribution_analyses.items():
        summary_data.append({
            'Cell_Type': celltype,
            'Cosine_Similarity': analysis['cosine_similarity'],
            'Top5_Genes_Pct': analysis['top_5_pct'],
            'Top10_Genes_Pct': analysis['top_10_pct'],
            'Top20_Genes_Pct': analysis['top_20_pct'],
            'Top50_Genes_Pct': analysis['top_50_pct'],
            'Marker_Genes_Pct': analysis['marker_contribution_pct'],
            'Learned_Top10_Expr_Pct': analysis['learned_top10_conc'],
            'Learned_Top50_Expr_Pct': analysis['learned_top50_conc'],
            'Reference_Top10_Expr_Pct': analysis['ref_top10_conc'],
            'Reference_Top50_Expr_Pct': analysis['ref_top50_conc']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, 'similarity_diagnostics_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")
    
    # Save detailed gene contributions for each cell type
    for celltype, analysis in all_contribution_analyses.items():
        celltype_clean = celltype.replace(' ', '_').replace('/', '_')
        detail_file = os.path.join(output_dir, f'gene_contributions_{celltype_clean}.csv')
        analysis['contribution_df'].head(100).to_csv(detail_file, index=False)
    
    print(f"Detailed contribution files saved for all {len(all_celltype_names)} cell types")
    print(f"\n{'='*80}")
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate validation figures for HGSOC distillation discovery results'
    )
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to the main output directory (e.g., outputs/hgsoc_Plasma_results)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for computation (cpu or cuda:0)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HGSOC DISTILLATION DISCOVERY VALIDATION")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Computation device: {args.device}")
    
    # Create validation results subdirectory
    validation_dir = os.path.join(args.output_dir, 'validation_results')
    os.makedirs(validation_dir, exist_ok=True)
    print(f"Validation results will be saved to: {validation_dir}")
    
    # Define paths based on project structure
    project_root = os.path.dirname(os.path.dirname(args.output_dir))
    data_dir = os.path.join(project_root, 'data')
    
    # Load all required data
    print("\n" + "="*80)
    print("LOADING DATA FILES")
    print("="*80)
    
    # Load characteristic genes
    # Change: Point to validation_inputs
    diagnostics_dir = os.path.join(args.output_dir, 'validation_inputs') 
    char_genes = load_characteristic_genes(diagnostics_dir)
    
    # Load virtual cell expression
    virtual_expr = load_virtual_cell_expression(diagnostics_dir)
    
    # Load reference signatures
    reference_dir = os.path.join(data_dir, 'reference')
    candidate_sigs, known_sigs = load_reference_signatures(reference_dir)
    
    # Load single-cell data
    sc_file = os.path.join(data_dir, 'processed', 'hgsoc_sc_processed.csv')
    sc_data = load_single_cell_data(sc_file)
    
    # Load bulk data
    bulk_file = os.path.join(data_dir, 'processed', 'hgsoc_bulk_processed.csv')
    bulk_data = pd.read_csv(bulk_file, index_col=0)
    print(f"Loaded bulk data: {bulk_data.shape}")
    
    gene_list = bulk_data.columns.tolist()

    # Load models
    print("\n" + "="*80)
    print("LOADING TRAINED MODELS")
    print("="*80)

    # Find initial model (there should be only one in checkpoints_initial)
    initial_checkpoint_dir = os.path.join(args.output_dir, 'checkpoints_initial')
    initial_model_files = [f for f in os.listdir(initial_checkpoint_dir) if f.endswith('.pth')]
    if not initial_model_files:
        raise FileNotFoundError(f"No initial model found in {initial_checkpoint_dir}")

    initial_model_path = os.path.join(initial_checkpoint_dir, initial_model_files[0])
    print(f"Loading initial model from: {initial_model_path}")
    initial_model = torch.load(initial_model_path, map_location=args.device)
    initial_model.eval()

    # Load round 1 model (after *second* discovery, this is the final 8-dim model)
    # Round 0 discovered Plasma cells (7-dim)
    # Round 1 discovered adipocyte (8-dim)
    final_model_path = os.path.join(args.output_dir, 'round_1_model.pth')
    print(f"Loading final model (round 1) from: {final_model_path}")
    if not os.path.exists(final_model_path):
        raise FileNotFoundError(f"Final model not found: {final_model_path}. Run ended early.")
    expanded_model = torch.load(final_model_path, map_location=args.device)
    expanded_model.eval()
    
    # # Extract learned adipocyte signature from expanded model
    # # The new cell type is the last dimension in the expanded model
    # n_celltypes = expanded_model.decoder[0].weight.shape[1]
    # learned_adipocyte_sig = extract_learned_signature_from_model(expanded_model, n_celltypes - 1)
    
    # # Get reference adipocyte signature
    # reference_adipocyte_sig = candidate_sigs.loc['adipocyte'].values
    
    # # Get list of genes (should match between bulk and signatures)
    # gene_list = bulk_data.columns.tolist()

    # Extract learned adipocyte signature from expanded model
    # The discovered adipocyte is the last cell type in the expanded model
    print("\n" + "="*80)
    print("EXTRACTING AND ALIGNING SIGNATURES")
    print("="*80)

    # First, extract the complete signature matrix from the expanded model
    # This gives us all cell types including the newly discovered one
    expanded_sig_matrix = expanded_model.sigmatrix().detach().cpu().numpy()
    n_celltypes = expanded_sig_matrix.shape[0]
    n_genes_in_model = expanded_sig_matrix.shape[1]

    print(f"Expanded model signature matrix: {expanded_sig_matrix.shape}")
    print(f"  Total cell types: {n_celltypes}")
    print(f"  Genes per signature: {n_genes_in_model}")

    # The discovered cell types are at the end.
    # Initial model = 6 types (index 0-5)
    # Round 0 (index 6) was Plasma cells
    # Round 1 (index 7) was adipocyte
    # We want to check Adipocyte (index 7) for Figure 3
    learned_adipocyte_index = 7 # This is the index for adipocyte in the 8-dim model
    discovered_type_for_fig3 = 'adipocyte' # This is the reference we compare against

    print(f"Extracting signature for Figure 3: '{discovered_type_for_fig3}' at index {learned_adipocyte_index}")

    learned_adipocyte_sig = extract_learned_signature_from_model(
        expanded_model,
        learned_adipocyte_index
    )

    # Signature is already at single-cell scale, no scaling needed
    print(f"    Mean expression: {learned_adipocyte_sig.mean():.4f}")

    # Now we need to get the reference adipocyte signature and align it with model genes
    # The model was trained on genes from bulk_data, so we use that gene list
    model_genes = bulk_data.columns.tolist()

    print(f"\nPreparing reference '{discovered_type_for_fig3}' signature:")
    print(f"  Model uses {len(model_genes)} genes")

    # Check if adipocyte exists in candidate signatures
    learned_adipocyte_index
    if discovered_type_for_fig3 not in candidate_sigs.index:
        raise ValueError(
            f"'{discovered_type_for_fig3}' cell type not found in candidate signatures. "
            f"Available types: {list(candidate_sigs.index)}"
        )

    # Extract the adipocyte signature from candidate signatures
    # This is a pandas Series with gene names as index
    reference_adipocyte_series = candidate_sigs.loc[discovered_type_for_fig3]
    print(f"  Reference signature original length: {len(reference_adipocyte_series)}")

    # Align the reference signature to match the model's gene list
    reference_adipocyte_sig = reference_adipocyte_series.reindex(
        model_genes,
        fill_value=0
    ).values

    print(f"  Reference signature after alignment: {len(reference_adipocyte_sig)}")
    print(f"  Genes matched: {(reference_adipocyte_sig != 0).sum()}")
    print(f"  Genes filled with zero: {(reference_adipocyte_sig == 0).sum()}")

    # Critical validation: ensure both signatures have exactly the same length
    print(f"\nFinal signature dimension check:")
    print(f"  Learned signature:    {len(learned_adipocyte_sig)} genes")
    print(f"  Reference signature:  {len(reference_adipocyte_sig)} genes")

    if len(learned_adipocyte_sig) != len(reference_adipocyte_sig):
        raise ValueError(
            f"Dimension mismatch detected! "
            f"Learned signature has {len(learned_adipocyte_sig)} genes "
            f"but reference signature has {len(reference_adipocyte_sig)} genes. "
            f"These must be identical for similarity calculation."
        )

    print("  ✓ Dimensions match - ready for similarity calculation")

    # Get list of genes for other functions
    gene_list = model_genes

    print(f"\nTotal genes in analysis: {len(gene_list)}")
    print(f"Initial model dimensions: {initial_model.sigmatrix().detach().cpu().numpy().shape[0]} cell types")
    print(f"Expanded model dimensions: {n_celltypes} cell types")
    
    # Generate all validation figures
    print("\n" + "="*80)
    print("GENERATING VALIDATION FIGURES")
    print("="*80)
    
    create_figure1_feature_heatmap(
        char_genes, virtual_expr, candidate_sigs, known_sigs, sc_data, validation_dir
    )
    
    try:
        create_figure2_residual_reduction(validation_dir, gene_list)
    except Exception as e:
        print(f"[ERROR] Failed to generate Figure 2 (Modified): {e}")
        import traceback
        traceback.print_exc()
    

    create_figure3_quality_scorecard(
        learned_adipocyte_sig, reference_adipocyte_sig, initial_model, 
        expanded_model, known_sigs, gene_list, validation_dir
    )
    
    # Generate Figure 4: Complete signature similarity heatmap
    # This figure shows the similarity between all learned and reference signatures
    # including both the 7 known types and the 1 discovered type
    discovered_types_ordered = ['Plasma cells', 'adipocyte']

    create_figure4_signature_similarity_heatmap(
        expanded_model,           # The final model with 14 cell types
        candidate_sigs,           # Candidate signatures (contains adipocyte reference)
        known_sigs,               # Known signatures (13 types)
        gene_list,                # List of genes used by the model
        discovered_types_ordered, # Pass the LIST of discovered names, e.g., ['adipocyte', 'Plasma cells']
        validation_dir            # Output directory
    )


    # --- Call the new Figure 5 function ---
    try:
        # Pass the directory where validation results should be saved
        create_figure5_ratio_reduction(validation_dir)
    except Exception as e:
        print(f"[ERROR] Failed to generate Figure 5: {e}")
        import traceback
        traceback.print_exc()




    print("\n" + "="*80)
    print("OPTIONAL: DETAILED SIMILARITY DIAGNOSTICS")
    print("="*80)
    print("This step performs in-depth analysis of gene contributions to similarity scores.")
    print("It will generate detailed reports for all cell types.")
    
    user_wants_diagnostics = True # Set to False to skip diagnostics

    if user_wants_diagnostics:
        perform_detailed_similarity_diagnostics(
            expanded_model,
            candidate_sigs,
            known_sigs,
            gene_list,
            discovered_types_ordered, # Pass the full list
            validation_dir
        )
    else:
        print("Skipping detailed diagnostics (user_wants_diagnostics = False)")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"All results saved to: {validation_dir}")
    print("Generated files:")
    print("  - Figure1_Feature_Expression_Heatmap.png")
    print("  - Figure2_Residual_Reduction.png")
    print("  - Figure3_Quality_Scorecard.png")
    print("  - Figure4_Complete_Signature_Similarity_Heatmap.png")
    print("  - residual_reduction_stats.csv")
    print("  - quality_metrics.csv")
    print("  - signature_similarity_statistics.csv")
    print("  - complete_similarity_matrix.csv")
    print("\nValidation pipeline completed successfully!")

if __name__ == '__main__':
    main()


# sbatch --job-name=validate_discovery --output="%x_%j.out" --error="%x_%j.err" --time=0:30:00 --ntasks=1 --cpus-per-task=2 --mem=16G --account=ctb-liyue --wrap="source /home/yiminfan/projects/ctb-liyue/yiminfan/project_yue/tape/bin/activate && cd /home/yiminfan/projects/ctb-liyue/yiminfan/project_yixuan/Distillation_Cut && python scripts/validate_discovery.py --output_dir outputs/hgsoc_Adipocytes_Plasma_results"