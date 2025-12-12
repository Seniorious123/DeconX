# File: scripts/run_distillation.py
# This is the main executable script for running the distillation workflow.

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# This is the standard way to make the 'src' directory visible to the script
# when run from the project's root directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from our new, clean, modularized code library
from src.distiller.decon.distillation import distillation
from src.distiller.generation.data_manager import read_sc_input, read_bulk_input
from src.distiller.decon.utils import find_sigmatrix, normalize_vectors

from configs import path_config

if __name__ == "__main__":

    print("--- Starting Distillation Pipeline ---")

    # --- Fixed Technical & Performance Parameters ---
    # These parameters control the training process and are set to optimal values
    # based on previous experiments. They do not need to be changed for typical use cases.
    SEED = 42
    SAMPLE = 3000
    EPOCHS = 200
    CPU = 1
    BATCH_SIZE = 256
    DEVICE = 'cuda:0'
    LEARNING_RATE = 1e-3
    LAMDA = 10.0
    USE_MSE = 0.0  # Corresponds to your --MSE argument
    
    # --- Command Line Argument Parser Setup ---
    # This section defines all the parameters that users can specify when running the script
    parser = argparse.ArgumentParser(description='Process single-cell data and perform cell type discovery.')

    # --- Group 1: Optional Flags (Required=False, No default value or boolean flags) ---
    parser.add_argument('--resume-from-checkpoint', type=str, default=None, 
                        help='Path to a checkpoint file to resume training.')
    parser.add_argument('--no-noise', action='store_true', 
                        help='Flag to disable noise addition.')
    parser.add_argument('--no-residual-restriction', action='store_true', 
                        help='Flag to disable residual restriction.')

    # --- Group 2: Optional Data Paths (Required=False, Have default=None) ---
    # These are skipped in real discovery mode
    parser.add_argument('--test_frac', type=str, required=False, default=None, 
                        help="Path to the ground truth fraction file (optional, for validation only).")
    parser.add_argument('--candidate_sig_path', type=str, required=False, default=None,
                        help="Path to the signature matrix containing candidate cell types (optional, for discovery annotation).")

    # --- Group 3: ALL REQUIRED PATHS AND LISTS (CRITICAL FOR PARSING STABILITY) ---
    # These must be provided for every successful run (simulation or real).
    
    # Core Data Paths
    parser.add_argument('--sc_data', type=str, required=True, 
                        help="Path to the preprocessed single-cell RNA-seq data (CSV/H5AD format).")
    parser.add_argument('--test_bulk', type=str, required=True, 
                        help="Path to the bulk RNA-seq test data (CSV format).")

    # Workflow & Output Configuration
    parser.add_argument('--output_path', type=str, required=True, 
                        help="Directory path where all results will be saved.")
    parser.add_argument('--target_celltypes', type=str, required=True, 
                        help="Comma-separated list of target cell types to discover (e.g., 'Plasmablast,DCs').")

    # Dataset Specific Parameters (The newly universalized parameters)
    parser.add_argument('--known_celltypes', type=str, required=True, 
                        help="Comma-separated list of known cell types for initial model training (e.g., 'B_cell,CD4,CD8').")
    parser.add_argument('--known_sig_path', type=str, required=True, 
                        help="Path to the signature matrix containing only known cell types (used for initialization).")

    # CRITICAL STEP: Parse the command line arguments
    # This must happen BEFORE we try to use any args.xxx variable
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Single-cell data: {args.sc_data}")
    print(f"  Bulk data: {args.test_bulk}")
    print(f"  Output directory: {args.output_path}")
    print(f"  Target cell types: {args.target_celltypes}")
    
    # --- Data Loading Phase ---
    # Now that we have parsed the arguments, we can safely use them
    
    # Define the known cell types that we will use for the initial model
    # These are the cell types we are confident about and will use as a baseline
    known_cell_types_for_sim = [ct.strip() for ct in args.known_celltypes.split(',')]
    print(f"\nLoading single-cell reference data...")
    print(f"  Using known cell types as baseline: {known_cell_types_for_sim}")
    
    # Read the single-cell reference data without any filtering
    # This ensures that sc_data is not an empty DataFrame.
    sc_data = read_sc_input(args.sc_data)
    # all_sc_data = read_sc_input(args.sc_data)
    all_sc_data = sc_data
    print(f"  Loaded single-cell data with shape: {sc_data.shape}")
    
    # Read the bulk RNA-seq data that we want to deconvolve
    bulk_data = read_bulk_input(args.test_bulk)

    print(f"\n=== DIMENSION ALIGNMENT CHECK ===")
    print(f"sc_data shape: {sc_data.shape}")
    print(f"bulk_data shape (before): {bulk_data.shape}")

    # 确保bulk_data和sc_data的基因列一致
    sc_genes = sc_data.drop(columns='celltype').columns
    bulk_data = bulk_data[sc_genes]

    print(f"bulk_data shape (after): {bulk_data.shape}")
    print(f"Genes aligned: {len(sc_genes)}")
    print("="*50)

    # import pdb; pdb.set_trace()

    
    print(f"  Loaded bulk data with shape: {bulk_data.shape}")

    # ==========================================================================
    # [STRATEGY] EARLY GLOBAL CAPPING (99.95%)
    # ==========================================================================
    print("\n" + "="*80)
    print("[STRATEGY] Applying Scale Alignment and Global Capping (EARLY STAGE)")
    print("="*80)

    # 1. Scale Alignment
    sc_total_counts = np.sum(sc_data.drop(columns='celltype').values, axis=1)
    target_scale = np.mean(sc_total_counts)
    bulk_total_counts = np.sum(bulk_data.values, axis=1)
    current_scale = np.mean(bulk_total_counts)
    
    if current_scale > 0:
        scaling_factor = target_scale / current_scale
        bulk_data = bulk_data * scaling_factor
        print(f"[INFO] Bulk data scaled down by factor: {scaling_factor:.4e}")

    # 2. Calculate Threshold (99.95%)
    sc_values = sc_data.drop(columns='celltype').values
    cap_threshold = np.percentile(sc_values, 99.95) 
    
    # Safety floor
    cap_threshold = max(cap_threshold, 300.0)
    print(f"[INFO] Calculated Capping Threshold (p99.95): {cap_threshold:.4f}")

    # 3. Apply Capping
    # Cap Training Data (Single Cell)
    sc_data_numeric = sc_data.drop(columns='celltype')
    sc_data_clipped = sc_data_numeric.clip(upper=cap_threshold)
    sc_data_clipped['celltype'] = sc_data['celltype']
    sc_data = sc_data_clipped
    all_sc_data = sc_data 

    # Cap Test Data (Bulk)
    bulk_data = bulk_data.clip(upper=cap_threshold)
    
    print(f"[SUCCESS] Data pipelines aligned and capped at {cap_threshold:.4f}.")
    print("="*80 + "\n")
    # ==========================================================================


    
    # Parse the target cell types from comma-separated string to list
    target_celltypes = [ct.strip() for ct in args.target_celltypes.split(',')]
    print(f"  Target cell types for discovery: {target_celltypes}")
    
    # Load ground truth if provided (optional, for validation purposes)
    test_groundtruth_y = None
    if args.test_frac is not None and os.path.exists(args.test_frac):
        print(f" 	Loading ground truth fractions from: {args.test_frac}")
        test_groundtruth_y = pd.read_csv(args.test_frac) 
        if test_groundtruth_y.columns[0] in ['Unnamed: 0', 'index']:
            test_groundtruth_y = test_groundtruth_y.drop(columns=[test_groundtruth_y.columns[0]])        
        print(f" 	Ground truth shape: {test_groundtruth_y.shape}")

        # import pdb; pdb.set_trace()


    else:
        print(f"  No ground truth provided - running in pure discovery mode")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    print(f"\nOutput directory created: {args.output_path}")

    # --- 1. Generate or Load Unified Signature Matrix ---
    print("\n" + "="*80)
    print("Step 1: Loading Reference Signatures")
    print("="*80)

    # Load known type signatures (for initial model training)
    KNOWN_SIG_PATH = args.known_sig_path 

    if not os.path.exists(KNOWN_SIG_PATH):
        print(f"WARNING: Known signatures file not found at: {KNOWN_SIG_PATH}")
        print("Attempting to dynamically calculate and save the signature matrix from SC data...")

        # 1. Check for missing cell types
        missing_types = [ct for ct in known_cell_types_for_sim if ct not in sc_data['celltype'].unique()]

        if missing_types:
            print(f"ERROR: SC data is missing required known cell types: {missing_types}")
            print("Cannot dynamically generate signature matrix. Exiting.")
            sys.exit(1)

        # 2. Filter the SC data down to only the known types
        sc_df_filtered = sc_data[sc_data['celltype'].isin(known_cell_types_for_sim)].copy()

        # CRITICAL CHECK: Ensure the filtered data only contains the expected number of types
        if len(sc_df_filtered['celltype'].unique()) != len(known_cell_types_for_sim):
            # This is the fail-safe if the filtration logic somehow didn't work as expected
            print(f"FATAL ERROR: Filtered SC data contains {len(sc_df_filtered['celltype'].unique())} types, expected {len(known_cell_types_for_sim)}.")
            print("Please check celltype naming in SC data.")
            sys.exit(1)

        # Calculate the core signature matrix (groupby + mean)
        expression_columns = sc_df_filtered.columns.drop('celltype')
        signature_matrix = sc_df_filtered.groupby('celltype')[expression_columns].mean()

        # Create directory if necessary and save the calculated signature
        os.makedirs(os.path.dirname(KNOWN_SIG_PATH), exist_ok=True)
        signature_matrix.to_csv(KNOWN_SIG_PATH)

        print(f"SUCCESS: Dynamically calculated and saved signature matrix to: {KNOWN_SIG_PATH}")

        # Assign the calculated matrix directly to known_signatures (No redundant read_csv)
        known_signatures = signature_matrix
        
        # --- MODIFICATION END ---
    else:
        # Original logic: Load the static file if it exists
        print(f"Loaded existing known signature file from: {KNOWN_SIG_PATH}")
        known_signatures = pd.read_csv(KNOWN_SIG_PATH, index_col=0)

    print(f"Loaded known signatures: {known_signatures.shape}")
    print(f" \tTypes: {list(known_signatures.index)}")

    # Load candidate type signatures (for residual annotation)
    CANDIDATE_SIG_PATH = args.candidate_sig_path 
    if CANDIDATE_SIG_PATH is None or not os.path.exists(CANDIDATE_SIG_PATH):
        print(f"WARNING: Candidate signatures not provided or not found: {CANDIDATE_SIG_PATH}")
        print("Will use only known types for residual annotation.")
        unified_signature_matrix = known_signatures.copy()
    else:
        candidate_signatures = pd.read_csv(CANDIDATE_SIG_PATH, index_col=0)
        print(f"Loaded candidate signatures: {candidate_signatures.shape}")
        print(f" \tTypes: {list(candidate_signatures.index)}")
        
        # Merge
        unified_signature_matrix = pd.concat([known_signatures, candidate_signatures], axis=0)

        # [CRITICAL] Ensure signatures match the capped data distribution
        unified_signature_matrix = unified_signature_matrix.clip(upper=cap_threshold)

    

    print(f"\nUnified signature matrix: {unified_signature_matrix.shape}")
    print("="*80)

    # --- 2. Run the Core Distillation Algorithm ---
    print("\n" + "="*80)
    print("Step 2: Starting Multi-Round Cell Type Discovery")
    print("="*80 + "\n")


    # import pdb; pdb.set_trace()
    
    # Now, we manually filter sc_data for the initial training model
    # We also ensure the passed lists for `scseq` and `all_scseq` are consistent.
    # The initial model will train on a subset of these cells.
    # initial_train_cells_for_model = ['B cells', 'T cells', 'Monocytes', 'Macrophages', 'NK cells', 'Epithelial cells', 'Endothelial cells', 'Fibroblasts', 'ILC',  'Plasma cells', 'Mast cells', 'Erythroid', 'DC']
    # initial_train_cells_for_model = ['B cells', 'T cells', 'Macrophages', 'Epithelial cells', 'Endothelial cells', 'Fibroblasts', 'DC']
    # initial_train_cells_for_model = ['B cells', 'T cells', 'Macrophages',  'Epithelial cells', 'Endothelial cells', 'Fibroblasts', 'Plasma cells']
    # initial_train_cells_for_model = ['B cells', 'T cells', 'Macrophages',  'Epithelial cells', 'Endothelial cells', 'Fibroblasts']
    # Define the training subset using the universally passed list
    initial_train_cells_for_model = known_cell_types_for_sim
    sc_data_for_initial_model = sc_data[sc_data['celltype'].isin(initial_train_cells_for_model)].copy()
    
    complete_model = distillation(
        test_x=bulk_data,
        test_groundtruth_y=test_groundtruth_y,
        scseq=sc_data_for_initial_model, 
        all_scseq=all_sc_data, 
        sigpath=path_config.SIGPATH,
        unified_signature_matrix=unified_signature_matrix,
        output_path=args.output_path,
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        act_lr=LEARNING_RATE, 
        seed=SEED, 
        sample=SAMPLE, 
        target_celltypes=target_celltypes,
        cpu=CPU,
        endtoend_epochs=150,
        initial_known_ctypes=known_cell_types_for_sim
    )
    
    # --- Save Final Models ---
    print("\n" + "="*80)
    print("Saving final trained models...")
    print("="*80)
    
    # Save the complete model (recommended for future use)
    complete_path = os.path.join(args.output_path, "final_complete_model.pth")
    torch.save(complete_model, complete_path)
    print(f"  Complete model saved to: {complete_path}")
    
    # Save separate encoder and decoder components for backward compatibility
    # This allows older scripts to load the model if needed
    encoder_path = os.path.join(args.output_path, "final_encoder_model.pth")
    decoder_path = os.path.join(args.output_path, "final_decoder_model.pth")
    torch.save(complete_model.encoder, encoder_path)
    torch.save(complete_model.decoder, decoder_path)
    print(f"  Encoder saved to: {encoder_path}")
    print(f"  Decoder saved to: {decoder_path}")
    
    print("\n" + "="*80)
    print("Distillation pipeline completed successfully!")
    print(f"All results have been saved to: {args.output_path}")
    print("="*80)