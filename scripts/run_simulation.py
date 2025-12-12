# File: scripts/run_simulation.py
"""
Main executable script for the Data Generation (Simulation) part of the project.

This script takes primary single-cell data and generates a pseudo-bulk dataset
with corresponding cell fractions based on specified parameters.

Example Usage (replaces 'gen_exp_single_CD16_alpha_0.5.sh'):
------------------------------------------------------------------
python scripts/run_simulation.py \
    --config configs/path_config.py \
    --experiment-name "single_CD16_alpha_0.5" \
    --unknown-celltypes CD16 \
    --alphas 0.5 \
    --num-samples 3000 \
    --seed 42
"""

import os
import sys
import argparse
import importlib.util
import pandas as pd

# This is the standard way to make the 'src' directory visible to the script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from our new, clean, modularized code library
from src.distiller.generation.data_manager import read_sc_input, simulate_data

def load_config_from_path(path):
    """Loads a Python configuration file as a module."""
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def main():
    """Main function to parse arguments and run the data simulation."""
    parser = argparse.ArgumentParser(
        description="Generate pseudo-bulk data from single-cell profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Configuration and I/O Arguments ---
    parser.add_argument('--config', type=str, required=True, help="Path to the python config file for file paths.")
    parser.add_argument('--experiment-name', type=str, required=True, help="A unique name for this simulation run. This will be used as the output folder name.")

    # --- Core Scientific Variables ---
    parser.add_argument('--known-celltypes', nargs='+', default=['B_cell', 'CD4', 'CD8', 'CD14'], help="List of known background cell types.")
    parser.add_argument('--unknown-celltypes', nargs='+', required=True, help="List of new cell types to simulate.")
    parser.add_argument('--alphas', nargs='+', type=float, required=False, help="List of alpha values for the unknown cell types' Dirichlet prior.")
    parser.add_argument('--alpha-known', type=int, default=2, help="Alpha value for all known cell types.")

    # --- Simulation Hyperparameters ---
    parser.add_argument('--num-samples', type=int, default=3000, help="Number of pseudo-bulk samples to generate.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--cpu', type=int, default=10, help="Number of CPU cores to use for data simulation.")

    args = parser.parse_args()

    # --- 1. Load Configuration and Set Up ---
    print("--- Step 1: Loading configuration and setting up ---")
    config = load_config_from_path(args.config)
    
    # Create a unique output directory for this simulation
    output_dir = os.path.join("outputs", "simulated_data", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generated data will be saved to: {output_dir}")

    # --- 2. Load Primary Data ---
    print(f"\n--- Step 2: Loading primary single-cell data from: {config.sc_data_path} ---")
    sc_data = read_sc_input(config.sc_data_path)

    # --- 3. Prepare Parameters for Simulation ---
    print("\n--- Step 3: Preparing parameters for simulation ---")
    all_celltypes = args.known_celltypes + args.unknown_celltypes

    if args.alphas:
        # --- [LEGACY METHOD] ---
        # If alphas are provided, use the original simulation method.
        print("Alphas provided via command line. Using standard simulation method.")

        full_alphas = [args.alpha_known] * len(args.known_celltypes) + args.alphas
        d_prior_tuple = tuple(full_alphas)
        print(f"Using Dirichlet prior: {d_prior_tuple}")

        bulk_data, frac_data = simulate_data(
            sc_data=sc_data,
            d_prior=d_prior_tuple,
            ctypes=all_celltypes,
            seed=args.seed,
            samples=args.num_samples,
            cpu=args.cpu,
            unknown_ctypes=args.unknown_celltypes
        )
        frac_df = pd.DataFrame(frac_data, columns=all_celltypes)

    else:
        # --- [NEW STRATIFIED METHOD] ---
        # If no alphas are provided, use the new internal stratified logic.
        print("No alphas provided via command line. Using stratified simulation method for exactly two unknown cell types.")

        # This safeguard ensures the stratified logic is only used for the case it was designed for.
        if len(args.unknown_celltypes) != 2:
            raise ValueError(f"Stratified sampling is currently implemented for exactly two unknown cell types, but {len(args.unknown_celltypes)} were provided.")

        # Dynamically get the names of the two unknown cell types
        unknown1, unknown2 = args.unknown_celltypes

        strata = [
            {'name': f'Batch 1 (Highlighting {unknown1})', 'samples': 1000, 'alphas': [0.5, 2.0]},
            {'name': 'Batch 2 (Mixed Signals)',            'samples': 1000, 'alphas': [2.0, 2.0]},
            {'name': f'Batch 3 (Highlighting {unknown2})', 'samples': 1000, 'alphas': [2.0, 0.5]}
        ]

        all_bulk_batches = []
        all_frac_batches = []

        for i, stratum in enumerate(strata):
            print(f"\nGenerating {stratum['name']} ({stratum['samples']} samples)...")

            batch_alphas = [args.alpha_known] * len(args.known_celltypes) + stratum['alphas']
            batch_d_prior = tuple(batch_alphas)
            print(f"Using Dirichlet prior: {batch_d_prior}")

            batch_bulk, batch_frac = simulate_data(
                sc_data=sc_data,
                d_prior=batch_d_prior,
                ctypes=all_celltypes,
                seed=args.seed + i,
                samples=stratum['samples'],
                cpu=args.cpu,
                unknown_ctypes=args.unknown_celltypes
            )

            all_bulk_batches.append(pd.DataFrame(batch_bulk, columns=sc_data.columns.drop('celltype')))
            all_frac_batches.append(pd.DataFrame(batch_frac, columns=all_celltypes))

        print("\nConcatenating results from all batches...")
        bulk_df = pd.concat(all_bulk_batches, ignore_index=True)
        frac_df = pd.concat(all_frac_batches, ignore_index=True)
        # Convert bulk_data back to a numpy array for downstream compatibility, if necessary
        # However, keeping it as a DataFrame is often safer. We'll leave the conversion to the end.
        bulk_data = bulk_df.values

    # --- 5. Save Outputs ---
    print("\n--- Step 5: Saving generated data ---")
    bulk_path = os.path.join(output_dir, "bulk.csv")
    frac_path = os.path.join(output_dir, "frac.csv")
    
    pd.DataFrame(bulk_data, columns=sc_data.columns.drop('celltype')).to_csv(bulk_path, index=False)
    frac_df.to_csv(frac_path, index=False)
    
    print(f"Successfully saved bulk data to: {bulk_path}")
    print(f"Successfully saved fraction data to: {frac_path}")
    print("\n--- Simulation Finished ---")

if __name__ == "__main__":
    main()