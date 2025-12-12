# File: src/distiller/generation/data_manager.py
#
# This module handles all data loading and simulation tasks.
# It reads raw single-cell data and simulates pseudo-bulk expression profiles.

import anndata
import numpy as np
import pandas as pd
import random
import scipy.sparse
from multiprocessing import Pool
from numpy.random import choice
from tqdm import tqdm
from copy import deepcopy


def read_sc_input(input_file, sample=None, includes_cells=None):
    """
    Reads single-cell data from .h5ad, .txt, or .csv formats.
    """
    if '.txt' in input_file or '.csv' in input_file:
        # This will correctly handle the .csv format you generated
        sc_data = pd.read_csv(input_file, index_col=0)
    elif ".h5ad" in input_file:
        sc_data_ann = anndata.read_h5ad(input_file)
        if isinstance(sc_data_ann.X, scipy.sparse.spmatrix):
            sc_data_df = pd.DataFrame(sc_data_ann.X.toarray(), columns=sc_data_ann.var.index)
        else:
            sc_data_df = pd.DataFrame(sc_data_ann.X, columns=sc_data_ann.var.index)
        sc_data_df["celltype"] = sc_data_ann.obs["celltypes"].values
        sc_data = sc_data_df
    

    if includes_cells and isinstance(includes_cells, list):
        print(f"    --> Filtering data to include only: {includes_cells}")
        if 'celltype' in sc_data.columns:
            sc_data = sc_data[sc_data['celltype'].isin(includes_cells)].copy()
        else:
            print("    --> WARNING: 'celltype' column not found, cannot filter.")


    if sample is not None:
        # Using groupby().sample() can be slow; a faster alternative might be needed for large datasets
        sampled_df = sc_data.groupby('celltype').apply(
            lambda x: x.sample(n=sample, replace=True, random_state=1)
        ).reset_index(drop=True)
        sc_data = deepcopy(sampled_df)
        
    return sc_data

def read_bulk_input(input_file, sample=None):
    """
    Reads bulk RNA-seq data from various formats.

    Args:
        input_file (str): Path to the bulk data file.
        sample (int, optional): If specified, subsamples the data. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing bulk gene expression.
    """
    if '.txt' in input_file or '.csv' in input_file:
        if ".txt" in input_file:
            bulk_data_csv = pd.read_csv(input_file, sep='\t')
        elif ".csv" in input_file:
            bulk_data_csv = pd.read_csv(input_file)
        bulk_data = bulk_data_csv

    elif ".h5ad" in input_file:
        bulk_data_ann = anndata.read_h5ad(input_file)
        if isinstance(bulk_data_ann.X, scipy.sparse.spmatrix):
            bulk_data_df = pd.DataFrame(bulk_data_ann.X.toarray(), columns=bulk_data_ann.var.index)
        else:
            bulk_data_df = pd.DataFrame(bulk_data_ann.X, columns=bulk_data_ann.var.index)
        
        if 'sample_ids' in bulk_data_ann.obs:
            bulk_data_df.index = bulk_data_ann.obs['sample_ids'].values
        bulk_data = bulk_data_df

    if sample is not None:
        bulk_data = bulk_data.sample(n=sample, replace=True, random_state=1)

    return bulk_data


def _generate_bulk_worker(args_tuple):
    """
    Internal worker function for parallel data generation.
    Note: Using a single tuple for arguments is a common pattern for multiprocessing.Pool.
    """
    sc_data, dirichlet_alpha, ctypes, n, sample, show, seed, min_cell_ratio, unknown_indices = args_tuple
    
    random.seed(seed)
    np.random.seed(seed)

    # Convert to DataFrame if ndarray
    subsets = [pd.DataFrame(sc) if isinstance(sc, np.ndarray) else sc for sc in sc_data]

    alpha = np.array(dirichlet_alpha)
    prop = np.random.dirichlet(alpha, sample)

    if unknown_indices: # Only run if there are specified unknown types
        for i in range(prop.shape[0]): # Iterate over each generated sample
            sample_prop = prop[i, :]
            total_excess = 0.0

            # Identify which unknown types exceed the 0.5 cap and calculate the total excess
            capped_indices = []
            for u_idx in unknown_indices:
                if sample_prop[u_idx] > 0.5:
                    total_excess += sample_prop[u_idx] - 0.5
                    sample_prop[u_idx] = 0.5
                    capped_indices.append(u_idx)

            # If there is excess proportion, redistribute it proportionally to other cell types
            if total_excess > 0:
                # Identify all other cell types that will receive the redistributed proportion
                redistribution_indices = [idx for idx in range(len(sample_prop)) if idx not in capped_indices]

                # Calculate the sum of proportions for the receiving cell types
                sum_for_redistribution = sample_prop[redistribution_indices].sum()

                if sum_for_redistribution > 1e-9: # Avoid division by zero
                    # Distribute the excess proportionally based on the current size of other cell types
                    proportions_for_redistribution = sample_prop[redistribution_indices] / sum_for_redistribution
                    sample_prop[redistribution_indices] += total_excess * proportions_for_redistribution
                elif len(redistribution_indices) > 0:
                    # Edge case: if other cells are all zero, distribute the excess evenly
                    add_amount = total_excess / len(redistribution_indices)
                    sample_prop[redistribution_indices] += add_amount

            # Update the proportion array for the current sample
            prop[i, :] = sample_prop

    # Final re-normalization to ensure everything sums exactly to 1 after adjustments
    prop /= prop.sum(axis=1)[:, np.newaxis]
    
    # Ensure a minimum proportion for each cell type to avoid zero-counts
    prop[prop < min_cell_ratio] = min_cell_ratio
    prop /= prop.sum(axis=1)[:, np.newaxis]  # Renormalize to sum to 1
    
    cell_num = np.floor(n * prop)
    cell_sum = np.sum(cell_num, axis=1, keepdims=True)
    
    # Avoid division by zero if a sample has no cells
    valid_samples = cell_sum > 0
    cell_sum = cell_sum[valid_samples]
    prop = cell_num[valid_samples.flatten()] / cell_sum[:, np.newaxis]
    cell_num = cell_num[valid_samples.flatten()]
    
    samples_array = np.zeros((cell_num.shape[0], sc_data[0].shape[1]))
    tqdm_range = tqdm(range(cell_num.shape[0])) if show else range(cell_num.shape[0])

    for i in tqdm_range:
        for cidx, ctype_name in enumerate(ctypes):
            n_cells_to_sample = int(cell_num[i, cidx])
            if n_cells_to_sample > 0:
                subset_df = subsets[cidx]
                select_indices = choice(subset_df.index, size=n_cells_to_sample, replace=True)
                samples_array[i] += subset_df.loc[select_indices].values.sum(axis=0)
        samples_array[i] /= cell_sum[i]

    prop_df = pd.DataFrame(prop, columns=ctypes)
    
    return samples_array, prop_df


def simulate_data(sc_data, d_prior, ctypes, seed=0, samples=600, n_cells_per_bulk=3000, 
                  min_cell_ratio=1e-3, cpu=10, unknown_ctypes=None):
    """
    Simulates pseudo-bulk data from a single-cell DataFrame.

    Args:
        sc_data (pd.DataFrame): DataFrame with expression and a 'celltype' column.
        d_prior (tuple or list): Dirichlet alpha parameters for proportion generation.
        ctypes (list): List of cell types to include in the simulation.
        seed (int): Random seed for reproducibility.
        samples (int): Total number of bulk samples to generate.
        n_cells_per_bulk (int): Number of single cells to pool for one bulk sample.
        min_cell_ratio (float): Minimum proportion for any cell type.
        cpu (int): Number of CPU cores to use for parallel generation.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the simulated bulk data (train_x)
                                       and the corresponding proportions (train_y).
    """
    print(f"Starting data simulation for {samples} samples...")

    # Split dataset by cell type and remove the label column for processing
    sc_subsets = [
        sc_data[sc_data['celltype'] == celltype].drop(columns='celltype').reset_index(drop=True)
        for celltype in ctypes
    ]
    
    # Setup for multiprocessing
    random.seed(seed)
    seeds = [random.randint(0, 2**32-1) for _ in range(cpu)]
    samples_per_process = samples // cpu
    
    args_list = []
    # Find the integer indices of the unknown cell types to pass to the worker
    unknown_indices = []
    if unknown_ctypes:
        unknown_indices = [i for i, ctype in enumerate(ctypes) if ctype in unknown_ctypes]
    for i in range(cpu):
        # The last process takes any remaining samples
        current_samples = samples_per_process if i < cpu - 1 else samples - (samples_per_process * (cpu - 1))
        if current_samples == 0:
            continue
            
        # The first process will show the tqdm progress bar
        show_progress = (i == 0)
        
        args_list.append((
            sc_subsets, d_prior, ctypes, n_cells_per_bulk, current_samples, 
            show_progress, seeds[i], min_cell_ratio, unknown_indices
        ))

    # Run generation in parallel
    with Pool(cpu) as pool:
        results = pool.map(_generate_bulk_worker, args_list)

    # Concatenate results from all processes
    all_data = np.concatenate([result[0] for result in results], axis=0)
    all_frac = pd.concat([result[1] for result in results], axis=0).reset_index(drop=True)

    print("Data simulation completed successfully.")
    return all_data, all_frac.values