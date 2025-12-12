import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import Counter
import umap
import matplotlib.pyplot as plt
import scanpy as sc
import anndata
import matplotlib
matplotlib.use('Agg')


from .models import SimpleDataset
from .utils import find_best_match
from .analysis import validate_voting_winner_reliability, compute_signature_similarity, convert_numpy_types

def plot_residual_umap(all_residuals, similarities_to_winner, output_path, round_num, winner_celltype):
    """
    Performs UMAP dimensionality reduction on residual data and visualizes it.

    Parameters:
    - all_residuals (np.ndarray): The residual matrix of shape (n_samples, n_genes).
    - similarities_to_winner (np.ndarray): An array of similarity scores for each sample against the winner's signature.
    - output_path (str): The directory to save the output plot.
    - round_num (int): The current discovery round number.
    - winner_celltype (str): The name of the winning cell type for the current round.
    """
    print("Generating UMAP plot for residuals...")
    if len(all_residuals) < 15:
        print("Warning: Not enough samples for robust UMAP (< 15). Skipping plot.")
        return

    # Initialize and train the UMAP model
    # n_neighbors and min_dist are key parameters for tuning
    reducer = umap.UMAP(
        n_neighbors=15,  # Controls the balance between local and global structure
        min_dist=0.1,    # Controls how tightly UMAP is allowed to pack points together
        n_components=2,
        random_state=42  # Ensures reproducibility
    )
    embedding = reducer.fit_transform(all_residuals)

    # Create the scatter plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=similarities_to_winner, # Use similarity scores for coloring
        cmap='viridis',           # 'viridis' colormap is clear from low (purple) to high (yellow)
        s=5,                      # Size of the points
        alpha=0.7                 # Opacity of the points
    )

    # Add a colorbar to show the mapping of colors to similarity scores
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'Similarity', rotation=270, labelpad=15, fontsize=15)

    # Add titles and labels
    # plt.title(f'UMAP of Residuals - Round {round_num + 1}\n(Colored by Similarity to "{winner_celltype}")', fontsize=16)
    # plt.xlabel('UMAP 1', fontsize=12)
    # plt.ylabel('UMAP 2', fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.5)

    # Save the figure
    umap_fig_path = os.path.join(output_path, f'round_{round_num}_residual_umap.png')
    plt.savefig(umap_fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"UMAP plot of residuals saved to: {umap_fig_path}")



def vote_for_celltype(ae_model, main_bulk, calibration_bulk, filtered_sig_matrix, batch_size, output_path, output_name, round_num, top_k=None):
    """
    Voting mechanism to find the best matching cell type.
    Analyzes residuals from each test sample to identify the most likely unknown cell type.
    """

    print("Starting geometric analysis for candidate discovery...")
    ae_model.eval()

    # Step 1: Calculate residuals for both datasets
    # Helper function to get residuals from a bulk dataset
    def get_residuals(model, bulk_data, batch_size):
        data_loader = DataLoader(SimpleDataset(bulk_data), batch_size=batch_size, shuffle=False)
        all_residuals = []
        with torch.no_grad():
            for data in data_loader:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                reconstructed_data, _, _ = model(data)
                residual = (data - reconstructed_data).cpu().numpy()
                all_residuals.append(residual)
        return np.vstack(all_residuals)

    print("Calculating residuals for the main analysis set...")
    residuals_A = get_residuals(ae_model, main_bulk, batch_size)
    print("Calculating residuals for the calibration set...")
    residuals_B = get_residuals(ae_model, calibration_bulk, batch_size)

    # Step 2: Create a joint AnnData object
    all_residuals = np.vstack([residuals_A, residuals_B])
    adata = anndata.AnnData(all_residuals)
    data_source_list = ['Test Bulk'] * len(residuals_A) + ['Calibration Set'] * len(residuals_B)
    adata.obs['data_source'] = pd.Categorical(data_source_list)

    # Step 3: Run Scanpy workflow
    print("Running Scanpy workflow (neighbors, umap, dpt)...")
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    sc.tl.diffmap(adata)
    sc.tl.umap(adata)

    # Step 4: Locate the root cell using the start of the diffusion trajectory
    # This method is more robust for complex-shaped starting clusters than the centroid method.
    # It identifies the cell at the most "stable" or "earliest" state of the diffusion process,
    # which typically corresponds to the minimum value of the first non-trivial diffusion component (DC1).
    # Find the indices of the cells at both ends (poles) of the first diffusion component.
    dc1_min_idx = np.argmin(adata.obsm['X_diffmap'][:, 1])
    dc1_max_idx = np.argmax(adata.obsm['X_diffmap'][:, 1])

    # Check which of these two poles falls within our known 'Calibration Set'.
    # This intelligently determines the true start of the trajectory, regardless of direction.
    if adata.obs['data_source'][dc1_min_idx] == 'Calibration Set':
        root_cell_index = dc1_min_idx
        print("Root cell identified at the minimum of DC1.")
    elif adata.obs['data_source'][dc1_max_idx] == 'Calibration Set':
        root_cell_index = dc1_max_idx
        print("Root cell identified at the maximum of DC1, trajectory is inverted.")
    else:
        # Fallback in case neither pole is in the calibration set (highly unlikely)
        # We find the cell in the calibration set that is closest to a pole.
        print("Warning: Neither DC1 pole is in the calibration set. Using fallback method.")
        calib_indices = np.where(adata.obs['data_source'] == 'Calibration Set')[0]
        calib_dc1_values = adata.obsm['X_diffmap'][calib_indices, 1]
        # Find the index within the calibration set that is most extreme
        most_extreme_local_idx = np.argmin(np.abs(calib_dc1_values - np.mean(calib_dc1_values)))
        root_cell_index = calib_indices[most_extreme_local_idx]

    adata.uns['iroot'] = root_cell_index

    # Step 5: Calculate DPT (Geodesic Distance)
    sc.tl.dpt(adata)

    # Step 6: Calculate unsupervised diagnostic metrics
    calib_indices = np.where(adata.obs['data_source'] == 'Calibration Set')[0]

    if len(calib_indices) > 0:
        calib_coords = adata.obsm['X_umap'][calib_indices]
        calib_centroid = np.mean(calib_coords, axis=0)

        # Calculate the distances of ONLY calibration points to their own centroid
        calib_distances_to_centroid = np.linalg.norm(calib_coords - calib_centroid, axis=1)
        origin_dispersion = np.mean(calib_distances_to_centroid)
    else:
        # Handle the edge case where there might be no calibration points
        origin_dispersion = 0.0 
    # Placeholder for k-NN stability check (can be implemented as discussed)
    # For now, we'll just report the dispersion.
    diagnostics_report = {
        'origin_cluster_dispersion': origin_dispersion
    }
    print(f"Diagnostic Report: Origin Cluster Dispersion = {origin_dispersion:.4f}")

    # Step 7: Filter samples using a hybrid score of pseudotime and spatial proximity.
    
    print("Filtering samples using hybrid score (pseudotime + proximity to endpoint)...")
    
    # 7.1: Identify the endpoint (the single cell with the maximum pseudotime).
    main_bulk_mask = (adata.obs['data_source'] == 'Test Bulk')
    # Use idxmax() to get the index label of the max value
    endpoint_index_label = adata.obs['dpt_pseudotime'][main_bulk_mask].idxmax()
    # Get the integer position (.iloc) of that label for coordinate lookup
    endpoint_iloc = adata.obs.index.get_loc(endpoint_index_label)
    endpoint_coords = adata.obsm['X_umap'][endpoint_iloc]

    # 7.2: Calculate the two core metrics for all main bulk samples.
    main_bulk_indices = np.where(main_bulk_mask)[0]
    pseudotime_values = adata.obs['dpt_pseudotime'].iloc[main_bulk_indices]
    distances_to_endpoint = np.linalg.norm(adata.obsm['X_umap'][main_bulk_indices] - endpoint_coords, axis=1)

    # 7.3: Normalize both metrics to a [0, 1] scale for fair combination.
    # Handle edge case where max == min to avoid division by zero.
    pt_range = pseudotime_values.max() - pseudotime_values.min()
    dist_range = distances_to_endpoint.max() - distances_to_endpoint.min()
    
    norm_pseudotime = (pseudotime_values - pseudotime_values.min()) / (pt_range if pt_range > 0 else 1.0)
    norm_distances = (distances_to_endpoint - distances_to_endpoint.min()) / (dist_range if dist_range > 0 else 1.0)

    # 7.4: Calculate the Hybrid Score.
    # We want high pseudotime (high norm_pseudotime) and low distance (high 1 - norm_distances).
    hybrid_score = norm_pseudotime + (1 - norm_distances)
    
    # Store the score in adata.obs for potential visualization or analysis
    # Use .loc for safe assignment to a slice of the DataFrame
    adata.obs.loc[main_bulk_mask, 'hybrid_score'] = hybrid_score

    # 7.5: Select the top 30% of samples based on the new hybrid score.
    score_threshold = hybrid_score.quantile(0.70)
    # Ensure to select from adata.obs where the score was actually calculated
    high_confidence_mask = (adata.obs['hybrid_score'] > score_threshold) & main_bulk_mask
    high_confidence_indices = np.where(high_confidence_mask)[0]
    
    print(f"Selected {len(high_confidence_indices)} high-confidence samples using Hybrid Score thresholding.")
    
    # Step 8: Annotation via Individual Sample Voting
    # Instead of averaging all high-confidence residuals, we let each sample vote
    # for its best match. This is more robust to noise and mixed signals.
    
    high_confidence_residuals = adata.X[high_confidence_indices]
    
    print(f"Performing individual voting on {len(high_confidence_residuals)} high-confidence samples...")
    votes = []
    # Use tqdm for a progress bar if the number of samples is large
    from tqdm import tqdm
    for residual_sample in tqdm(high_confidence_residuals, desc="Annotating samples"):
        # For each individual sample, find its best match
        best_match, _, _ = find_best_match(residual_sample, filtered_sig_matrix)
        votes.append(best_match)
    
    # Tally the votes using collections.Counter
    if not votes:
        # Handle edge case where no high-confidence samples were found
        best_matched_celltype = "Unknown"
        print("Warning: No high-confidence samples to vote on.")
    else:
        vote_counts = Counter(votes)
        # The winner is the cell type with the most votes
        best_matched_celltype = vote_counts.most_common(1)[0][0]
        
        print("\n--- Annotation Voting Results ---")
        for celltype, count in vote_counts.most_common(5): # Print top 5 candidates
            print(f"  {celltype}: {count} votes")
        print("-------------------------------")

    print(f"Final annotated signal: {best_matched_celltype}")


    # Step 9: Visualization
    # Call the new plotting function we defined
    plot_discovery_manifold(adata, round_num, output_path, output_name)
    
    # The original function returned several items. We simplify the return.
    # The main distillation loop will need to be adjusted to unpack this new return signature.
    # For now, we return the essentials.
    all_residuals = adata.X[main_bulk_mask] # Return only residuals from the main bulk
    celltype_counts = {best_matched_celltype: len(high_confidence_indices)} # Dummy value

    
    return {
        "best_matched_celltype": best_matched_celltype,
        "high_confidence_residuals": high_confidence_residuals,
        "diagnostics_report": diagnostics_report,
        "all_residuals": all_residuals,  # Pass along all residuals from the main bulk
        "voting_counts": celltype_counts # Pass along the dummy voting counts
    }


def plot_discovery_manifold(adata, round_num, output_path, output_name):
    """
    Generates and saves UMAP plots for discovery analysis.
    1. A plot colored by data source to show the location of the calibration set.
    2. A plot colored by DPT pseudotime to show the discovered trajectory.
    
    Args:
        adata (anndata.AnnData): The AnnData object containing UMAP and DPT results.
        round_num (int): The current discovery round number.
        output_path (str): The directory to save the plots.
        output_name (str): The base name for the output files.
    """
    if 'X_umap' not in adata.obsm:
        print("Warning: UMAP coordinates not found. Skipping plotting.")
        return

    # Plot 1: Color by data source
    # This plot helps verify that the calibration set forms a distinct cluster.
    fig, ax = plt.subplots(figsize=(12, 10))
    ax_source = sc.pl.umap(
        adata,
        color='data_source',
        palette={'Test Bulk': '#1f77b4', 'Calibration Set': '#d62728'}, # Blue for test, Red for calibration
        s=50,
        title=f'Round {round_num + 1}: Data Source Location',
        show=False,
        ax=ax
    )

    # Use the robust plt.savefig() to save the figure to a precise path.
    plot_path = os.path.join(output_path, f'{output_name}round_{round_num}_source_location.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved data source plot to: {plot_path}")

    plt.close()

    # Plot 2: Color by DPT pseudotime with the root cell marked
    # This plot shows the trajectory away from the origin.
    if 'dpt_pseudotime' in adata.obs:
        plt.figure()
        # Create the base UMAP plot colored by pseudotime
        ax = sc.pl.umap(
            adata,
            color='dpt_pseudotime',
            cmap='viridis',
            s=50,
            title=f'Round {round_num + 1}: Pseudotime Trajectory',
            show=False
        )

        # Overlay a marker for the root cell
        if 'iroot' in adata.uns:
            root_index = adata.uns['iroot']
            root_coords = adata.obsm['X_umap'][root_index]
            ax.scatter(root_coords[0], root_coords[1], s=60, c='red', marker='o', zorder=10, label='Start Point')
            ax.legend()
        
        plt.savefig(os.path.join(output_path, f'{output_name}round_{round_num}_trajectory.png'), dpi=300, bbox_inches='tight')
        plt.close()