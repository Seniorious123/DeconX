import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from ..generation.data_manager import simulate_data
from .utils import device

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def plot_residual_pca(adata_pca, round_num, output_path):
    """Generates the basic PCA scatter plot of residuals."""
    plot_dir = os.path.join(output_path, 'pca_plots_per_round')
    ensure_dir(plot_dir)
    
    pc1 = adata_pca.obsm['X_pca'][:, 0]
    pc2 = adata_pca.obsm['X_pca'][:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pc1, pc2, alpha=0.6, s=15, label='Residual Samples')
    ax.set_title(f'PCA of Residuals - Round {round_num + 1}', fontsize=16)
    ax.set_xlabel('PC1', fontsize=12); ax.set_ylabel('PC2', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5); ax.legend()
    
    path = os.path.join(plot_dir, f'round_{round_num}_pca_of_residuals.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved PCA plot to: {path}")

def plot_pca_colored_by_ground_truth(adata_pca, groundtruth, targets, round_num, output_path):
    """Generates PCA plots colored by ground truth proportions."""
    if groundtruth is None: return
    plot_dir = os.path.join(output_path, 'pca_plots_per_round')
    ensure_dir(plot_dir)

    pc1, pc2 = adata_pca.obsm['X_pca'][:, 0], adata_pca.obsm['X_pca'][:, 1]
    valid_types = [t for t in targets if t in groundtruth.columns][:2]
    props = {t: groundtruth[t].values for t in valid_types}

    for ctype, vals in props.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.scatter(pc1, pc2, c=vals, cmap='coolwarm', alpha=0.6, s=20, edgecolors='black', linewidth=0.2)
        plt.colorbar(im, ax=ax, label=f'{ctype} Proportion')
        ax.set_title(f'PCA Colored by {ctype} - Round {round_num + 1}')
        plt.savefig(os.path.join(plot_dir, f'round_{round_num}_pca_colored_by_{ctype}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    if len(valid_types) == 2:
        t1, t2 = valid_types
        ratio = np.log10((props[t1] + 1e-6) / (props[t2] + 1e-6))
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.scatter(pc1, pc2, c=ratio, cmap='RdBu_r', alpha=0.6, s=20, vmin=-2, vmax=2)
        plt.colorbar(im, ax=ax, label=f'Log10({t1}/{t2})')
        ax.set_title(f'PCA Ratio {t1}/{t2} - Round {round_num + 1}')
        plt.savefig(os.path.join(plot_dir, f'round_{round_num}_pca_colored_by_ratio.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_pca_selection(adata_pca, indices, round_num, output_path, suffix, title, label):
    """Generates PCA plots highlighting selected samples (Red dots)."""
    plot_dir = os.path.join(output_path, 'pca_plots_per_round')
    ensure_dir(plot_dir)

    mask = np.zeros(len(adata_pca.obsm['X_pca']), dtype=bool)
    mask[indices] = True
    pc1, pc2 = adata_pca.obsm['X_pca'][:, 0], adata_pca.obsm['X_pca'][:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pc1[~mask], pc2[~mask], alpha=0.6, s=15, label='All Residuals')
    ax.scatter(pc1[mask], pc2[mask], c='red', alpha=0.9, s=15, edgecolors='black', linewidth=0.2, label=label)
    
    ax.set_title(f'PCA {title} - Round {round_num + 1}', fontsize=16)
    ax.legend()
    
    path = os.path.join(plot_dir, f'round_{round_num}_{suffix}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved selection plot to: {path}")

def plot_projection_validation(adata_pca, indices, filtered_res, scseq, model, top_genes, test_x, seed, cpu, round_num, output_path):
    """Generates the projection validation plot with synthetic green triangles."""
    plot_dir = os.path.join(output_path, 'pca_plots_per_round')
    ensure_dir(plot_dir)

    fig, ax = plt.subplots(figsize=(12, 10))
    coords = adata_pca.obsm['X_pca']
    mask = np.zeros(len(coords), dtype=bool)
    mask[indices] = True

    ax.scatter(coords[~mask, 0], coords[~mask, 1], c='blue', alpha=0.3, s=40, label='Rest')
    ax.scatter(coords[mask, 0], coords[mask, 1], c='red', alpha=1.0, s=80, edgecolors='black', label='Selected')

    # Synthetic Mix Generation (Green Triangles)
    valid_types = scseq['celltype'].unique().tolist()
    # SAFE CALL: Using keyword arguments to match data_manager.py signature exactly
    sim_x, _ = simulate_data(
        sc_data=scseq,
        d_prior=tuple([0.5]*len(valid_types)),
        seed=seed + 40000 + round_num,
        ctypes=valid_types,
        cpu=cpu,
        samples=50
    )
    sim_norm = sim_x / 1000.0
    
    with torch.no_grad():
        recon, _, _ = model(torch.from_numpy(sim_norm).float().to(device))
    
    res = (sim_norm - recon.cpu().numpy())[:, top_genes]
    proj = (res - np.mean(filtered_res, axis=0)) @ adata_pca.varm['PCs']
    ax.scatter(proj[:, 0], proj[:, 1], c='green', marker='^', s=60, alpha=0.6, label='Synthetic Known Mix')

    # Annotate specific targets
    targets = {2.0: "sample_2", 0.0: "sample_0", 14.0: "sample_14", 12.0: "sample_12"}
    for tidx, txt in targets.items():
        locs = np.where(test_x.index == tidx)[0]
        if len(locs) > 0:
            x, y = coords[locs[0], :2]
            ax.scatter(x, y, s=200, facecolors='none', edgecolors='black', linewidth=1.5, zorder=10)
            ax.annotate(txt, (x, y), xytext=(10, 10), textcoords='offset points', bbox=dict(boxstyle="round", fc="yellow", alpha=0.7))

    ax.set_title(f'PCA Projection Validation - Round {round_num + 1}')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    path = os.path.join(plot_dir, f'round_{round_num}_pca_validation_projection.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved projection plot to: {path}")