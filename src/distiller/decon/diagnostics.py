"""
distillation_diagnostics.py

Independent diagnostic functions module for cell type discovery process.
Separated from main distillation function to improve code modularity and maintainability.
"""

import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


# Use relative imports to load from sibling modules in the 'decon' package
from .models import SimpleDataset
from .utils import find_best_match, evaluate
from scipy.stats import pearsonr


def evaluate_initial_model(ae_model, test_x, test_groundtruth_y, sigpath, output_path, train_cells, batch_size):
    """
    Initial model evaluation diagnostic
    """
    if test_groundtruth_y is None:
        print("No test groundtruth provided, skipping initial model evaluation")
        return
    
    print("\n=== Initial Model Evaluation Report ===")
    eval_dir = os.path.join(output_path, 'eval_initial/')
    os.makedirs(eval_dir, exist_ok=True)
    
    try:
        evaluate(ae_model, test_x, test_groundtruth_y, sigpath, eval_dir,
                cell_types=train_cells.copy(), batch_size=batch_size)
        print(f"Initial model evaluation completed successfully")
        print(f"Results saved to: {eval_dir}")
    except Exception as e:
        print(f"Initial model evaluation failed: {str(e)}")


def generate_similarity_ranking_report(avg_residual, filtered_sig_matrix, output_path, round_num):
    """
    Generate similarity ranking report
    """
    print("\n=== Similarity Ranking Analysis ===")
    
    _, _, similarity_series = find_best_match(avg_residual, filtered_sig_matrix)
    similarities = similarity_series.to_dict()
    sorted_sim = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    print("Average Residual Similarity Ranking:")
    for i, (celltype, sim) in enumerate(sorted_sim):
        status = "TOP1" if i == 0 else "TOP2" if i == 1 else "TOP3" if i == 2 else "    "
        print(f"  {status} {i+1:2d}. {celltype:<20}: {sim:.4f}")
    
    ranking_df = pd.DataFrame(sorted_sim, columns=['Cell_Type', 'Similarity_Score'])
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)
    ranking_file = os.path.join(output_path, f'round_{round_num}_similarity_ranking.csv')
    ranking_df.to_csv(ranking_file, index=False)
    
    print(f"Similarity ranking saved to: {ranking_file}")
    
    return {
        'similarities': similarities,
        'sorted_similarities': sorted_sim,
        'best_match': sorted_sim[0][0] if sorted_sim else None,
        'best_score': sorted_sim[0][1] if sorted_sim else 0.0
    }


def diagnose_weight_initialization(model, test_loader, extended_train_cells, device, round_num, output_path):
    """
    Diagnose weight initialization state
    """
    print("\n=== Weight Initialization Diagnostic ===")
    print(f"Extended cell type sequence: {extended_train_cells}")
    
    model.eval()
    test_sample = next(iter(test_loader))
    if isinstance(test_sample, (list, tuple)):
        test_sample = test_sample[0]
    
    diagnosis_results = {}
    warnings = []
    
    with torch.no_grad():
        x = test_sample.to(device)
        
        for layer in model.encoder[:-2]:
            x = layer(x)
        
        initial_raw_outputs = x.cpu().numpy()
        
        print("Neuron output states after weight copying, before training:")
        for i, cell_type in enumerate(extended_train_cells):
            if i >= initial_raw_outputs.shape[1]:
                continue
                
            neuron_values = initial_raw_outputs[:, i]
            mean_val = neuron_values.mean()
            negative_ratio = (neuron_values < 0).mean()
            std_val = neuron_values.std()
            zero_ratio = (neuron_values == 0).mean()
            
            status = "Normal"
            if negative_ratio > 0.9:
                status = "Critical"
                warnings.append(f"{cell_type}: Almost all negative values ({negative_ratio:.1%})")
            elif negative_ratio > 0.5:
                status = "Warning" 
                warnings.append(f"{cell_type}: High negative ratio ({negative_ratio:.1%})")
            elif mean_val < -1.0:
                status = "Warning"
                warnings.append(f"{cell_type}: Severely negative mean ({mean_val:.6f})")
            
            print(f"  {cell_type:<20}: Mean={mean_val:8.6f}, Neg%={negative_ratio:5.1%}, "
                  f"Std={std_val:7.4f}, Zero%={zero_ratio:4.1%} {status}")
            
            diagnosis_results[cell_type] = {
                'mean': mean_val,
                'std': std_val,
                'negative_ratio': negative_ratio,
                'zero_ratio': zero_ratio,
                'status': status
            }
    
    model.train()
    
    diag_df = pd.DataFrame(diagnosis_results).T
    diag_df.index.name = 'Cell_Type'
    diag_file = os.path.join(output_path, f'round_{round_num}_weight_initialization_diagnosis.csv')
    diag_df.to_csv(diag_file)
    
    if warnings:
        print(f"\nWeight Initialization Warnings ({len(warnings)} issues found):")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nAll neurons show healthy initialization states")
    
    print(f"Weight diagnosis saved to: {diag_file}")
    
    return {
        'results': diagnosis_results,
        'warnings': warnings,
        'healthy_neurons': len([r for r in diagnosis_results.values() if r['status'] == 'Normal']),
        'total_neurons': len(diagnosis_results)
    }


def generate_training_loss_plot(losses, output_path, round_num, plot_name='training_loss'):
    """
    Generate training loss curve plot
    """
    if not losses:
        print("No loss data to plot")
        return None
    
    print(f"\n=== Generating Training Loss Plot ===")
    
    plt.figure(figsize=(12, 6))
    plt.plot(losses, 'b-', linewidth=1, alpha=0.7)
    
    if len(losses) > 10:
        window_size = min(50, len(losses) // 10)
        moving_avg = pd.Series(losses).rolling(window=window_size, center=True).mean()
        plt.plot(moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.legend()
    
    plt.title(f'Training Loss Curve - Round {round_num}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(output_path, f'round_{round_num}_{plot_name}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    final_loss = losses[-1]
    min_loss = min(losses)
    max_loss = max(losses)
    avg_loss = np.mean(losses)
    
    print(f"Loss Statistics:")
    print(f"  Final Loss: {final_loss:.6f}")
    print(f"  Min Loss: {min_loss:.6f}")
    print(f"  Max Loss: {max_loss:.6f}")
    print(f"  Average Loss: {avg_loss:.6f}")
    print(f"Loss plot saved to: {plot_file}")
    
    return plot_file


def compare_learned_vs_reference_signatures(final_model, discovered_types, filtered_sig_matrix, 
                                           high_confidence_residuals, output_path, round_num):
    """
    Compare learned signatures vs reference signatures
    """
    if not discovered_types:
        print("No discovered types to compare")
        return {}
    
    print(f"\n=== Learned vs Reference Signature Comparison ===")
    
    results = {}
    
    final_learned_sig = final_model.sigmatrix()[-1, :].detach().cpu().numpy()
    last_discovered = discovered_types[-1]
    
    if last_discovered not in filtered_sig_matrix.index:
        print(f"Reference signature for '{last_discovered}' not found")
        return {}
    
    reference_sig = filtered_sig_matrix.loc[last_discovered].values
    reference_sig_df = pd.DataFrame([reference_sig], index=[last_discovered], 
                                   columns=filtered_sig_matrix.columns)
    
    _, sim_learned_ref, _ = find_best_match(final_learned_sig, reference_sig_df)
    
    print(f"\n[Diagnostic Report 1: Final Learned Signature vs Reference Signature]")
    print(f"Final Grade: {sim_learned_ref:.4f}")
    print(f"   Comparison: Learned '{last_discovered}' signature vs Reference '{last_discovered}' signature")
    
    avg_initial_residual_sig = np.mean(high_confidence_residuals, axis=0)
    _, sim_source_ref, _ = find_best_match(avg_initial_residual_sig, reference_sig_df)
    
    print(f"\n[Diagnostic Report 2: Simulation Source vs Reference Signature]")
    print(f"Textbook Quality: {sim_source_ref:.4f}")
    print(f"   Comparison: Initial residual (simulation source) vs Reference '{last_discovered}' signature")
    
    improvement = sim_learned_ref - sim_source_ref
    improvement_pct = (improvement / sim_source_ref) * 100 if sim_source_ref > 0 else 0
    
    print(f"\nLearning Progress Analysis:")
    print(f"   Initial Quality: {sim_source_ref:.4f}")
    print(f"   Final Quality:   {sim_learned_ref:.4f}")
    print(f"   Improvement:     {improvement:+.4f} ({improvement_pct:+.1f}%)")
    
    if improvement > 0.1:
        print("   Significant learning improvement achieved")
    elif improvement > 0.05:
        print("   Moderate learning improvement")
    elif improvement > 0:
        print("   Minimal learning improvement")
    else:
        print("   No improvement or degradation detected")
    
    comparison_data = {
        'Metric': ['Final_Learned_vs_Reference', 'Initial_Residual_vs_Reference', 'Learning_Improvement'],
        'Score': [sim_learned_ref, sim_source_ref, improvement],
        'Description': [
            f'Final learned signature vs {last_discovered} reference',
            f'Initial residual vs {last_discovered} reference', 
            'Improvement from initial to final'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = os.path.join(output_path, f'round_{round_num}_signature_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"Signature comparison saved to: {comparison_file}")
    
    results = {
        'learned_vs_reference': sim_learned_ref,
        'source_vs_reference': sim_source_ref,
        'improvement': improvement,
        'improvement_percent': improvement_pct,
        'discovered_type': last_discovered
    }
    
    return results


def generate_signature_similarity_table(mixed_scseq, final_sim_cells, anonymous_celltype, 
                                       sigpath, discovered_types, output_path):
    """
    Generate signature similarity comparison table
    """
    print(f"\n=== Signature Similarity Table Generation ===")
    
    if not discovered_types:
        print("No discovered types, skipping similarity table generation")
        return pd.DataFrame()
    
    def compute_simulated_signatures(mixed_scseq, cell_types):
        return mixed_scseq.groupby('celltype').mean().reindex(cell_types)
    
    def compute_signature_similarity(sim_sig, ref_sig):
        sim_norm = sim_sig / (np.linalg.norm(sim_sig) + 1e-8)
        ref_norm = ref_sig / (np.linalg.norm(ref_sig) + 1e-8)
        return np.dot(sim_norm, ref_norm)
    
    simulated_signatures = compute_simulated_signatures(mixed_scseq, final_sim_cells)
    reference_signatures = pd.read_csv(sigpath, index_col=0)
    
    results = []
    last_discovered = discovered_types[-1] if discovered_types else "Unknown"
    
    for cell_type in final_sim_cells:
        if cell_type == anonymous_celltype:
            comparison_type = last_discovered
        else:
            comparison_type = cell_type
            
        if comparison_type not in reference_signatures.index:
            print(f"Reference signature for '{comparison_type}' not found, skipping")
            continue
            
        sim_vec = simulated_signatures.loc[cell_type].values
        ref_vec = reference_signatures.loc[comparison_type].values
        similarity = compute_signature_similarity(sim_vec, ref_vec)
        
        results.append({
            'Cell_Type': cell_type,
            'Reference_Type': comparison_type,
            'Similarity': similarity,
            'Is_Unknown': cell_type == anonymous_celltype,
            'Category': 'Unknown' if cell_type == anonymous_celltype else 'Known'
        })
    
    similarity_table = pd.DataFrame(results)
    
    print("\nSimulated vs Reference Signature Similarity:")
    print("=" * 70)
    
    for _, row in similarity_table.iterrows():
        marker = "UNKNOWN" if row['Is_Unknown'] else "KNOWN  "
        print(f"{marker} {row['Cell_Type']:<20} vs {row['Reference_Type']:<15}: {row['Similarity']:.4f}")
    
    known_sims = similarity_table[~similarity_table['Is_Unknown']]['Similarity']
    unknown_sims = similarity_table[similarity_table['Is_Unknown']]['Similarity']
    
    print("=" * 70)
    
    if len(known_sims) > 0:
        print(f"Known types statistics:")
        print(f"   Average similarity: {known_sims.mean():.4f}")
        print(f"   Min similarity: {known_sims.min():.4f}")
        print(f"   Max similarity: {known_sims.max():.4f}")
    
    if len(unknown_sims) > 0:
        print(f"Unknown type similarity: {unknown_sims.iloc[0]:.4f}")
        
        avg_known = known_sims.mean() if len(known_sims) > 0 else 0.8
        unknown_quality = unknown_sims.iloc[0]
        
        if unknown_quality >= avg_known * 0.9:
            quality_status = "Excellent"
        elif unknown_quality >= avg_known * 0.8:
            quality_status = "Good"
        elif unknown_quality >= avg_known * 0.7:
            quality_status = "Acceptable"
        else:
            quality_status = "Poor"
            
        print(f"Unknown type quality: {quality_status}")
    
    print("=" * 70)
    
    similarity_file = os.path.join(output_path, 'simulated_vs_reference_similarity.csv')
    similarity_table.to_csv(similarity_file, index=False)
    print(f"Similarity table saved to: {similarity_file}")
    
    return similarity_table


def generate_final_discovery_summary(discovered_types, train_cells, output_path):
    """
    Generate final discovery summary report
    """
    print(f"\n{'='*80}")
    print("FINAL DISCOVERY SUMMARY")
    print(f"{'='*80}")
    
    summary = {
        'total_discovered': len(discovered_types),
        'discovered_types': discovered_types.copy(),
        'final_cell_types': train_cells.copy(),
        'initial_known_types': len(train_cells) - len(discovered_types)
    }
    
    print(f"Discovery Statistics:")
    print(f"   Initial known types: {summary['initial_known_types']}")
    print(f"   Newly discovered types: {summary['total_discovered']}")
    print(f"   Total final types: {len(train_cells)}")
    
    if discovered_types:
        print(f"\nDiscovered Types:")
        for i, dtype in enumerate(discovered_types, 1):
            print(f"   {i}. {dtype}")
    else:
        print(f"\nNo unknown types were discovered")
    
    print(f"\nFinal Cell Type Inventory:")
    for i, ctype in enumerate(train_cells, 1):
        status = "DISCOVERED" if ctype in discovered_types else "KNOWN    "
        print(f"   {i:2d}. {ctype:<20} {status}")
    
    summary_file = os.path.join(output_path, 'final_discovery_summary.json')
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDiscovery summary saved to: {summary_file}")
    print(f"{'='*80}")
    
    return summary


def generate_similarity_ranking_report_with_reduced_comparison(avg_residual, filtered_sig_matrix, 
                                                               output_path, round_num,
                                                               candidate_sig_reduced=None, 
                                                               feature_genes=None):
    """
    Generate similarity ranking report with support for reduced-dimension comparison on candidate types.
    
    This function compares residuals against reference signatures. For candidate cell types from 
    external datasets (like GSE176171), it uses only the discriminatory genes for comparison to 
    avoid signal dilution from artificially zero-padded gene dimensions. For known cell types 
    from the original dataset, it uses full-dimensional comparison.
    
    Parameters:
        avg_residual: Average residual vector (full dimension)
        filtered_sig_matrix: Filtered signature matrix (full dimension) 
        output_path: Directory path for saving results
        round_num: Current discovery round number
        candidate_sig_reduced: Reduced candidate signature matrix (optional, only discriminatory genes)
        feature_genes: List of feature gene names used in reduced comparison (optional)
    
    Returns:
        Dictionary containing similarity results and rankings
    """
    
    print("\n=== Similarity Ranking Analysis (with Reduced-Dimension Support) ===")
    avg_residual = pd.Series(avg_residual, index=filtered_sig_matrix.columns)
    
    # Define candidate types that should use reduced comparison
    # These types are from external datasets and have zero-padded signatures in the full matrix
    candidate_types = ['adipocyte', 'ASPC', 'mesothelium', 'endothelial', 'macrophage']
    
    similarities = {}
    
    for celltype in filtered_sig_matrix.index:
        # Determine comparison mode based on feature_genes source
        # Two modes:
        # 1. Virtual cell characteristic genes mode: feature_genes from virtual cells, use full matrix
        # 2. Predefined discriminatory genes mode: feature_genes from external dataset, use reduced matrix
        if feature_genes is not None:
            # Virtual cell characteristic genes provided - use full signature matrix
            should_use_reduced = False
            use_virtual_genes = True
        else:
            # No feature_genes or using predefined genes - use reduced matrix for candidates
            should_use_reduced = (candidate_sig_reduced is not None and 
                                celltype in candidate_types and 
                                celltype in candidate_sig_reduced.index)
            use_virtual_genes = False
        
        if use_virtual_genes:
            # Mode 1: Virtual cell characteristic genes with full signature matrix
            full_signature = filtered_sig_matrix.loc[celltype]
            
            # Find common genes between virtual cell features and signature
            common_features = [g for g in feature_genes 
                            if g in full_signature.index and g in avg_residual.index]
            
            if len(common_features) == 0:
                print(f"  Warning: No common features found for {celltype}, skipping")
                continue
            
            # Extract expression values for common genes
            residual_vec = avg_residual[common_features].values.reshape(1, -1)
            signature_vec = full_signature[common_features].values.reshape(1, -1)
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sim_score = cosine_similarity(residual_vec, signature_vec)[0, 0]
            similarities[celltype] = sim_score
            
            print(f"  {celltype:<20} (virtual genes, {len(common_features)} genes): {sim_score:.4f}")
            
        elif should_use_reduced:
            # For candidate types: use reduced-dimension comparison
            # Extract the reduced signature for this candidate type
            candidate_signature = candidate_sig_reduced.loc[celltype]
            
            # Find which feature genes are available in both residual and signature
            available_features = [g for g in feature_genes if g in avg_residual.index]
            
            if len(available_features) == 0:
                # Fallback to full comparison if no common features found
                print(f"  Warning: No common features for {celltype}, using full comparison")
                reference_sig = filtered_sig_matrix.loc[celltype].values.reshape(1, -1)
                residual_vec = avg_residual.values.reshape(1, -1)
            else:
                # Ensure both vectors use the same gene set
                common_features = [g for g in available_features if g in candidate_signature.index]
                
                residual_vec = avg_residual[common_features].values.reshape(1, -1)
                signature_vec = candidate_signature[common_features].values.reshape(1, -1)
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sim_score = cosine_similarity(residual_vec, signature_vec)[0, 0]
            similarities[celltype] = sim_score
            
            print(f"  {celltype:<20} (reduced, {len(common_features)} genes): {sim_score:.4f}")
            
        else:
            # For known types: use full-dimension comparison
            reference_sig = filtered_sig_matrix.loc[celltype].values.reshape(1, -1)
            residual_vec = avg_residual.values.reshape(1, -1)
            
            from sklearn.metrics.pairwise import cosine_similarity
            sim_score = cosine_similarity(residual_vec, reference_sig)[0, 0]
            similarities[celltype] = sim_score
            
            print(f"  {celltype:<20} (full dimension): {sim_score:.4f}")
    
    # Sort by similarity score in descending order
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Print ranking report
    print(f"\nAverage Residual Similarity Ranking:")
    for rank, (celltype, score) in enumerate(sorted_similarities, 1):
        if rank <= 3:
            print(f"  TOP{rank}  {rank}. {celltype:20s}: {score:.4f}")
        else:
            print(f"        {rank}. {celltype:20s}: {score:.4f}")
    
    # Save results to CSV file
    results_df = pd.DataFrame(sorted_similarities, columns=['CellType', 'Similarity'])
    results_df['Rank'] = range(1, len(results_df) + 1)
    results_file = os.path.join(output_path, f'round_{round_num}_similarity_ranking.csv')
    results_df.to_csv(results_file, index=False)
    
    print(f"Similarity ranking saved to: {results_file}")
    
    return {
        'sorted_similarities': sorted_similarities,
        'all_similarities': similarities,
        'best_match': sorted_similarities[0][0] if sorted_similarities else None,
        'best_score': sorted_similarities[0][1] if sorted_similarities else 0.0
    }
