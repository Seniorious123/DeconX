import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from copy import deepcopy
from scipy.optimize import nnls
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import scanpy as sc
import torch.nn as nn

from .models import SimpleDataset, reproducibility, AutoEncoderPlus
from .trainer import train_model
from .utils import device, evaluate, find_best_match
from .diagnostics import (
    generate_similarity_ranking_report, 
    generate_final_discovery_summary,
    compare_learned_vs_reference_signatures
)
from ..generation.data_manager import simulate_data
from .distillation_plots import (
    plot_residual_pca, 
    plot_pca_colored_by_ground_truth,
    plot_pca_selection,
    plot_projection_validation
)

def export_round_plot_data(round_num, output_path, **data_dict):
    plot_data_dir = os.path.join(output_path, 'plot_data')
    os.makedirs(plot_data_dir, exist_ok=True)
    save_path = os.path.join(plot_data_dir, f'round_{round_num}.npz')
    np.savez_compressed(save_path, **data_dict)
    print(f"[PLOT DATA] Exported round {round_num} data")

def diagnose_virtual_data_scale(mixed_scseq, final_sim_cells, anonymous_celltype, output_path):
    """Analyzes scale issues and outputs CSVs required for characteristic gene extraction."""
    print("\n" + "="*80 + "\nVIRTUAL DATA SCALE DIAGNOSTIC MODULE\n" + "="*80)
    
    real_data = mixed_scseq[mixed_scseq['celltype'] != anonymous_celltype].drop(columns='celltype')
    virtual_data = mixed_scseq[mixed_scseq['celltype'] == anonymous_celltype].drop(columns='celltype')
    
    real_means, virtual_means = real_data.mean(), virtual_data.mean()
    gene_ratios = virtual_means / (real_means + 1e-8)
    
    # Save gene-level ratios (Critical for V2 validation)
    pd.DataFrame({
        'gene': real_means.index, 'real_mean': real_means.values,
        'virtual_mean': virtual_means.values, 'ratio': gene_ratios.values
    }).to_csv(os.path.join(output_path, 'gene_level_scale_analysis.csv'), index=False)
    
    # Bulk generation test (Legacy output required for consistency)
    bulk_stats = []
    for prop in [0.1, 0.2, 0.3, 0.5]:
        n_real, n_virt = int(1000 * (1 - prop)), int(1000 * prop)
        if len(real_data) >= n_real and len(virtual_data) >= n_virt:
            s_real, s_virt = real_data.sample(n=n_real, replace=True), virtual_data.sample(n=n_virt, replace=True)
            bulk_stats.append({
                'proportion': prop, 'bulk_mean': (s_real.sum() + s_virt.sum()).sum() / 1000 / len(real_means),
                'real_contribution': s_real.sum().mean()/1000, 'virtual_contribution': s_virt.sum().mean()/1000
            })
    pd.DataFrame(bulk_stats).to_csv(os.path.join(output_path, 'bulk_generation_scale_test.csv'), index=False)
    return {'scale_ratios': {'mean': virtual_means.mean()/real_means.mean()}}

def create_mixed_scseq_data(scseq, unknown_residuals, train_cells, unknown_celltype, **kwargs):
    """Generates mixed single-cell dataset by aligning residuals."""
    print("\n--- Starting virtual cell generation ---")
    real_vals = scseq.drop(columns=['celltype']).values
    target_mean, target_std = real_vals.mean(), real_vals.std()
    
    src_mean, src_std = unknown_residuals.mean(), unknown_residuals.std()
    scaled = ((unknown_residuals - src_mean) / src_std * target_std + target_mean) if src_std > 1e-8 else np.full_like(unknown_residuals, target_mean)
    
    final_virtual = np.clip(np.maximum(scaled, 0), None, 1000.0)
    unknown_df = pd.DataFrame(final_virtual, columns=scseq.columns[:-1])
    unknown_df['celltype'] = unknown_celltype
    
    print(f"[SUCCESS] Generated {len(unknown_df)} virtual cells for type '{unknown_celltype}'.")
    return pd.concat([scseq, unknown_df], axis=0, ignore_index=True)

def distillation(test_x, test_groundtruth_y=None, scseq=None, all_scseq=None, sigpath=None, 
                 d_prior=None, target_celltypes=None, batch_size=128, epochs=100, act_lr=1e-4, seed=114514, 
                 sample=1500, cpu=1, endtoend_epochs=150, max_discovery_rounds=5, discovery_threshold=0.02, 
                 unified_signature_matrix=None, initial_known_ctypes=None, **kwargs):
    
    reproducibility(seed)
    output_path = kwargs.get('output_path', '.')
    val_dir = os.path.join(output_path, 'validation_inputs')
    os.makedirs(val_dir, exist_ok=True)
    all_previous_characteristic_genes = set()
    
    if unified_signature_matrix is None:
        unified_signature_matrix = pd.read_csv("data/reference/unified_signature_matrix.csv", index_col=0)
    
    # Phase I: Initial Model
    print("\n--- Phase I: Initial Model ---")
    train_cells = initial_known_ctypes
    ckpt_path = os.path.join(output_path, 'checkpoints_initial', f"initial_model_{len(train_cells)}_cells.pth")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
    if os.path.exists(ckpt_path):
        initial_model = torch.load(ckpt_path, map_location=device)
    else:
        # SAFE CALL 1: Explicit keyword arguments for simulate_data
        tx, ty = simulate_data(
            sc_data=scseq,
            d_prior=tuple([0.5]*len(train_cells)),
            seed=seed,
            ctypes=train_cells,
            cpu=cpu,
            samples=sample
        )
        # SAFE CALL 2: Explicit keyword arguments for train_model to avoid positional string error
        initial_model = train_model(
            model=AutoEncoderPlus(tx.shape[1], ty.shape[1]).to(device), 
            train_loader=DataLoader(SimpleDataset(tx, ty), batch_size, True), 
            epochs=epochs, 
            act_lr=act_lr, 
            output_dir=output_path, 
            output_name='round_0_'
        )
        torch.save(initial_model, ckpt_path)
    
    # Normalize Test X
    model_genes = scseq.drop(columns='celltype').columns
    scale_fac = np.mean(np.sum(scseq.drop(columns='celltype').values, axis=1)) / np.mean(np.sum(test_x.values, axis=1))
    if scale_fac > 1e-8: test_x = test_x * scale_fac
    
    baseline_sigs = initial_model.sigmatrix().detach().cpu().numpy()
    

    if test_groundtruth_y is not None:
        print("\n--- Evaluating Initial Model (Ground Truth available) ---")
        try:
            eval_dir_init = os.path.join(output_path, 'eval_initial/')
            os.makedirs(eval_dir_init, exist_ok=True)
            evaluate(
                ae_model=initial_model,
                test_x=test_x,
                test_groundtruth_y=test_groundtruth_y,
                real_sig=unified_signature_matrix.reindex(columns=model_genes, fill_value=0),
                output_path=eval_dir_init,
                cell_types=deepcopy(train_cells),
                batch_size=batch_size
            )
        except Exception as e:
            print(f"[WARN] Initial evaluation failed: {e}")
    else:
        print("\n--- Skipping Initial Model Evaluation (No Ground Truth) ---")


    # Phase II: Discovery Loop
    print("\n--- Phase II: Discovery Loop ---")
    discovered, current_scseq, current_model = [], deepcopy(scseq), initial_model
    last_purified_residuals = None
    
    for round_num in range(max_discovery_rounds):
        print(f"\n{'='*80}\nDISCOVERY ROUND {round_num + 1}\n{'='*80}")
        current_train = train_cells + discovered
        filtered_sigs = unified_signature_matrix.loc[~unified_signature_matrix.index.isin(current_train)].reindex(columns=model_genes, fill_value=0)
        
        if filtered_sigs.empty: break
        
        # Residuals & Calibration (SAFE CALL 3)
        calibration_ctypes = train_cells + [f"Unknown_R{i+1}" for i in range(len(discovered))]
        
        prop_zero, _ = simulate_data(
            sc_data=current_scseq, 
            d_prior=tuple([2] * len(calibration_ctypes)), 
            seed=seed+round_num+9999, 
            ctypes=calibration_ctypes, 
            cpu=cpu, 
            samples=int(len(test_x) * 0.25)
        )
        
        with torch.no_grad():
            recon, _, _ = current_model(torch.from_numpy(test_x.values).float().to(device))
        residuals = np.maximum(test_x.values - recon.detach().cpu().numpy(), 0)
        if np.mean(np.sum(residuals, axis=1)) < discovery_threshold: break
        
        # Feature Selection & PCA
        adata = sc.AnnData(np.vstack([residuals, prop_zero]))
        adata.obs['g'] = ['S']*len(residuals) + ['N']*len(prop_zero)
        sc.tl.rank_genes_groups(adata, 'g', groups=['S'], reference='N', method='wilcoxon')
        top_genes = [int(n) for n in pd.DataFrame(adata.uns['rank_genes_groups']['names'])['S'].head(2000) if int(n) < len(model_genes)]
        
        res_pca = residuals[:, top_genes]
        adata_pca = sc.AnnData(res_pca)
        sc.tl.pca(adata_pca, n_comps=min(50, min(adata_pca.shape)-1))
        
        # Plots
        plot_residual_pca(adata_pca, round_num, output_path)
        plot_pca_colored_by_ground_truth(adata_pca, test_groundtruth_y, target_celltypes, round_num, output_path)
        
        # Ghost Detection (SAFE CALL 4)
        valid_types = current_scseq['celltype'].unique().tolist()
        anchor, _ = simulate_data(
            sc_data=current_scseq,
            d_prior=tuple([0.5]*len(valid_types)),
            seed=seed+55555+round_num,
            ctypes=valid_types,
            cpu=cpu,
            samples=200
        )
        with torch.no_grad():
            arecon, _, _ = current_model(torch.from_numpy(anchor/1000).float().to(device))
        green_center = np.mean(((anchor/1000 - arecon.cpu().numpy())[:, top_genes] - np.mean(res_pca, 0)) @ adata_pca.varm['PCs'], 0)
        
        scores = adata_pca.obsm['X_pca'][:, 0]
        indices = np.argsort(scores)[-int(len(scores)*0.1):] if abs(scores.max() - green_center[0]) > abs(scores.min() - green_center[0]) else np.argsort(scores)[:int(len(scores)*0.1)]
        
        # Refinement
        with torch.no_grad():
            xsub_recon, _, _ = current_model(torch.from_numpy(test_x.values[indices]).float().to(device))
        purified = np.maximum(test_x.values[indices] - xsub_recon.cpu().numpy(), 0)
        last_purified_residuals = purified

        export_round_plot_data(
            round_num, output_path,
            adata_pca_obsm=adata_pca.obsm['X_pca'],
            adata_pca_varm=adata_pca.varm['PCs'],
            res_pca=res_pca,
            residuals=residuals,
            top_genes=np.array(top_genes),
            indices=indices,
            test_x_values=test_x.values,
            test_x_index=test_x.index.values,
            test_groundtruth_y=(test_groundtruth_y.values if test_groundtruth_y is not None else None),
            test_groundtruth_columns=(test_groundtruth_y.columns.tolist() if test_groundtruth_y is not None else None),
            target_celltypes=target_celltypes,
            current_scseq_celltypes=current_scseq['celltype'].unique().tolist(),
            model_genes=model_genes.tolist(),
            seed=seed,
            cpu=cpu
        )
        
        # Plots
        plot_pca_selection(adata_pca, indices, round_num, output_path, "pca_with_selected", "Selection", "Selected")
        plot_projection_validation(adata_pca, indices, res_pca, current_scseq, current_model, top_genes, test_x, seed, cpu, round_num, output_path)
        
        # # Similarity
        # sim_res = generate_similarity_ranking_report(np.mean(purified, 0), filtered_sigs, output_path, round_num)
        # best_match = sim_res['sorted_similarities'][0][0]
        
        # sim_scores = np.array([find_best_match(r, filtered_sigs)[1] for r in residuals]) 

        # Similarity - using all genes
        sim_res = generate_similarity_ranking_report(np.mean(purified, 0), filtered_sigs, output_path, round_num)
        best_match = sim_res['sorted_similarities'][0][0]

        # Extract similarity scores for all candidate celltypes (for ranking output later)
        sims = []
        for idx, celltype_name in enumerate(filtered_sigs.index):
            sig_vec = filtered_sigs.iloc[idx].values
            purif_mean = np.mean(purified, 0)
            
            sig_norm = np.linalg.norm(sig_vec)
            purif_norm = np.linalg.norm(purif_mean)
            
            if sig_norm > 0 and purif_norm > 0:
                cos_sim = np.dot(purif_mean, sig_vec) / (sig_norm * purif_norm)
            else:
                cos_sim = 0.0
            
            sims.append(cos_sim)

        sims = np.array(sims)

        sim_scores = np.array([find_best_match(r, filtered_sigs)[1] for r in residuals])

        if len(sim_scores) > 0:
            plot_pca_selection(adata_pca, np.where(sim_scores >= np.percentile(sim_scores, 90))[0], round_num, output_path, "pca_sim_selected", "Sim Selection", "Sim Selected")
        
        discovered.append(best_match)
        anon_type = f"Unknown_R{round_num+1}"

        # Print similarity ranking
        print(f"\n[SIMILARITY RANKING] Round {round_num} - {anon_type} vs Candidate Cell Types:")
        sorted_indices = np.argsort(sims)[::-1]  # Sort in descending order
        for rank, idx in enumerate(sorted_indices[:10], 1):  # Print top 10
            celltype_name = filtered_sigs.index[idx]
            similarity_value = sims[idx]
            marker = " <- SELECTED" if celltype_name == best_match else ""
            print(f"  {rank}. {celltype_name}: {similarity_value:.4f}{marker}")
        
        # Virtual Data & Characteristic Genes
        mixed_scseq = create_mixed_scseq_data(current_scseq, purified, train_cells + discovered[:-1], anon_type)
        current_scseq = deepcopy(mixed_scseq)
        
        diagnose_virtual_data_scale(mixed_scseq, train_cells + discovered[:-1] + [anon_type], anon_type, val_dir)
        # try:
        #     os.rename(os.path.join(val_dir, 'gene_level_scale_analysis.csv'), os.path.join(val_dir, f'round_{round_num}_gene_level_scale_analysis.csv'))
        #     ga = pd.read_csv(os.path.join(val_dir, f'round_{round_num}_gene_level_scale_analysis.csv'))
        #     high = ga[(ga['ratio'] > 10) & (ga['virtual_mean'] > 1.0)].sort_values('ratio', ascending=False)
        #     if not high.empty:
        #         cgenes = high.head(5)['gene'].tolist()
        #         with open(os.path.join(val_dir, f'round_{round_num}_characteristic_genes.txt'), 'w') as f:
        #             f.write('\n'.join(cgenes))
        # except Exception as e: print(f"Gene extraction error: {e}")

        try:
            os.rename(os.path.join(val_dir, 'gene_level_scale_analysis.csv'), os.path.join(val_dir, f'round_{round_num}_gene_level_scale_analysis.csv'))
            ga = pd.read_csv(os.path.join(val_dir, f'round_{round_num}_gene_level_scale_analysis.csv'))
            high = ga[(ga['ratio'] > 10) & (ga['virtual_mean'] > 1.0)].sort_values('ratio', ascending=False)
            if not high.empty:
                # Select the top 50 most prominent genes (or fewer if less than 50 are available)
                cgenes = high.head(50)['gene'].tolist()
                
                # Calculate the percentage of genes that are "new" (not seen in previous rounds)
                cgenes_set = set(cgenes)
                new_genes = cgenes_set - all_previous_characteristic_genes
                num_new_genes = len(new_genes)
                total_genes = len(cgenes)
                new_gene_percentage = (num_new_genes / total_genes * 100) if total_genes > 0 else 0
                
                # Print output in the specified format
                print(f"Round {round_num + 1}: {new_gene_percentage:.1f}% genes are new ({num_new_genes}/{total_genes} genes)")
                
                # Add this round's characteristic genes to the global tracking set for future comparison
                all_previous_characteristic_genes.update(cgenes_set)
                
                # Save characteristic genes to file with updated filename
                with open(os.path.join(val_dir, f'round_{round_num}_characteristic_genes_top50.txt'), 'w') as f:
                    f.write('\n'.join(cgenes))
        except Exception as e: print(f"Gene extraction error: {e}")
        
        # Expansion & Training
        train_types = train_cells + [f"Unknown_R{i+1}" for i in range(len(discovered))]

        current_scseq = current_scseq.copy() 
        import gc
        gc.collect()

        # SAFE CALL 5: Explicit keywords for simulate_data
        sx, sy = simulate_data(
            sc_data=mixed_scseq,
            d_prior=tuple([1]*len(train_types)),
            seed=seed+round_num*100,
            ctypes=train_types,
            cpu=cpu,
            samples=sample
        )
        
        exp_model = AutoEncoderPlus(len(model_genes), len(train_types)).to(device)
        prev_dim = current_model.encoder[-3].weight.shape[0]
        
        # Copy weights
        for ol, nl in zip(current_model.encoder[:-3], exp_model.encoder[:-3]):
            if hasattr(ol, 'weight'): nl.weight.data = ol.weight.data.clone()
            if hasattr(ol, 'bias') and ol.bias is not None: nl.bias.data = ol.bias.data.clone()
            
        exp_model.encoder[-3].weight.data[:prev_dim] = current_model.encoder[-3].weight.data.clone()

        if current_model.encoder[-3].bias is not None:
            exp_model.encoder[-3].bias.data[:prev_dim] = current_model.encoder[-3].bias.data.clone()
            if len(train_types) > prev_dim:
                exp_model.encoder[-3].bias.data[prev_dim:] = 0.0

        init_sig, _ = nnls(current_model.raw_sigmatrix2().detach().cpu().numpy().T, np.mean(purified, 0))
        if len(train_types) > prev_dim:
             scaled = (torch.from_numpy(init_sig).float().to(device) / (np.linalg.norm(init_sig)+1e-8)) * torch.mean(torch.norm(current_model.encoder[-3].weight.data, dim=1))
             exp_model.encoder[-3].weight.data[prev_dim] = scaled

        # Decoder Copy
        od, nd = [l for l in current_model.decoder if isinstance(l, nn.Linear)], [l for l in exp_model.decoder if isinstance(l, nn.Linear)]
        for i, (ol, nl) in enumerate(zip(od, nd)):
            if i > 0: 
                # Copy weights
                nl.weight.data = ol.weight.data.clone()
                # Safe copy bias: only if both source and destination have bias
                if ol.bias is not None and nl.bias is not None:
                    nl.bias.data = ol.bias.data.clone()
            else:
                # First layer
                nl.weight.data[:, :prev_dim] = ol.weight.data.clone()
                if len(train_types) > prev_dim: 
                    nl.weight.data[:, prev_dim] = torch.from_numpy(init_sig).float().to(device)
                
                # Safe copy bias for first layer
                if ol.bias is not None and nl.bias is not None:
                    nl.bias.data = ol.bias.data.clone()
        
        # Train - Two-phase progressive training
        exp_model.train()
        loader = DataLoader(SimpleDataset(sx, sy), batch_size, True)
        test_loader = DataLoader(SimpleDataset(test_x.values), batch_size, True)
        ref_w = exp_model.decoder[0].weight.data[:, :-1].clone()
        freeze_old = True

        for ep in tqdm(range(endtoend_epochs), desc=f"Training R{round_num+1}"):
            if ep == 0:
                opt = Adam([p for p in exp_model.parameters() if p.requires_grad], lr=act_lr)
            elif ep == endtoend_epochs // 3:
                for p in exp_model.parameters(): 
                    p.requires_grad = True
                opt = Adam(exp_model.parameters(), lr=act_lr * 0.5)
                freeze_old = False
            
            for batch_idx, (bx, by) in enumerate(loader):
                opt.zero_grad()
                rec, frac, sigs = exp_model(bx.to(device))
                loss = F.mse_loss(frac, by.to(device)) + F.mse_loss(rec, bx.to(device)) + \
                    (F.mse_loss(sigs[:len(baseline_sigs)], torch.from_numpy(baseline_sigs).float().to(device))*5 if len(baseline_sigs)>0 else 0) + \
                    F.mse_loss(exp_model.decoder[0].weight[:, :-1], ref_w)*0.1
                
                if batch_idx % 5 == 0:
                    test_data_iter = iter(test_loader)
                    try:
                        test_batch = next(test_data_iter)
                        if isinstance(test_batch, tuple): 
                            test_batch = test_batch[0]
                        test_rec, _, _ = exp_model(test_batch.to(device))
                        loss += 0.5 * F.mse_loss(test_rec, test_batch.to(device))
                    except StopIteration:
                        pass
                
                loss.backward()
                if freeze_old:
                    if exp_model.encoder[-3].weight.grad is not None: 
                        exp_model.encoder[-3].weight.grad[:prev_dim] = 0
                    if exp_model.encoder[-3].bias is not None and exp_model.encoder[-3].bias.grad is not None: 
                        exp_model.encoder[-3].bias.grad[:prev_dim] = 0
                    if exp_model.decoder[0].weight.grad is not None: 
                        exp_model.decoder[0].weight.grad[:, :prev_dim] = 0
                opt.step()
        
        current_model = exp_model
        torch.save(current_model, os.path.join(output_path, f'round_{round_num}_model.pth'))
        baseline_sigs = np.vstack([baseline_sigs, current_model.sigmatrix().detach().cpu().numpy()[-1:]])


        # =================================================================================
        # [MODIFIED] Internal Model Signature Similarity (新发现 vs 模型内所有已知)
        # =================================================================================
        print(f"\n[ANALYSIS] Calculating INTERNAL signature similarity (Model-Learned) for Round {round_num}...")
        
        # 1. 获取模型当前所有的 Signature 矩阵 (Rows = Cell Types, Cols = Genes)
        all_current_sigs = current_model.sigmatrix().detach().cpu().numpy()
        
        # 2. 定义主角：这一轮刚刚训练出来的那个 Signature (矩阵最后一行)
        target_label = f"Unknown_R{round_num+1}"
        new_sig_vector = all_current_sigs[-1]
        new_sig_norm = np.linalg.norm(new_sig_vector)

        # 3. 定义对比对象：模型里现有的所有 Signature (包括它自己，包括初始已知的，包括之前发现的)
        current_model_labels = train_cells + [f"Unknown_R{i+1}" for i in range(len(discovered))]
        
        sim_results = []
        
        for idx, label in enumerate(current_model_labels):
            ref_vec = all_current_sigs[idx]
            ref_norm = np.linalg.norm(ref_vec)
            
            # Cosine Similarity Calculation
            if new_sig_norm == 0 or ref_norm == 0:
                cos_sim = 0.0
            else:
                cos_sim = np.dot(new_sig_vector, ref_vec) / (new_sig_norm * ref_norm)
            
            sim_results.append({
                'Round': round_num,
                'Target_New_Signature': target_label, 
                'Compared_To_Signature': label,       # 这是模型内部的已知类型或旧发现
                'Internal_Cosine_Similarity': cos_sim
            })

        # 4. 保存结果
        sim_df = pd.DataFrame(sim_results)
        save_name = f'round_{round_num}_internal_model_similarity.csv'
        sim_df.to_csv(os.path.join(output_path, save_name), index=False)
        print(f"[ANALYSIS] Saved Internal Similarity report to: {save_name}")
        # =================================================================================

        

        if test_groundtruth_y is not None:
            print(f"\n--- Running End-of-Round Evaluation (Ground Truth available) ---")
            try:
                os.makedirs(os.path.join(output_path, f'eval_round_{round_num}/'), exist_ok=True)
                evaluate(
                    ae_model=current_model, 
                    test_x=test_x, 
                    test_groundtruth_y=test_groundtruth_y, 
                    real_sig=unified_signature_matrix.reindex(columns=model_genes, fill_value=0), 
                    output_path=os.path.join(output_path, f'eval_round_{round_num}/'),
                    cell_types=deepcopy(train_cells + discovered),
                    batch_size=batch_size
                )
            # except (KeyError, ValueError) as e:
            #     print(f"\n[INFO] Evaluation stopped at new cell type (Expected): {e}")
            #     print("Exiting discovery loop to finalize results.")
            #     break


            except (KeyError, ValueError) as e:
                print(f"\n[WARN] Evaluation failed (New type not in GT). Ignoring and CONTINUING loop.")
                # break

        else:
            # Real Data (No Ground Truth) -> SKIP evaluation -> CONTINUE loop
            print(f"\n--- Skipping End-of-Round Evaluation (Real Data / No GT) ---")



    # Phase III: Final
    print("\n--- Phase III: Finalization ---")

    if len(discovered) > 0:
        print(f"Total discovered: {len(discovered)} -> {discovered}")
        print("Rolling back last incorrect discovery...")
        valid_discoveries = discovered[:-1]
        
        if not valid_discoveries:
            final_model = initial_model
            final_cell_types_for_eval = train_cells
        else:
            last_valid_idx = len(valid_discoveries) - 1
            model_path = os.path.join(output_path, f'round_{last_valid_idx}_model.pth')
            try:
                final_model = torch.load(model_path, map_location=device)
                final_cell_types_for_eval = train_cells + valid_discoveries
            except FileNotFoundError:
                final_model = initial_model
                final_cell_types_for_eval = train_cells
    else:
        final_model = initial_model
        final_cell_types_for_eval = train_cells

    print(f"\n--- Final Evaluation (Cell Types: {final_cell_types_for_eval}) ---")

    if test_groundtruth_y is not None:
        final_eval_dir = os.path.join(output_path, 'eval_final_best_model/')
        os.makedirs(final_eval_dir, exist_ok=True)
        gt_cols = [c for c in final_cell_types_for_eval if c in test_groundtruth_y.columns]
        evaluate(final_model, test_x, test_groundtruth_y[gt_cols], 
                unified_signature_matrix.reindex(columns=model_genes, fill_value=0), 
                final_eval_dir, cell_types=deepcopy(gt_cols), batch_size=batch_size)

    if discovered:
        compare_learned_vs_reference_signatures(final_model, discovered, unified_signature_matrix, 
                                            last_purified_residuals, output_path, len(discovered)-1)

    generate_final_discovery_summary(discovered, train_cells + discovered, output_path)
    torch.save(final_model, os.path.join(output_path, 'final_complete_model.pth'))

    return final_model