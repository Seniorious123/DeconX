# ===================================================================
#         Complete Import Section for: src/distiller/decon/utils.py
# ===================================================================

# --- Standard Libraries ---
import os
from collections import Counter

# --- Third-party Libraries ---
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# --- Device Configuration (from your original file) ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def L1error(pred, true):
    return np.mean(np.abs(pred - true))


def CCCscore(y_pred, y_true, mode='all'):
    # pred: shape{n sample, m cell}
    if mode == 'all':
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
    elif mode == 'avg':
        pass
    ccc_value = 0
    for i in range(y_pred.shape[1]):
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        # Mean
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        # Variance
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        # Standard deviation
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        ccc_value += ccc
    return ccc_value / y_pred.shape[1]



def score(pred, label):
    print('L1 error is', L1error(pred, label))
    print('CCC is ', CCCscore(pred, label))



### A better version of showloss
def showloss_plot(loss, filename='loss_plot.png', y_axis=''):
    plt.plot(loss)
    plt.xticks(range(0,len(loss),20))
    plt.xlabel('iteration')
    plt.ylabel(y_axis+'Loss')
    plt.savefig(filename, format='png')
    plt.close()


        
def find_best_match(residual_sig, sig_matrix):
    """
    Compares the similarity between residual_sig and each celltype in sig_matrix to find the best match.
    
    Args:
        residual_sig (np.ndarray): Residual vector, with shape (n_genes,).
        sig_matrix (pd.DataFrame): Signature matrix, with shape (n_celltypes, n_genes).

    Returns:
        str: The name of the celltype with the highest similarity.
        float: The corresponding similarity value.
        pd.Series: The similarity scores for all celltypes.
    """

    if residual_sig.ndim == 1:
        residual_sig = residual_sig.reshape(1, -1)

    similarity_scores = cosine_similarity(residual_sig, sig_matrix)[0]
    
    # print(similarity_scores.shape)  (19,)

    similarity_series = pd.Series(similarity_scores, index=sig_matrix.index)

    best_match_celltype = similarity_series.idxmax()
    best_match_score = similarity_series.max()
    
    return best_match_celltype, best_match_score, similarity_series

### Find signature matrix of every cell types, return signature and cell types
def find_sigmatrix(sc_data):
    groups = sc_data.groupby("celltype")
    sorted_groups = sorted(groups, key=lambda x: x)
    # print(sorted_groups)
    subsets = [group for _, group in sorted_groups]
    cell_types = [celltype for celltype, _ in sorted_groups]

    for i in range(len(subsets)):
        subsets[i].drop(columns="celltype", inplace=True)

    sig_matrix = pd.DataFrame(
        np.zeros((len(cell_types), subsets[0].shape[1])),
        index = cell_types, columns = subsets[0].columns,
    )
    for i in range(len(subsets)):
        sig_matrix.iloc[i] = subsets[i].mean()

    return cell_types, sig_matrix



### Normalize the vectors to fix the length of the vectors to 1
def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

### Dot plot to check the similarity between two matrixs.
### If Matrix A is (M, dim) and Matrix B is (N, dim), the output matrix is (M, N)
def matrix_similarity(matrix_A, matrix_B):
    A_matrix = normalize_vectors(matrix_A)
    B_matrix = normalize_vectors(matrix_B)
    similarity_matrix = np.dot(A_matrix, B_matrix.T)
    return similarity_matrix

def evaluate(ae_model, test_x, test_groundtruth_y, real_sig, output_path, cell_types=None, batch_size=128):
    from .models import SimpleDataset
    from .trainer import predict_ae
    print(f"Evaluating...")
    ae_model.eval()

    if not isinstance(test_x, pd.DataFrame):
        test_x = pd.DataFrame(test_x)

    if not isinstance(real_sig, pd.DataFrame):
        real_sig = pd.DataFrame(real_sig, index=cell_types, columns=test_x.columns)
    
    if cell_types:
        test_groundtruth_y = test_groundtruth_y[cell_types]


        real_sig = real_sig.reindex(index=cell_types, columns=test_x.columns, fill_value=0)
    
    test_bulk_loader = DataLoader(SimpleDataset(test_x), batch_size=batch_size, shuffle=False)
    predicted_test_y = predict_ae(ae_model, test_bulk_loader)

    CCC_score = []
    L1_error = []

    with open(os.path.join(output_path, 'eval_score.txt'), 'w') as f:
        for cell_type in cell_types:
            true_values = test_groundtruth_y[cell_type].values.reshape(-1, 1)
            predicted_values = predicted_test_y[:, cell_types.index(cell_type)].reshape(-1, 1)
            
            ccc = CCCscore(true_values, predicted_values)
            l1 = L1error(true_values, predicted_values)

            CCC_score.append(ccc)
            L1_error.append(l1)

            f.write(f"{cell_type}: CCC = {ccc}, L1 = {l1}\n")
            print(f"CCC Score for {cell_type}: {ccc}")
            print(f"L1 Error for {cell_type}: {l1}")

    with open(os.path.join(output_path, 'ccc_scores.txt'), 'w') as f:
        for cell_type, ccc in zip(cell_types, CCC_score):
            f.write(f"{cell_type}: {ccc}\n")

    with open(os.path.join(output_path, 'l1_errors.txt'), 'w') as f:
        for cell_type, l1 in zip(cell_types, L1_error):
            f.write(f"{cell_type}: {l1}\n")

    plot_prediction_vs_truth(predicted_test_y, test_groundtruth_y.values, output_path, cell_types)
    plot_distribution_violin(predicted_test_y, test_groundtruth_y.values, os.path.join(output_path, f'distribution_violin_epoch.png'))
    
    sig_matrix = ae_model.sigmatrix().detach().cpu().numpy() 
    
    real_matrix = real_sig.values

    similarity_matrix = matrix_similarity(sig_matrix, real_matrix)
    reordered_similarity_matrix = similarity_matrix
    heatmap_data_df = pd.DataFrame(
        reordered_similarity_matrix,
        index=cell_types,      # Set row titles to cell types
        columns=cell_types     # Set column titles to cell types
    )
    csv_path = os.path.join(output_path, 'similarity_matrix_for_prism.csv')
    heatmap_data_df.to_csv(csv_path)
    print(f"Heatmap data saved for Prism: {csv_path}")

    with open(os.path.join(output_path, 'similarity.txt'), 'a') as f:
        f.write(str(reordered_similarity_matrix))
        f.write('\n')

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        reordered_similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        cbar=True,
        xticklabels=cell_types,
        yticklabels=cell_types 
    )
    plt.title('Signature Matrix Similarity Heatmap')
    plt.xlabel('Real Signatures')
    plt.ylabel('Predicted Signatures')

    heatmap_path = os.path.join(output_path, 'similarity_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_vs_truth(pred, truth, fig_dir, cell_types):
    if len(pred.shape) != 2 or len(truth.shape) != 2:
        raise ValueError("Both pred and truth should be 2D matrices.")
    if pred.shape[1] != len(cell_types):
        raise ValueError("The number of columns in pred must match the length of cell_types.")

    os.makedirs(fig_dir, exist_ok=True)

    for i, cell_type in enumerate(cell_types):
        plt.figure(figsize=(6, 6))
        plt.scatter(truth[:, i], pred[:, i], color='blue', alpha=0.5, s=5)
        # plt.title(f'Prediction vs. Truth for {cell_type}')
        # plt.xlabel('Truth')
        # plt.ylabel('Prediction')
        plt.xlabel('Ground_truth', fontname='Arial', fontsize=24)
        plt.ylabel('Prediction', fontname='Arial', fontsize=18)
        plt.plot([truth[:, i].min(), truth[:, i].max()], [truth[:, i].min(), truth[:, i].max()], 'k--', lw=2, label='Perfect Prediction')
        # plt.legend()


        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig_path = os.path.join(fig_dir, f'prediction_vs_truth_{cell_type}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {fig_path}")

    

def plot_distribution_violin(pred, truth, fig_path):
    """
    Plots violin distributions for each feature in pred and truth matrices.

    Parameters:
    - pred (numpy.ndarray or pandas.DataFrame): Predicted values matrix.
    - truth (numpy.ndarray or pandas.DataFrame): Ground truth values matrix.
    - fig_path (str): Path to save the resulting figure.
    """
    
    # Ensure pred and truth are numpy arrays
    pred = np.array(pred)
    truth = np.array(truth)
    
    # Number of features (columns)
    num_features = pred.shape[1]
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(num_features, 2, figsize=(12, 6 * num_features))
    
    # If there's only one feature, ensure axes is 2D
    if num_features == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i in range(num_features):
        feature_pred = pred[:, i]
        feature_truth = truth[:, i]
        
        # **Case 1:** truth is zero
        mask_zero = feature_truth == 0
        pred_zero = feature_pred[mask_zero]
        
        ax_zero = axes[i, 0]
        sns.violinplot(y=pred_zero, ax=ax_zero, color="skyblue")
        ax_zero.set_title(f'Feature {i+1} - Pred when Truth = 0')
        ax_zero.set_ylabel('Predicted Values')
        
        # **Case 2:** truth is non-zero
        mask_nonzero = feature_truth != 0
        pred_nonzero = feature_pred[mask_nonzero]
        truth_nonzero = feature_truth[mask_nonzero]
        
        ax_nonzero = axes[i, 1]
        
        # Prepare DataFrame for seaborn
        df_nonzero = pd.DataFrame({
            'Predicted': pred_nonzero,
            'Truth': truth_nonzero
        })
        df_melted = df_nonzero.melt(var_name='Type', value_name='Value')
        
        sns.violinplot(x='Type', y='Value', data=df_melted, ax=ax_nonzero, palette="Set2")
        ax_nonzero.set_title(f'Feature {i+1} - Pred vs Truth when Truth != 0')
        ax_nonzero.set_ylabel('Values')
        
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def evaluate_nnls_deconvolution(learned_signatures, test_x, test_groundtruth_y, 
                                real_sig, output_path, cell_types=None):
    """
    Evaluate NNLS-based deconvolution results.
    
    Args:
        learned_signatures: The signature matrix learned by the model (numpy array or tensor)
                          Shape: (n_celltypes, n_genes)
        test_x: Test bulk RNA-seq data (DataFrame or numpy array)
        test_groundtruth_y: Ground truth cell type proportions (DataFrame)
        real_sig: Reference signature matrix (DataFrame)
        output_path: Directory to save evaluation results
        cell_types: List of cell type names (should match order in learned_signatures)
    
    Returns:
        predicted_fractions: NNLS-predicted cell type fractions for all test samples
        evaluation_metrics: Dictionary containing CCC scores and L1 errors
    """
    from scipy.optimize import nnls
    
    print(f"\n{'='*60}")
    print("NNLS DECONVOLUTION EVALUATION")
    print(f"{'='*60}")
    
    # Step 1: Data Preparation
    print("\n[Step 1] Preparing data...")
    
    # Convert test_x to numpy array if it's a DataFrame
    if isinstance(test_x, pd.DataFrame):
        test_x_array = test_x.values
        gene_names = test_x.columns
    else:
        test_x_array = test_x
        gene_names = None
    
    # Convert learned_signatures to numpy if it's a tensor
    if isinstance(learned_signatures, torch.Tensor):
        learned_signatures = learned_signatures.detach().cpu().numpy()
    
    # Ensure cell_types is properly set
    if cell_types is None:
        if isinstance(test_groundtruth_y, pd.DataFrame):
            cell_types = test_groundtruth_y.columns.tolist()
        else:
            cell_types = [f"CellType_{i}" for i in range(learned_signatures.shape[0])]
    
    print(f"Number of test samples: {test_x_array.shape[0]}")
    print(f"Number of genes: {test_x_array.shape[1]}")
    print(f"Number of cell types: {len(cell_types)}")
    print(f"Cell types: {cell_types}")
    
    # Step 2: NNLS Deconvolution
    print("\n[Step 2] Running NNLS deconvolution...")
    
    n_samples = test_x_array.shape[0]
    n_celltypes = learned_signatures.shape[0]
    predicted_fractions = np.zeros((n_samples, n_celltypes))
    
    for i in range(n_samples):
        bulk_sample = test_x_array[i, :]
        fractions, residual = nnls(learned_signatures.T, bulk_sample)
        predicted_fractions[i, :] = fractions
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples} samples...")
    
    print(f"NNLS deconvolution completed for all {n_samples} samples")
    
    # Step 3: Normalize Predictions
    print("\n[Step 3] Normalizing predictions to sum to 1...")
    
    row_sums = predicted_fractions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    predicted_fractions_normalized = predicted_fractions / row_sums
    
    print(f"Normalization complete")
    print(f"Sample prediction sum range: [{predicted_fractions_normalized.sum(axis=1).min():.4f}, "
          f"{predicted_fractions_normalized.sum(axis=1).max():.4f}]")
    
    # Step 4: Calculate Metrics
    print("\n[Step 4] Calculating evaluation metrics...")
    
    if isinstance(test_groundtruth_y, pd.DataFrame):
        test_groundtruth_y_filtered = test_groundtruth_y[cell_types].values
    else:
        test_groundtruth_y_filtered = test_groundtruth_y
    
    CCC_scores = []
    L1_errors = []
    
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, 'nnls_eval_scores.txt'), 'w') as f:
        f.write("NNLS Deconvolution Evaluation Results\n")
        f.write("="*50 + "\n\n")
        
        for i, cell_type in enumerate(cell_types):
            true_values = test_groundtruth_y_filtered[:, i].reshape(-1, 1)
            predicted_values = predicted_fractions_normalized[:, i].reshape(-1, 1)
            
            ccc = CCCscore(true_values, predicted_values)
            CCC_scores.append(ccc)
            
            l1 = L1error(true_values, predicted_values)
            L1_errors.append(l1)
            
            result_line = f"{cell_type}: CCC = {ccc:.4f}, L1 = {l1:.4f}\n"
            f.write(result_line)
            print(f"  {cell_type}: CCC = {ccc:.4f}, L1 = {l1:.4f}")
        
        avg_ccc = np.mean(CCC_scores)
        avg_l1 = np.mean(L1_errors)
        f.write(f"\n{'='*50}\n")
        f.write(f"Average CCC: {avg_ccc:.4f}\n")
        f.write(f"Average L1:  {avg_l1:.4f}\n")
        
        print(f"\n  Average CCC: {avg_ccc:.4f}")
        print(f"  Average L1:  {avg_l1:.4f}")
    
    with open(os.path.join(output_path, 'nnls_ccc_scores.txt'), 'w') as f:
        for cell_type, ccc in zip(cell_types, CCC_scores):
            f.write(f"{cell_type}: {ccc:.4f}\n")
    
    with open(os.path.join(output_path, 'nnls_l1_errors.txt'), 'w') as f:
        for cell_type, l1 in zip(cell_types, L1_errors):
            f.write(f"{cell_type}: {l1:.4f}\n")
    
    # Step 5: Generate Visualizations
    print("\n[Step 5] Generating visualizations...")
    
    plot_prediction_vs_truth(
        predicted_fractions_normalized, 
        test_groundtruth_y_filtered, 
        output_path, 
        cell_types
    )
    
    plot_distribution_violin(
        predicted_fractions_normalized, 
        test_groundtruth_y_filtered, 
        os.path.join(output_path, 'nnls_distribution_violin.png')
    )
    
    if real_sig is not None:
        print("\n[Step 5.3] Generating signature similarity heatmap...")
        
        if isinstance(real_sig, pd.DataFrame):
            real_sig_array = real_sig.values
        else:
            real_sig_array = real_sig
        
        similarity_matrix = matrix_similarity(learned_signatures, real_sig_array)
        
        heatmap_df = pd.DataFrame(
            similarity_matrix,
            index=cell_types,
            columns=cell_types
        )
        csv_path = os.path.join(output_path, 'nnls_signature_similarity.csv')
        heatmap_df.to_csv(csv_path)
        print(f"  Similarity matrix saved to: {csv_path}")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            cbar=True,
            xticklabels=cell_types,
            yticklabels=cell_types
        )
        plt.title('NNLS: Learned vs Reference Signature Similarity')
        plt.xlabel('Reference Signatures')
        plt.ylabel('Learned Signatures')
        
        heatmap_path = os.path.join(output_path, 'nnls_signature_similarity_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Heatmap saved to: {heatmap_path}")
    
    print(f"\n{'='*60}")
    print("NNLS EVALUATION COMPLETED")
    print(f"All results saved to: {output_path}")
    print(f"{'='*60}\n")
    
    evaluation_metrics = {
        'CCC_scores': CCC_scores,
        'L1_errors': L1_errors,
        'avg_CCC': avg_ccc,
        'avg_L1': avg_l1,
        'cell_types': cell_types
    }
    
    return predicted_fractions_normalized, evaluation_metrics
