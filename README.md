# DeconX: Iterative Distillation Framework for Cell Type Discovery

DeconX is a specialized computational toolbox designed to identify and extract features of unknown cell types from bulk RNA-seq data by leveraging single-cell RNA-seq (scRNA-seq) references. It integrates data preprocessing, pseudo-bulk simulation, and an iterative distillation algorithm to discover novel biological components within complex samples.

## 1. Project Architecture

The project is structured to ensure clear separation between data management, core algorithms, and execution scripts:

* **scripts/**: Main execution entry points.
    * 01_preprocess_hgsoc.py: Handles raw data cleaning, HVG selection (default 6000), and specific noise gene removal (e.g., MALAT1, MT-CO1, NEAT1).
    * 02_build_signature_library.py: Constructs signature matrices by combining internal HGSOC data with external datasets (e.g., GSE176171).
    * run_distillation.py: The primary workflow runner that manages scale alignment, global 99.95% capping, and the discovery process.
    * run_simulation.py: Generates pseudo-bulk data via parallel Dirichlet sampling for algorithm benchmarking.
    * validate_discovery.py: Produces publication-quality validation figures, including characteristic gene heatmaps and residual reduction charts.
* **src/distiller/decon/**: Core computational logic.
    * distillation.py: Implements the multi-round discovery loop, "Ghost Detection," and virtual cell generation logic.
    * models.py: Defines the AutoEncoderPlus architecture featuring custom normalization and ReLU layers for probability output.
    * diagnostics.py: Handles similarity ranking, weight initialization diagnostics, and training loss analysis.
    * trainer.py: Manages the composite loss function (Reconstruction + Fraction + Signature preservation loss).
    * utils.py: Provides evaluation metrics such as Concordance Correlation Coefficient (CCC) and L1 error.
* **src/distiller/generation/**:
    * data_manager.py: High-performance data engine for parallel pseudo-bulk simulation and H5AD/CSV loading.
* **configs/**:
    * path_config.py: Centralized management for scRNA-seq paths, training data, and signature file locations.

## 2. Technical Workflow

1. Initial Alignment: The model is first trained on known cell types to establish a baseline for gene expression reconstruction.
2. Residual Analysis: After reconstruction, the difference between the bulk input and the model output (residuals) is calculated to isolate unexplained biological signals.
3. Virtual Cell Generation: High-confidence residuals are scaled and clipped to generate "Virtual Cells," which are then integrated back into the training dataset.
4. Iterative Expansion: The AutoEncoderPlus model dynamically expands its output layer to incorporate newly discovered cell types, followed by fine-tuning to learn their specific features.

## 3. Installation & Dependencies

DeconX requires a Python 3.8+ environment. Core dependencies include:

* Deep Learning: torch
* Single-cell Analysis: scanpy, anndata
* Data Science: pandas, numpy, scipy
* Visualization: matplotlib, seaborn

Install all requirements via pip:
pip install torch scanpy anndata pandas numpy matplotlib seaborn tqdm scipy

## 4. Usage Guide

### Step 1: Preprocessing
Clean raw single-cell data and identify highly variable genes:
python scripts/01_preprocess_hgsoc.py

### Step 2: Build Reference Library
Construct the signature matrix for both known and candidate cell types:
python scripts/02_build_signature_library.py

### Step 3: Run Discovery Pipeline
Execute the distillation process. You must specify known types and target bulk data via command-line arguments:
python scripts/run_distillation.py \
    --sc_data data/processed/hgsoc_sc_processed.csv \
    --test_bulk data/processed/hgsoc_bulk_processed.csv \
    --known_celltypes "B cells,T cells,Macrophages,Epithelial cells,Endothelial cells,Fibroblasts" \
    --target_celltypes "Plasma cells,adipocyte" \
    --known_sig_path data/reference/known_signatures_hgsoc.csv \
    --output_path outputs/discovery_results

### Step 4: Validation
Generate diagnostic reports and figures:
python scripts/validate_discovery.py --output_dir outputs/discovery_results

## 5. Primary Outputs

* final_complete_model.pth: The fully trained model containing learned signatures for both known and discovered cell types.
* validation_results/: A directory containing PCA of residuals, signature similarity heatmaps, and residual reduction trends.
* gene_level_scale_analysis.csv: Detailed diagnostic data used to identify characteristic genes for the discovered cell types.
