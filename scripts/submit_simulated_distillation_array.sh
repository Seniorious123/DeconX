#!/bin/bash
#
# =================================================================================
# SBATCH SCRIPT: SUBMIT SIMULATED DISTILLATION ARRAY
# =================================================================================
# Usage: sbatch scripts/submit_simulated_distillation_array.sh
# Description: 
#   Runs the full distillation pipeline on 23 pairs of SIMULATED data.
#   1. Checks if simulated bulk/frac data exists.
#   2. If not, generates it using run_simulation.py.
#   3. Runs run_distillation.py using the Part 1 signature file.
#   4. Outputs results to 'outputs/simulated_result/'.
# =================================================================================

# --- SBATCH Directives ---
#SBATCH --job-name=distill_sim_array
#SBATCH --output=%x_%A_%a.out        # Logs to submission directory
#SBATCH --error=%x_%A_%a.err         # Logs to submission directory
#SBATCH --time=3:00:00
#SBATCH --mem=300G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=ctb-liyue
#SBATCH --array=0-22                 # Array index for 23 tasks

# --- 1. Environment & Directory Setup ---
source /home/yiminfan/projects/ctb-liyue/yiminfan/project_yue/tape/bin/activate

# [CRITICAL] Explicitly jump to project root to ensure relative paths work
PROJECT_ROOT="/home/yiminfan/projects/ctb-liyue/yiminfan/project_yixuan/Distillation_Cut"
cd "${PROJECT_ROOT}" || { echo "Error: Could not cd to ${PROJECT_ROOT}"; exit 1; }

echo "Working Directory: $(pwd)"

# Define Local Paths (Relative to PROJECT_ROOT)
CONFIG_FILE="configs/path_config.py"
SC_DATA_PATH="data/raw/preprocessed_covid-19_sc.h5ad"
SIG_PATH="data/reference/AllsigOfCNS_neurous.csv"

# Define Output Directories
SIM_BASE_DIR="outputs/simulated_data"
DISTILL_RESULT_BASE="outputs/simulated_result"

# Ensure output directories exist
mkdir -p "${SIM_BASE_DIR}"
mkdir -p "${DISTILL_RESULT_BASE}"

# --- 2. Define the 23 Cell Type Pairs ---
declare -a CELLTYPE_PAIRS
CELLTYPE_PAIRS[0]="RBC Mono_prolif"
CELLTYPE_PAIRS[1]="DCs RBC"
CELLTYPE_PAIRS[2]="Platelets RBC"
CELLTYPE_PAIRS[3]="DCs pDC"
CELLTYPE_PAIRS[4]="pDC Mono_prolif"
CELLTYPE_PAIRS[5]="RBC CD16"
CELLTYPE_PAIRS[6]="RBC HSC"
CELLTYPE_PAIRS[7]="RBC pDC"
CELLTYPE_PAIRS[8]="Plasmablast CD16"
CELLTYPE_PAIRS[9]="gdT Treg"
CELLTYPE_PAIRS[10]="MAIT gdT"
CELLTYPE_PAIRS[11]="MAIT pDC"
CELLTYPE_PAIRS[12]="DCs CD16"
CELLTYPE_PAIRS[13]="DCs Plasmablast"
CELLTYPE_PAIRS[14]="Lymph_prolif CD16"
CELLTYPE_PAIRS[15]="Lymph_prolif DCs"
CELLTYPE_PAIRS[16]="Lymph_prolif Plasmablast"
CELLTYPE_PAIRS[17]="Lymph_prolif RBC"

# --- 3. Get Current Task Configuration ---
CURRENT_PAIR=${CELLTYPE_PAIRS[$SLURM_ARRAY_TASK_ID]}
ALPHAS_VAL="2 2" 

# Construct Experiment Name (e.g., DCs_Plasmablast_alphas_2_2)
PAIR_UNDERSCORE=$(echo "${CURRENT_PAIR}" | tr ' ' '_')
ALPHA_UNDERSCORE=$(echo "${ALPHAS_VAL}" | tr ' ' '_')
EXPERIMENT_NAME="${PAIR_UNDERSCORE}_alphas_${ALPHA_UNDERSCORE}"

# Dynamic Job Name Update (e.g., Sim_DCs_Plasmablast)
NEW_JOB_NAME="Sim_${EXPERIMENT_NAME}"
scontrol update JobID=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} JobName=${NEW_JOB_NAME}

echo "================================================================"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Target Pair: ${CURRENT_PAIR}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "================================================================"

# --- 4. Logic Step A: Check or Generate Data ---
SIM_DIR="${SIM_BASE_DIR}/${EXPERIMENT_NAME}"
BULK_FILE="${SIM_DIR}/bulk.csv"
FRAC_FILE="${SIM_DIR}/frac.csv"

if [ -f "${BULK_FILE}" ] && [ -f "${FRAC_FILE}" ]; then
    echo "[INFO] Simulation data found locally at: ${SIM_DIR}"
    echo "[INFO] Skipping generation step."
else
    echo "[WARN] Simulation data NOT found. Generating now..."
    
    python scripts/run_simulation.py \
        --config "${CONFIG_FILE}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        --unknown-celltypes ${CURRENT_PAIR} \
        --alphas ${ALPHAS_VAL} \
        --alpha-known 2 \
        --num-samples 3000 \
        --seed 42 \
        --cpu 1
        
    echo "[INFO] Generation completed."
fi

# --- 5. Logic Step B: Run Distillation ---
echo "----------------------------------------------------------------"
echo "Starting Distillation..."
echo "----------------------------------------------------------------"

TARGET_COMMA=$(echo "${CURRENT_PAIR}" | tr ' ' ',')

# Output to: outputs/simulated_result/<ExperimentName>
DISTILL_OUTPUT_DIR="${DISTILL_RESULT_BASE}/${EXPERIMENT_NAME}"
mkdir -p "${DISTILL_OUTPUT_DIR}"

python scripts/run_distillation.py \
    --sc_data "${SC_DATA_PATH}" \
    --test_bulk "${BULK_FILE}" \
    --test_frac "${FRAC_FILE}" \
    --output_path "${DISTILL_OUTPUT_DIR}" \
    --target_celltypes "${TARGET_COMMA}" \
    --known_celltypes B_cell,CD4,CD8,CD14 \
    --known_sig_path "${SIG_PATH}" 

echo "================================================================"
echo "Pipeline Finished for Task ${SLURM_ARRAY_TASK_ID}"
echo "Results saved to: ${DISTILL_OUTPUT_DIR}"
echo "================================================================"