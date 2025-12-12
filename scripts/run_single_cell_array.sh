#!/bin/bash
#
# SBATCH SCRIPT FOR RUNNING A SINGLE-CELLTYPE DISTILLATION JOB ARRAY
#
# This script submits a job for each of the 14 candidate unknown celltypes,
# dynamically renames each task, and creates descriptively named log files.

# --- SBATCH Directives ---
#SBATCH --job-name=distill_single_parent
#SBATCH --time=2:00:00
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --account=ctb-liyue
#SBATCH --array=0-13
#SBATCH --requeue

# --- 1. Define all 14 candidate unknown celltypes for the array ---
#    (Excludes the 4 base known types: B_cell, CD14, CD4, CD8)
declare -a SINGLE_CELLTYPES
SINGLE_CELLTYPES[0]="CD16"
SINGLE_CELLTYPES[1]="DCs"
SINGLE_CELLTYPES[2]="HSC"
SINGLE_CELLTYPES[3]="Lymph_prolif"
SINGLE_CELLTYPES[4]="MAIT"
SINGLE_CELLTYPES[5]="Mono_prolif"
SINGLE_CELLTYPES[6]="NK_16hi"
SINGLE_CELLTYPES[7]="NK_56hi"
SINGLE_CELLTYPES[8]="Plasmablast"
SINGLE_CELLTYPES[9]="Platelets"
SINGLE_CELLTYPES[10]="RBC"
SINGLE_CELLTYPES[11]="Treg"
SINGLE_CELLTYPES[12]="gdT"
SINGLE_CELLTYPES[13]="pDC"


# --- 2. Get the cell type for the current job task ---
CURRENT_CELLTYPE=${SINGLE_CELLTYPES[$SLURM_ARRAY_TASK_ID]}

# --- 3. Dynamically generate Job Name AND Log File Names ---
#    (We use a default alpha of 2 for the name, as this is a common setting for single-cell discoveries)
NEW_NAME="distill_single_${CURRENT_CELLTYPE}_alpha_2"

# Set the Job Name using scontrol
scontrol update JobID=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} JobName=${NEW_NAME}

# Define log file paths to be created in the directory where you run 'sbatch'
OUTPUT_FILE="${NEW_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
ERROR_FILE="${NEW_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

# --- 4. Execute the command with manual output redirection ---
{
    echo "--- Starting Slurm Job Array Task #${SLURM_ARRAY_TASK_ID} for single celltype: ${CURRENT_CELLTYPE} ---"
    echo "Job name updated to: ${NEW_NAME}"
    echo "Standard output will be logged to: $(pwd)/${OUTPUT_FILE}"
    echo "Standard error will be logged to: $(pwd)/${ERROR_FILE}"
    
    # Activate your environment and navigate to the project directory
    source /home/yiminfan/projects/ctb-liyue/yiminfan/project_yue/tape/bin/activate
    cd /home/yiminfan/projects/ctb-liyue/yiminfan/project_yixuan/Distillation_Project

    echo "Running single-celltype experiment for: ${CURRENT_CELLTYPE}"
    # Call run_experiment.sh, passing the single celltype and a default alpha value
    bash scripts/run_experiment.sh --unknown-celltypes ${CURRENT_CELLTYPE} --alphas 2
    
    echo "--- Finished Slurm Job Array Task #${SLURM_ARRAY_TASK_ID} ---"

} > "${OUTPUT_FILE}" 2> "${ERROR_FILE}"