#!/bin/bash
#SBATCH --job-name=deepMGEhost
#SBATCH --partition=small-gpu
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/deepMGEhost_%j.out
#SBATCH --error=logs/deepMGEhost_%j.err

# Load modules
module load cuda/11.4

# Activate virtual environment
source /scratch/liu_price_lab/mlaziz/incubator_2025/deepMGEhost/bin/activate

# Run using the venv Python (replace with your actual path if different)
VENV_PYTHON=/scratch/liu_price_lab/mlaziz/incubator_2025/deepMGEhost/bin/python

# Sanity check (optional)
$VENV_PYTHON -c "import pandas as pd; print('✅ pandas is available:', pd.__version__)"

#
# Optional: Set any env vars needed for PyTorch/FAISS or logging
#export OMP_NUM_THREADS=4
#export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run your script
$VENV_PYTHON mge_llm_cluster.py \
   generate_embeddings \
  --host-labels /scratch/liu_price_lab/mlaziz/incubator_2025/deepMGEhost_project/data/sampletest/samplelist_1.txt \
  --genomad-dir /scratch/liu_price_lab/mlaziz/incubator_2025/deepMGEhost_project/data/sampletest \
  --output-prefix /scratch/liu_price_lab/mlaziz/incubator_2025/deepMGEhost_project/output/debug_070625 \
  --esm-model esmc_600m \
  --device cuda \
  --batch-size 4


