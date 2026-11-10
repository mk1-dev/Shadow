#!/bin/bash
#SBATCH -J shadow_motion_train
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/shadow_motion_train_%j.out
#SBATCH -e logs/shadow_motion_train_%j.err

set -euo pipefail
source "${SHADOW_CONDA_SH:-/public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh}"
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
ASSETS="${SHADOW_FSD_ASSETS:?}"
export PYTHONPATH="$ASSETS/python_deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u scripts/run_shadow_completion_training.py run \
  --plan "${SHADOW_COMPLETION_TRAIN_PLAN:?}" --assets "$ASSETS" \
  --annotation-workers "${SHADOW_ANNOTATION_WORKERS:-4}"
