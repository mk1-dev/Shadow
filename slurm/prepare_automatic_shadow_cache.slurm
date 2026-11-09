#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
set -euo pipefail
source "${SHADOW_CONDA_SH:-/public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh}"
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u scripts/submit_automatic_shadow_pipeline.py worker \
  --job "${SHADOW_AUTO_JOB:?Set pipeline.json}" --phase "${SHADOW_CACHE_PHASE:?Set cpu/bev/plan}" \
  --index "${SLURM_ARRAY_TASK_ID:-0}"
