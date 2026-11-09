#!/bin/bash
#SBATCH -J shadow_E2_suite
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_E2_%j.out
#SBATCH -e logs/shadow_E2_%j.err

set -euo pipefail
source "${SHADOW_CONDA_SH:-/public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh}"
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
EXTRA=()
if [[ -n "${SHADOW_EVAL_GT:-}" ]]; then
  EXTRA+=(--ground-truth "$SHADOW_EVAL_GT")
fi
python -u scripts/run_shadow_segmenter_suite.py \
  --plan "${SHADOW_SEG_SUITE:?Set the aligned suite.json}" \
  --output-root "${SHADOW_SEG_SUITE_OUTPUT:?Set a new training output directory}" \
  --gpus "${SHADOW_SEG_GPUS:-1}" "${EXTRA[@]}"
