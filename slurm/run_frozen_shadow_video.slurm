#!/bin/bash
#SBATCH -J shadow_frozen_scenes
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH -o logs/shadow_frozen_%j.out
#SBATCH -e logs/shadow_frozen_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

# BEV and DIS caches must already exist; this job performs no model training/export.
python -u scripts/run_frozen_shadow_video.py run \
  --lock "${SHADOW_VIDEO_FREEZE_LOCK:?Set the frozen v8+ lock JSON}" \
  --batches "${SHADOW_VIDEO_NEW_BATCHES:?Set new-scene batches JSONL}" \
  --output-root "${SHADOW_VIDEO_NEW_OUTPUT:?Set a new output directory}"
