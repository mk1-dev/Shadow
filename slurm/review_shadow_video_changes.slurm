#!/bin/bash
#SBATCH -J shadow_review
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o logs/shadow_review_%j.out
#SBATCH -e logs/shadow_review_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m pytest -q tests/test_shadow_video_review.py
PAIRS=("${SHADOW_VIDEO_PAIRS:?Set original pairs.jsonl}")
if [[ -n "${SHADOW_VIDEO_VAL_PAIRS:-}" ]]; then
  PAIRS+=("${SHADOW_VIDEO_VAL_PAIRS}")
fi
python -u scripts/review_shadow_video_changes.py \
  --before-root "${SHADOW_REVIEW_BEFORE:?Set frozen Geometry-Anchored run root}" \
  --after-root "${SHADOW_REVIEW_AFTER:?Set frozen Anchor-first run root}" \
  --pairs "${PAIRS[@]}" \
  --output-root "${SHADOW_REVIEW_OUTPUT:?Set a new directory outside both inputs}" \
  --context-radius "${SHADOW_REVIEW_CONTEXT_RADIUS:-1}"
