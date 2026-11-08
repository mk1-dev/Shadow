#!/bin/bash
#SBATCH -J shadow_video
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH -o logs/shadow_video_%j.out
#SBATCH -e logs/shadow_video_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m pytest -q tests/test_shadow_video.py tests/test_receiver_explanations.py tests/test_physics_shadow_labels.py
EXTRA=()
if [[ -n "${SHADOW_VIDEO_REVIEW_REGIONS:-}" ]]; then
  EXTRA+=(--review-regions "$SHADOW_VIDEO_REVIEW_REGIONS")
fi
python -u scripts/discover_shadow_video.py \
  --pairs "${SHADOW_VIDEO_PAIRS:?Set pairs.jsonl}" \
  --analysis-root "${SHADOW_VIDEO_FLOW:?Set the matching cached DIS analysis directory}" \
  --feature-root "${SHADOW_VIDEO_FEATURES:?Set the frozen mature BEV cache}" \
  --clips "${SHADOW_VIDEO_CLIPS:?Set the selected track clip JSONL}" \
  --output-root "${SHADOW_VIDEO_OUTPUT:?Set a new output directory}" \
  --stage "${SHADOW_VIDEO_STAGE:-full}" \
  --bev-role "${SHADOW_VIDEO_BEV_ROLE:-context}" "${EXTRA[@]}"
