#!/bin/bash
#SBATCH -J shadow_entity_ab
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH -o logs/shadow_entity_ab_%j.out
#SBATCH -e logs/shadow_entity_ab_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m pytest -q tests/test_shadow_entity_ab.py tests/test_entity_memory.py tests/test_shadow_video.py
EXTRA=()
if [[ -n "${SHADOW_VIDEO_REVIEW_REGIONS:-}" ]]; then
  EXTRA+=(--review-regions "$SHADOW_VIDEO_REVIEW_REGIONS")
fi
python -u scripts/compare_shadow_entity_ab.py \
  --pairs "${SHADOW_VIDEO_PAIRS:?Set pairs.jsonl}" \
  --analysis-root "${SHADOW_VIDEO_FLOW:?Set matching DIS analysis directory}" \
  --feature-root "${SHADOW_VIDEO_FEATURES:?Set frozen BEV cache}" \
  --clips "${SHADOW_VIDEO_CLIPS:?Set selected clip JSONL}" \
  --output-root "${SHADOW_VIDEO_OUTPUT:?Set a new output directory}" "${EXTRA[@]}"
