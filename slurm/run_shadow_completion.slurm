#!/bin/bash
#SBATCH -J shadow_v9_complete
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/shadow_v9_%j.out
#SBATCH -e logs/shadow_v9_%j.err

set -euo pipefail
source "${SHADOW_CONDA_SH:-/public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh}"
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
ASSETS="${SHADOW_FSD_ASSETS:?}"
export PYTHONPATH="$ASSETS/python_deps:${PYTHONPATH:-}"
OUTPUT="${SHADOW_V9_OUTPUT:?Set a new output directory}"
mkdir "$OUTPUT"
python -u scripts/export_fsd_shadow_priors.py \
  --pairs "${SHADOW_PAIRS:?}" --clips "${SHADOW_CLIPS:?}" \
  --fsd-root "$ASSETS/source" --checkpoint "$ASSETS/FSD_best.ckpt" \
  --output-root "$OUTPUT/prior" --max-width "${SHADOW_PRIOR_WIDTH:-800}"
python -u scripts/discover_shadow_video.py \
  --pairs "$SHADOW_PAIRS" --clips "$SHADOW_CLIPS" \
  --analysis-root "${SHADOW_FLOW_ROOT:?}" \
  --feature-root "${SHADOW_CONTEXT_BEV:?}" --spatial-feature-root "${SHADOW_SPATIAL_BEV:?}" \
  --shadow-prior-manifest "$OUTPUT/prior/manifest.jsonl" \
  --output-root "$OUTPUT/discovery" --stage full --discovery-mode cross-space
