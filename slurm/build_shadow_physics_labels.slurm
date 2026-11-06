#!/bin/bash
#SBATCH -J shadow_physics_labels
# CPU-only physics audit: do not request a GPU.
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/physics_labels_%j.out
#SBATCH -e logs/physics_labels_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-${SHADOW_ENV:-shadow_occ}}"
cd "${SHADOW_PROJECT_ROOT:-${SHADOW_ROOT:-${SLURM_SUBMIT_DIR:?}}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ROOT="${PHYSICS_ROOT:-outputs/physics_bev_flow_${SLURM_JOB_ID}}"
DATA="${DATAROOT:-/public/home/eng_zhihongli/project/data/nuscenes-full}"

if python -c "import pytest" >/dev/null 2>&1; then
  python -m pytest -q tests/test_physics_shadow_motion.py tests/test_physics_shadow_labels.py tests/test_bev_flow.py tests/test_cached_bev.py
else
  echo "pytest is unavailable in ${SHADOW_ENV:-shadow_occ}; skipping tests for this audit job." >&2
fi

# A few whole scenes are needed for independent other-track light support.
# No fixed scene is silently borrowed from official val for tuning.
python -u scripts/export_nuscenes_motion_pairs.py \
  --dataroot "$DATA" --split train --cams CAM_FRONT \
  --max-pairs "${PAIRS:-120}" --max-attempts "${ATTEMPTS:-600}" \
  --output-root "$ROOT/pairs"

python -u scripts/analyze_shadow_motion_pairs.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --flow-backend dis --device cpu \
  --output-root "$ROOT/flow"

echo "Dense flow is ready. BEV_FEATURE_ROOT must contain mature spatial features for these samples."
: "${BEV_FEATURE_ROOT:?Export mature BEV features before this BEV selection stage}"
python -u scripts/analyze_bev_flow_discrepancy.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --analysis-root "$ROOT/flow" \
  --feature-root "$BEV_FEATURE_ROOT" --output-root "$ROOT/bev_flow"

# Audit-only on purpose. Explicit weak export is a separate post-review command.
python -u scripts/build_physics_shadow_labels.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --analysis-root "$ROOT/flow" \
  --bev-flow-root "$ROOT/bev_flow" \
  --minimum-temporal-gain "${MINIMUM_TEMPORAL_GAIN:-0.1}" \
  --decision-policy "${SHADOW_PHYSICS_POLICY:-factorized-v3}" \
  --output-root "$ROOT/labels_audit"
