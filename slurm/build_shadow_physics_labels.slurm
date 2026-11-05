#!/bin/bash
#SBATCH -J shadow_physics_labels
#SBATCH -p debugpc
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/physics_labels_%j.out
#SBATCH -e logs/physics_labels_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_ENV:-shadow_occ}"
cd "${SHADOW_ROOT:-/public/home/eng_zhihongli/project/Shadow}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ROOT="${PHYSICS_ROOT:-outputs/physics_joint_${SLURM_JOB_ID}}"
DATA="${DATAROOT:-/public/home/eng_zhihongli/project/data/nuscenes-full}"

python -m pytest -q tests/test_physics_shadow_motion.py tests/test_physics_shadow_labels.py

# A few whole scenes are needed for independent other-track light support.
# No fixed scene is silently borrowed from official val for tuning.
python -u scripts/export_nuscenes_motion_pairs.py \
  --dataroot "$DATA" --split train --cams CAM_FRONT \
  --max-pairs "${PAIRS:-120}" --max-attempts "${ATTEMPTS:-600}" \
  --output-root "$ROOT/pairs"

python -u scripts/analyze_shadow_motion_pairs.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --flow-backend dis --device cpu \
  --output-root "$ROOT/flow"

# Audit-only on purpose. Explicit weak export is a separate post-review command.
python -u scripts/build_physics_shadow_labels.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --analysis-root "$ROOT/flow" \
  --output-root "$ROOT/labels_audit"
