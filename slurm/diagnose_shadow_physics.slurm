#!/bin/bash
#SBATCH -J Shadow_physics_pilot
#SBATCH -p debugpc
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/physics_%j.out
#SBATCH -e logs/physics_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_ENV:-shadow_occ}"
cd "${SHADOW_ROOT:-/public/home/eng_zhihongli/project/Shadow}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

RUN="${SHADOW_PHYSICS_ROOT:-outputs/physics_motion_${SLURM_JOB_ID:?}}"
DATA="${SHADOW_DATAROOT:-/public/home/eng_zhihongli/project/data/nuscenes-full}"
SCENE="${SHADOW_PHYSICS_SCENE:-scene-0001}"

python -m pytest -q tests/test_physics_shadow_motion.py
python -u scripts/export_nuscenes_motion_pairs.py \
  --dataroot "$DATA" \
  --version v1.0-trainval --split train \
  --scene "$SCENE" --cams CAM_FRONT \
  --max-pairs "${SHADOW_PHYSICS_PAIRS:-20}" \
  --working-width 800 \
  --output-root "$RUN/pairs"

python -u scripts/analyze_shadow_motion_pairs.py \
  --pairs "$RUN/pairs/pairs.jsonl" \
  --flow-backend dis --device cpu --review-limit 20 \
  --output-root "$RUN/dis"

echo "Stage 1/2 evidence written to $RUN; not approved shadow training labels."
