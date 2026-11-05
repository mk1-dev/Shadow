#!/bin/bash
#SBATCH -J shadow_mature_bev
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH -o logs/mature_bev_%j.out
#SBATCH -e logs/mature_bev_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
PROJECT="${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
cd "$PROJECT"
PROJECT="$PWD"
ROOT="${PHYSICS_ROOT:?Set PHYSICS_ROOT to the completed CPU preparation root}"
ROOT="$(realpath "$ROOT")"
FEATURE_ROOT="${BEV_FEATURE_ROOT:-$ROOT/mature_bev}"
FEATURE_ROOT="$(realpath -m "$FEATURE_ROOT")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

conda activate "${UNIAD_CONDA_ENV:?Set the existing working UniAD/OpenDriveVLA conda environment}"
python -u "$PROJECT/scripts/export_mature_bev.py" \
  --external-root "${UNIAD_ROOT:?Set the mature OpenDriveVLA repository path}" \
  --model-config "${UNIAD_MODEL_CONFIG:-projects/configs/stage1_track_map/base_track_map.py}" \
  --checkpoint "${UNIAD_CHECKPOINT:?Set the mature model checkpoint}" \
  --ann-file "${UNIAD_ANN_FILE:?Set ordered train/val nuScenes infos for these pairs}" \
  --pairs "$ROOT/pairs/pairs.jsonl" --output-root "$FEATURE_ROOT" \
  --dataroot "${DATAROOT:-/public/home/eng_zhihongli/project/data/nuscenes-full}"

conda activate "${SHADOW_CONDA_ENV:-shadow_occ}"
cd "$PROJECT"
python -u scripts/analyze_bev_flow_discrepancy.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --analysis-root "$ROOT/flow" \
  --feature-root "$FEATURE_ROOT" --output-root "$ROOT/bev_flow_mature"
python -u scripts/build_physics_shadow_labels.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --analysis-root "$ROOT/flow" \
  --bev-flow-root "$ROOT/bev_flow_mature" --decision-policy factorized-v3 \
  --output-root "$ROOT/labels_bev_mature_audit"
