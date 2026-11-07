#!/bin/bash
#SBATCH -J shadow_mature_3d
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_mature_3d_%j.out
#SBATCH -e logs/shadow_mature_3d_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${UNIAD_CONDA_ENV:?Set the working mature-model environment}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
PROJECT="$PWD"
SOURCE="${SHADOW_REBUILD_SOURCE:?Set existing pairs/flow/feature preparation root}"
OUTPUT="${SHADOW_MATURE_OBJECT_ROOT:?Set a new decoded-objects cache root}"
for SPLIT in train val; do
  [[ -f "$SOURCE/$SPLIT/pairs/pairs.jsonl" ]] || {
    echo "Missing pair manifest: $SOURCE/$SPLIT/pairs/pairs.jsonl" >&2
    exit 2
  }
done
for SPLIT in train val; do
  if [[ "$SPLIT" == train ]]; then
    INFOS="${UNIAD_TRAIN_ANN_FILE:?Set train temporal infos}"
  else
    INFOS="${UNIAD_VAL_ANN_FILE:?Set val temporal infos}"
  fi
  python -u scripts/export_mature_bev.py \
    --external-root "${UNIAD_ROOT:?Set mature OpenDriveVLA repository}" \
    --model-config "${UNIAD_MODEL_CONFIG:-projects/configs/stage1_track_map/base_track_map.py}" \
    --checkpoint "${UNIAD_CHECKPOINT:?Set the matching mature checkpoint}" \
    --ann-file "$INFOS" \
    --dataroot "${SHADOW_NUSC_ROOT:-/public/home/eng_zhihongli/project/data/nuscenes-full}" \
    --pairs "$SOURCE/$SPLIT/pairs/pairs.jsonl" --output-root "$OUTPUT/$SPLIT" \
    --history-mode "${SHADOW_BEV_HISTORY_MODE:-temporal}" \
    --minimum-object-score "${SHADOW_MIN_OBJECT_SCORE:-0.25}" --objects-only
  cd "$PROJECT"
done
