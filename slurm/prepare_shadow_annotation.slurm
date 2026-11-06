#!/bin/bash
#SBATCH -J shadow_bev_flow_data
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_annotation_data_%j.out
#SBATCH -e logs/shadow_annotation_data_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
PROJECT="$PWD"
ROOT_OUT="${SHADOW_ANNOTATION_ROOT:-$PROJECT/outputs/shadow_bev_flow}"
mkdir -p "$ROOT_OUT"
ROOT_OUT="$(realpath "$ROOT_OUT")"
NUSC_ROOT="${SHADOW_NUSC_ROOT:-/public/home/eng_zhihongli/project/data/nuscenes-full}"
ANNOTATION_ENV="${SHADOW_CONDA_ENV:-shadow_annotator}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
read -r -a CAMERAS <<< "${SHADOW_CAMERAS:-CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT}"

conda activate "$ANNOTATION_ENV"
if [[ "${SHADOW_RUN_TESTS:-1}" == 1 ]]; then
  python -m pytest -q tests/test_bev_flow.py tests/test_cached_bev.py tests/test_bev_teacher.py
fi
for SPLIT in train val; do
  echo "[1/5] Export $SPLIT raw geometry and dense flow"
  python -u scripts/export_nuscenes_motion_pairs.py \
    --dataroot "$NUSC_ROOT" --split "$SPLIT" --cams "${CAMERAS[@]}" \
    --max-pairs "${SHADOW_PAIRS_PER_SPLIT:-1200}" --max-attempts "${SHADOW_PAIR_ATTEMPTS:-12000}" \
    --max-pairs-per-scene "${SHADOW_PAIRS_PER_SCENE:-24}" \
    --output-root "$ROOT_OUT/$SPLIT/pairs"
  python -u scripts/analyze_shadow_motion_pairs.py \
    --pairs "$ROOT_OUT/$SPLIT/pairs/pairs.jsonl" --flow-backend dis --device cpu \
    --output-root "$ROOT_OUT/$SPLIT/flow"
done

conda activate "${UNIAD_CONDA_ENV:?Set the existing working UniAD/OpenDriveVLA environment}"
for SPLIT in train val; do
  echo "[2/5] Export $SPLIT frozen mature BEVFormer features"
  if [[ "$SPLIT" == train ]]; then
    INFOS="${UNIAD_TRAIN_ANN_FILE:?Set train temporal infos}"
  else
    INFOS="${UNIAD_VAL_ANN_FILE:?Set val temporal infos}"
  fi
  python -u "$PROJECT/scripts/export_mature_bev.py" \
    --external-root "${UNIAD_ROOT:?Set the mature OpenDriveVLA repository}" \
    --model-config "${UNIAD_MODEL_CONFIG:-projects/configs/stage1_track_map/base_track_map.py}" \
    --checkpoint "${UNIAD_CHECKPOINT:?Set the mature encoder checkpoint}" \
    --ann-file "$INFOS" --dataroot "$NUSC_ROOT" \
    --pairs "$ROOT_OUT/$SPLIT/pairs/pairs.jsonl" --output-root "$ROOT_OUT/$SPLIT/mature_bev"
done

conda activate "$ANNOTATION_ENV"
cd "$PROJECT"
for SPLIT in train val; do
  echo "[3/5] Select $SPLIT receivers using learned BEV, flow and illumination"
  python -u scripts/analyze_bev_flow_discrepancy.py \
    --pairs "$ROOT_OUT/$SPLIT/pairs/pairs.jsonl" --analysis-root "$ROOT_OUT/$SPLIT/flow" \
    --feature-root "$ROOT_OUT/$SPLIT/mature_bev" --minimum-bev-cosine "${SHADOW_BEV_MIN_COSINE:-0.3}" \
    --output-root "$ROOT_OUT/$SPLIT/bev_flow"
  python -u scripts/build_physics_shadow_labels.py \
    --pairs "$ROOT_OUT/$SPLIT/pairs/pairs.jsonl" --analysis-root "$ROOT_OUT/$SPLIT/flow" \
    --bev-flow-root "$ROOT_OUT/$SPLIT/bev_flow" --decision-policy factorized-v3 \
    --export-weak-labels --output-root "$ROOT_OUT/$SPLIT/labels"
  echo "[4/5] Export observed shadows and lit endpoints for $SPLIT"
  python -u scripts/build_bev_flow_annotation_manifest.py \
    --pairs "$ROOT_OUT/$SPLIT/pairs/pairs.jsonl" --labels-root "$ROOT_OUT/$SPLIT/labels" \
    --output-root "$ROOT_OUT/$SPLIT/teacher"
done

echo "[5/5] Build shared segmentation and unified VLM datasets"
python -u scripts/build_shadow_annotation_dataset.py \
  --manifest "$ROOT_OUT/train/teacher/canonical.jsonl" "$ROOT_OUT/val/teacher/canonical.jsonl" \
  --output-root "$ROOT_OUT/data" --seed 0
python -u scripts/check_shadow_annotation_data.py --data-root "$ROOT_OUT/data"
python -u scripts/build_joint_shadow_annotation_dataset.py \
  --region-data-root "$ROOT_OUT/data" --output-root "$ROOT_OUT/joint"
python -u scripts/check_joint_shadow_annotation_data.py --data-root "$ROOT_OUT/joint" --require-both-classes
echo "New BEV-flow data ready: $ROOT_OUT/data and $ROOT_OUT/joint"
