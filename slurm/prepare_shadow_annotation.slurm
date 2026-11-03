#!/bin/bash
#SBATCH -J shadow_annotation_data
#SBATCH -p Intel
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH -o logs/shadow_annotation_data_%j.out
#SBATCH -e logs/shadow_annotation_data_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
ROOT_OUT="${SHADOW_ANNOTATION_ROOT:-outputs/shadow_annotation}"
NUSC_ROOT="${SHADOW_NUSC_ROOT:-/public/home/eng_zhihongli/project/data/nuscenes-full}"
LABEL_ROOT="${SHADOW_LABEL_ROOT:-outputs_temporal/shadow_labels}"
EXCLUSIONS=()
if [[ -n "${SHADOW_EXCLUDE_SCENES:-}" ]]; then
  EXCLUSIONS=(--exclude-scenes "$SHADOW_EXCLUDE_SCENES")
fi

for SPLIT in train val; do
  python scripts/export_nuscenes_caster_gt.py \
    --dataroot "$NUSC_ROOT" --split "$SPLIT" \
    --output "$ROOT_OUT/casters_$SPLIT.jsonl" \
    --max-samples "${SHADOW_EXPORT_SAMPLES:-1000}" "${EXCLUSIONS[@]}"
done
python scripts/export_caster_shadow_weak_data.py \
  --casters "$ROOT_OUT/casters_train.jsonl" "$ROOT_OUT/casters_val.jsonl" \
  --dataroot "$NUSC_ROOT" --label-root "$LABEL_ROOT" \
  --output-root "$ROOT_OUT/weak_images"
python scripts/add_shadow_temporal_evidence.py \
  --manifest "$ROOT_OUT/weak_images/manifest.jsonl" \
  --output-root "$ROOT_OUT/temporal_images"
python scripts/audit_shadow_temporal.py \
  --manifest "$ROOT_OUT/temporal_images/manifest.jsonl" \
  --output "$ROOT_OUT/temporal_images/audit_summary.json"
python scripts/build_shadow_annotation_dataset.py \
  --manifest "$ROOT_OUT/temporal_images/manifest.jsonl" \
  --output-root "$ROOT_OUT/data" --seed 0 "${EXCLUSIONS[@]}"
python scripts/check_shadow_annotation_data.py --data-root "$ROOT_OUT/data"
EXTERIOR_ARGS=()
if [[ "${SHADOW_INCLUDE_EXTERIOR_ASSOCIATIONS:-1}" == "1" ]]; then
  for SPLIT in train val; do
    python scripts/export_nuscenes_caster_gt.py \
      --dataroot "$NUSC_ROOT" --split "$SPLIT" --max-distance 100 \
      --min-lidar-points 0 --include-offscreen \
      --max-samples "${SHADOW_EXPORT_SAMPLES:-1000}" \
      --output "$ROOT_OUT/exterior_casters_$SPLIT.jsonl" "${EXCLUSIONS[@]}"
    python scripts/export_offbev_shadow_observations.py \
      --dataroot "$NUSC_ROOT" --split "$SPLIT" --label-root "$LABEL_ROOT" \
      --max-samples "${SHADOW_EXPORT_SAMPLES:-1000}" \
      --output-root "$ROOT_OUT/exterior_observations_$SPLIT" "${EXCLUSIONS[@]}"
  done
  python scripts/build_offbev_shadow_dataset.py \
    --observations "$ROOT_OUT/exterior_observations_train/observations.jsonl" "$ROOT_OUT/exterior_observations_val/observations.jsonl" \
    --casters "$ROOT_OUT/exterior_casters_train.jsonl" "$ROOT_OUT/exterior_casters_val.jsonl" \
    --teach-ambiguous-abstention --output-root "$ROOT_OUT/exterior_associations" --seed 0
  EXTERIOR_ARGS=(--offbev-data-root "$ROOT_OUT/exterior_associations")
fi
python scripts/build_joint_shadow_annotation_dataset.py \
  --region-data-root "$ROOT_OUT/data" --output-root "$ROOT_OUT/joint" "${EXTERIOR_ARGS[@]}"
python scripts/check_joint_shadow_annotation_data.py --data-root "$ROOT_OUT/joint"
