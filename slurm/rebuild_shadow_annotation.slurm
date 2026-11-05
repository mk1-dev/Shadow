#!/bin/bash
#SBATCH -J shadow_rebuild_latest
# CPU-only rebuild: use the current data builders without re-exporting raw nuScenes.
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -o logs/shadow_rebuild_%j.out
#SBATCH -e logs/shadow_rebuild_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SOURCE_ROOT="${SHADOW_SOURCE_ROOT:?Set SHADOW_SOURCE_ROOT to an existing annotation root}"
OUTPUT_ROOT="${SHADOW_ANNOTATION_ROOT:?Set SHADOW_ANNOTATION_ROOT to a new output root}"
OFFBEV_ROOT="${SHADOW_OFFBEV_ROOT:-}"

if [[ -e "$OUTPUT_ROOT" ]] && [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Output root must be new or empty: $OUTPUT_ROOT" >&2
  exit 1
fi

python scripts/build_shadow_annotation_dataset.py \
  --manifest "$SOURCE_ROOT/temporal_images/manifest.jsonl" \
  --output-root "$OUTPUT_ROOT/data" --seed 0
python scripts/check_shadow_annotation_data.py --data-root "$OUTPUT_ROOT/data"

JOINT_ARGS=()
if [[ -n "$OFFBEV_ROOT" ]]; then
  JOINT_ARGS=(--offbev-data-root "$OFFBEV_ROOT")
fi
python scripts/build_joint_shadow_annotation_dataset.py \
  --region-data-root "$OUTPUT_ROOT/data" \
  --output-root "$OUTPUT_ROOT/joint" "${JOINT_ARGS[@]}"
python scripts/check_joint_shadow_annotation_data.py --data-root "$OUTPUT_ROOT/joint"
