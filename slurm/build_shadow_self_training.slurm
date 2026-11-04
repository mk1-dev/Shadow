#!/bin/bash
#SBATCH -J shadow_self_label_pilot
#SBATCH -p Intel
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH -o logs/shadow_self_label_%j.out
#SBATCH -e logs/shadow_self_label_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
BASE="${SHADOW_ANCHOR_ROOT:-outputs/shadow_annotation}"
RUN="${SHADOW_SELF_ROOT:-outputs/shadow_self_training_fold0}"
MODE="${SHADOW_SELF_MODE:-dry-run}"
EXTRA_ARGS=()
if [[ "${SHADOW_ALLOW_WEAK_PILOT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-weak-pilot)
fi
python scripts/build_shadow_self_training_round.py \
  --anchor-train "$BASE/joint/annotation_train.jsonl" \
  --dev "$BASE/joint/annotation_dev.jsonl" \
  --seg-anchor-train "$BASE/data/seg_train.jsonl" \
  --seg-dev "$BASE/data/seg_dev.jsonl" \
  --teacher-scores "$RUN/teacher_scores/scores.jsonl" \
  --score-protocol "$RUN/teacher_scores/scoring_protocol.json" \
  --teacher-annotations "$RUN/teacher_annotations/annotations.jsonl" \
  --annotation-protocol "$RUN/teacher_annotations/annotation_protocol.json" \
  --seg-predictions "$RUN/segmentation/predictions.jsonl" \
  --seg-metrics "$RUN/segmentation/segmentation_metrics.json" \
  --policy "$RUN/self_training_policy.json" \
  --mode "$MODE" --output-root "${SHADOW_SELF_OUTPUT:-$RUN/student_$MODE}" \
  "${EXTRA_ARGS[@]}"
