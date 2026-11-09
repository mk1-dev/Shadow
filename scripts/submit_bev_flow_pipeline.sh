#!/bin/bash
set -euo pipefail
cd "${SHADOW_PROJECT_ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}"
export SHADOW_PROJECT_ROOT="$PWD"
if [[ -n "${SHADOW_AUTO_PIPELINE_CONFIG:-}" ]]; then
  EXTRA=()
  if [[ "${SHADOW_AUTO_SUBMIT:-0}" == 1 ]]; then EXTRA+=(--submit); fi
  exec python -u scripts/submit_automatic_shadow_pipeline.py prepare \
    --config "$SHADOW_AUTO_PIPELINE_CONFIG" \
    --output-root "${SHADOW_AUTO_PIPELINE_OUTPUT:?Set a new automatic pipeline directory}" \
    --fraction "${SHADOW_AUTO_FRACTION:-25}" "${EXTRA[@]}"
fi
if [[ "${SHADOW_LEGACY_PIPELINE:-0}" != 1 ]]; then
  echo "Use scripts/submit_automatic_shadow_pipeline.py for the frozen v8+ mainline." >&2
  echo "The legacy teacher/VLM pipeline requires explicit SHADOW_LEGACY_PIPELINE=1." >&2
  exit 2
fi
export SHADOW_ANNOTATION_ROOT="${SHADOW_ANNOTATION_ROOT:-$PWD/outputs/shadow_bev_flow_$(date +%Y%m%d_%H%M%S)}"
export SHADOW_DATA_ROOT="$SHADOW_ANNOTATION_ROOT/data"
export SHADOW_ANNOTATION_DATA_ROOT="$SHADOW_ANNOTATION_ROOT/joint"
export SHADOW_SEG_ROOT="$SHADOW_ANNOTATION_ROOT/segmenter"
export SHADOW_VLM_ROOT="$SHADOW_ANNOTATION_ROOT/vlm_lora"
export SHADOW_VLM_ADAPTER="$SHADOW_VLM_ROOT/adapter"
export SHADOW_DIAG_ROOT="$SHADOW_ANNOTATION_ROOT/diagnostics"
export SHADOW_EVAL_ROOT="$SHADOW_ANNOTATION_ROOT/evaluation"
unset SHADOW_INIT_ADAPTER SHADOW_SEG_INIT_CHECKPOINT
if [[ "${SHADOW_RECEIVER_AUDIT_ONLY:-0}" == 1 ]]; then
  echo "Audit-only does not create training data; submit slurm/rebuild_bev_flow_labels.slurm directly." >&2
  exit 1
fi
if [[ -z "${SHADOW_REBUILD_SOURCE:-}" ]]; then
  : "${UNIAD_ROOT:?Set the mature OpenDriveVLA repository}"
  : "${UNIAD_CONDA_ENV:?Set the working mature encoder environment}"
  : "${UNIAD_CHECKPOINT:?Set the mature encoder checkpoint}"
  : "${UNIAD_TRAIN_ANN_FILE:?Set train temporal infos}"
  : "${UNIAD_VAL_ANN_FILE:?Set val temporal infos}"
fi
: "${SHADOW_VLM_MODEL:?Set the local Qwen3-VL model}"
mkdir -p logs
if [[ -n "${SHADOW_REBUILD_SOURCE:-}" ]]; then
  DATA_JOB=$(sbatch --parsable slurm/rebuild_bev_flow_labels.slurm)
else
  DATA_JOB=$(sbatch --parsable slurm/prepare_shadow_annotation.slurm)
fi
SEG_JOB=$(sbatch --parsable --dependency="afterok:$DATA_JOB" slurm/train_shadow_segmenter.slurm)
VLM_JOB=$(sbatch --parsable --dependency="afterok:$DATA_JOB" slurm/train_shadow_vlm.slurm)
SEG_EVAL=$(sbatch --parsable --dependency="afterok:$SEG_JOB" --export=ALL,SHADOW_EVAL_TASK=segmenter slurm/evaluate_shadow_annotators.slurm)
VLM_DEV=$(sbatch --parsable --dependency="afterok:$VLM_JOB" --export=ALL,SHADOW_DIAG_SPLIT=dev slurm/diagnose_shadow_vlm.slurm)
CALIBRATION="${SHADOW_DIAG_ROOT:-$SHADOW_ANNOTATION_ROOT/diagnostics}/dev/analysis/calibration.json"
VLM_EVAL=$(sbatch --parsable --dependency="afterok:$VLM_DEV" --export="ALL,SHADOW_DIAG_SPLIT=eval,SHADOW_CALIBRATION=$CALIBRATION" slurm/diagnose_shadow_vlm.slurm)
JOINT_EVAL=$(sbatch --parsable --dependency="afterok:$VLM_JOB" --export=ALL,SHADOW_EVAL_TASK=vlm slurm/evaluate_shadow_annotators.slurm)
printf 'root=%s\ndata=%s\nsegmenter=%s\nvlm=%s\nsegmenter_eval=%s\nvlm_dev=%s\nvlm_frozen_eval=%s\njoint_eval=%s\n' \
  "$SHADOW_ANNOTATION_ROOT" "$DATA_JOB" "$SEG_JOB" "$VLM_JOB" "$SEG_EVAL" "$VLM_DEV" "$VLM_EVAL" "$JOINT_EVAL"
