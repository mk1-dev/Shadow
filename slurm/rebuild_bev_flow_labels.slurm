#!/bin/bash
#SBATCH -J shadow_static_labels
#SBATCH -p AMD
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_rebuild_%j.out
#SBATCH -e logs/shadow_rebuild_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${SHADOW_RUN_TESTS:-1}" == 1 ]]; then
  python -m pytest -q tests/test_bev_flow.py tests/test_physics_shadow_labels.py tests/test_bev_teacher.py tests/test_static_shadow_teacher.py tests/test_receiver_explanations.py tests/test_mature_objects.py tests/test_rebuild_shadow_analysis.py
fi
REUSE_ARGS=()
if [[ "${SHADOW_REUSE_ANALYSIS:-0}" == 1 ]]; then
  REUSE_ARGS+=(--reuse-analysis)
fi
if [[ "${SHADOW_REUSE_SOURCE_ANALYSIS:-0}" == 1 ]]; then
  REUSE_ARGS+=(--reuse-source-analysis)
fi
if [[ "${SHADOW_RECEIVER_AUDIT_ONLY:-0}" == 1 ]]; then
  REUSE_ARGS+=(--audit-only)
fi
if [[ -n "${SHADOW_BASELINE_MANIFEST:-}" ]]; then
  REUSE_ARGS+=(--shadow-baseline-manifest "$SHADOW_BASELINE_MANIFEST")
fi
if [[ -n "${SHADOW_MATURE_OBJECT_ROOT:-}" ]]; then
  REUSE_ARGS+=(--mature-object-root "$SHADOW_MATURE_OBJECT_ROOT")
fi
python -u scripts/rebuild_bev_flow_labels.py \
  --source-root "${SHADOW_REBUILD_SOURCE:?Set the existing cached preparation root}" \
  --output-root "${SHADOW_ANNOTATION_ROOT:?Set a new output directory}" \
  --split-workers "${SHADOW_REBUILD_SPLIT_WORKERS:-2}" \
  --positive-fusion "${SHADOW_POSITIVE_FUSION:-consensus}" \
  --negative-policy "${SHADOW_NEGATIVE_POLICY:-endpoints+lit-receivers}" \
  --receiver-competition "${SHADOW_RECEIVER_COMPETITION:-on}" \
  --control-shift-m "${SHADOW_CONTROL_SHIFT_M:-3.0}" \
  --minimum-projection-gain "${SHADOW_MIN_PROJECTION_GAIN:-0.05}" \
  --minimum-explanation-margin "${SHADOW_MIN_EXPLANATION_MARGIN:-0.05}" \
  --appearance-prior-weight "${SHADOW_APPEARANCE_PRIOR_WEIGHT:-0.2}" \
  --mature-object-mode "${SHADOW_MATURE_OBJECT_MODE:-required}" \
  "${REUSE_ARGS[@]}"
