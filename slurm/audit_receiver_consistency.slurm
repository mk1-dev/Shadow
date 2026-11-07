#!/bin/bash
#SBATCH -J receiver_consistency
#SBATCH -p AMD
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o logs/receiver_consistency_%j.out
#SBATCH -e logs/receiver_consistency_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${SHADOW_RUN_TESTS:-1}" == 1 ]]; then
  python -m pytest -q tests/test_receiver_consistency.py tests/test_receiver_explanations.py
fi
EXTRA_ARGS=()
if [[ -n "${SHADOW_EXISTING_SAMPLING_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--sampling-root "$SHADOW_EXISTING_SAMPLING_ROOT")
fi
python -u scripts/audit_receiver_consistency.py \
  --source-root "${SHADOW_REBUILD_SOURCE:?Set the frozen pair/BEV source root}" \
  --output-root "${SHADOW_ANNOTATION_ROOT:?Set a new consistency report directory}" \
  --temporal-radius "${SHADOW_TEMPORAL_RADIUS:-3}" \
  --review-limit "${SHADOW_AUDIT_CASES_PER_TYPE:-20}" \
  "${EXTRA_ARGS[@]}"
