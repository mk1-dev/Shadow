#!/bin/bash
#SBATCH -J shadow_observability
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH -o logs/shadow_observability_%j.out
#SBATCH -e logs/shadow_observability_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
: "${SHADOW_REBUILD_SOURCE:?Set the existing frozen pair/BEV/label root}"
: "${SHADOW_OBSERVABILITY_ROOT:?Set a new tagging output directory}"
: "${SHADOW_VLM_MODEL:?Set the untuned local Qwen3-VL base model directory}"
if [[ "${SHADOW_RUN_TESTS:-1}" == 1 ]]; then
  python -m pytest -q tests/test_shadow_observability.py tests/test_receiver_explanations.py tests/test_rebuild_shadow_analysis.py
fi
PREPARE_ARGS=()
if [[ -n "${SHADOW_OBSERVABILITY_AUDIT_ROOT:-}" ]]; then
  PREPARE_ARGS+=(--audit-root "$SHADOW_OBSERVABILITY_AUDIT_ROOT"
    --random-windows "${SHADOW_OBSERVABILITY_RANDOM_WINDOWS:-10}")
fi
python -u scripts/prepare_shadow_observability.py \
  --source-root "$SHADOW_REBUILD_SOURCE" \
  --output-root "$SHADOW_OBSERVABILITY_ROOT/inputs" "${PREPARE_ARGS[@]}"
python -u scripts/run_shadow_observability.py \
  --manifest "$SHADOW_OBSERVABILITY_ROOT/inputs/manifest.jsonl" \
  --model "$SHADOW_VLM_MODEL" --load-in-4bit \
  --output-root "$SHADOW_OBSERVABILITY_ROOT/vlm"
python -u scripts/analyze_shadow_observability.py \
  --source-root "$SHADOW_REBUILD_SOURCE" \
  --manifest "$SHADOW_OBSERVABILITY_ROOT/inputs/manifest.jsonl" \
  --votes "$SHADOW_OBSERVABILITY_ROOT/vlm/votes.jsonl" \
  --output-root "$SHADOW_OBSERVABILITY_ROOT/analysis"
