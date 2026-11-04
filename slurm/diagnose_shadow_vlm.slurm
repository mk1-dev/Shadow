#!/bin/bash
#SBATCH -J shadow_vlm_diagnose
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH -o logs/shadow_diagnose_%j.out
#SBATCH -e logs/shadow_diagnose_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
DATA="${SHADOW_ANNOTATION_DATA_ROOT:-outputs/shadow_annotation/joint}"
ADAPTER="${SHADOW_VLM_ADAPTER:-${SHADOW_VLM_ROOT:-outputs/shadow_annotation/vlm_lora}/adapter}"
OUT="${SHADOW_DIAG_ROOT:-outputs/shadow_annotation/diagnostics}"
SPLIT="${SHADOW_DIAG_SPLIT:-dev}"
if [[ "$SPLIT" != "dev" && "$SPLIT" != "eval" ]]; then
  echo "SHADOW_DIAG_SPLIT must be dev or eval" >&2
  exit 1
fi
python scripts/score_shadow_annotation_decisions.py \
  --model "${SHADOW_VLM_MODEL:?Set local Qwen3-VL model path}" --load-in-4bit \
  --adapter "$ADAPTER" --manifest "$DATA/annotation_$SPLIT.jsonl" \
  --output-root "$OUT/$SPLIT/scores"
if [[ "$SPLIT" == "dev" ]]; then
  python scripts/analyze_shadow_decision_scores.py fit-dev \
    --scores "$OUT/dev/scores/scores.jsonl" --max-fpr "${SHADOW_MAX_DEV_FPR:-0.1}" \
    --output-root "$OUT/dev/analysis"
else
  python scripts/analyze_shadow_decision_scores.py evaluate \
    --scores "$OUT/eval/scores/scores.jsonl" \
    --calibration "${SHADOW_CALIBRATION:?Set frozen dev calibration.json}" \
    --output-root "$OUT/eval/analysis"
fi
python scripts/audit_shadow_decisions.py \
  --predictions "$OUT/$SPLIT/analysis/predictions.jsonl" \
  --output-root "$OUT/$SPLIT/review"
