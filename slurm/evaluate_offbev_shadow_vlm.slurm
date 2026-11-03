#!/bin/bash
#SBATCH -J shadow_offbev_eval
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_offbev_eval_%j.out
#SBATCH -e logs/shadow_offbev_eval_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
ROOT_OUT="${SHADOW_OFFBEV_ROOT:-outputs/shadow_offbev}"
python scripts/evaluate_offbev_shadow_vlm.py \
  --model "${SHADOW_VLM_MODEL:?Set SHADOW_VLM_MODEL}" --load-in-4bit \
  --manifest "$ROOT_OUT/data/offbev_eval.jsonl" --output-root "$ROOT_OUT/eval/base"
python scripts/evaluate_offbev_shadow_vlm.py \
  --model "$SHADOW_VLM_MODEL" --load-in-4bit --adapter "$ROOT_OUT/vlm_lora/adapter" \
  --manifest "$ROOT_OUT/data/offbev_eval.jsonl" --output-root "$ROOT_OUT/eval/tuned"
python scripts/compare_offbev_shadow_vlm.py \
  --base "$ROOT_OUT/eval/base/hypotheses.jsonl" --tuned "$ROOT_OUT/eval/tuned/hypotheses.jsonl" \
  --output "$ROOT_OUT/eval/comparison.json"
