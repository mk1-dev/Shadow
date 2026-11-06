#!/bin/bash
#SBATCH -J shadow_annotation_eval
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_annotation_eval_%j.out
#SBATCH -e logs/shadow_annotation_eval_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ROOT="${SHADOW_ANNOTATION_ROOT:-outputs/shadow_bev_flow}"
DATA_ROOT="${SHADOW_DATA_ROOT:-$ROOT/data}"
ANNOTATION_DATA="${SHADOW_ANNOTATION_DATA_ROOT:-$ROOT/joint}"
SEG_ROOT="${SHADOW_SEG_ROOT:-$ROOT/segmenter}"
VLM_ROOT="${SHADOW_VLM_ROOT:-$ROOT/vlm_lora}"
EVAL_ROOT="${SHADOW_EVAL_ROOT:-$ROOT/evaluation}"
TASK="${SHADOW_EVAL_TASK:-both}"
if [[ "$TASK" != both && "$TASK" != segmenter && "$TASK" != vlm ]]; then
  echo "SHADOW_EVAL_TASK must be segmenter, vlm or both" >&2
  exit 1
fi

if [[ "$TASK" != vlm ]]; then
python scripts/evaluate_shadow_segmenter.py \
  --manifest "$DATA_ROOT/seg_eval.jsonl" --checkpoint "$SEG_ROOT/best.pth" \
  --output-root "$EVAL_ROOT/segmentation"
fi
if [[ "$TASK" != segmenter ]]; then
python scripts/evaluate_shadow_annotator.py \
  --model "${SHADOW_VLM_MODEL:?Set SHADOW_VLM_MODEL}" --load-in-4bit \
  --manifest "$ANNOTATION_DATA/annotation_eval.jsonl" --output-root "$EVAL_ROOT/base"
python scripts/evaluate_shadow_annotator.py \
  --model "$SHADOW_VLM_MODEL" --load-in-4bit --adapter "$VLM_ROOT/adapter" \
  --manifest "$ANNOTATION_DATA/annotation_eval.jsonl" --output-root "$EVAL_ROOT/tuned"
python scripts/compare_shadow_annotations.py \
  --base "$EVAL_ROOT/base/annotations.jsonl" \
  --tuned "$EVAL_ROOT/tuned/annotations.jsonl" \
  --output "$EVAL_ROOT/comparison.json"
fi
