#!/bin/bash
#SBATCH -J shadow_vlm_lora
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_vlm_%j.out
#SBATCH -e logs/shadow_vlm_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
DATA_ROOT="${SHADOW_ANNOTATION_DATA_ROOT:-outputs/shadow_annotation/joint}"
OUTPUT_ROOT="${SHADOW_VLM_ROOT:-outputs/shadow_annotation/vlm_lora}"

python scripts/train_shadow_vlm.py \
  --model "${SHADOW_VLM_MODEL:?Set SHADOW_VLM_MODEL to the local Qwen3-VL directory}" \
  --primary-task shadow_annotation \
  --train "$DATA_ROOT/annotation_train.jsonl" \
  --dev "$DATA_ROOT/annotation_dev.jsonl" \
  --output-root "$OUTPUT_ROOT" \
  --load-in-4bit --max-pixels 147456 --max-length 4096 \
  --epochs 3 --accumulation 16 \
  --allow-weak-labels --weak-weight 0.25 --seed 0
