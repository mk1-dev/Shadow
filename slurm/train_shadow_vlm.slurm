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
EXTRA_ARGS=()
if [[ -n "${SHADOW_MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max-steps "$SHADOW_MAX_STEPS")
fi

GPU_MONITOR_PID=""
if [[ -n "${SHADOW_GPU_LOG:-}" ]]; then
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits --loop-ms=2000 > "$SHADOW_GPU_LOG" &
  GPU_MONITOR_PID=$!
  trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
fi

python scripts/train_shadow_vlm.py \
  --model "${SHADOW_VLM_MODEL:?Set SHADOW_VLM_MODEL to the local Qwen3-VL directory}" \
  --primary-task shadow_annotation \
  --train "$DATA_ROOT/annotation_train.jsonl" \
  --dev "$DATA_ROOT/annotation_dev.jsonl" \
  --output-root "$OUTPUT_ROOT" \
  --load-in-4bit --max-pixels 147456 --max-length 4096 \
  --epochs "${SHADOW_EPOCHS:-3}" --accumulation "${SHADOW_ACCUMULATION:-16}" \
  --allow-weak-labels --weak-weight 0.25 --balance-shadow --seed 0 "${EXTRA_ARGS[@]}"
