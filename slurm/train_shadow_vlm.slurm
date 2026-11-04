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
BALANCE_ARGS=()
INIT_ARGS=()
TEMPORAL_ARGS=()
if [[ -n "${SHADOW_MAX_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--max-steps "$SHADOW_MAX_STEPS")
fi
if [[ "${SHADOW_BALANCE:-1}" == "1" ]]; then
  BALANCE_ARGS+=(--balance-shadow --balance-power "${SHADOW_BALANCE_POWER:-0.5}" --max-class-weight "${SHADOW_MAX_CLASS_WEIGHT:-5}")
fi
if [[ -n "${SHADOW_INIT_ADAPTER:-}" ]]; then
  INIT_ARGS+=(--init-adapter "$SHADOW_INIT_ADAPTER")
fi
if [[ -n "${SHADOW_TEMPORAL_POLICY:-}" ]]; then
  TEMPORAL_ARGS+=(--temporal-policy "$SHADOW_TEMPORAL_POLICY")
fi

GPU_MONITOR_PID=""
if [[ -n "${SHADOW_GPU_LOG:-}" ]]; then
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits --loop-ms=2000 > "$SHADOW_GPU_LOG" &
  GPU_MONITOR_PID=$!
  trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
fi

python scripts/check_joint_shadow_annotation_data.py --data-root "$DATA_ROOT"

python scripts/train_shadow_vlm.py \
  --model "${SHADOW_VLM_MODEL:?Set SHADOW_VLM_MODEL to the local Qwen3-VL directory}" \
  --primary-task shadow_annotation \
  --train "$DATA_ROOT/annotation_train.jsonl" \
  --dev "$DATA_ROOT/annotation_dev.jsonl" \
  --output-root "$OUTPUT_ROOT" \
  --load-in-4bit --max-pixels 147456 --max-length 4096 \
  --epochs "${SHADOW_EPOCHS:-3}" --accumulation "${SHADOW_ACCUMULATION:-16}" \
  --lr "${SHADOW_LR:-1e-4}" --lora-rank "${SHADOW_LORA_RANK:-16}" \
  --loss-mode "${SHADOW_LOSS_MODE:-component}" \
  --decision-loss-weight "${SHADOW_DECISION_LOSS_WEIGHT:-1.0}" \
  --association-loss-weight "${SHADOW_ASSOCIATION_LOSS_WEIGHT:-0.5}" \
  --format-loss-weight "${SHADOW_FORMAT_LOSS_WEIGHT:-0.1}" \
  --allow-weak-labels --weak-weight 0.25 --seed 0 \
  "${BALANCE_ARGS[@]}" "${INIT_ARGS[@]}" "${TEMPORAL_ARGS[@]}" "${EXTRA_ARGS[@]}"
