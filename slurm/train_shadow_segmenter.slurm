#!/bin/bash
#SBATCH -J shadow_segmenter
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_seg_%j.out
#SBATCH -e logs/shadow_seg_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
DATA_ROOT="${SHADOW_DATA_ROOT:-outputs/shadow_annotation/data}"
OUTPUT_ROOT="${SHADOW_SEG_ROOT:-outputs/shadow_annotation/segmenter}"
INIT_ARGS=()
if [[ -n "${SHADOW_SEG_INIT_CHECKPOINT:-}" ]]; then
  INIT_ARGS+=(--init-checkpoint "$SHADOW_SEG_INIT_CHECKPOINT")
fi

torchrun --standalone --nproc_per_node=2 scripts/train_shadow_segmenter.py \
  --train "$DATA_ROOT/seg_train.jsonl" \
  --dev "$DATA_ROOT/seg_dev.jsonl" \
  --output-root "$OUTPUT_ROOT" \
  --epochs "${SHADOW_SEG_EPOCHS:-10}" --batch-size "${SHADOW_SEG_BATCH_SIZE:-2}" \
  --accumulation "${SHADOW_SEG_ACCUMULATION:-4}" --temporal-weight "${SHADOW_SEG_TEMPORAL_WEIGHT:-0.2}" \
  --seed 0 --allow-weak-labels "${INIT_ARGS[@]}"
