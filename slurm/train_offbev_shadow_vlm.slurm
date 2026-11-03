#!/bin/bash
#SBATCH -J shadow_offbev_vlm
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_offbev_vlm_%j.out
#SBATCH -e logs/shadow_offbev_vlm_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
ROOT_OUT="${SHADOW_OFFBEV_ROOT:-outputs/shadow_offbev}"
AUXILIARY=()
if [[ -n "${SHADOW_CASTER_AUX_DATA:-}" ]]; then
  AUXILIARY=(--auxiliary-train "$SHADOW_CASTER_AUX_DATA/vlm_train.jsonl" --auxiliary-dev "$SHADOW_CASTER_AUX_DATA/vlm_dev.jsonl")
fi
python scripts/train_shadow_vlm.py \
  --model "${SHADOW_VLM_MODEL:?Set SHADOW_VLM_MODEL}" \
  --primary-task offbev_caster \
  --train "$ROOT_OUT/data/offbev_train.jsonl" --dev "$ROOT_OUT/data/offbev_dev.jsonl" \
  --output-root "$ROOT_OUT/vlm_lora" --load-in-4bit --allow-weak-labels \
  --weak-weight 0.25 --caster-ratio 0.25 --seed 0 "${AUXILIARY[@]}"
