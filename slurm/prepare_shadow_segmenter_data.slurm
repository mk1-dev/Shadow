#!/bin/bash
#SBATCH -J shadow_seg_data
#SBATCH -p AMD
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH -o logs/shadow_seg_data_%j.out
#SBATCH -e logs/shadow_seg_data_%j.err

set -euo pipefail
source "${SHADOW_CONDA_SH:-/public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh}"
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

python -u scripts/build_shadow_mainline_experiments.py export \
  --runs "${SHADOW_AUTO_RUNS:?Set the existing frozen experiment_runs.jsonl}" \
  --freeze-lock "${SHADOW_VIDEO_FREEZE_LOCK:?Set the original teacher freeze lock}" \
  --auto-negatives --negative-weight "${SHADOW_NEGATIVE_WEIGHT:-0.5}" \
  --size 384 704 --output-root "${SHADOW_AUTO_DATA_ROOT:?Set a new dataset directory}"

read -r -a METHODS <<< "${SHADOW_SEG_METHODS:-appearance_only anchor_first_v8plus}"
INIT=()
if [[ -n "${SHADOW_SEG_INIT_CHECKPOINT:-}" ]]; then
  INIT+=(--init-checkpoint "$SHADOW_SEG_INIT_CHECKPOINT")
fi
python -u scripts/build_shadow_mainline_experiments.py suite \
  --dataset-root "$SHADOW_AUTO_DATA_ROOT" --methods "${METHODS[@]}" \
  --reference-method anchor_first_v8plus --skip-untrainable \
  --epochs "${SHADOW_SEG_EPOCHS:-10}" --batch-size "${SHADOW_SEG_BATCH_SIZE:-2}" \
  --accumulation "${SHADOW_SEG_ACCUMULATION:-4}" --seed "${SHADOW_SEED:-0}" \
  --backbone-weights "${SHADOW_SEG_BACKBONE_WEIGHTS:-imagenet}" \
  --scene-sampling supervision --output-root "${SHADOW_AUTO_SUITE_ROOT:?Set a new suite directory}" "${INIT[@]}"
