#!/bin/bash
#SBATCH -J shadow_light
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH -o logs/shadow_light_%j.out
#SBATCH -e logs/shadow_light_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
ROOT="${SHADOW_LIGHT_ROOT:?Set a new output directory for light estimation}"
if [[ "${SHADOW_RUN_TESTS:-1}" == 1 ]]; then
  python -m pytest -q tests/test_light_direction.py tests/test_caster_window.py tests/test_physics_shadow_labels.py tests/test_receiver_explanations.py
fi
python -u scripts/prepare_light_direction_data.py \
  --source-root "${SHADOW_REBUILD_SOURCE:?Set the completed BEV/3D audit root}" \
  --dataroot "${SHADOW_NUSC_ROOT:-/public/home/eng_zhihongli/project/data/nuscenes-full}" \
  --output-root "$ROOT/data"
python -u scripts/train_light_direction.py --data-root "$ROOT/data" --output-root "$ROOT/model" \
  --epochs "${SHADOW_LIGHT_EPOCHS:-10}" --crossfit-folds "${SHADOW_LIGHT_FOLDS:-2}" --device cuda
python -u scripts/predict_light_direction.py --data-root "$ROOT/data" --checkpoint-root "$ROOT/model" \
  --output-root "$ROOT/predictions" --device cuda
python -u scripts/predict_light_direction.py --data-root "$ROOT/data" \
  --method solar --output-root "$ROOT/solar_only" --device cpu
