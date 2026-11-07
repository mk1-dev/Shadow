#!/bin/bash
#SBATCH -J shadow_sampling
#SBATCH -p AMD
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o logs/shadow_sampling_%j.out
#SBATCH -e logs/shadow_sampling_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${SHADOW_RUN_TESTS:-1}" == 1 ]]; then
  python -m pytest -q tests/test_caster_window.py tests/test_receiver_explanations.py
fi
python -u scripts/audit_receiver_sampling.py \
  --source-root "${SHADOW_REBUILD_SOURCE:?Set the existing cached preparation root}" \
  --output-root "${SHADOW_ANNOTATION_ROOT:?Set a new output directory}" \
  --temporal-radius "${SHADOW_TEMPORAL_RADIUS:-3}" \
  --max-windows "${SHADOW_SAMPLING_MAX_WINDOWS:-0}"
