#!/bin/bash
#SBATCH -J shadow_fsd_prior
#SBATCH -p NV_RTX4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH -o logs/shadow_fsd_prior_%j.out
#SBATCH -e logs/shadow_fsd_prior_%j.err

set -euo pipefail
source "${SHADOW_CONDA_SH:-/public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh}"
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
ASSETS="${SHADOW_FSD_ASSETS:?Set the external FSD asset directory}"
export PYTHONPATH="$ASSETS/python_deps:${PYTHONPATH:-}"
EXTRA=()
if [[ -n "${SHADOW_CLIPS:-}" ]]; then
  EXTRA+=(--clips "$SHADOW_CLIPS")
fi
python -u scripts/export_fsd_shadow_priors.py \
  --pairs "${SHADOW_PAIRS:?}" --fsd-root "$ASSETS/source" \
  --checkpoint "$ASSETS/FSD_best.ckpt" --output-root "${SHADOW_PRIOR_OUTPUT:?}" \
  --max-width "${SHADOW_PRIOR_WIDTH:-800}" "${EXTRA[@]}"
