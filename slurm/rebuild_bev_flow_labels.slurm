#!/bin/bash
#SBATCH -J shadow_static_labels
#SBATCH -p AMD
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_rebuild_%j.out
#SBATCH -e logs/shadow_rebuild_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ "${SHADOW_RUN_TESTS:-1}" == 1 ]]; then
  python -m pytest -q tests/test_bev_flow.py tests/test_physics_shadow_labels.py tests/test_bev_teacher.py tests/test_static_shadow_teacher.py
fi
python -u scripts/rebuild_bev_flow_labels.py \
  --source-root "${SHADOW_REBUILD_SOURCE:?Set the existing cached preparation root}" \
  --output-root "${SHADOW_ANNOTATION_ROOT:?Set a new output directory}" \
  --positive-fusion "${SHADOW_POSITIVE_FUSION:-consensus}" \
  --negative-policy "${SHADOW_NEGATIVE_POLICY:-endpoints+lit-receivers}"
