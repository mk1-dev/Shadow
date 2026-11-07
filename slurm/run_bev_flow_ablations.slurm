#!/bin/bash
#SBATCH -J shadow_bev_ablations
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_ablations_%j.out
#SBATCH -e logs/shadow_ablations_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
SOURCE="${SHADOW_ABLATION_SOURCE:?Set the completed shared BEV-flow preparation directory}"
OUTPUT="${SHADOW_ABLATION_ROOT:-${SOURCE}_ablations_$(date +%Y%m%d_%H%M%S)}"
read -r -a VARIANTS <<< "${SHADOW_ABLATION_VARIANTS:-full without_mature_3d projection_coverage geometry_only shuffled}"
EXTRA=()
if [[ -n "${SHADOW_CURRENT_BEV_ROOT:-}" ]]; then
  EXTRA+=(--current-cache-root "$SHADOW_CURRENT_BEV_ROOT")
fi
if [[ "${SHADOW_ABLATION_LABELS_ONLY:-0}" == 1 ]]; then
  EXTRA+=(--labels-only)
fi
python -m pytest -q tests/test_cached_bev.py tests/test_bev_teacher.py tests/test_bev_flow_ablations.py tests/test_physics_shadow_labels.py tests/test_physics_shadow_motion.py tests/test_receiver_explanations.py tests/test_mature_objects.py
python -u scripts/run_bev_flow_ablations.py --source-root "$SOURCE" --output-root "$OUTPUT" \
  --variants "${VARIANTS[@]}" --seed "${SHADOW_ABLATION_SEED:-0}" "${EXTRA[@]}"
