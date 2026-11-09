#!/bin/bash
#SBATCH -J shadow_auto_data
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH -o logs/shadow_auto_%j.out
#SBATCH -e logs/shadow_auto_%j.err

set -euo pipefail
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate "${SHADOW_CONDA_ENV:-shadow_annotator}"
cd "${SHADOW_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

PLAN="${SHADOW_AUTO_PLAN:?Set the generated automatic scene-plan directory}"
LOCK="${SHADOW_VIDEO_FREEZE_LOCK:?Set the frozen v8+ lock JSON}"
STAGE="${SHADOW_AUTO_STAGE:-stage_025}"
case "$STAGE" in
  shared|stage_025|stage_050|stage_100) ;;
  *) echo "Unknown automatic annotation stage: $STAGE" >&2; exit 1 ;;
esac
python - "$PLAN/plan.json" "$STAGE" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
stage = sys.argv[2]
missing = plan.get("missing_shared_scenes", []) if stage == "shared" else plan.get("missing_stage_scenes", {}).get(str(int(stage[-3:])), [])
if missing:
    raise SystemExit(f"Target official scene caches are incomplete for {stage}: {len(missing)} missing; prepare caches before annotation")
PY
ANNOTATIONS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["annotation_output_root"])' "$PLAN/plan.json")
if [[ -s "$PLAN/$STAGE/batches.jsonl" ]]; then
  python -u scripts/run_frozen_shadow_video.py run \
    --lock "$LOCK" --batches "$PLAN/$STAGE/batches.jsonl" --output-root "$ANNOTATIONS/$STAGE"
else
  echo "No additional scene batches in $STAGE; cumulative manifests reuse earlier annotations."
fi
if [[ "$STAGE" == shared ]]; then
  exit 0
fi

DATA="${SHADOW_AUTO_DATA_ROOT:?Set a new automatic dataset output directory}"
SUITE="${SHADOW_AUTO_SUITE_ROOT:?Set a new segmenter suite directory}"
python -u scripts/build_shadow_mainline_experiments.py export \
  --runs "$PLAN/$STAGE/experiment_runs.jsonl" --freeze-lock "$LOCK" \
  --auto-negatives --negative-weight "${SHADOW_NEGATIVE_WEIGHT:-0.5}" --size 384 704 --output-root "$DATA"
INIT=()
if [[ -n "${SHADOW_SEG_INIT_CHECKPOINT:-}" ]]; then
  INIT+=(--init-checkpoint "$SHADOW_SEG_INIT_CHECKPOINT")
fi
read -r -a METHODS <<< "${SHADOW_SEG_METHODS:-$(python -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["methods"]))' "$PLAN/plan.json")}"
python -u scripts/build_shadow_mainline_experiments.py suite \
  --dataset-root "$DATA" --methods "${METHODS[@]}" --reference-method anchor_first_v8plus --skip-untrainable \
  --epochs "${SHADOW_SEG_EPOCHS:-10}" --batch-size "${SHADOW_SEG_BATCH_SIZE:-2}" \
  --accumulation "${SHADOW_SEG_ACCUMULATION:-4}" --seed "${SHADOW_SEED:-0}" \
  --backbone-weights "${SHADOW_SEG_BACKBONE_WEIGHTS:-none}" \
  --scene-sampling uniform \
  --output-root "$SUITE" "${INIT[@]}"
