#!/bin/bash
set -euo pipefail
cd "${SHADOW_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export SHADOW_PROJECT_ROOT="$PWD"
mkdir -p logs
TAG="$(date +%Y%m%d_%H%M%S)"
PREP=""
if [[ -z "${SHADOW_REBUILD_SOURCE:-}" ]]; then
  export SHADOW_ANNOTATION_ROOT="$PWD/outputs/shadow_light_bootstrap_$TAG"
  export SHADOW_LIGHT_BOOTSTRAP_ONLY=1
  PREP="$(sbatch --parsable slurm/prepare_shadow_annotation.slurm)"
  export SHADOW_REBUILD_SOURCE="$SHADOW_ANNOTATION_ROOT"
fi
export SHADOW_LIGHT_ROOT="${SHADOW_LIGHT_ROOT:-$PWD/outputs/shadow_light_$TAG}"
DEP=()
if [[ -n "$PREP" ]]; then DEP+=(--dependency="afterok:$PREP"); fi
LIGHT="$(sbatch --parsable "${DEP[@]}" slurm/train_light_direction.slurm)"
export SHADOW_LIGHT_PRIOR_MANIFEST="$SHADOW_LIGHT_ROOT/predictions/light_priors.jsonl"
export SHADOW_ANNOTATION_ROOT="${SHADOW_LIGHT_REFINED_ROOT:-$PWD/outputs/shadow_light_refined_$TAG}"
export SHADOW_REUSE_SOURCE_ANALYSIS=1
export SHADOW_REUSE_ANALYSIS=0
export SHADOW_CANDIDATE_SOURCE="${SHADOW_CANDIDATE_SOURCE:-caster}"
export SHADOW_TEMPORAL_RADIUS="${SHADOW_TEMPORAL_RADIUS:-3}"
export SHADOW_RECEIVER_AUDIT_ONLY="${SHADOW_RECEIVER_AUDIT_ONLY:-1}"
AUDIT="$(sbatch --parsable --dependency="afterok:$LIGHT" slurm/rebuild_bev_flow_labels.slurm)"
printf 'bootstrap=%s\nlight=%s\nrefinement=%s\nsource=%s\nlight_root=%s\nrefined_root=%s\n' \
  "$PREP" "$LIGHT" "$AUDIT" "$SHADOW_REBUILD_SOURCE" "$SHADOW_LIGHT_ROOT" "$SHADOW_ANNOTATION_ROOT"
