# Zero-Shot Scene Observability Teacher

This is an RGB observation-priority task, not a shadow-label teacher. It runs in
parallel with the Receiver Geometry Consistency Audit. No VLM decision creates a
shadow positive, a negative, a caster association, a mask or a light direction.

## Inputs And Policy

The unit is scene x camera x short time segment: three clean chronological RGB
frames from exported nuScenes pairs. Ground masks, GT boxes, optical flow,
existing teacher decisions and colored diagnostic panels never reach the model.
Use an untuned local Qwen3-VL base model, not the existing shadow adapter. The
runner has no adapter argument and rejects an adapter directory.

The frozen prompt requests illumination, shadow observability, visual receiver
visibility, temporal usability, a coarse object-shadow relation, a short evidence
sentence and an uncertainty boolean. It does not request calibrated confidence.

- `PRIORITY`: clear cast-shadow observations, sufficient visible receiver, usable
  temporal context and no declared uncertainty. Stable temporal context qualifies;
  a visible moving boundary is not mandatory. Weather alone is not a hard gate.
- `LOW_SIGNAL`: valid but weak/not-observed shadow evidence or limited receiver.
  `not_observed` is not a true NO and never becomes negative supervision.
- `UNCERTAIN`: unknown shadow/receiver evidence, insufficient temporal context,
  declared uncertainty, or invalid model output. Parse errors remain in reports.
- `UNRATED`: source windows not covered by this tagging run. They are retained.

The coarse object-shadow relation does not gate priority or provide a caster ID.
It cannot establish whether an object is inside or outside BEV.

## First Run On Existing Audit Cases

The job below deduplicates the 85 audit CASES by physical window, preserving a
separate case-to-window index. It adds ten unscreened random windows per split
by default. The resulting unique-window count is not necessarily 85. These
failure-enriched cases are a pilot, not a representative nuScenes test set.

```bash
cd /public/home/eng_zhihongli/project/Shadow
export SHADOW_PROJECT_ROOT="$PWD"
export SHADOW_CONDA_ENV=shadow_annotator
export SHADOW_REBUILD_SOURCE="$PWD/outputs/shadow_transition_stages_20261107_211522"
export SHADOW_OBSERVABILITY_AUDIT_ROOT="$PWD/outputs/shadow_geometry_audit_20261107_224244/visual"
export SHADOW_OBSERVABILITY_ROOT="$PWD/outputs/shadow_observability_$(date +%Y%m%d_%H%M%S)"
export SHADOW_VLM_MODEL=/path/to/untuned/Qwen3-VL-8B-Instruct
mkdir -p logs
sbatch --export=ALL slurm/tag_shadow_observability.slurm
```

This requests one RTX 4090, runs CPU regression tests, prepares clean inputs,
runs 4-bit zero-shot inference, then compares existing physical outputs. It does
not train any model or rerun BEV/flow/caster search. Receiver consistency auditing
must still use the full, unscreened source independently.

Remove `SHADOW_OBSERVABILITY_AUDIT_ROOT` to tag all consecutive RGB windows in the
pair caches. Current input coverage is the existing pair export, not every raw
nuScenes camera frame; no broader dataset coverage claim follows automatically.

## Individual Commands

```bash
python scripts/prepare_shadow_observability.py \
  --source-root "$SHADOW_REBUILD_SOURCE" \
  --audit-root "$SHADOW_OBSERVABILITY_AUDIT_ROOT" --random-windows 10 \
  --output-root "$SHADOW_OBSERVABILITY_ROOT/inputs"

python scripts/run_shadow_observability.py \
  --model "$SHADOW_VLM_MODEL" --load-in-4bit \
  --manifest "$SHADOW_OBSERVABILITY_ROOT/inputs/manifest.jsonl" \
  --output-root "$SHADOW_OBSERVABILITY_ROOT/vlm"

python scripts/analyze_shadow_observability.py \
  --source-root "$SHADOW_REBUILD_SOURCE" \
  --manifest "$SHADOW_OBSERVABILITY_ROOT/inputs/manifest.jsonl" \
  --votes "$SHADOW_OBSERVABILITY_ROOT/vlm/votes.jsonl" \
  --output-root "$SHADOW_OBSERVABILITY_ROOT/analysis"
```

Append `--resume` to the inference command to continue the same manifest/model
settings. Finished votes, including parse errors, are not silently regenerated.
The runner streams raw responses, parsed fields, timing and input signatures.

## Reports And Physical-Stage Integration

`analysis/summary.json` reports `ALL_SOURCE`, `A_ALL_TAGGED`, `PRIORITY`,
`LOW_SIGNAL`, `UNCERTAIN` and `UNRATED` separately. It includes receiver-supported
windows, uncorrected central brightness changes, supported central events,
event/projection matches, physical-gate passes, exported positives per 100
windows, unresolved positives and uniquely associated positives. Uncorrected
events are evaluated on tagged windows only, with the coverage denominator
reported; they are not shadow truth. Existing outputs with missing per-window
physics timing retain null timing rather than fabricated runtimes.

`analysis/priority_windows.jsonl` contains ALL source windows: priority first,
then a seeded 10% sample of nonpriority windows for audit, then the remainder.
It is a scheduling report, NOT a replacement pair manifest or a votes file.
No throughput improvement or shadow precision is claimed before validation.

An optional integration is available when rebuilding physical labels later:

```bash
export SHADOW_OBSERVABILITY_MANIFEST="$SHADOW_OBSERVABILITY_ROOT/vlm/votes.jsonl"
export SHADOW_OBSERVABILITY_ORDER=priority
```

`slurm/rebuild_bev_flow_labels.slurm` forwards these to the label builder. Its
default is `original` ordering, and both modes retain every window. Votes only
attach metadata and optionally reorder work; they never enable a physical gate,
change scores, delete cloudy scenes or supply training targets. Use a new output
root for a changed manifest/order. These flags do not prioritize or skip upstream
BEV export or DIS computation. Budgeted upstream scheduling is not implemented.

Per-window processing seconds in NEW label outputs cover the window-processing
stage, not the later cross-window light consensus or full end-to-end training.

Inspect representative RGBs from every stratum and retain random unscreened
examples. A high-observability group with zero receiver support is a priority
geometry failure case, not something to discard. Group comparisons are the same
physical method under different observation conditions, not separate algorithms
or evidence of true shadow precision/recall.
