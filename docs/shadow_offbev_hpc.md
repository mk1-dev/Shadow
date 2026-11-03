# Potential outside-BEV casters inferred from visible shadows

The main workflow is the single GT-constrained shadow annotation task in
[shadow_joint_annotator_hpc.md](shadow_joint_annotator_hpc.md). Potential exterior
casters are association fields of that annotation. This document describes
boundary-observation data preparation and the earlier standalone selector as a
diagnostic; it is not an additional required VLM task.

```text
camera frames -> shadow segmentation and conditional temporal evidence
  -> visible boundary-shadow component INSIDE the nominal BEV
  -> geometry-only uncertain candidate regions OUTSIDE the BEV
  -> VLM selects a potential caster region or ABSTAINS
  -> independent GT association/localization evaluation
```

For example, a caster footprint at x=43m can lie outside [-40,40]m while part
of its shadow falls at x=38m. The hypothesis output can describe a region beyond
x=40m. It is not an occupied voxel inside the original grid, and cannot be
evaluated as an Occ3D voxel beyond Occ3D's supplied spatial bounds.

## What the code now provides

- `export_nuscenes_caster_gt.py --include-offscreen --max-distance 100` retains
  distant or camera-invisible object GT. The older 30m visible-only export must
  not be reused for outside-BEV target discovery.
- `export_offbev_shadow_observations.py` generates crops, previous-frame context
  and outside candidate banks without reading object GT. Its temporal view uses
  ego-motion ground alignment only; it never uses the hidden target's GT motion
  as an inference input.
- `build_offbev_shadow_dataset.py` uses object annotations only to build training
  targets and evaluation associations. By default it excludes cases where the
  associated caster appears directly in any supplied current/previous crop.
- `train_shadow_vlm.py --primary-task offbev_caster` trains structured
  POTENTIAL/ABSTAIN decisions. Existing visible-caster GT auxiliary tasks can
  still be mixed in using `--auxiliary-train/--auxiliary-dev`.
- `evaluate_offbev_shadow_vlm.py` outputs uncertain outside regions and compares
  them with GT associations, including a geometry-only Top-1 baseline.

Object GT proves object existence and location. It does not prove which shadow
belongs to it. Automatic unique geometry matches are therefore `geometry_weak`.
No matching annotation remains unknown, never an empty-space negative. Multiple
compatible casters can supply an explicitly enabled weak ABSTAIN target: this
means insufficient evidence to select one caster, not that no caster exists.

The current hypothesis bank extrapolates both directions of the shadow principal
axis near the BEV edge into 0-4m, 4-8m and 8-16m exterior bands. This is a first
geometric baseline without measured sunlight or a known caster height. VLM
selects among these uncertainty regions; it does not freely invent exact 3D boxes.
Objects beyond the bank are counted as bank misses when an independent positive
association is available. A fixed-sun rendered prior is not treated as GT.

## HPC workflow

Use the annotation environment and dependencies described in
[shadow_annotation_hpc.md](shadow_annotation_hpc.md). Preserve segmentation and
its temporal training; those are still the front end.

```bash
cd /public/home/eng_zhihongli/project/Shadow_update
mkdir -p logs
export SHADOW_VLM_MODEL=/path/to/Qwen3-VL-8B-Instruct
export SHADOW_OFFBEV_ROOT=outputs/shadow_offbev_v1

sbatch slurm/prepare_offbev_shadow.slurm
# After checking the generated data_check.json and coverage:
sbatch slurm/train_offbev_shadow_vlm.slurm
# After training succeeds:
sbatch slurm/evaluate_offbev_shadow_vlm.slurm
```

CPU preparation defaults to 1000 samples per split. Change
`SHADOW_OFFBEV_EXPORT_SAMPLES` to select a pilot size. `SHADOW_OFFBEV_MASK_ROOT`
can point to existing or refined BEV masks. The observation exporter also accepts
`--seg-predictions` to intersect projected shadow evidence with actual image
segmentation probabilities; the prediction index must cover the requested split.
As with the previous stage, supplying `SHADOW_EXCLUDE_SCENES` excludes tuning or
already examined scenes. New output roots are required instead of overwriting.

The preparation job reports scene-disjoint train/dev/eval counts, unknown rows,
visible-target exclusions, and reachable associations. Check that there are
enough hidden-target POTENTIAL examples and meaningful ABSTAIN targets in both
train and dev before training. If none survive, inspect the observations and GT
coverage first; changing the language model cannot create missing supervision.

Optional existing visible-caster auxiliary data:

```bash
export SHADOW_CASTER_AUX_DATA=outputs/shadow_annotation/data_temporal_v1
sbatch slurm/train_offbev_shadow_vlm.slurm
```

Auxiliary and primary manifests must use the same scene partition. No official
val scene or prior frame from a different scene is admitted to training.

## Independent association input

For independently established shadow-caster pairs, pass `--associations` to the
dataset builder. The JSONL format is:

```json
{"observation_id":"sample:camera:component","annotation_token":"nuScenes_annotation_token","association_source":"manual_caster_association_gt","decision":"POTENTIAL"}
```

`synthetic_caster_association_gt` is also supported. Known ambiguous/unresolvable
examples can use `decision: ABSTAIN` without an annotation token. Preserve the
association source: geometry/temporal weak links must not be renamed GT just
because the linked object has an accurate 3D box.

GT supervision is excluded from the VLM prompt. Its inputs contain images,
nominal BEV bounds, and geometry-generated candidate regions only. GT target
coordinates, identity, category and association answers remain target metadata.
The default hidden-only filter checks current crops and the actual warped
previous context for a directly visible caster. Missing/ambiguous visibility
evidence is excluded rather than assumed hidden. `--allow-visible-caster` is a
diagnostic override and must be reported as such.

## Outputs and metrics

`hypotheses.jsonl` includes `decision`, `potential_caster_region_xyxy_ego`,
`boundary_side`, candidate choice scores, GT association provenance and nominal
BEV bounds. The spatial output is a possible exterior region, not a confirmed
object category, shape, height or exact center. `ABSTAIN` does not mark space free.

`metrics.json` reports, separately by association source:

- candidate-bank recall: upper bound from whether the GT location is covered;
- outside region localization recall;
- geometry-only Top-1 recall;
- positive hypothesis rate and abstention accuracy on explicit targets;
- cell-center error on matched regions, with misses represented by recall.

`comparison.json` compares the base VLM with its LoRA adapter on identical input
IDs/signatures and uses scene-cluster bootstrap for localization recall changes.
Scores are uncalibrated conditional likelihoods; tune the choice threshold on
dev and freeze it for eval. Pure geometry matches only support weak consistency
claims. Independent association GT is needed to measure actual shadow-based
caster localization accuracy.

Do not write these hypotheses directly into the existing OCC voxel labels.
The next research question is whether a calibrated exterior hypothesis improves
early target discovery or prediction when the target later enters the BEV.
That requires a separate trajectory/time-to-entry experiment beyond this code's
current region-association evaluator.

## HPC checks

```bash
python -m compileall shadow_occ/annotation scripts
python -m pytest tests/test_shadow_offbev.py tests/test_shadow_annotation_temporal.py -q
python scripts/check_offbev_shadow_data.py --data-root outputs/shadow_offbev_v1/data
```

No local model validation or HPC job submission was performed for this addition.
