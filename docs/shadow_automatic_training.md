# Automatic Weak Supervision to Shadow Models

Current mainline: freeze v8+ -> automatic scene expansion -> physical weak
trimaps -> Segmenter -> held-out teacher-agreement inference. No manual shadow
mask, human positive/negative selection, or completed review pack is required.
GT Box/Track and LiDAR are offline teacher inputs, not shadow mask truth.

M1-M4 positive discovery and acceptance rules remain unchanged. The new data
adapter reuses the existing `lit_receiver_negative_mask` for weak negative
supervision; it does not add a network or turn all unlabeled pixels into zeros.
Student Shadow-BEV and OCC integration are later stages, not implemented or
validated merely by running this annotation/training workflow.

## 1. Prepared Input Catalogue

Prepare the existing nuScenes pairs, DIS analysis, temporal/context BEV and
current-only BEV caches for the intended train/val scenes. The planning command
does not export missing RGB, GT, flow or BEV caches.

`automatic_inputs.jsonl`:

```json
{"pairs":"/data/prepared_train/pairs.jsonl","analysis_root":"/data/prepared_train/flow","feature_root":"/data/bev_train_context","spatial_feature_root":"/data/bev_train_current_only"}
{"pairs":"/data/prepared_val/pairs.jsonl","analysis_root":"/data/prepared_val/flow","feature_root":"/data/bev_val_context","spatial_feature_root":"/data/bev_val_current_only"}
```

An old cache covering seven scenes is still a seven-scene population, not the
complete dataset. Optional `--expected-train-scenes` checks the expected official
train scene count before any plan is written. It does not infer completeness
from the presence of a trainval database directory.

Before expanding caches, estimate disk using existing files, without submitting
training or changing discovery:

```bash
python scripts/estimate_shadow_pipeline_storage.py \
  --catalogue /data/automatic_inputs.jsonl --lock outputs/v8plus_freeze/lock.json \
  --destination /data/new_shadow_cache --output-root outputs/shadow_storage_025 \
  --reference-run /data/frozen_v8plus_run --fraction 25 --methods 2
```

This writes `population.json`, `cache_requirements.jsonl` and
`storage_estimate.json`. Without `--population`, official scene names come from
the installed nuScenes SDK. No raw nuScenes tables are loaded. The estimate
measures actual Pair/Flow/native BEV file sizes and reference teacher outputs;
it includes overlapping seven-frame/stride-three windows, two student checkpoints
per method, deduplicated targets and held-out predictions. Missing measurements
remain unknown. A 30% headroom is a planning allowance, not a disk reservation.
It does not delete old files, export caches or submit jobs.

## 2. Automatic Scene and Clip Planning

Use the existing v8+ freeze lock. Include all previous teacher-development scenes
in its exclusion registry; these can be training data but not held-out eval.
When creating a lock, `run_frozen_shadow_video.py freeze --development-pairs ...`
registers actual Pair manifests from earlier development/threshold experiments,
including scenes not present in the final twelve demonstration clips. Do not
pass future, never-used cache catalogues as development evidence.

```bash
python -u scripts/plan_automatic_shadow_scenes.py \
  --catalogue /data/automatic_inputs.jsonl \
  --lock outputs/v8plus_freeze/lock.json \
  --output-root outputs/shadow_auto_plan \
  --annotation-root /public/home/eng_zhihongli/project/.shadow_auto_annotations \
  --seed 0 --dev-fraction 0.15 --num-frames 7 --stride 3 \
  --population outputs/shadow_storage_025/population.json --expected-train-scenes 700 \
  --methods appearance_only anchor_first_v8plus
```

The plan uses only existing scene identities, timestamps and GT track continuity:

- Assign scene-level train/dev once from official train, using a seeded hash.
- Keep official val as eval, excluding prior teacher-development scenes.
- Build nested 25%, 50%, 100% subsets of the remaining training scene pool.
- Select contiguous 3-7-frame windows and one deterministic eligible vehicle
  track per window, without inspecting shadow masks, RGB quality or VLM votes.
- Record windows without a usable GT track rather than inventing one. This
  remains a track-centric annotator, not guaranteed full-image coverage.

Metadata is sharded per source/scene so the discovery CLI does not repeatedly
load every training scene for every clip. Pair image/support paths and flow
evidence paths become absolute; numeric geometry and raw flow NPZs are unchanged.
Subset flow protocols record parent hashes and `raw_flow_recomputed=false`.
The existing evidence hash checks still apply when replay reads each NPZ.

Outputs:

```text
shared/batches.jsonl              # dev + eval: annotate once
stage_025/batches.jsonl           # first 25% training scenes
stage_050/batches.jsonl           # additional scenes to reach 50%
stage_100/batches.jsonl           # additional scenes to reach 100%
stage_025/experiment_runs.jsonl   # cumulative train + shared dev/eval
stage_050/experiment_runs.jsonl
stage_100/experiment_runs.jsonl
scene_inputs/                    # metadata-only pair/flow shards + clips
scene_coverage.jsonl
plan.json
```

The 50% and 100% steps do not rerun earlier training scenes. Dev/eval annotation
outputs are reused, keeping the reference teacher and split fixed across stages.
Train subset percentages refer to the scene pool after the train-derived dev
holdout, not percentages of all official train+val scenes.
With `--population`, roles and incremental subsets are fixed from the complete
official population before caches are available. Partial caches never cause a
new dev split or a smaller population denominator. Missing stage scenes are
listed explicitly; the Slurm annotation entry waits for complete target coverage.
At 700 train scenes and dev fraction 0.15, dev has 105 scenes, the train pool has
595, and its 25% stage has 149 scenes. Eval excludes prior teacher-development
scenes listed in the lock.

`appearance_only` switches only the existing proposal discovery mode. Receiver,
physics acceptance, caster association and memory remain identical to v8+.
This is an appearance-proposal ablation, not a geometry-free shadow system.
Geometry/Flow and historical v7 comparisons require their real frozen outputs;
the planner does not simulate those methods using renamed v8+ results.

## 3. CPU Annotation and Automatic Dataset Construction

Shared labels first:

```bash
mkdir -p logs
export SHADOW_AUTO_PLAN="$PWD/outputs/shadow_auto_plan"
export SHADOW_VIDEO_FREEZE_LOCK="$PWD/outputs/v8plus_freeze/lock.json"
SHARED=$(sbatch --parsable -p AMD --export=ALL,SHADOW_AUTO_STAGE=shared \
  slurm/prepare_automatic_shadow_stage.slurm)
```

Then the first training stage:

```bash
export SHADOW_AUTO_DATA_ROOT="$PWD/outputs/shadow_auto_025_data"
export SHADOW_AUTO_SUITE_ROOT="$PWD/outputs/shadow_auto_025_suite"
DATA=$(sbatch --parsable -p AMD --dependency=afterok:$SHARED \
  --export=ALL,SHADOW_AUTO_STAGE=stage_025 \
  slurm/prepare_automatic_shadow_stage.slurm)
```

The CPU job runs unchanged discovery, then:

```bash
python scripts/build_shadow_mainline_experiments.py export \
  --runs outputs/shadow_auto_plan/stage_025/experiment_runs.jsonl \
  --freeze-lock outputs/v8plus_freeze/lock.json \
  --auto-negatives --output-root outputs/shadow_auto_025_data
```

The adapter generates zero pixels automatically only when the existing physical
lit-receiver rule is supported by actual saved observations:

- A physically accepted positive core and saved light hypothesis exist.
- The same receiver is visible in all three local observations.
- Real BEV descriptor support and saved ground-motion support are present.
- Raw sampled Lab-L stays locally lit across those observations, with current
  contrast relative to the accepted shadow core.
- The negative patch is outside plausible shadows of all available annotated
  object volumes and outside current observed shadow proposals.

The existing defaults for lit ratio, contrast, radius and area budget are reused.
This is weak evidence, not absolute exclusion of every unannotated tree/building
shadow. Missing prerequisites yield no negative; they do not become default zeros.
Boundary frames without three observations remain IGNORE unless another real
window supplies evidence. Frozen positive discovery NPZs are never changed.

Training targets retain `1=weak positive`, `0=physical weak negative`,
`255=unknown/occluded/unsupported`. Overlapping clips are deduplicated per camera
frame, conflicting evidence becomes IGNORE, and memory-only states emit no masks.
Scene summaries count deduplicated image pixels/frames, not independent shadows.

The original confidence, receiver, caster, source and region-track evidence is
retained through exact frame/image NPZ and `frames.jsonl` references; no missing
field is silently promoted into a training target. Outputs include:

- Per-method `seg_train/dev/eval.jsonl` and trimaps.
- `dataset_summary.json` and `scene_dataset_counts.jsonl`.
- Stable delivery names `annotation_summary.json`, `dataset_manifest.json`,
  `class_distribution.json`, including resized per-image and per-scene counts.
- Per-method `automatic_negatives.jsonl`, usable as a common frozen negative pool
  in later supervision comparisons.
- Saved discovery predictions/scores, separately from training targets.

No manual task list is generated by default. `--manual-eval-template` is optional
and does not gate annotation or training. If actual support still produces no
known negatives, the dataset is exported with that fact, but two-class training
does not start on fabricated labels.

## 4. Segmenter Training Without Manual GT

The CPU stage prepares the actual methods in `plan.json` (S0/S3 when both were
planned). `SHADOW_SEG_METHODS` can request additional actual exported methods.
`method_availability.json` records absent or one-class arms instead of inventing
labels. A remaining arm still needs two known classes after the paired-image
intersection. Start with S0/S3 rather than waiting for historical methods.

```bash
export SHADOW_SEG_SUITE="$SHADOW_AUTO_SUITE_ROOT/suite.json"
export SHADOW_SEG_SUITE_OUTPUT="$PWD/outputs/shadow_auto_025_segmenter"
unset SHADOW_EVAL_GT
sbatch --dependency=afterok:$DATA slurm/run_shadow_segmenter_suite.slurm
```

This uses the existing DeepLabV3-MobileNetV3-Large balanced BCE/Dice training on
known pixels only. Sample scale, initialization, scene split and budget are
recorded. The suite uses uniform scene sampling, then uniform image sampling
within each scene, with replacement and a fixed draw count per epoch. DDP ranks
shard the same seeded global draw list; there is no class oversampling. Class
loss weights account for this expected scene sampling distribution. Legacy
single-model training retains ordinary image sampling unless requested.

Exported weak negatives have `negative_weight=0.5` by default, configurable by
`SHADOW_NEGATIVE_WEIGHT` or export `--negative-weight`. This is a fixed loss
coefficient, not a calibrated probability, and affects only known negatives in
BCE/Dice. Positive evidence and IGNORE do not change. Class balancing is computed
before this reliability coefficient so it does not cancel the discount. All
compared arms must use the same setting. Set a shared `SHADOW_SEG_INIT_CHECKPOINT` before preparing the CPU suite
to use a local pretrained segmenter. A checkpoint previously trained with manual
shadow masks is not part of the no-manual-shadow-supervision main result; record
its source as an explicit pretraining control. Default initialization and an
ImageNet-only backbone do not add shadow-mask supervision. Offline ImageNet initialization requires
already-cached torchvision weights. No target-domain manual shadow mask enters
this default path.

The current bridge does not create new temporal-loss evidence. Temporal and
receiver-pair loss stay zero in the first same-supervision comparison unless real
compatible evidence is exported; a `temporal_weak` provenance tag alone does not
activate temporal training.

After training, the runner automatically infers on the held-out frozen teacher
reference configured in `suite.json`, even without `--ground-truth`. All methods
in a comparison share one reference teacher and the same eval images.

Outputs are deliberately named:

```text
metrics/table_teacher_agreement.csv
metrics/metrics.json               # reference_kind=frozen-teacher
metrics/teacher_agreement.csv
metrics/cross_scene_summary.json
<method>/segmenter_history.json
<method>/checkpoint_best.pth        # hardlink to noncollapsed best.pth on HPC
training_status.csv
suite_status.json
```

Columns such as `teacher_agreement_iou` and `teacher_agreement_recall` quantify
student imitation on known teacher pixels, **not true shadow IoU/Precision/Recall**.
The JSON records `is_true_shadow_accuracy=false`. No Table E1/E2 manual-GT accuracy
table is silently generated from pseudo-labels. Optional future independent GT
evaluation uses the separate `--ground-truth` path.
The agreement table includes teacher-positive recall, teacher-negative recall,
training resized weak-positive count and train/eval scene counts. Per-scene
confusions and negative recall are saved separately. A failed/collapsed arm has
its status and history, but no fabricated best-checkpoint alias or agreement row.

Collapsed checkpoints remain failures and are not reused for inference or pseudo
label cleaning. Training success, teacher agreement and downstream benefit are
different results; a completed job alone does not prove any of them.

## 5. Scale to 50%, Then 100%

After the 25% dataset/training run is usable, submit `stage_050`, then `stage_100`
with new dataset/suite/training output roots. Stage 50 requires the existing stage
25 annotation outputs; stage 100 additionally requires stage 50. Reuse `shared`.
The cumulative manifests automatically reference the earlier immutable outputs.

For S0-S3 comparisons, supply real Appearance, Geometry/Flow, v7 and v8+ frozen
run manifests to the same dataset/suite tools. Unsupported variants are not
approximated by relabeling an Anchor-first output. If all arms use the v8+-derived
`automatic_negatives.jsonl` as a common pool, report that explicitly as a
**common-physical-negative, differing-positive-label comparison**, not a pure
appearance-only pipeline. Keep initialization, image IDs, scene roles and budget
fixed. The existing suite enforces paired usable training images.

## 6. Later VLM, Student Shadow-BEV and OCC

Once pixel supervision is usable, train Unified VLM from generated region-level
YES/NO/UNCERTAIN and actual owner supervision, never from its old unverified
predictions. No outside-BEV learning claim is made without actual examples.

The offline teacher may consume GT tracks, LiDAR and future frames. Online
student/OCC inference must not. Student Shadow-BEV must keep receiver shadow,
caster evidence and reliability separate; an illuminated ground shadow is not
an occupied voxel. Freeze the deployment input contract before exporting it.

Occ3D supervision belongs to the later OCC loss, not this shadow teacher.
O0 versus shadow/reliability-gated inputs must use the same mature BEV encoder,
OCC head, split, seeds and training budget. True Occ3D occupancy metrics are a
separate independent downstream endpoint, not evidence of measured shadow mask
accuracy. These deployment/fusion stages are not started by the current CPU or
Segmenter scripts.

## Tests

```bash
python -m pytest -q tests/test_shadow_automatic_pipeline.py tests/test_shadow_mainline_experiments.py
```

Tests cover nested incremental scene selection, immutable raw flow reuse, real
synthetic lit-negative computation, missing-evidence IGNORE, weak-reference metric
names and no-manual-GT training orchestration. GPU calls are mocked locally;
these tests are not actual HPC training or full-scene discovery results.
