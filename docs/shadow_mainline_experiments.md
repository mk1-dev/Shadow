# Shadow_OCC Mainline Experiments

The default route now follows [automatic weak supervision and training](shadow_automatic_training.md).
Manual masks below are optional independent accuracy evaluation, not a prerequisite
for annotation, model training or held-out teacher-agreement inference.

Priority: **E1 independent-scene discovery quality, then E2 learning from weak
labels**. Freeze v8+ and reuse DeepLabV3-MobileNetV3. Video exports, owner ranking
experiments and temporal ablations are not prerequisites for these two tasks.
No M1-M4 discovery/acceptance formulas are changed by this workflow.

## 1. Expand Frozen Automatic Annotation

Use `scripts/run_frozen_shadow_video.py` and the existing prepared pair/flow/BEV
caches on new scenes, as described in [shadow_video_release.md](shadow_video_release.md).
The replay does not export fresh nuScenes/BEV caches itself. All prior tuning
scenes belong in the lock's development exclusion set, not just the last 12 clips.

Appearance-only can use the existing discovery CLI's `--discovery-mode appearance`
on the exact same inputs. Other proposed methods need their own real frozen
outputs. The following bridge does **not** manufacture Flow-only, BEV-only,
Geometry/Flow or v7 baselines by renaming v8+ outputs.

Assign train-derived train/dev scenes before preparing masks. Official val/test
scenes are eval-only. An eval frame cannot also appear as train/dev under another
method. Use new scene manifests, not another random frame split of the old clips.

Create `experiment_runs.jsonl` with one row per actual method/run/split:

```json
{"method":"appearance_only","split":"train","run_root":"/runs/appearance_train","pairs":["/data/new_train/pairs.jsonl"]}
{"method":"appearance_only","split":"dev","run_root":"/runs/appearance_dev","pairs":["/data/new_dev/pairs.jsonl"]}
{"method":"appearance_only","split":"eval","run_root":"/runs/appearance_eval","pairs":["/data/new_eval/pairs.jsonl"]}
{"method":"anchor_first_v8plus","split":"train","run_root":"/runs/anchor_train","pairs":["/data/new_train/pairs.jsonl"]}
{"method":"anchor_first_v8plus","split":"dev","run_root":"/runs/anchor_dev","pairs":["/data/new_dev/pairs.jsonl"]}
{"method":"anchor_first_v8plus","split":"eval","run_root":"/runs/anchor_eval","pairs":["/data/new_eval/pairs.jsonl"]}
```

Each run must already contain `protocol.json`, `frames.jsonl` and per-frame
image/grid NPZs. Paths may be absolute or relative to `experiment_runs.jsonl`.
Do not put the export directory inside a frozen input run.

## 2. Export E1 Predictions and E2 Weak Data

```bash
python -u scripts/build_shadow_mainline_experiments.py export \
  --runs /data/experiment_runs.jsonl \
  --freeze-lock outputs/v8plus_freeze/lock.json \
  --negative-manifests /data/physical_train/canonical.jsonl /data/physical_dev/canonical.jsonl \
  --size 384 704 \
  --output-root outputs/shadow_E1_E2_data
```

Output for every supplied method:

- `seg_train.jsonl`, `seg_dev.jsonl`, `seg_eval.jsonl` and 0/1/255 image trimaps.
- `discovery_train.jsonl`, `discovery_dev.jsonl`, `discovery_eval.jsonl` with saved
  accepted masks and projected frozen ranking scores when available.
- Dataset scene/pixel counts, resized class counts and source fingerprints.
- Optional `--manual-eval-template`: clean image references with `target=null` and
  `independent=false`; this is an annotation task list, **not generated GT**.

Overlapping clips are merged per physical camera frame, not counted as new
training examples. Saved positive observations are unioned; contradiction with
explicit negative evidence becomes IGNORE. Merging uses disk staging rather
than retaining all image masks/scores in memory during scene expansion.

### Negative Labels

Frozen v8+ video targets encode only `1=weak positive, 255=IGNORE`. Exporting
more clips does not automatically create known non-shadow pixels. Neither dark
absence, BEV non-occupancy nor a brighter endpoint alone becomes a negative.

The optional negative manifests reuse **already-exported physical weak trimaps**,
such as existing `build_bev_flow_annotation_manifest.py` canonical outputs.
They must retain `label_source=temporal_weak` or `geometry_weak`,
`physics_export_enabled=true`, `physics_protocol`, `mask_encoding=trimap` and
`shadow_mask`/`target`. The bridge takes only their explicit zero pixels. Match
by `scene_name`, `camera`, `frame_token`; legacy canonical rows without a camera
token are matched by the exact original `image` path. Existing target hashes are
checked when supplied. Manual eval masks are rejected as negative sources.

Without negative evidence, export still supports E1 and records missing known
background. The formal two-class E2 suite cannot start until positive and
negative labels survive the actual training resize in both train and dev. This
is a real data requirement, not an invitation to label all IGNORE as background.
Use `--auto-negatives` to obtain those weak negatives automatically from saved
receiver/BEV/motion observations with the existing physical lit-receiver rule,
without an external negative manifest or manual selection.

## 3. Prepare Independent Manual Eval

Annotate new eval scenes only. Use the exported clean image task list without
copying teacher masks, model owner guesses or projected footprints as answers.
For each completed row, provide an original-resolution manual target and mark
`independent=true`. Target values are 0/1/255; 255 means unreviewed/uncertain.

```json
{"id":"frame_id","scene_name":"scene-XXXX","camera":"CAM_FRONT","frame_token":"CAMERA_SAMPLE_DATA_TOKEN","image":"/data/image.jpg","target":"/manual/mask.png","mask_encoding":"trimap","label_source":"manual_shadow_gt","split":"eval","independent":true}
```

Explicit binary 0/255 masks may declare `mask_encoding=binary`; suite inference
normalizes them into separate 0/1/255 files outside training directories. The
manual test scene list is checked against weak train/dev and development scenes.
Partial reviews measure only known pixels, so report review coverage separately.

This manual eval set is never used for automatic label generation, early
stopping, checkpoint selection or threshold tuning. An S4 supervised upper bound
requires a **separate manual training set** and is not synthesized by this suite.

## 4. E1 Discovery Table

Create `E1_methods.jsonl` for actual predictions on the same manual frames:

```json
{"experiment":"E1","method":"appearance_only","predictions":"/outputs/shadow_E1_E2_data/appearance_only/discovery_eval.jsonl"}
{"experiment":"E1","method":"anchor_first_v8plus","predictions":"/outputs/shadow_E1_E2_data/anchor_first_v8plus/discovery_eval.jsonl"}
```

```bash
python -u scripts/evaluate_shadow_mainline_experiments.py \
  --ground-truth /manual/eval.jsonl \
  --methods /data/E1_methods.jsonl \
  --dataset-summary outputs/shadow_E1_E2_data/dataset_summary.json \
  --freeze-lock outputs/v8plus_freeze/lock.json \
  --output-root outputs/shadow_E1_metrics
```

This writes `metrics.json` and `table_E1.csv`: Precision, Recall, F1, IoU, FPR,
Boundary F, predicted-positive coverage and score-ranking AP. Boundary tolerance
defaults to two original-image pixels with a square neighbourhood; unknown-mask
edges are excluded. Empty boundary support is reported as unavailable, not a
perfect result. GT is never resized to fit predictions, and missing method
predictions are errors rather than silently dropping frames.

The operating point uses the saved accepted binary mask. Ranking evaluation uses
the saved score field separately; it does not re-threshold accepted labels or
change v8+ physics. Projected scores are ranking evidence, not calibrated shadow
probabilities. Pixels outside valid score support receive lowest rank zero;
support coverage is reported separately. Without saved scores, AP stays null.

PR integration is stepwise average precision on a fixed 4096-bin histogram,
explicitly named `average_precision_binned`, not an exact/trapezoidal PR-AUC.
`score_diagnostics.pr_curve` and `recall_at_fpr` support common-FPR comparisons.
Those are descriptive test-curve points, **not deployable thresholds fitted on
test GT**. Positive coverage is `(TP+FP)/known GT pixels`, not a substitute for
Recall. Use the same audited image population for coverage comparisons.

## 5. E2 Shared Segmenter Training

```bash
python scripts/build_shadow_mainline_experiments.py suite \
  --dataset-root outputs/shadow_E1_E2_data \
  --methods appearance_only anchor_first_v8plus \
  --size 384 704 --epochs 10 --batch-size 2 --accumulation 4 --seed 0 \
  --output-root outputs/shadow_E2_suite
```

The suite intersects usable camera frames across supplied methods for train and
dev separately. Every arm sees the same image IDs, input resolution, epoch
budget, batch/accumulation, seed, optimizer settings and augmentation. Their
weak trimaps differ. Coverage beyond that shared image pool belongs in the full
data-expansion experiment, not an unequal-budget supervision comparison.

The existing balanced BCE/Dice trainer is reused. This first E2 comparison sets
temporal/receiver-pair loss weights to zero to isolate label quality; it does not
claim a T0-T3 temporal result. Weak sample scale is shared at 1.0 across arms.
Current random initialization is preserved by default. To use a common existing
pretraining checkpoint add `--init-checkpoint /checkpoints/segmenter.pth`; its
hash is recorded and checked for every arm. `--backbone-weights imagenet` is also
supported, but requires the torchvision weight file already cached on offline HPC.
No pretrained weight is silently downloaded by the suite's default configuration.

```bash
mkdir -p logs
export SHADOW_SEG_SUITE="$PWD/outputs/shadow_E2_suite/suite.json"
export SHADOW_SEG_SUITE_OUTPUT="$PWD/outputs/shadow_E2_training"
export SHADOW_EVAL_GT=/manual/eval.jsonl
sbatch slurm/run_shadow_segmenter_suite.slurm
```

This job uses one 4090 and runs the arms sequentially. For a two-GPU allocation,
submit `sbatch --gres=gpu:2 --export=ALL,SHADOW_SEG_GPUS=2 ...`; all arms use the
same process count. No physical GPU indices are hardcoded. H100 can use the same
runner with the site's H100 partition/allocation. Existing cached annotation
replay stays a separate CPU stage.

If manual GT is not ready, omit `SHADOW_EVAL_GT`: training and frozen held-out
teacher-agreement inference run without it. The separate
`metrics/table_teacher_agreement.csv` is not true shadow accuracy. With GT provided, successful arms
run frozen-threshold inference and generate `metrics/table_E2.csv` automatically.
Checkpoints are selected on weak train-derived dev only. Collapsed arms produce
no substituted `best.pth` and are preserved in `training_status.csv` and
`suite_status.json`; they must not be hidden from paper comparisons. They are
not converted into numerical zero-quality rows without actual evaluation.

## Scope and Verification

These entries prepare and execute E1/E2; they do not claim algorithm improvement
before independent GT results exist. E3 needs independent caster identities and
ranking results. E4 needs explicit valid temporal correspondences and a fixed
temporal metric protocol. Neither is approximated by owner counts or memory
lifespans here. Existing visualization/review tools remain optional.

```bash
python -m pytest -q tests/test_shadow_mainline_experiments.py
```

Local tests cover frozen-output export, class/scene isolation, IGNORE semantics,
overlap merging, fixed-budget planning and genuine pixel/boundary metric
calculations on synthetic masks. GPU training orchestration is mocked in tests;
real training, independent manual GT and HPC results are not local test outputs.
