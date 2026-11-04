# Controlled offline self-annotation pilot

Date: 2026-11-04. Implementation and unit tests are synchronized in
`Shadow_update`; data/model runs remain on HPC.
This extends the SAME GT-constrained shadow annotator and RGB segmenter. It does
not add another VLM detection task and does not replace segmentation.

## What is borrowed from self-labeling research

- [FixMatch](https://arxiv.org/abs/2001.07685): high-confidence teacher selections
  and student training under a different view/perturbation.
- [Soft Teacher](https://arxiv.org/abs/2106.09018): unreliable pseudo supervision
  should not have the same strength as original supervision.
- [Confirmation-bias analysis](https://arxiv.org/abs/1908.02983): a model can
  reinforce its own incorrect labels; retaining anchors and independent checks
  matters more than repeatedly expanding a noisy pool.

These are design adaptations, NOT reproductions of those methods. There is no
online EMA teacher, RandAugment policy, full soft-target KL loss, or Soft Teacher
box-jittering implementation. The teacher is frozen offline; the student gets
mild shared photometric perturbation and reduced, agreement-weighted losses.
The current poor 1000-sample teachers are not made reliable by this code.

```text
original train-derived supervision + frozen dev
    -> scene-cross-fit segmentation/VLM teachers
    -> teacher predictions on unseen TRAIN-pool scenes
    -> generated JSON agrees with frozen shadow-score decision
    -> observed-ground + pixel + temporal teacher agreement
    -> diversity/budget limits + provenance checks
    -> original anchors plus one limited pseudo-label round
    -> student training; unchanged dev; independent final eval stays frozen
```

## Gates and honest boundaries

1. The dev score-gate policy requires precision >= 0.9, recall >= 0.2, at least
   20 correctly accepted dev positives across at least 3 scenes by default.
   These are initial experimental gates, not universally valid safety levels.
   In particular, nine dev positives cannot pass the default gate. Do not lower
   requirements merely to make a failed teacher start self-training.
2. If dev labels are weak, the gate measures weak-label agreement, NOT real
   shadow precision. Building a pilot then requires --allow-weak-pilot. No policy
   grants production approval. Final multi-evidence selection precision is not
   implied by the score gate's precision.
3. Both the segmenter and VLM must exclude candidate scenes from training AND
   checkpoint-selection dev. Initialization history is checked/propagated too.
   A full-train nuScenes checkpoint cannot initialize a supposedly unseen-fold
   teacher. Student initialization may use a seed trained on its own anchors.
4. Score and generation artifacts must have identical candidate IDs, teacher
   fingerprints, image-content hashes and proposal-mask hashes. Probability,
   source image, target and observed-ground hashes are checked before reuse.
   Older artifacts without provenance must be re-inferred, not patched by hand.
5. Only unlabeled TRAIN-pool regions are eligible. Existing labeled rows are not
   overwritten. Pixel supervision fills only original 255 pixels on observed
   ground. Known masks remain unchanged; new false-negative corrections of
   existing 0/1 pixels are intentionally outside this pilot.
6. Default expansion is positive-only. Optional negative pseudo labels need a
   separate dev negative-prediction gate; no GT match is never treated as empty.
7. Static consistency permits unchanged shadows. Moving positives require
   caster-aligned teacher probabilities to agree and outperform world alignment.
   This is evidence, not proof: stable dark surfaces and correlated teachers
   can still fool the pipeline. Missing temporal predictions mean rejection.
8. Per-class/total additions are capped at half the original anchor counts;
   defaults retain at most one per track/class, one per image/class and ten per
   scene. Sorting uses a confidence proxy, not calibrated correctness probability.
9. Pixel and temporal agreement modulate a capped pseudo-loss factor <= 0.25,
   on top of the ordinary weak-source factor. Class weights are fitted on the
   ORIGINAL anchors only, never on an increasingly biased pseudo distribution.
10. Reliable caster supervision is retained only when teacher localization agrees
    with existing GT-constrained association evidence. Otherwise only shadow
    tokens are supervised; the placeholder unresolved value is NOT trained as
    a refusal. Zero outside_bev seed support cannot be bootstrapped into a claimed
    trained outside state.
11. Only one pilot round is supported. Known self-trained teachers or recursive
    anchor/pool metadata are rejected. Legacy checkpoints lacking round history
    are explicitly assumed to be seeds; verify this assumption yourself. These
    artifacts are provenance records, not cryptographic proof of training history.

## 1. Prepare genuinely cross-fit teachers

First use `build_shadow_crossfit_folds.py` as described in the decision/annotation
docs. Train each teacher independently on its fold train manifest and the fixed
train-derived dev. Do not use --allow-train-scenes to bypass a failed check.
The code below illustrates fold 0; it does not train teachers automatically.

```bash
cd /public/home/eng_zhihongli/project/Shadow
conda activate shadow_annotator
mkdir -p logs
BASE=outputs/shadow_annotation
FOLD=$BASE/crossfit/fold_0
RUN=outputs/shadow_self_training_fold0
MODEL=/path/to/Qwen3-VL-8B-Instruct
TEACHER=outputs/vlm_fold0/adapter
SEG_TEACHER=outputs/seg_fold0/best.pth
```

Use the existing decision-diagnostics workflow to score THIS teacher on dev and
fit its frozen calibration. Then create its pilot gate:

```bash
python scripts/make_shadow_self_training_policy.py \
  --calibration outputs/vlm_fold0_dev/analysis/calibration.json \
  --dev-scores outputs/vlm_fold0_dev/scores/scores.jsonl \
  --output "$RUN/self_training_policy.json"
```

If eligible_for_pilot=false, stop expansion and inspect blocking_reasons. The
policy creation command intentionally writes a blocked report instead of
pretending class balancing or more GPUs made the teacher usable.

## 2. Teacher inference on heldout training scenes

```bash
python scripts/evaluate_shadow_segmenter.py \
  --manifest "$FOLD/seg_heldout.jsonl" --checkpoint "$SEG_TEACHER" \
  --output-root "$RUN/segmentation"
python scripts/propose_shadow_annotation_regions.py \
  --seg-predictions "$RUN/segmentation/predictions.jsonl" \
  --output-root "$RUN/predicted_regions"
python scripts/build_shadow_annotation_dataset.py \
  --manifest "$RUN/predicted_regions/manifest.jsonl" \
  --output-root "$RUN/proposal_views" --seed 0
python scripts/build_joint_shadow_annotation_dataset.py \
  --region-data-root "$RUN/proposal_views" --output-root "$RUN/proposal_joint"
python scripts/score_shadow_annotation_decisions.py \
  --model "$MODEL" --load-in-4bit --adapter "$TEACHER" \
  --manifest "$RUN/proposal_joint/annotation_train.jsonl" \
  --output-root "$RUN/teacher_scores"
python scripts/evaluate_shadow_annotator.py \
  --model "$MODEL" --load-in-4bit --adapter "$TEACHER" \
  --manifest "$RUN/proposal_joint/annotation_train.jsonl" \
  --output-root "$RUN/teacher_annotations"
```

Use the SAME original split seed when rebuilding proposal views. New proposal
answers remain unlabeled; confidence does not automatically turn them into GT.
Keep source images/masks immutable. Do not use eval scenes as this pool.

## 3. CPU dry-run, then optional limited build

```bash
export SHADOW_ANCHOR_ROOT="$BASE"
export SHADOW_SELF_ROOT="$RUN"
sbatch slurm/build_shadow_self_training.slurm
```

Default output is `$RUN/student_dry-run/selection_audit.jsonl` and a summary.
No training manifests are written in dry-run, even when many regions satisfy
individual rules. Inspect EVERY rejection stage and independent track coverage.
Do not interpret zero accepted regions as a reason to remove all gates.

Only after a passing gate and audit, for a weak-label experiment:

```bash
SHADOW_SELF_MODE=build SHADOW_ALLOW_WEAK_PILOT=1 \
  sbatch slurm/build_shadow_self_training.slurm
```

Outputs under `$RUN/student_build/`:

- annotation_train.jsonl: unchanged original annotation anchors plus pseudo rows.
- annotation_dev.jsonl: original dev rows, unchanged; no eval manifest imported.
- seg_train.jsonl: original segmenter anchors plus sparse new-pixel trimaps.
- seg_dev.jsonl: original dev rows, unchanged.
- pseudo_masks/: only accepted originally unknown pixels; other pixels are 255.
- self_training_protocol.json: sources, selection counts, budgets and assumptions.

The selector uses an 8-parent LRU cache and sparse accepted pixel indices. It
does not retain all full-resolution probability/mask volumes in memory. New
segmenter rows do not reuse temporal caches tied to OLD labeled pixels; anchor
temporal training remains, while new rows have teacher-temporal selection.

```bash
python scripts/check_joint_shadow_annotation_data.py \
  --data-root "$RUN/student_build" --training-only
```

This is a single-fold pilot. For a later multi-fold study, do not naively
concatenate outputs containing repeated anchors or bypass the global pseudo
budget. A separately audited deduplicated merger would be needed.

## 4. Train students without dropping original supervision

```bash
SHADOW_ANNOTATION_DATA_ROOT="$RUN/student_build" \
SHADOW_INIT_ADAPTER="$TEACHER" \
SHADOW_VLM_ROOT="$RUN/vlm_student" \
SHADOW_VLM_MODEL="$MODEL" \
  sbatch slurm/train_shadow_vlm.slurm

SHADOW_DATA_ROOT="$RUN/student_build" \
SHADOW_SEG_INIT_CHECKPOINT="$SEG_TEACHER" \
SHADOW_SEG_ROOT="$RUN/seg_student" \
  sbatch slurm/train_shadow_segmenter.slurm
```

The VLM receives shared mild exposure/contrast changes only for pseudo examples;
pixel masks and GT box coordinates are not geometrically transformed. The
segmenter keeps its flip augmentation and applies shared mild exposure changes
to pseudo views. Its supervised and temporal objectives both respect example
weights; temporal downweighting cannot cancel at batch size one.

An initial adapter's rank/config is retained when warm-starting; --lora-rank
does not resize an existing adapter. Re-measure GPU memory for these paths.

Compare student versus seed on identical unchanged dev observations, including
PR-AUC, fixed-FPR recall, abstention coverage and association performance. Do
not retune on eval or declare success because student training loss decreased.

## HPC checks and limitations

```bash
python -m compileall shadow_occ/annotation scripts
python -m pytest tests/test_shadow_self_training.py tests/test_shadow_decision_diagnostics.py tests/test_shadow_supervision_policy.py tests/test_shadow_annotation_pipeline.py tests/test_shadow_annotation_temporal.py -q
```

Run a small real-model prefix/QLoRA/student smoke test before allocating a full
job. This pilot does not fix missing true shadow GT, create independent positives
from duplicates, complete outside-BEV training coverage, merge full OCC geometry
targets, or establish that shadow cues improve occupancy.
