# Shadow annotator: decision diagnostics before expansion

Date: 2026-11-04. All commands below run on HPC, not the local Windows workspace.
No new training, tests or data/model evaluation were executed locally.

After a passing diagnosis, the optional
[controlled offline self-training pilot](shadow_self_training_hpc.md) provides
cross-fit pseudo-label selection and anchored student manifests. It is disabled
for a failed teacher and does not turn weak validation into real shadow GT.

The reported 1000-sample pilot collapses to NO without balance and has high weak
negative false positives with balance. Do not expand labels or retrain OCC from
that model yet. First distinguish a poor operating threshold from poor ranking.
GT-assisted contexts remain allowed for this annotation-training stage; they are
not evidence of autonomous GT-free inference. Segmentation is retained.

## 1. Audit what is actually trained

The current sampler does NOT oversample classes. `--balance-shadow` changes
training loss weights only. The new default exponent is 0.5 (tempered inverse
frequency), capped at 5. These are experiment settings, not validated choices.
`--balance-power 1 --max-class-weight 10` reproduces the old full class-weight
formula, but the new dev loss is deliberately not class-weighted.

Training writes `balance_audit.json` and `loss_audit.json`, and prints class
counts, effective weights and observed per-class losses. The latter is rank-0
cumulative exposure, not a distributed estimate of all workers' samples. One GPU
is the intended VLM configuration. Unknown metadata is never guessed into GT.

```bash
cd /public/home/eng_zhihongli/project/Shadow
conda activate shadow_annotator
mkdir -p logs
DATA=outputs/shadow_annotation/joint
MODEL=/path/to/Qwen3-VL-8B-Instruct
ADAPTER=outputs/shadow_annotation/vlm_lora/adapter
OUT=outputs/shadow_diagnostics_v1

python scripts/audit_shadow_training_data.py \
  --train "$DATA/annotation_train.jsonl" \
  --dev "$DATA/annotation_dev.jsonl" \
  --eval "$DATA/annotation_eval.jsonl" \
  --output-root "$OUT/data_audit"
```

Inspect unique positive scenes/tracks, missing track IDs, repeated source images,
camera/range/illumination/ground coverage and caster-state counts. A large count
of adjacent-frame crops is not a large count of independent positives. Older
manifests without distance/crop metadata must be rebuilt for those diagnostics.

## 2. Score existing models before retraining

```bash
python scripts/score_shadow_annotation_decisions.py \
  --model "$MODEL" --load-in-4bit --adapter "$ADAPTER" \
  --manifest "$DATA/annotation_dev.jsonl" \
  --output-root "$OUT/dev_scores"
python scripts/analyze_shadow_decision_scores.py fit-dev \
  --scores "$OUT/dev_scores/scores.jsonl" \
  --max-fpr 0.1 --abstain-margin 0 --max-uncertainty 1 \
  --output-root "$OUT/dev_analysis"
```

All three alternatives share the same multimodal prompt and token prefix. Only
the shadow-value tokens (including their closing quote boundary token) contribute
to likelihood, not caster category, box coordinates, remaining JSON or EOS.
Hypothetical complete JSON answers preserve the trained token boundary. Prefix
or offset mismatches fail rather than silently score different contexts.

The binary score normalizes YES/NO likelihoods; a separate three-way relative
UNCERTAIN score is available. Neither is a calibrated real-world probability.
`--abstain-margin`/`--max-uncertainty` define the fixed rejection rule.

Outputs contain AP (step-interpolated PR-AUC, explicitly named average precision),
AUROC, weak precision, recall, FPR, coverage, selective accuracy and scene-cluster
intervals. Abstentions remain in class recall denominators, so rejecting all
samples cannot earn perfect balanced accuracy. Small-positive-count intervals
are unstable; a single scene does not produce a meaningful cluster interval.

`calibration.json` is fitted only on train-derived dev. Eval scenes, mixed model
scores and training-scene scoring are rejected. The policy is bound to adapter
weights, model path/config, quantization, image-token budget and view mode.
Changing those requires a new dev calibration. Base-model shard weights are not
hashed: keep the base-model directory immutable/versioned between dev and eval.

The initial FPR limit 0.1 is a diagnostic constraint, not a paper acceptance
criterion. A feasible all-NO policy still has no useful positive recall. Every
policy starts `approved_for_relabeling=false`; fitting a threshold never grants
automatic approval to create training labels.

## 3. Review false positives/negatives without turning predictions into GT

```bash
python scripts/audit_shadow_decisions.py \
  --predictions "$OUT/dev_analysis/predictions.jsonl" \
  --output-root "$OUT/dev_review" --max-images 100
```

The review queue and PNGs separate false positives, false negatives, and
abstentions/parse failures. Optional `--reviews reviewed.jsonl` accepts only IDs,
notes and fixed categories: dark_pavement, vehicle_underbody, wet_ground,
vegetation, suspected_missing_shadow_label, other, unreviewed. No category is
automatically assigned by the same VLM, no mask is changed, and no review is
claimed to be shadow GT. When using generation results, supply `--manifest` if
they do not include their input image paths.

## 4. Paired input ablations

```bash
for MODE in full region context occlude-region; do
  python scripts/make_shadow_input_ablation.py \
    --manifest "$DATA/annotation_dev.jsonl" --mode "$MODE" \
    --output-root "$OUT/views_$MODE"
  python scripts/score_shadow_annotation_decisions.py \
    --model "$MODEL" --load-in-4bit --adapter "$ADAPTER" \
    --manifest "$OUT/views_$MODE/manifest.jsonl" \
    --output-root "$OUT/scores_$MODE"
  python scripts/analyze_shadow_decision_scores.py fit-dev \
    --scores "$OUT/scores_$MODE/scores.jsonl" \
    --max-fpr 0.1 --output-root "$OUT/analysis_$MODE"
done
```

- full: unchanged clean/reference/context/history slots.
- region: retain clean/reference; gray out context AND history.
- context: retain current context; gray out history only.
- occlude-region: mask the projected candidate in ALL supplied views, including
  its reference outline. Requires exact source crop coordinates; legacy data
  without them is rejected, not approximated by a rectangle.

Prompts and targets are unchanged. These are shadow-score diagnostics, not
caster-box evaluation: gray context has no valid visible-box localization task.
Only examples with available history can test temporal contribution. Gray
masking is out-of-distribution and is not proof of causality. Candidate masks
are still pseudo masks, not independent true-shadow masks.

## 5. Small loss ablations, same data/seed/epochs

The unified SFT task now supports `--loss-mode component` (default): average
decision, association and format losses separately, then combine them. Decision
weighting cannot multiply the entire association/box response. UNCERTAIN targets
do not force an association loss. `sequence` retains the old whole-answer loss
for a controlled baseline. Weak provenance still downweights examples.

Suggested 2x2: sequence/component x no balance/tempered balance. Do not also add
oversampling. Keep adapter rank, image budget, split, learning rate and seed fixed.

```bash
export SHADOW_VLM_MODEL="$MODEL"
export SHADOW_ANNOTATION_DATA_ROOT="$DATA"
SHADOW_LOSS_MODE=sequence SHADOW_BALANCE=0 SHADOW_VLM_ROOT=outputs/diag_seq_raw \
  sbatch slurm/train_shadow_vlm.slurm
SHADOW_LOSS_MODE=sequence SHADOW_BALANCE=1 SHADOW_VLM_ROOT=outputs/diag_seq_bal \
  sbatch slurm/train_shadow_vlm.slurm
SHADOW_LOSS_MODE=component SHADOW_BALANCE=0 SHADOW_VLM_ROOT=outputs/diag_comp_raw \
  sbatch slurm/train_shadow_vlm.slurm
SHADOW_LOSS_MODE=component SHADOW_BALANCE=1 SHADOW_VLM_ROOT=outputs/diag_comp_bal \
  sbatch slurm/train_shadow_vlm.slurm
```

Use saved checkpoint directories with the score script, then select checkpoint
and threshold from identical dev observations rather than overall generation
loss alone:

```bash
python scripts/select_shadow_checkpoint.py \
  --scores outputs/ckpt_a_dev/scores.jsonl outputs/ckpt_b_dev/scores.jsonl \
  --max-fpr 0.1 --output-root "$OUT/selected"
```

The helper selects a policy and scorer ID; it does not silently copy adapters or
launch training. Generation dev loss remains a convenient training checkpoint
signal, not the final scientific selection criterion. New loss/scoring paths
need a GPU smoke run and fresh peak-memory measurement; prior 13.12 GiB does not
certify their memory usage.

## 6. Frozen eval, after dev diagnosis

Score the selected adapter on eval, then apply its frozen calibration:

```bash
python scripts/score_shadow_annotation_decisions.py \
  --model "$MODEL" --load-in-4bit --adapter "$ADAPTER" \
  --manifest "$DATA/annotation_eval.jsonl" --output-root "$OUT/eval_scores"
python scripts/analyze_shadow_decision_scores.py evaluate \
  --scores "$OUT/eval_scores/scores.jsonl" \
  --calibration "$OUT/dev_analysis/calibration.json" \
  --output-root "$OUT/eval_analysis"
```

For checkpoint selection use `selected/calibration.json` and the corresponding
adapter, not the example ADAPTER variable accidentally left at another model.
Eval does not search thresholds and rejects dev-scene/ID overlap. Final eval
results must not feed back into training or threshold selection.

`slurm/diagnose_shadow_vlm.slurm` runs one-GPU dev scoring, fitting and review.
For eval set SHADOW_DIAG_SPLIT=eval and SHADOW_CALIBRATION to a frozen file. Use
SHADOW_VLM_ADAPTER to choose a checkpoint and a NEW SHADOW_DIAG_ROOT each run.

## 7. Temporal evidence and outside-BEV gates

Training-data audit reports temporal feature availability and AUROC against
existing targets. This is a biased weak audit, not independent confirmation.
`temporal_policy_template.json` starts unvalidated. Only after dev validation and
threshold freezing may `--temporal-policy` (or SHADOW_TEMPORAL_POLICY) be enabled.
The policy must attest validation and use disjoint train-derived dev scenes.
It can only reduce weights on contradictory moving-caster evidence; static small
differences and missing evidence are never penalized. It cannot generate YES,
override a mask or establish an association. The code cannot independently
verify the human/external validation attestation in the policy JSON.

Exterior observation/association exports now record stage funnel counts: sample
coverage, component area/boundary availability, camera projection visibility,
unique/multiple/no GT matches and retained trainable associations. Supply these
protocols using `audit_shadow_training_data.py --stage-protocols ...`.
Do not compare component counts with camera-component counts as one denominator.

Training protocols record caster-state support. When outside_bev has zero train
targets, generation evaluation reports its learned recall as null with an
unsupported-state warning. Older adapters lacking a support audit also receive
null rather than an unverified learned-state recall. Keep the interface, but do not claim it is trained.
One or a few eval examples do not establish that capability either.

## HPC validation checklist

```bash
python -m compileall shadow_occ/annotation scripts
python -m pytest tests/test_shadow_decision_diagnostics.py tests/test_joint_shadow_annotation.py tests/test_shadow_supervision_policy.py tests/test_shadow_annotation_pipeline.py tests/test_shadow_annotation_temporal.py -q
```

Then use a small train-derived dev manifest for real Qwen token-offset/prefix
checks and a short component-loss GPU run. Unit tests do not establish model
discrimination. No automated label expansion or OCC integration is enabled by
these diagnostic commands.
