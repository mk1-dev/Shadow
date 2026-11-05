# Receiver-Anchored Track-Contrast Shadow Annotation

## Scope

This is a proposed weak-supervision method, not a demonstrated improvement.
No manual nuScenes shadow masks, no Occ3D, no OCC training, no new flow network.
Keep both the pixel segmenter and the unified VLM annotator. Existing HPC jobs
and outputs are not updated in place; rebuild into a fresh v2 output root.

## BEV-Flow Redesign

The learned encoder is the existing UniAD/OpenDriveVLA BEVFormer, not Shadow's
older local BEVFormer-style model. The inspected local implementation is
`E:/VLA_baseline/mulit_VLA/OpenDriveVLAre`:

- `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py:get_bevs` outputs
  `[H*W, B, C]`; the base config is 200 x 200 with 256 channels.
- `projects/mmdet3d_plugin/uniad/modules/encoder.py:get_reference_points`
  confirms rows follow y and columns follow x. The configured metric range
  belongs to LIDAR_TOP, not an interchangeable ego frame.
- `scripts/export_uniad_pth.py` supplies the existing model construction,
  preprocessing and inference helpers. Its compact VLM cache omits spatial
  BEV; `--full-results` retains `result_track.bev_embed`.

`scripts/export_mature_bev.py` calls those helpers in the mature environment,
replays needed scenes in timestamp order with the existing test pipeline,
and saves only the requested samples' spatial BEV. Earlier frames in each
needed scene initialize temporal memory. `--ann-file` selects train/val infos
without switching to the training/augmentation path. The two external projects
are not modified. A MACR checkout can be added through its actual encoder
interface once its local directory is identified; it is not a second
dependency for this implementation.

### Common Representation

1. Export all-class box volumes, camera-time 3D vehicle tracks, ground support
   and calibration. A fixed first-sample ego frame defines the metric grid.
2. Sample the SAME world ground points in three images. Compose both affine
   exposure corrections into the first frame's Lab-L reference.
3. Store expected geometric flow and measured dense flow at each receiver;
   their units remain source-image pixels per actual camera interval.
4. Store box footprints/height envelopes, visibility, receiver support,
   brightness onset/recovery, static dark regions and flow registration support.
5. Transform each common world point into each cached BEV's LIDAR_TOP frame
   before sampling features. Save three-frame receiver features and temporal
   cosine similarity. Feature variation is recorded, not used as a replacement
   for the measured lighting transitions.
6. Extract components in metric BEV. Match all three caster-box shadow
   footprints under the same light in that BEV. Keep the existing real-time
   versus frozen/permuted trajectory comparisons and factorized decisions.
7. Project the observed BEV component back to the supported middle camera.
   Export image-space trimaps to the existing segmenter/VLM data builders.
   Each candidate also stores BEV area, feature means, and its leading caster
   footprint hypotheses for inspection and later feature fusion.

The `explanation_state` map is 0=unknown, 1=stable receiver observation,
2=physical box support, 3=receiver illumination change. Candidate formation
uses illumination and receiver evidence, not a large-flow-residual threshold.
Static dark components are retained for audit; the moving-caster acceptance
stage still needs observed onset/recovery. The evidence includes all supported
receivers before any shadow component is extracted.

### HPC Sequence

No local model execution is required. Run the tests and data/model jobs on HPC.
Use an unused output directory for `ROOT`.

```bash
cd /public/home/eng_zhihongli/project/Shadow_update
conda activate shadow_occ
python -m pytest -q tests/test_bev_flow.py tests/test_cached_bev.py \
  tests/test_physics_shadow_motion.py tests/test_physics_shadow_labels.py

ROOT="$PWD/outputs/bev_flow_v1"
DATA=/public/home/eng_zhihongli/project/data/nuscenes-full
python -u scripts/export_nuscenes_motion_pairs.py \
  --dataroot "$DATA" --split train --cams CAM_FRONT \
  --max-pairs 120 --max-attempts 600 --output-root "$ROOT/pairs"
python -u scripts/analyze_shadow_motion_pairs.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --flow-backend dis --device cpu \
  --output-root "$ROOT/flow"
```

Use the already working OpenDriveVLA environment and its actual cloud paths:

```bash
export UNIAD_ROOT=/path/to/OpenDriveVLAre
export UNIAD_CHECKPOINT=/path/to/uniad_base_track_map.pth
export UNIAD_ANN_FILE=/path/to/nuscenes_infos_temporal_train.pkl
conda activate YOUR_EXISTING_UNIAD_ENV
python -u scripts/export_mature_bev.py \
  --external-root "$UNIAD_ROOT" --checkpoint "$UNIAD_CHECKPOINT" \
  --ann-file "$UNIAD_ANN_FILE" --dataroot "$DATA" \
  --pairs "$ROOT/pairs/pairs.jsonl" --output-root "$ROOT/mature_bev"

conda activate shadow_occ
python -u scripts/analyze_bev_flow_discrepancy.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --analysis-root "$ROOT/flow" \
  --feature-root "$ROOT/mature_bev" --output-root "$ROOT/bev_flow"
python -u scripts/build_physics_shadow_labels.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --analysis-root "$ROOT/flow" \
  --bev-flow-root "$ROOT/bev_flow" --decision-policy factorized-v3 \
  --output-root "$ROOT/labels_audit"
```

The export stage needs one GPU; optical-flow DIS, physical BEV construction,
feature-cache sampling and label matching run on CPU. A Slurm version of the
second block is `slurm/export_mature_shadow_bev.slurm`, configured through
`PHYSICS_ROOT`, `UNIAD_ROOT`, `UNIAD_CHECKPOINT`, `UNIAD_ANN_FILE` and
`UNIAD_CONDA_ENV`. `slurm/build_shadow_physics_labels.slurm` prepares a fresh
CPU-only physical-BEV audit without requiring the feature cache first.

Existing full-results caches can instead be converted using
`scripts/import_uniad_bev_cache.py --external-root ... --cache-root ... --output-root ... --pairs ...
--dataroot ... --model-config ... --checkpoint ...` in the mature environment.

### Outputs And Comparisons

- `mature_bev/<sample>.npz/json`: spatial features, range, LiDAR-to-world pose,
  timestamp, model/config identity.
- `bev_flow/windows.jsonl` and `windows/*.npz`: common receivers, 3D motion,
  geometric/observed flow, reliability, three-frame brightness, illumination
  transitions, learned receiver features and explanations.
- `bev_flow/review/*.jpg`: aligned three-frame BEV brightness, changes, states.
- `labels_audit/bev_regions/*.npz`: individual BEV components and feature means.
- `labels_audit/decisions.jsonl`: light/caster matches, motion controls and links
  to the corresponding BEV evidence; image masks remain in camera coordinates.

For the image-space ablation, run the same label command without
`--bev-flow-root` and choose another output directory. For the physical-BEV-only
ablation, omit `--feature-root` from the BEV analyzer. These retain the same
frames, tracks and flow. Review labels before using the existing explicit
`--export-weak-labels` option and the segmentation/joint-annotator builders.
OCC experiments and injecting shadow features into the mature model are the
next stage; this update covers BEV-based observation and annotation.

## Recovery Update: 2026-11-06

The reported fresh run completed execution but failed quality checks: segmenter
TP=FP=0 in all ten epochs, VLM positives 116/5741, and zero accepted physical
labels in 15 windows. These are three separate failures, not evidence that
fresh data or small total loss implies a working annotator. Cloud artifacts
were not accessible from this Windows workspace. Local HEAD was `1f1621e`, not
the reported HPC `8d65711`; merge these edits with server-only commits before
running, rather than overwriting the server tree.

### Segmenter

- `--pixel-balance loss` estimates foreground/background risk from effective
  known-class mass AFTER the actual resize, including weak/sample weights.
  Inverse-frequency loss weights are capped at 50. There is no class
  oversampling. `--pixel-balance none` is the unbalanced-loss ablation.
- Foreground Dice defaults to 0.5 and is applied to examples with known positive
  pixels; negative-only examples retain BCE. Original/resized pixel counts and
  actual weights are printed and recorded. Both dev classes must survive resize.
- All-background, all-foreground, zero-TP and zero-TN dev models cannot become
  `best.pth`. If every epoch fails, keep diagnostic `last.pth`, write
  `readiness.json`, and fail the job rather than export a misleading best model.
  Available GT sources take precedence over weak sources in this check.
- Save per-source positive/negative probability histograms and a fixed
  quantized threshold sweep on dev. This distinguishes subthreshold score
  separation from representation failure; a sweep does not automatically
  lower the threshold or approve a checkpoint.
- Prediction rejects collapsed/unvalidated checkpoints unless explicitly using
  `--diagnostics-only`. Those outputs, and old predictions lacking readiness
  metadata, are blocked from proposals, refinement and self-training builds.
  Non-collapse remains only a necessary check, NOT quality approval.
- Optional `--receiver-pair-weight 0.1` adds relative ordering at observed onset
  and recovery pixels inside positive cores. Registered previous/next views
  and evidence are hash-checked. The weak objective prefers higher middle
  shadow logits than the appropriate endpoint, using a margin of 0.5. It does
  NOT impose mask invariance or label either endpoint absolutely non-shadow.
  The default weight is zero; compare zero versus 0.1 on identical data first.
  Additional earlier/later forwards increase memory; start the paired pilot
  with per-GPU batch size 1 and adjust accumulation instead of assuming the
  old two-view memory budget still applies.

### Unified VLM

- Source/reliability weights now multiply ALL component losses. Class balance
  multiplies decision tokens only; association/format tokens do not inherit
  minority-class amplification. Previously the combined source/class multiplier
  reached only the decision term. New and old total losses are NOT comparable.
- Slurm defaults use one loss-weighting mechanism: inverse effective class mass
  (`--balance-power 1 --max-class-weight 50`), not oversampling plus weighting.
  Print raw counts, expected sampling fractions and effective decision-weight
  fractions. A cap can prevent exact balance; inspect actual fractions.
- Choose component checkpoints by `eval_macro_decision_loss`, the equally
  weighted mean of per-class decision-token NLL, not micro total JSON loss.
  Distributed padding is deduplicated by example ID. This still measures
  teacher-forced agreement, NOT generated balanced accuracy or calibration.
- Keep the frozen-adapter independent dev confusion/score diagnostic. No adapter
  becomes automatically approved because macro NLL improves. Inverse weighting
  can also increase false positives, as the earlier run demonstrated.
- Both teachers preserve initialization scene history so warm starts cannot
  hide prior exposure to current dev or pseudo-label-pool scenes. Training
  protocols also record critical source-code hashes.

### Experimental Factorized Physics Policy

`--decision-policy strict-v2` preserves the original gate. The explicit
`--decision-policy factorized-v3` separates existence from association:

1. Receiver visibility, registration, exposure and observed onset/recovery
   remain requirements. Invalid observations never become negatives.
2. An observed component with an adequate physical projection fit can supply
   experimental weak shadow evidence without other-track light votes or a
   unique trajectory explanation. This is not a calibrated probability.
3. Real-versus-control contrast, ownership competition and independent shared
   light support still constrain a UNIQUE caster association. Missing votes
   leave that field unresolved instead of vetoing existence.
4. Retain near-best explanations and a normalized light-weight table. Positive
   cores still intersect all near-best plausible volumes; no box polygon
   becomes a target. The exporter reports receiver/component/absolute-fit/
   contrast/accepted-shadow/unique-association counts separately.

The CPU Slurm pilot now defaults to factorized-v3, audit-only, in a fresh
`outputs/physics_factorized_v3_JOBID` root. The Python builder still defaults to
strict-v2 for reproducible old calls. Explicit weak export must repeat the
chosen policy. Preserve the old zero-acceptance audit; do not overwrite it.

**Negative supervision remains an evidence problem.** A brighter endpoint can
still be in a weaker shadow, so Lit-Shadow-Lit cannot automatically mean 0-1-0.
This update exports relative transition evidence, NOT new hard negative pixels.
Independent lit/non-shadow evidence or declared public/synthetic supervision
is still needed for complete binary training. Class balance cannot repair
incorrect NO labels, missing true shadows or only one independent positive track.

### Relation To The Baselines

Following the noisy-label concern in SILT, keep uncertain pixels ignored and
block unverified teacher feedback. Following cast/attached distinctions, keep
the receiver-ground cast-shadow target separate from vehicle self-shadow.
ViShadow and MetaShadow remain external comparison references, not implementations
silently incorporated here. No full SILT, MetaShadow, ViShadow or
ShadowDetectionAttached model was downloaded or reproduced in this update.

### HPC Check And Retry Order

Run on HPC, not the local Windows workspace:

```bash
python -m compileall -q shadow_occ/annotation scripts
python -m pytest -q tests/test_shadow_training_recovery.py tests/test_shadow_annotation_temporal.py
python -m pytest -q tests/test_physics_shadow_motion.py tests/test_physics_shadow_labels.py
python -m pytest -q tests/test_shadow_annotation_pipeline.py tests/test_shadow_self_training.py tests/test_shadow_decision_diagnostics.py
```

First wait for the frozen fresh adapter diagnostic (reported job 366419); do not
replace that adapter or its evaluation outputs. Retry the segmenter on the SAME
fresh manifests in a new output root with balanced loss and receiver-pair weight
zero. Use a compatible audited segmenter initialization if available; external
SILT/PVT weights do NOT load directly into the MobileNet architecture.

Run a tiny QLoRA smoke with `--max-steps 2 --accumulation 1` before the full run,
including at least an evaluation step/epoch to exercise macro checkpoint
selection. Retain the original train/dev split, use the revised VLM Slurm
script, and inspect its balance audit before a new full training job. Eval must
remain frozen while changing dev operating points.

For physics, compare policies on EXACTLY the same pairs/flow archives:

```bash
ROOT=outputs/physics_factorized_v3_JOBID
python -u scripts/build_physics_shadow_labels.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --analysis-root "$ROOT/flow" \
  --decision-policy strict-v2 --output-root "$ROOT/strict_audit"
python -u scripts/build_physics_shadow_labels.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --analysis-root "$ROOT/flow" \
  --decision-policy factorized-v3 --output-root "$ROOT/factorized_audit"
```

Only after reviewing stage counts and rejection cases, opt into a factorized
weak-label pilot with `--decision-policy factorized-v3 --export-weak-labels` in
another fresh root. Build shared data with the existing builder, including
independently justified negative manifests. V3 data records `physics_transition`;
old fresh data without that field cannot enable the nonzero receiver-pair loss
(the trainer now rejects that configuration rather than silently ignoring it).
Check the active-transition counts in training history before claiming it ran.

Successful training, non-collapse and agreement with weak labels do not prove
real nuScenes accuracy. No automatic self-training or OCC integration is enabled.

## What Changed

1. **Receiver registration, not shadow flow.** Geometric ground correspondence
   fixes the same physical receiver across frames. DIS or RAFT-small provides
   forward/backward consistency. Only textured, geometry-consistent ground
   pixels become flow anchors. A radius-7 neighbourhood requires at least 16
   anchors and 60% agreement among observable textured pixels. Votes cannot
   cross disconnected supported-ground components. Record direct anchors,
   neighbourhood counts and support fractions in the evidence archive.
2. **Separate illumination observations.** Exposure-compensated Lab-L onset,
   persistence and recovery are computed using geometric receiver coordinates,
   not optical-flow endpoints. A changing shadow boundary need not have reliable
   material flow. No flow residual by itself is a positive shadow label. The
   new local flow support validates registration, not physical shadow truth.
3. **Track contrast under one light.** For each light and caster, score observed
   middle coverage, newly covered onset and newly uncovered recovery:
   `S = 0.4 * cover + 0.3 * onset_fit + 0.3 * recovery_fit`.
   Reuse the same three projected box volumes and SAME light for five nonidentity
   time permutations and a frozen-middle control. Require
   `S_real - max(S_controls) >= 0.1` as well as absolute gates. This is a
   temporal-explanation test, not a physically rendered intervention.
4. **Shared-light support without self-voting.** Require two other distinct
   tracks in the same official split/scene, excluding every plausible owner
   and every query camera frame and nuScenes sample. Different cameras at the
   same physical sample do not count as independent query-excluding evidence.
   This remains internal consistency, not an independent correctness test.
5. **Decouple existence and ownership.** A shadow may pass while caster identity
   remains ambiguous. Emit YES with unresolved ownership; only a unique weak
   association supplies a caster target. Positive pixels are the observed
   component intersected with ALL near-best explanatory volumes. Never fill
   those volumes as masks. Insufficient common support becomes IGNORE.
6. **Remove the crop oracle.** Default wider VLM context is a fixed enlargement
   of the proposal bounds. GT caster boxes cannot expand it. If the target
   caster falls outside this context, the visible-association target abstains.
   `--context-policy gt-assisted` preserves the old setting as an explicit
   diagnostic ablation, not the main comparison.

Shadow probabilities are uncalibrated rule scores. Failure or low flow support
means UNKNOWN, never NO. Static shadows are retained as unknown observations,
not accepted by this moving-track gate. When the three states are temporally
indistinguishable (for example disjoint footprints fit both time orders), the
contrast gate correctly has no evidence to choose the real order.

## Relation To Prior Work

| Prior work | Reused idea | Proposed difference, requiring evidence |
|---|---|---|
| [Temporal Feature Warping for Video Shadow Detection](https://arxiv.org/abs/2107.14287) | Optical-flow temporal correspondence | Flow validates receiver registration; labels come from light transitions tested against real/wrong-time/frozen 3D tracks, rather than claiming new feature warping |
| [Instance Shadow Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Instance_Shadow_Detection_CVPR_2020_paper.pdf) | Distinct shadow instances and caster associations | No manually paired nuScenes shadow instances; separate weak existence/ownership decisions and abstention |
| [RenDetNet](https://arxiv.org/abs/2408.17143) | Caster verification through scene physics | Box/plane bounds and observed temporal contrast in real driving data, NOT caster removal or differentiable re-rendering |
| [Automatic Training Supervision from 3D geometry](https://openaccess.thecvf.com/content/ICCV2021W/WAAMI/papers/Ufuktepe_Learning-Based_Shadow_Detection_in_Aerial_Imagery_Using_Automatic_Training_Supervision_ICCVW_2021_paper.pdf) | 3D geometry can supply weak shadow supervision | Preserve observed image support and test temporal competing explanations instead of treating a projected geometric footprint as a true mask |

These references motivate components; they do not establish novelty priority.
The candidate contribution is the receiver-anchored, track-contrast annotation
protocol with ownership abstention, not DIS/RAFT, GT projection, or a VLM alone.

## Baseline Reference Set

Reference check: 2026-11-06. This is the comparison specification, NOT a claim
that the external models have been installed, adapted or evaluated in this repo.

| Method | Verified task and supervision | Role in our comparison |
|---|---|---|
| [ShadowDetectionAttached](https://github.com/cvlab-stonybrook/ShadowDetectionAttached), paper: [Cast and Attached Shadow Detection via Iterative Light and Geometry Reasoning](https://arxiv.org/abs/2512.06179) | Author repository reports ECCV 2026 acceptance. Predicts cast/attached shadows with iterative light and geometry reasoning; trained using manually annotated shadow data. | Physics-informed external detector. Compare CAST output on receiver ground, not attached shadows or the union of both types. |
| [ViShadow](https://arxiv.org/abs/2211.12827), [author code](https://github.com/HarryHsing/Video-Instance-Shadow-Detection) | Semi-supervised video instance shadow detection, pairing and tracking; labeled images plus unlabeled video. | Main temporal/caster-association reference. Distinguish learned image/video associations from our GT-track-constrained weak annotation. |
| [MetaShadow](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_MetaShadow_Object-Centered_Shadow_Detection_Removal_and_Synthesis_CVPR_2025_paper.pdf), [method details](https://arxiv.org/html/2412.02635v1) | CVPR 2025 object-conditioned detection/removal/synthesis with supervised datasets. Shadow Analyzer takes RGB plus an object mask; an empty mask enables general-shadow detection. | Object-conditioned external reference, not an unconditioned detector with identical inputs. Benchmark detection only; do not silently credit removal/synthesis as our task. |
| [SILT](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_SILT_Shadow-Aware_Iterative_Label_Tuning_for_Learning_to_Detect_Shadows_ICCV_2023_paper.html) | ICCV 2023 noisy-label training/self-training framework with iterative label refinement, not just a backbone. | Segmentation/noisy-label reference. Separate original pretrained inference from a SILT-style adaptation using our frozen weak labels. |
| Ours: receiver-anchored 3D-to-2D track contrast | Existing 3D object GT and ground geometry, observed light transitions, competing time-order explanations, independent-track support; no manually labeled nuScenes shadows. | Annotation method under test. Its downstream segmenter is a separate learned model; direct weak trimaps and learned dense masks are different outputs. |

### Fair Comparison Rules

- **Separate three questions.** Detector transfer compares pretrained image/video
  predictions; annotation compares weak-label construction and abstention;
  downstream learning fixes the segmenter architecture, initialization and
  training budget while changing only label generation/refinement. Do not put
  all three into a single undifferentiated leaderboard.
- **Declare manual-label usage.** External shadow-supervised checkpoints remain
  supervised even when nuScenes has no manual shadow annotations. If ours uses
  those checkpoints, describe the setting as no manual *nuScenes* shadow labels,
  not no manual shadow supervision anywhere. Record source data and checkpoint
  hashes, and keep test datasets out of pretraining/fine-tuning when claiming
  held-out generalization.
- **Match inputs explicitly.** For MetaShadow, predicted object masks, oracle
  object masks and projected-box proxies are separate protocols. A nuScenes
  3D Box is NOT a pixel-accurate object mask. An empty-object-mask run is a
  general-shadow baseline, not the same object-conditioned task. No object mask
  should be derived from the target shadow answer.
- **Match evaluation support.** Use the same scene split, frames, receiver ROI,
  distances and resolution. Evaluate a fixed pool including rejected cases;
  report coverage/abstention separately so ours cannot win by evaluating only
  its own accepted regions. If geometry post-filtering is used on model outputs,
  report model-only and model-plus-common-filter variants separately.
- **Distinguish cast and attached.** Our receiver-ground target is cast shadow.
  Keep vehicle self-shadow as a separate diagnostic or UNKNOWN, never silently
  merge it with the cast-shadow target. For ShadowDetectionAttached, record the
  upstream commit and use its corrected `standard` evaluation protocol for new
  comparisons; `reported` is a separate historical-reproduction protocol.
- **Do not grade against the teacher.** A pretrained detector's output, our own
  trimaps or VLM answers cannot become independent shadow truth. On nuScenes
  without independent labels, report coverage and weak agreement honestly.
  BER/IoU/PR-AUC and association accuracy need an independently labeled benchmark
  or synthetic truth under the declared domain; such results do not establish
  nuScenes accuracy. SOAP-VID additionally needs instance/temporal association
  truth and compatible prediction formatting.

For the subsequent BEV/OCC study, keep the BEV encoder, feature adapter, fusion
locations and training conditions fixed; change only the declared evidence
source. Report inference-time inputs and prevent offline 3D-GT/future-frame
teacher evidence from silently entering an online camera-only evaluation.
Do not turn cast-shadow receiver pixels directly into occupied targets.

### What This Means For Novelty

Physical illumination reasoning, video tracking, caster association and noisy
label refinement already have strong precedents above. Our testable difference
is **using observed receiver illumination transitions to discriminate true-time
3D track explanations from frozen/wrong-time competitors, with separate shadow
and ownership abstention**. This is a candidate contribution, not proof of
priority or superiority. The frozen-control ablation and matched downstream
segmenter experiment are necessary to isolate its value.

Runtime availability, licensing, checkpoint compatibility and adapters still
need to be checked before launching these baselines on HPC. No external source
repository or checkpoint has been downloaded by this documentation update.

## HPC Validation

Run tests and the default audit-only CPU pilot:

```bash
mkdir -p logs
sbatch slurm/build_shadow_physics_labels.slurm
```

The script runs focused regression tests first. Then it exports raw nuScenes
pairs, rebuilds flow evidence and applies the explicit selected policy. For the
original v2 behavior submit with `SHADOW_PHYSICS_POLICY=strict-v2`. To run tests separately:

```bash
python -m pytest -q tests/test_physics_shadow_motion.py tests/test_physics_shadow_labels.py
python -m pytest -q tests/test_shadow_supervision_policy.py tests/test_joint_shadow_annotation.py
```

After inspecting the audit, export experimental weak labels into a NEW root:

```bash
ROOT=outputs/physics_factorized_v3_JOBID
python -u scripts/build_physics_shadow_labels.py \
  --pairs "$ROOT/pairs/pairs.jsonl" --analysis-root "$ROOT/flow" \
  --minimum-temporal-gain 0.1 --decision-policy strict-v2 --export-weak-labels \
  --output-root "$ROOT/labels_weak"

python scripts/build_shadow_annotation_dataset.py \
  --manifest "$ROOT/labels_weak/canonical.jsonl" \
  --context-policy proposal --seed 17 --dev-fraction 0.15 \
  --output-root "$ROOT/region_data"

python scripts/build_joint_shadow_annotation_dataset.py \
  --region-data-root "$ROOT/region_data" --output-root "$ROOT/joint_data"
```

Replace JOBID with the actual pilot directory. This builds data, not a ready
training set: physics exports are positive/IGNORE only. Supply independently
justified negative supervision through additional canonical `--manifest`
inputs before training. Existing class-coverage guards remain mandatory.
Do not use time-permuted controls as NO shadow images.

## Experiments And Stop Criteria

- Freeze scene-disjoint train/dev/eval before threshold selection. Real shadow
  accuracy stays unavailable without independent truth; held-out pseudo-label
  agreement is NOT real precision, recall, IoU or caster accuracy.
- On identical observed candidates, report absolute fit, real-minus-control
  scores, rejection reasons, distinct accepted tracks/scenes, accepted area and
  unresolved-owner coverage. Inspect rejected cases too; a tiny accepted set
  is not evidence of general reliability.
- Ablate trajectory contrast with `--minimum-temporal-gain 0` in a NEW output
  root. The other absolute gates stay active. Keep seeds, pair manifest,
  light bank, observation masks and class coverage fixed.
- Compare DIS vs RAFT-small on identical pairs with explicitly provisioned
  weights, not by assuming a bigger flow model is better. Record geometry
  residuals, anchor coverage, low-texture rejection and transition support.
- Compare proposal-only vs GT-expanded context as a separate input ablation;
  do not mix it into the trajectory-contrast comparison.
- Test A=appearance, B=appearance+receiver temporal evidence, C=appearance+3D
  bounds, D=all+track contrast as a research ablation. A/B/C need separately
  defined exporters; the code does NOT pretend that the v2 CLI implements
  all four complete annotation methods.

## Remaining Limits

The 3D box is not the caster silhouette; light is hypothesized rather than
measured; the receiver is locally planar. Nonuniform illumination, wet roads,
box/ground errors, low-texture receivers and static/turning-in-place casters
can defeat the gates. VLM earlier/later views are offline inputs including
future context, not causal deployment evidence.

The v2 change only affected pseudo-label generation and VLM inputs. The recovery
update adds an optional relative receiver-order loss on its own metadata path;
legacy temporal consistency remains separate. A moving-shadow mask must not be
forced invariant in world-ground coordinates. No runtime success, real-label
quality, outside-BEV coverage or OCC gain is established until HPC verification.
