# 3D-to-2D Physics-Guided Shadow Annotation

## Scope And Status

Current priority: shadow annotation, without a neural BEV encoder, OCC, Occ3D or
a shadow GT mask. A metric ground/BEV coordinate system IS used for geometry.
Existing segmentation/VLM/self-training code is retained, but this experimental
path deliberately does not feed its candidates into training automatically.
No training, tests or nuScenes processing for this update have been run locally;
the commands below are for HPC validation.

| Stage | This update | Not established |
|---|---|---|
| 1: object motion | camera-time GT interpolation; center/corner motion decomposition | dense object surface motion accuracy |
| 2: ground / flow / light | local LiDAR plane; sparse visibility; DIS or local-weight RAFT; affine exposure fit | true shadow precision / recall |
| 3: physical association | joint three-frame receiver evidence, light hypotheses, caster competition, independent light support | real association accuracy or complete static-shadow coverage |
| 4: annotator learning | explicit positive-only weak export; segmenter/VLM data bridge with provenance checks | sufficient negative supervision, trained/validated annotation quality |

Call the method **weak supervision without manual shadow annotations**, not
fully unsupervised learning: annotated 3D boxes and tracks are used.

## What Is Actually Measured

For the same tracked box landmark, with world positions X0 and X1:

```text
p0        = project(camera_t, X0)
p_static  = project(camera_next, X0)
p1        = project(camera_next, X1)

total apparent displacement = p1 - p0
camera contribution         = p_static - p0
object residual             = p1 - p_static
```

The residual is at the **object's depth**. Subtracting ground flow at a box
center would mix parallax and object motion. Box center/corners are landmarks,
not pixel flow labels. Their algebraic sum is a consistency check, not an
independent calibration-accuracy measurement.

Camera poses come from their own sample_data records. Object centers, size and
rotation are interpolated only within adjacent same-track annotation brackets.
No extrapolation: endpoints and missing links lose motion evidence. Reported
velocities use actual camera delta-time and are box-level estimates, not an
independent velocity sensor. The keyframe spacing is often challenging for flow;
this pilot does not fabricate high-frequency object GT for camera sweeps.

Ground is fitted to object-excluded local LiDAR lower-envelope cells, rather
than assuming ego z=0. A seeded robust plane fit rejects insufficient support,
excessive tilt or disagreement between adjacent fits. Support is limited to
LiDAR observations within 35 metres in each ego frame. Ground projection requires
positive plane-intersection depths in both cameras and in-image endpoints.
LiDAR/camera offsets above 0.1 seconds are rejected by default. Remaining sensor
offset and LiDAR scan motion are not fully compensated, and remain limitations.

Visibility is **sparse local planar support**, not semantic ground GT. Supports
are dilated by three working pixels, depth-tested, and exclude projections of
all annotated objects, not just candidate vehicles. Boxes straddling the camera
near plane are clipped at positive depth before projection, then intersected
with the image viewport (including the exclusion margin). Only their local
projected regions remove ground support; an off-screen box no longer invalidates
the whole view. `ground_evidence` records `objects_crossing_near_plane`,
`near_plane_objects_clipped` and `object_exclusion_pixels` for auditing. Invalid
nonfinite box geometry rejects the pair instead of being silently ignored.
Missing annotations, vegetation, road slopes and reflective surfaces can still
invalidate the approximation. No inferred dense ground beyond local support.

All Stage 2 maps live in image t coordinates. Ground correspondences sample the
next frame at the same receiver-surface locations. Forward/backward flow checks
and ground-flow residuals select background for a trimmed affine exposure fit.
The fit assumes that most selected background pixels have stable lighting;
spatially varying exposure or broad moving shadows can violate this assumption.
The fit is frame-pair affine, **not** a full spatial local-exposure model.

```text
delta_L = corrected_L_next_at_same_ground - L_t
negative: darkening/onset candidate
positive: brightening/recovery candidate
small delta + dark at both times: stable-dark candidate
```

Lab L uses [0,100]. Darkness ratios and chroma consistency support these maps.
Flow inconsistency alone is never positive shadow evidence; low confidence may
be texturelessness, occlusion or brightness-constancy failure. Candidate pixels
are not forced to follow flow, since shadow boundaries are illumination changes,
not material points. Small changes do not reject static shadows.

Onset and recovery here are **pairwise residual signs**, not proof that the
same moving caster caused a ground patch to darken and recover over three frames.
That temporal/caster hypothesis test is implemented separately in Stage 3. Stage 2 candidates retain
IGNORE/UNCERTAIN and no calibrated probability or accepted caster ID.

## 1. Check Environment And Tests On HPC

Use a CPU allocation for the DIS pilot. No model download, GPU, mmcv or Occ3D is
needed. Use an environment containing the repository's base requirements and
pytest. RAFT is optional and additionally requires compatible torch/torchvision.

```bash
cd /public/home/eng_zhihongli/project/Shadow
source /public/home/eng_zhihongli/anaconda3/etc/profile.d/conda.sh
conda activate shadow_occ
python -m pytest -q tests/test_physics_shadow_motion.py
```

Tests cover projection decomposition, depth/parallax, timestamp interpolation,
ground validity, flow consistency, residual signs, static retention, invalid
exposure, resize coordinates, diagnostic rendering and CPU DIS execution.
The suite is synthetic; passing it does not prove nuScenes shadow accuracy.

## 2. Export Twenty Pairs

Choose tuning scenes from official train only. Defaults cap the export at twenty
valid camera pairs, not twenty samples or scenes. At most 200 camera pairs are
attempted (`--max-attempts`); inspect skipped-pair reasons if fewer survive.
Use a fresh output directory.
First inspect one scene; then explicitly add independent scenes for coverage.

```bash
python -u scripts/export_nuscenes_motion_pairs.py \
  --dataroot /public/home/eng_zhihongli/project/data/nuscenes-full \
  --version v1.0-trainval \
  --split train \
  --scene scene-0001 \
  --cams CAM_FRONT \
  --max-pairs 20 \
  --working-width 800 \
  --output-root outputs/physics_motion_pilot/pairs
```

For nuScenes-mini change `--version v1.0-mini --split mini_train` and select a
scene belonging to that split, or omit `--scene`. Do not use official val scenes
to choose thresholds. The manifest keeps the official split and deterministic
train/dev role; a one-scene pilot is not a valid train/dev evaluation dataset.

Output:

- `pairs.jsonl`: paths, camera-time projections, plane basis, track landmarks.
- `images/`, `ground/`: resized clean images and sparse supported-ground masks.
- `export_audit.jsonl`: failed pairs and reasons, not silently converted to NO.
- `export_protocol.json`: thresholds, counts and pair-manifest hash.

The exporter checks next-sample scene identity, clock offsets and plane quality.
At endpoints, missing object interpolation is counted even when the pair itself
remains usable for ground diagnostics. An empty export exits with an error.

## 3. CPU Flow And Light Diagnostics

```bash
python -u scripts/analyze_shadow_motion_pairs.py \
  --pairs outputs/physics_motion_pilot/pairs/pairs.jsonl \
  --flow-backend dis \
  --device cpu \
  --review-limit 20 \
  --output-root outputs/physics_motion_pilot/dis
```

DIS is a reproducible CPU baseline, not a claim of the best flow quality. Compare
it with a frozen pretrained engine before interpreting flow failure as physics.

Optional RAFT-small, on an allocated GPU, with an explicit **torchvision
raft_small state dict** at a local path (not an arbitrary RAFT checkpoint):

```bash
python -u scripts/analyze_shadow_motion_pairs.py \
  --pairs outputs/physics_motion_pilot/pairs/pairs.jsonl \
  --flow-backend raft-small \
  --flow-weights /path/to/torchvision_raft_small.pth \
  --device cuda \
  --output-root outputs/physics_motion_pilot/raft
```

No automatic download or CPU fallback. Images must be at least 128x128; the
adapter normalizes RGB to [-1,1], pads to multiples of eight and crops the result
back without altering pixel scale. The weight-file hash is recorded.

## 4. Read The Results

- `review/*.jpg`: clean t / next, next on supported ground, signed light
  residual, flow-minus-ground residual, candidate maps and GT landmark arrows.
  These panels are diagnostics, not VLM inputs or targets.
- `pair_metrics.jsonl`: ground coverage, flow forward/backward agreement,
  exposure fit, candidate counts and failure reasons.
- `evidence/*.npz`: full forward/backward flow and evidence arrays in t pixels.
- `candidates.jsonl`: component references, all `IGNORE/UNCERTAIN`, no caster.
- `summary.json`: evidence counts; shadow precision/recall/IoU and caster
  accuracy are explicitly null, not agreement with the generated candidates.
- `analysis_protocol.json`: frozen configuration, input and weight hashes,
  coordinate/sign conventions and missing stages.

An onset component is dark in **next**, but its stored mask is in receiver
coordinates at **t**. Do not feed that mask to a next-frame segmenter without
explicitly transforming it. Stage 2 candidates are intentionally not connected
directly to training.

Inspect ground support and plane/flow failures before candidate appearance.
Repeated dark asphalt can pass the stable-dark diagnostic. Reflections, wet
ground, cast shadows from unannotated structures and illumination shifts remain
ambiguous. No candidate count, exposure fit or near-zero geometric round trip
establishes true shadow accuracy.

## 5. Slurm And Next Gate

```bash
# Run inside the repository; create logs BEFORE sbatch opens its log files.
mkdir -p logs
sbatch slurm/diagnose_shadow_physics.slurm
```

The script defaults to `NV_RTX4090`, one process, four CPUs, 32G host memory and
no GPU request. Adjust the partition for the cluster; there is no GPU library override.
The run root contains the Slurm job ID and is not overwritten.

Inspect Stage 1/2 images and failure rates before enabling Stage 3 weak export.
Unknown light direction gives ranges of plausible projections, not one supposedly
exact shadow mask. Stage 3 uses those hypotheses as constraints, not target masks.
Do not equate shadow and object velocity: their velocities *can* coincide for
fixed parallel light, rigid translation and planar ground, but that special case
is not a universal constraint or evidence of non-shadow status.

Compare A appearance, B appearance+flow/time, C appearance+physical geometry,
D all evidence on the same scene split and matched acceptance coverage. Without
independent shadow truth, call the results coverage, stability and weak-label
agreement, not true precision/recall. Do not calibrate pseudo precision against
the same rules that generated it. A held-out human audit, if later available,
is evaluation-only; no manual shadow masks are required for this training design.

## 6. Joint Three-Frame Gate And Weak Export

This is the new coupling, not parallel object-flow diagnostics:

```text
GT box corners at t0, t1, t2 ----> fixed world-light ray hypotheses
                                      |
common receiver plane / geometric BEV  +--> three projected shadow volumes
                                      |         |
images + flow + exposure fit ----------+--> onset / middle dark / recovery
                                                |
                       competing-caster ranking + three-frame fit
                                                |
              other-track, disjoint-frame scene-light support
                                                |
                   observed image region -> positive / IGNORE trimap
```

Re-export pairs with the updated exporter: old Stage 1 manifests lack the world
box corners and annotation tokens now required by `window_tracks`. Do not patch
old JSONs with made-up states. Re-run Stage 2 too because it records the exact
pair-manifest hash. Pairs must be consecutive; scattered isolated pairs cannot
make a three-frame window. Export several complete train scenes, not only one
vehicle, to test independent light support. More cameras multiply processing
cost, not the number of independent physical tracks.

```bash
python -m pytest -q tests/test_physics_shadow_motion.py tests/test_physics_shadow_labels.py

# New run root; Stage 1 images and Stage 2 evidence must not be overwritten.
python -u scripts/export_nuscenes_motion_pairs.py \
  --dataroot /public/home/eng_zhihongli/project/data/nuscenes-full \
  --split train --cams CAM_FRONT --max-pairs 120 --max-attempts 600 \
  --output-root outputs/physics_joint/pairs

python -u scripts/analyze_shadow_motion_pairs.py \
  --pairs outputs/physics_joint/pairs/pairs.jsonl \
  --flow-backend dis --device cpu --output-root outputs/physics_joint/flow

python -u scripts/build_physics_shadow_labels.py \
  --pairs outputs/physics_joint/pairs/pairs.jsonl \
  --analysis-root outputs/physics_joint/flow \
  --output-root outputs/physics_joint/labels_audit
```

Or submit `sbatch slurm/build_shadow_physics_labels.slurm` after creating `logs/`.
This script runs tests on HPC, then all three CPU stages. It does not start
training or request GPUs. Default paths include the Slurm job ID.

### Gate Details

The current implementation is `three-frame-track-contrast-v2`; read
[the v2 changes and related-work boundaries](shadow_track_contrast_v2.md).
Rebuild flow evidence: old archives without `receiver-support-v2` are rejected
by the new label builder. Do not overwrite an earlier experiment directory.

- Join only identical middle camera records, images, timestamps, projections,
  scene and official split. Reject inconsistent local planes and track states.
- Align the first pair's maps to the middle camera using world-ground geometry.
  Middle dark support must be observed in BOTH pairwise comparisons.
- Require onset and recovery evidence in the same component, but NOT at exactly
  the same pixels: the entering and leaving edges can be spatially disjoint.
  `same_patch_cycle` separately counts stronger darken-then-recover evidence.
- Appearance and chroma candidates come from Stage 2. Forward/backward flow,
  texture and agreement with ground geometry validate receiver anchors. Local
  support may cover a changing boundary within the SAME observed ground patch;
  completely unobserved/textureless patches stay unsupported. Neither flow
  disagreement nor box velocity equality supplies a positive label.
- Test 96 fixed world-ray directions (24 azimuths, four elevations), holding
  each direction fixed across the three interpolated boxes. Ray/plane
  intersections use box corners at their actual world heights.
- Compare middle observed coverage, onset against newly covered receiver, and
  recovery against receiver no longer covered. Require a margin over competing
  casters, including competing explanations under different light hypotheses.
  Also require the real order to beat the hardest of five wrong-time permutations
  and a frozen-middle trajectory by `--minimum-temporal-gain` (default 0.1).
  These controls are explanations, NOT negative shadow labels or causal renders.
- A caster cannot support its own scene-light hypothesis. Require at least two
  OTHER distinct GT tracks, with disjoint camera frames AND physical sample sets
  from the query. Exclude every near-best candidate owner, not just the winner.
  Repeated windows of one track cannot count twice. This is still a
  weak internal-consistency gate, NOT independent shadow validation.
- Full GT box volumes overestimate silhouettes. Their projections constrain
  selection; they NEVER become the output mask. Only observed middle-image
  component pixels inside ALL near-best plausible volumes can receive a positive
  label. The unsupported tail of a partially covered component stays IGNORE,
  and an intersection below the minimum area is rejected.
- Shadow existence and ownership are separate decisions. An ambiguous owner
  yields `YES` + `unresolved` when the observed shadow passes the independent
  gates; it does not automatically turn a supported shadow into IGNORE.
- Rasterize only candidate bounding regions and cache projected box polygons
  within each window, rather than repeating full-image volume rasterization.

Inspect these files before export:

| File | Meaning |
|---|---|
| `summary.json` | window failures, candidate counts, static/change/flow rejection reasons, passed gate count |
| `window_audit.jsonl` | bad joins, inconsistent tracks, excessive components, unstable exposure |
| `decisions.jsonl` | all light/caster hypotheses, competing margins, independent track support |
| `review/*.jpg` | earlier / clean middle / later / diagnostic outline and proposed caster box; never a VLM training overlay |
| `ground_points/*.npz` | sampled OBSERVED receiver points in metric world coordinates |
| `context/*.png` | clean earlier/later images registered to the middle ground, unsupported pixels black |
| `trimaps/*.png` | audit mode: all 255; weak export: accepted pixels 1, all others 255 |
| `scores/*.npz` | uncalibrated rule score; -1 is unknown, NOT a calibrated probability |

All true precision/recall/IoU/association accuracy fields stay null without
independent shadow truth. An empty accepted set is a valid gate outcome, not a
reason to automatically loosen thresholds. Static shadows, long occlusions,
turning-in-place casters, coarse ground support, wet roads and nonparallel or
changing light can fail these conservative gates; static darkness is never NO.
There is no claim of complete shadow annotation coverage.

After reviewing the evidence, explicitly enable a weak-label PILOT in a fresh
directory (no automatic promotion of the audit run):

```bash
python -u scripts/build_physics_shadow_labels.py \
  --pairs outputs/physics_joint/pairs/pairs.jsonl \
  --analysis-root outputs/physics_joint/flow \
  --export-weak-labels --output-root outputs/physics_joint/labels_weak

python scripts/build_shadow_annotation_dataset.py \
  --manifest outputs/physics_joint/labels_weak/canonical.jsonl \
  --seed 17 --dev-fraction 0.15 \
  --output-root outputs/physics_joint/region_data

python scripts/build_joint_shadow_annotation_dataset.py \
  --region-data-root outputs/physics_joint/region_data \
  --output-root outputs/physics_joint/joint_data
```

The shared builder rejects audit-only physics rows and checks trimap, observed
region and temporal-image hashes. Weak provenance survives into segmenter and
VLM rows. IGNORE-only regions do not dilute supervised per-image sample weights.
VLM views 1-3 are clean crop, thin region outline, clean wider context;
views 4-5 are earlier/later GROUND-aligned context, not caster translation.
`--no-temporal-context` removes these extra views for an input ablation without
changing the annotation targets. Geometry/GT diagnostic arrows are not inputs.
The future view is an **offline annotation** input; this is not a causal online
model or evidence of GT-free deployment. Default crop bounds now depend only on
the observed region; GT-expanded context requires `--context-policy gt-assisted`.
GT still constrains offline candidate selection, visibility and weak targets.

**Training is not automatically ready:** this protocol exports NO negative
pixels or NO regions. Existing segmenter class-coverage checks intentionally
block positive-only BCE training. Do not label all remaining pixels background,
train an all-YES VLM, or treat uncertain static shadows as negatives. Add
independently justified negative supervision in a separate canonical manifest
(synthetic/public sources do not require manual nuScenes shadow annotation),
keep its provenance, and inspect actual scene-disjoint train/dev class coverage.
Never use generated pseudo agreement as real evaluation. The exporter does not
feed weak eval labels back into training; official val always stays eval in the
shared builders. Freeze thresholds on train-derived dev before exporting val.

This update has NOT been executed locally: run the supplied regression tests
and nuScenes pilot on HPC before trusting the pipeline or training an annotator.
OCC integration remains deferred until annotation quality is established.

## Engine References

- [nuScenes SDK](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py): camera metadata and object tracks.
- [OpenCV DIS](https://docs.opencv.org/4.1.1/javadoc/org/opencv/video/DISOpticalFlow.html): existing CPU dense-flow engine.
- [torchvision RAFT-small](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.optical_flow.raft_small.html): optional existing pretrained flow engine and input normalization.

These references support the reused engines, not proof of this shadow-annotation
method. New experiment code and thresholds still require HPC validation.
