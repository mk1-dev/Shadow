# Shadow_OCC v9.3: Event-Anchored Region Completion And Student Training

## Implemented Information Flow

```
RGB + calibrated receiver + GT tracks + frozen current-only/context BEV + DIS
  -> unchanged M1 entity suppression and Caster Memory
  -> anchor-first M2 appearance / BEV / flow discrepancy proposals
  -> same-world signed illumination observations on native camera pixels
  -> independent illumination proposals (source bit 64)
  -> RGB-supported, ground-warped FSD state transitions (source bit 128)
RGB -> pinned official FSD -> independent semantic proposals (source bit 16)
  -> proposal UNION, not multiplication or intersection
  -> unchanged M3 caster/light competition and physical support/owner
Pre-FSD v8+ outputs -> inherited physical cores (separate reference)
  -> near-feasible rendered caster/light envelope in calibrated image pixels
  -> image-space illumination correspondence + contrast-sensitive Graph Cut
  -> visible candidate / strict core / supported expansion / IGNORE
  -> scene-isolated, deduplicated 1/0/255 dataset + confidence/tier sidecar
  -> core BCE + Dice; curriculum-weighted expansion BCE
```

`--shadow-prior-manifest` and `--core-reference-root` together enable completion on
`scripts/discover_shadow_video.py`. The reference must contain the same clips
and camera timestamps, without an external shadow prior. This prevents FSD
proposals accepted by static projection from becoming their own trusted cores.
Without it, v8+ discovery and targets retain the existing path. No OCC/VLM
training, SAM2 or ViShadow inference is added.

The cross-space branch remains independent: a low FSD response cannot remove
its seed. A high FSD response alone creates a candidate, not a training label
or a caster ID. Flow residual magnitude is not shadow probability.

v9.2 additionally compares the official sigmoid `semantic_probability` at
the same ground location, not clipped/saturated ReLU responses. A semantic
increase of at least 0.05 together with qualified RGB darkening contributes
onset/recovery observations to the existing caster trajectory competition.
Semantic changes alone never create labels; missing or stable semantic states
do not veto independent RGB evidence. The current dark frame receives the
observation, not the recovered bright frame. Image outputs retain state
increase, conflict, joint score and signed event maps separately.

## Completion And Supervision

- M3 acceptance, trajectory competition, light bank and owner thresholds are
  unchanged. Image cores are inherited pre-FSD v8+ weak labels, not GT. New
  image positives require independent illumination evidence. Physical BEV
  positives/owners unsupported by the final image label are cleared; dense
  image extensions do not invent BEV owners.
- Dense completion support and sparse BEV evidence are separate. Metric
  Delaunay triangles join observed receiver anchors only when every edge is
  at most 3 m. Observed local support has confidence 0.8. Interpolation has
  confidence 0.5 through 1.5 m, then falls with squared inverse edge length;
  long gaps and outside-hull pixels remain unknown. GT convex-volume
  ray intersections remove actual occluded ground, not full 2D rectangles.
  This is estimated planar support, not dense semantic ground GT. No missing
  depth or semantic model is silently inferred.
- The plausible image envelope is the union of the selected and near-feasible
  M3 hypotheses, rendered from their actual box/light geometry. There is no
  arbitrary mask dilation or new box uncertainty distribution.
- Native calibrated RGB pixels are registered by the ground homography.
  Valid existing exposure fits, chroma agreement and high-pass log-luminance
  texture agreement support signed log darkening/recovery. Only a current
  dark observation relative to a registered brighter neighbor can supply a
  new positive. A recovered bright pixel is not labelled shadow. Native BEV
  validity establishes anchor availability independently of sparse LiDAR
  support. BEV cosine supports spatial identity, never darkness evidence.
  Missing exposure is missing evidence. Missing/low-confidence/reliable flow
  remain separate states; geometry-only correspondence can retain an event,
  while a FB-consistent observation that contradicts ground motion is weighted
  by `exp(-residual_px / 3)`. Missing flow does not inherit this contradiction.
  Large residuals never boost shadow evidence or become negative labels.
- Graph Cut uses the union of semantic, cross-space and native illumination responses for unary
  costs, Lab contrast-sensitive Potts edges, fixed strict-core positive seeds,
  optional explicit negative seeds, and the visible receiver domain. It does
  not render shadow shape from a box. Only seeded connected cut components survive.
- Expansions require a completed visible region, near-feasible physical support,
  qualified independent illumination, and semantic/cross-space or connected
  observed support. Static appearance is never combined with illumination by
  taking their maximum. Their weight is `0.25 * illumination_score`, including
  correspondence and interpolation confidence, not the clipped FSD response.
  FSD + static projection without an observed light event stays candidate/255,
  not a new positive or a negative. Inherited static cores remain positive.
- v9.3 adds a separate inferred-interior path. At least four jointly supported
  event pixels must anchor a connected Graph Cut component inside the physical
  envelope. Growth is restricted by FSD support, Lab distance, receiver/BEV
  availability, absence of semantic contradiction and a 12-pixel radius at
  width 800. It cannot cross an occlusion or disconnected region. The result
  is region-level inference, not a fabricated per-pixel illumination observation.
  Its weight is at most 0.1 times seed illumination, correspondence quality
  and distance decay. Static cores or FSD alone cannot seed this path.
- `shadow_amodal_hypothesis_bev` stores historical regions projected by caster
  geometry onto currently entity-occluded ground. It is an inferred hidden-state
  hypothesis, not a current image observation, training positive, or new owner.
- This version does not implement a learned amodal shape model or 12 Hz sweeps.
  It consumes the actual camera frames in the existing pair/clip cache.

Image NPZ adds `shadow_core_image`, `shadow_expanded_image`,
`shadow_visible_completed_image`, `shadow_confidence_image`,
`shadow_amodal_hypothesis_image` (visible candidates plus inferred hidden area),
`shadow_tier_image`, `shadow_prior_response_image`,
`shadow_plausible_projection_image`, `shadow_native_illumination_image`,
`shadow_native_correspondence_valid_image`, and source bits. Tiers are
`0=unknown, 1=core, 2=expansion, 3=explicit negative`. Expanded image labels are
not silently promoted into strict BEV/owner labels.

`shadow_direct_expanded_image` and `shadow_region_inferred_image` separate
the two expansion sources; their union is `shadow_expanded_image`.
Image-completion source bit 64 means inferred interior, distinct from BEV
proposal bit 64. Both source legends are recorded separately in the protocol.

v9.1 additionally saves `shadow_receiver_state_image`,
`shadow_receiver_confidence_image`, `shadow_illumination_qualified_image`,
`shadow_log_darkening_image`, `shadow_log_recovery_image`,
`shadow_anchor_available_image`, `shadow_anchor_consistency_image`,
`shadow_flow_state_image`, `shadow_correspondence_quality_image`, and
`shadow_static_prior_ignore_image`. Receiver states are unknown/observed local
support/bounded interpolation/GT-volume occluded. Flow states are
missing/low-confidence/reliable. `protocol.json` pins the reference hashes,
receiver configuration and illumination formula.

The existing dataset exporter merges overlapping clips, preserves conflicts
as IGNORE, records core/expanded pixel counts, and writes `pixel_supervision`
NPZ references with SHA256. The dataset loader resizes and flips confidence and
tiers with the label. Reliable negatives retain the existing independent lit
receiver rule, excluding completed proposals and FSD response >= 0.25.

Training first uses core/negative supervision for two epochs. Expansion BCE
then ramps over three epochs; `--expansion-warmup-epochs` and
`--expansion-ramp-epochs` control this. Expansion is excluded from core Dice.
Its BCE is normalized by expansion pixel count, not confidence mass, so a
uniform 0.25 confidence is not accidentally cancelled. Both reductions support
DDP, including ranks with no expanded pixels. Existing non-layered datasets
have no expansion term and keep the previous loss behavior.

## Data And Weights

| Resource | Required Now | Source / Reuse |
| --- | --- | --- |
| nuScenes images, metadata, poses, GT tracks, existing ground/LiDAR cache | Yes | Reuse existing official data; no duplicate download |
| Pair/receiver, DIS and frozen UniAD BEV caches | Yes | Reuse current clip inputs |
| Pre-FSD v8+ output clips | Yes | Same camera tokens/timestamps; inherited core reference |
| FSD source | Yes | Official IMU-Group/FSD, commit `b22927c5a2ec25c878a15ecf76344dde474fe503` |
| `FSD_best.ckpt` | Yes | 209,842,750 bytes; SHA256 below |
| EfficientNet ImageNet checkpoint | No | Full FSD checkpoint loads strictly; no backbone download |
| FSD training dataset | No | Frozen inference only |
| SAM2 / ViShadow weights | No | Not part of this implementation |
| nuScenes camera sweeps | Later | Not used by this v9 completion path |

Official FSD SHA256:
`a408c3e9f6cb4c6b304c5cfe742346e97f6434c73e9116b78a362a7180cd58f2`.

FSD `binary_mask` is a nonnegative ReLU response, not a logit or a calibrated
probability. The adapter stores the raw response and uses the official 0.5
threshold. Clipping to [0,1] for completion is a bounded response, not
probability calibration. Its `logit` field is already sigmoid; it is saved
separately without applying sigmoid twice. RGB [0,1] is passed to the model;
ImageNet normalization is internal. Checkpoints load with `strict=True`.

Weights and external source live outside the Shadow Git tree. On the current
HPC they are under:
`/public/home/eng_zhihongli/project/models/shadow_priors/FSD_b22927c/`
with `source/`, `FSD_best.ckpt`, and isolated `python_deps/`.

Reproducible download on another machine:

```bash
python scripts/fetch_shadow_prior_assets.py --output-root /path/outside/Shadow/fsd_assets
python -m pip install --no-deps --target /path/outside/Shadow/fsd_assets/python_deps \
  efficientnet-pytorch==0.7.1 PyMaxflow==1.3.2
```

The full external checkpoint is about 200 MiB. A three-field float32 response
cache has an uncompressed upper bound of `12 * H * W` bytes per unique image
(about 4.12 MiB at 450x800), before NPZ compression. Cache by camera token once,
not once per overlapping clip. Existing raw nuScenes and BEV caches are not
duplicated. Dense completion/confidence outputs need additional space measured
from the pilot; `summary.json` records actual prior cache bytes.

## HPC Execution

Use a new code snapshot and new output roots. Leave the current v8+ training
jobs and frozen results untouched. Set these paths to one existing clip batch:

```bash
export SHADOW_PROJECT_ROOT=/path/to/new/code_snapshot
export SHADOW_FSD_ASSETS=/public/home/eng_zhihongli/project/models/shadow_priors/FSD_b22927c
export SHADOW_PAIRS=/path/to/existing/pairs.jsonl
export SHADOW_CLIPS=/path/to/existing/clips.jsonl
export SHADOW_PRIOR_OUTPUT=/path/to/new/fsd_prior_cache
export SHADOW_CORE_REFERENCE=/path/to/existing/pre_fsd_v8plus_outputs
mkdir -p "$SHADOW_PROJECT_ROOT/logs"
sbatch --chdir="$SHADOW_PROJECT_ROOT" slurm/export_fsd_shadow_priors.slurm
```

After prior export completes, run the existing discovery command with its same
pair, analysis, feature and clip inputs, adding:

```bash
export PYTHONPATH="$SHADOW_FSD_ASSETS/python_deps:${PYTHONPATH:-}"
python scripts/discover_shadow_video.py \
  --pairs "$SHADOW_PAIRS" --clips "$SHADOW_CLIPS" \
  --analysis-root /path/to/existing/flow_analysis \
  --feature-root /path/to/existing/context_bev \
  --spatial-feature-root /path/to/existing/current_only_bev \
  --shadow-prior-manifest "$SHADOW_PRIOR_OUTPUT/manifest.jsonl" \
  --core-reference-root "$SHADOW_CORE_REFERENCE" \
  --output-root /path/to/new/v91_discovery --stage full
```

An existing FSD cache can be reused with no GPU inference. Set
`SHADOW_PRIOR_MANIFEST=/path/to/existing/prior/manifest.jsonl` when using
`slurm/run_shadow_completion.slurm`; it skips prior export. The discovery
command itself runs on CPU with the cached prior and PyMaxflow. It does not
rebuild nuScenes, optical flow or BEV caches, and does not train a model.

Create a new runs JSONL with `method`, `split`, `run_root`, and `pairs`, using
the same scene-isolated train/dev/eval assignment. Then use the existing
`build_shadow_mainline_experiments.py export --auto-negatives`, `suite`, and
`run_shadow_segmenter_suite.py` commands. Do not pass an old v8+ freeze lock as
authorization for v9; the prior and completion protocol are new teacher inputs.

## Multi-Scene Student Training

`scripts/run_shadow_completion_training.py` joins the existing cached-scene
plan, prior exporter, discovery, trimap exporter and segmenter suite. It samples
scene identities deterministically, keeps original train/dev/eval roles and
round-robins cameras. It does not choose scenes by teacher positives or eval
performance. The default pilot uses 8 train, 3 dev, 3 eval scenes and three clips
per scene, not the full 25% experiment.

```bash
python scripts/run_shadow_completion_training.py prepare \
  --source-plan /path/to/completed/automatic_scene_plan \
  --output-root /path/to/new/motion_illumination_training \
  --train-scenes 8 --dev-scenes 3 --eval-scenes 3 \
  --clips-per-scene 3 --epochs 10
export SHADOW_COMPLETION_TRAIN_PLAN=/path/to/new/motion_illumination_training/plan.json
sbatch slurm/run_shadow_completion_training.slurm
```

The GPU job exports FSD once for all unique selected camera tokens, annotates
each scene using its independent v8+ core reference, exports explicit
positive/negative/IGNORE data and confidence sidecars, then runs two equal-budget
student arms: `v8_core` and `motion_illumination`. Both use existing DeepLabV3
MobileNetV3 ImageNet initialization, scene-aware sampling, balanced BCE + 0.5
Dice, the same image IDs and ten epochs. Expansion supervision retains the
two-epoch warmup/three-epoch ramp. No VLM or OCC training is included.

Artifacts are `plan.json`, `pipeline_*_status.json`, `prior/manifest.jsonl`,
`discovery/`, `dataset/{annotation_summary,class_distribution,dataset_manifest}.json`,
`suite/suite.json`, and `training/` containing both model histories, checkpoints,
readiness results and held-out teacher-agreement metrics. Both students use the
same frozen new-teacher eval reference. These are weak-label agreement results,
not independent shadow accuracy. A failed or collapsed arm is recorded rather
than substituted with another checkpoint.

For separate allocations, `run --stages prior` needs one GPU,
`run --stages annotation --annotation-workers 4 --resume` uses four CPUs and
no GPU, `run --stages dataset` exports trimaps and the paired training plan on
CPU, and `run --stages train` consumes that plan on one GPU. Submit them with `afterok`
dependencies. Resume checks completed clip identities and teacher protocol;
incomplete scene directories are archived with a `.partial_...` suffix rather
than deleted. Input hashes are checked again on each stage. GPU prior inference
is shared across all scenes, not repeated for every annotation worker.
Both export arms declare the same `clip_ids`, so extra windows in a larger
v8+ reference directory cannot contribute additional labels to common frames.

## Literature And Scope

- [FSD, NeurIPS 2025](https://github.com/IMU-Group/FSD): official pretrained
  shadow/opacity responses provide an external semantic and shape prior.
- [Learning Shadow Correspondence, ECCV 2022](https://arxiv.org/abs/2208.00150):
  shadow correspondence motivates separating material flow from shadow changes;
  this code retains geometric correspondence rather than importing its network.
- [ViShadow, TIP 2024](https://github.com/HarryHsing/Video-Instance-Shadow-Detection):
  instance/temporal association motivates separating current visibility from
  hidden state; ViShadow weights and its tracking model are not used here.
- [PyMaxflow](https://pmneila.github.io/PyMaxflow/): existing graph-cut solver
  for the explicitly defined binary completion energy.

The resulting protocol is **no target-domain manual shadow annotation, with
an externally supervised pretrained shadow prior**. It is not shadow-supervision
free across all source datasets. Model responses, projection fits, and held-out
teacher agreement are not independent shadow GT.
