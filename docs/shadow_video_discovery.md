# Entity-Shadow Disentangled Cross-Space Discovery

This is a new executable path, not a replacement of the frozen A/B/C/D audit.
It does not train a segmenter, VLM, lighting estimator, BEV encoder or OCC model.

## Current Mainline: M1-M4

```text
RGB + 3D tracks + cached DIS + frozen mature BEV
  M1  surface-motion entity decomposition -> suppression + caster Track Tokens
  M2  current-only BEV matching <-> flow lifted to metric ground + Lab change
  M3  caster/light projection + signed boundary motion + shadow-track history
  M4  observed region memory + RGB boundary refinement + separate Shadow-BEV
```

The executable default is `--discovery-mode cross-space`. It actively unions
metric cross-space proposals with the appearance/static/caster proposal paths.
BEV is no longer just a region-linking descriptor in this mode.

Two feature roles are deliberately separate:

- `--feature-root`: existing temporal/context features for region identity and
  caster descriptors.
- `--spatial-feature-root`: independently exported `--history-mode current-only`
  features for discovery. A current-only `--feature-root` can also supply both
  roles. Temporal features are never silently relabeled as spatial evidence.

Export the missing current-only cache with the existing mature model, not a new
network. Use the SAME checkpoint/config and the matching train/val info file
used for the temporal export:

```bash
conda activate drivevla
python scripts/export_mature_bev.py \
  --external-root "$UNIAD_ROOT" \
  --checkpoint "$UNIAD_CHECKPOINT" \
  --pairs "$SHADOW_VIDEO_PAIRS" \
  --dataroot "$SHADOW_NUSC_ROOT" \
  --ann-file "$UNIAD_ANN_FILE" \
  --history-mode current-only \
  --output-root outputs/mature_bev_current
```

Existing RGB, pair manifests, DIS, GT states and temporal BEV remain reusable.
No pseudo-label dataset rebuild or model training is required. The main command
rejects a temporal-only spatial cache with an actionable export instruction;
`--discovery-mode appearance` explicitly selects the older base channel.

M2 lifts each measured flow endpoint through the NEXT camera onto the receiver
plane, then expresses its displacement in horizontal anchor-grid x/y meters.
Current-only BEV descriptors are locally cosine-matched with uniqueness and
reverse-cycle checks. `flow_bev_displacement_m` and `spatial_bev_displacement_m`
therefore share units and axes. Their difference is never computed between
image pixels and feature vectors. `spatial_bev_feature_change` retains raw
`1-cosine`; ranking normalizes it to [0,1]. Missing/ambiguous matches remain NaN
with false validity, not an anomalous displacement.

The proposal channel combines existing observed appearance with confidence-
weighted metric disagreement (weight 0.35) and normalized BEV change (0.25).
It is a candidate score, not shadow truth. Unsupported entity explanation is
reported as NaN and retained as partial evidence, not silently asserted absent.
For physical acceptance, a proposed ground cell still needs observed appearance
support (`minimum_observed_appearance=0.05`) and caster/light fit. Static
appearance retains its own path, including when no temporal event is visible.

M3 retains projection/light and signed-transition fits in each caster token's
`shadow_projection_history`. Local owner scoring combines projection coverage,
observed boundary fit against temporal counterfactuals (weight 0.25 when events
exist), repeated support for the same shadow track/light, and caster identity
consistency. No event is not a failed motion observation. The default CLI owner
mode is `geometry+appearance+bev`; `physical` retains the older independent-
shared-light owner policy. Ambiguous ownership does not remove shadow existence.

M4 links with metric overlap/distance, boundary shape, Lab histogram, context-BEV
descriptor, material correspondence and caster candidates. A caster/light
projection supplies an ALTERNATIVE shifted old mask for linking/weak seeding,
alongside the unshifted observation. It is not an owner label and never paints
new pixels into a currently unobserved region. Material flow is not treated as
shadow boundary speed. `visible -> occluded/unknown -> visible` keeps identity
through the configured short gap without emitting missing-frame masks.

Each observed region is refined on a small current-RGB ROI with a ground-
restricted darkness/boundary watershed. The image mask is not just an enlarged
BEV cell. Final weak labels require both this RGB support and the physical BEV
projection. Three separate payloads are retained:

- `*_image.npz`: actual image proposal, weak target and region IDs.
- Per-frame BEV NPZ: raw proposals, RGB-supported continuous proposals, physical
  weak targets, owner uncertainty and world coordinates.
- `caster_memory.jsonl`: 3D caster states, appearance/BEV and projection history.

`region_memory.jsonl` contains boundary/appearance histories, correspondence,
caster candidates, visibility and separate existence/owner scores. None of these
representations is merged into an occupancy label. Endpoints can have observed
continuous proposals while physical weak targets remain unknown.

```bash
conda activate shadow_annotator
python -u scripts/discover_shadow_video.py \
  --pairs "$SHADOW_VIDEO_PAIRS" \
  --analysis-root "$SHADOW_VIDEO_FLOW" \
  --feature-root "$SHADOW_VIDEO_FEATURES" \
  --spatial-feature-root outputs/mature_bev_current \
  --clips "$SHADOW_VIDEO_CLIPS" \
  --stage full --discovery-mode cross-space \
  --owner-scoring geometry+appearance+bev \
  --output-root outputs/shadow_mainline_v7
```

For Slurm, set `SHADOW_VIDEO_SPATIAL_FEATURES` in addition to the existing input
variables and submit `slurm/discover_shadow_video.slurm` on a CPU partition.
Frozen mature feature EXPORT uses the existing GPU environment; the subsequent
cached-feature discovery path uses CPU. This is an algorithm implementation,
not a claim of measured shadow quality or OCC improvement.

## Information Flow

1. Read canonical camera observations and build one metric world receiver grid
   per clip. Keep visibility separately for every frame and every adjacent pair.
2. Sample frozen mature BEV features into that grid. Compute cell and local
   neighbourhood temporal changes; these are representation evidence, not
   occupancy or a calibrated free-ground classifier.
3. Compare exposure-corrected appearance, measured DIS flow and geometrically
   expected flow. Compare available GT box-surface motion with ground motion.
   Match high-confidence entity cores before discovery; retain entity identity
   and observations independently in caster memory (details below).
4. Propose from discrepancy, supplementary static appearance, and observed
   appearance inside caster/light search regions. A light projection or a flow
   error alone cannot create shadow evidence.
5. Run seeded OpenCV watershed using observed score support and boundary maps.
   Rank regions and link them with Hungarian assignment using world overlap,
   metric distance and pooled BEV descriptors. Short missing intervals preserve
   region identity, not fabricated foreground pixels.
6. Apply the existing `rank_hypotheses` / `factorized_shadow_decision` functions
   with all competing GT vehicle tracks. The requested clip track is a search
   prior, never an automatically assigned owner. Existence and association remain
   separate. Positive pixels must also lie in the selected physical projection.

The appearance/geometry BASE discrepancy is registered illumination change multiplied by one
minus supported entity explanation, where a physical motion reference is valid.
There are two deliberately distinct domains:

- Box-surface motion lives on visible entities and remains an exclusion or
  competing-motion explanation. It is never copied or dilated onto ground.
- Ground receivers use expected geometric ground motion validated by existing
  `receiver_flow_supported` evidence: reliable textured DIS anchors or local
  same-connected-ground-patch support. Boundary flow itself may be unreliable.

`physical_reference_source` distinguishes `0=missing`, `1=box_surface_motion`,
and `2=validated_ground_motion`. A ground reference requires visibility in both
frames, finite expected motion, no volume occlusion, and actual local motion
support. A projected grid or absence of a predicted object alone cannot provide
this reference. Ground validation does not imply that a region is a shadow or
that all unannotated entities have been ruled out.

`entity_explanation_state` distinguishes `0=missing`, `1=explained` (a supported
nonzero explanation), and `2=unexplained` (a supported prediction with zero fit).
It correctly remains missing on receivers lacking box-surface predictions;
ground support does not relabel it as an entity prediction. The complete branch
uses either supported physical reference, avoiding the v3 contradiction of
requiring box-surface coverage inside an explicitly entity-free receiver.

No new acceptance threshold is introduced. When both references are missing, observed
appearance still proposes regions through `appearance_partial_score_map`, while
`discrepancy_score_map` is NaN and `discrepancy_evidence_complete` is false.
The candidate union uses both channels, so missing coverage does not delete
otherwise observable candidates or masquerade as entity exclusion. The source
bit for discrepancy is reserved for the complete-evidence branch.

Raw image-flow error alone does not establish a shadow:
material flow can follow the ground while a shadow boundary moves. A separate
`ground_consistent_illumination_score` exposes exactly this case. Flow remains
part of correspondence checking and exposure fitting.

BEV feature changes and neighbourhood changes remain explicit signals, and BEV
descriptors participate in region assignment. `--bev-role context` (default)
does not suppress proposals when BEV changes. `--bev-role stability` explicitly
tests the old illumination-invariance hypothesis; `--bev-role off` also removes
BEV descriptors from linking. Do not increase BEV weights to manufacture a gain.

## Motion-Matched Entity Suppression and Caster Memory

M1 first separates explained entity surfaces from the receiver search domain:

`GT surface motion + DIS -> entity suppression + caster memory -> receiver discrepancy -> caster/light association`

`entity_suppression_mask` requires a front box-surface ray hit, a projected
silhouette-core margin of 2 image pixels, FB-valid flow with at most 1.5 px
cycle error, and at most 2 px surface-motion residual. Predicted surface points
must remain in the next image and not be hidden by the same or another box.
Surface motion uses corresponding 3D points under an affine box transform,
not a constant box-center displacement. This is a bounding-volume proxy,
not pixel-perfect instance segmentation. All thresholds use the calibrated
working image coordinates used by the cached DIS flow.

The maps use the world receiver-RAY grid: an entity surface lies along a camera
ray whose ground endpoint is occluded. They are not dense image instance masks.
No match is spread onto nearby ground or propagated to the next ground cell.
The last frame lacks outgoing motion evidence and has no new suppression claim.
`entity_occlusion_mask` keeps all GT-volume ray support unavailable as ground;
unmatched/boundary rays remain `entity_uncertain_mask`. Neither becomes a
negative shadow training label. Adjacent observable shadow regions remain in
the receiver search domain even if their flow resembles the vehicle's motion.

`caster_memory.jsonl` retains every observed GT identity, including those with
no matched pixels or temporarily outside the image. It stores camera-time 3D
states, observed finite-difference velocities, projection history, normalized
Lab histograms from matched surfaces, and BEV descriptors sampled separately
at entity centers. Missing descriptors remain null. Existing checked GT triplets
are retrieved FROM this memory for the physical search and final footprint;
the requested clip track is not assigned as owner. Per-region physics decisions
retain light compatibility, while `caster_memory_candidates` identifies the
actual memory records used by that search. The `physical` owner policy retains
descriptors without using them; optional local-memory scoring is described below. This path
uses GT identities; it does not silently equate model-predicted IDs with GT IDs.

`--entity-motion-suppression off` disables the explicit motion-core suppression,
not the geometric occlusion exclusion or caster memory. It is a same-input
control, not permission to treat hidden ground as free space. The current GT
receiver already excludes volume rays, so incremental candidate removal can
be zero. `additional_search_removed_cells` reports this honestly. The cached
BEV backbone is unchanged; no FLOPs/speedup or association-quality gain is
claimed without a measured comparison and independent reference.

## Existing A/B Helper

`--discovery-domain image-rays` adds observed entity-surface rays to candidate
search BEFORE suppression. This is distinct from `receiver` (the default v5
behavior). `search_valid_by_frame` can include entity observations, while
`receiver_valid_by_frame` remains visible ground only. Box rays do not become
observable ground. Entity appearances can generate partial proposals, never
complete ground discrepancy. Physical ranking sees only the receiver portion
of each region; final positive targets are also intersected with that domain.

Run one paired experiment with shared input sampling:

| Arm | Candidate domain | Entity suppression | Owner scoring |
| --- | --- | --- | --- |
| A_no_suppression | image-rays | off | geometry |
| B_suppression_geometry | image-rays | on | geometry |
| B_suppression_memory | image-rays | on | geometry+appearance+bev |

A/B changes only explicit motion-core suppression. Caster memory, frozen BEV,
raw DIS, exposure fit, ground reference, physical acceptance and light bank
are identical. The third arm isolates descriptor weighting; it must retain
exactly the same proposals and positive targets as B. Mask hashes verify this.
The receiver-only default remains available for reproducing older results;
changing only `--entity-motion-suppression` there is not the informative A/B.

Owner modes (`--owner-scoring`):

- `physical`: original factorized/independent-shared-light policy.
- `geometry`: same linked shadow track, same candidate caster and same light
  hypothesis across frames. Average current fit and other-frame mean fit.
- `geometry+appearance+bev`: multiply that geometry score by caster identity
  continuity: `1 - 0.2*(1-appearance_cosine) - 0.2*(1-BEV_cosine)`. Each cosine
  compares the CASTER with its own matched observations in other supported
  frames, never car appearance with a shadow or ground descriptor. Missing
  descriptor terms contribute no penalty and provide no invented evidence.

The two new modes are an experimental offline local-memory owner policy, not
the old independent-light policy with a bonus attached. Both use the same
minimum score 0.65, margin 0.05 and at least two distinct linked frames. Multiple
regions in one frame do not supply extra temporal votes. Different shadow IDs,
failed physical fits and mismatched light IDs cannot supply owner support.
The new modes do not change shadow existence. An assigned owner's pixel mask
is the intersection of the existing positive target and its own physical
footprint; other positive pixels remain unresolved. None of these scores is
a calibrated probability or a measured association accuracy.

```bash
python -u scripts/compare_shadow_entity_ab.py \
  --pairs "$SHADOW_VIDEO_PAIRS" \
  --analysis-root "$SHADOW_VIDEO_FLOW" \
  --feature-root "$SHADOW_VIDEO_FEATURES" \
  --clips "$SHADOW_VIDEO_CLIPS" \
  --output-root outputs/shadow_entity_ab
# Or: sbatch -p AMD slurm/compare_shadow_entity_ab.slurm
```

This CPU-only runner samples each clip once and gives the three arms identical
arrays and caster observations. `summary.json` reports candidate cells,
entity-contaminated candidate cells, weak-positive cells, assigned owner cells,
resolved regions, and stage timings. Shared BEV/flow sampling is timed separately;
total runtime includes visualization/export. Fixed-order single-pass timings
are diagnostics, not a repeated runtime benchmark or backbone FLOP reduction.

Optional `--review-regions` uses the same independent-mask manifest as the
single-run script. Shadow preservation and proposal coverage use eligible
review cells on the common ray grid, not image-resolution IoU. Without
independent masks these metrics are null, not inferred from pseudo labels.
The report does not call weak acceptance a measured precision/recall gain.

## Three Decisive Stages

Use the same clips and inputs, with a separate output directory for each stage:

- `--stage signal`: no box/light search or physical acceptance. View A: signed
  illumination change, B: flow residual, C: BEV change, D: entity explanation,
  and E: actual discrepancy proposals in the synchronized video.
- `--stage projection`: add all 96 box/light hypotheses for the requested track,
  without physical acceptance. Report best single-direction coverage, best IoU
  and union coverage separately. Invalid geometry is counted separately, and
  `*_light_bank.npz` retains every direction, including validity.
- `--stage full` (default): run discovery, existing physical explanation and
  temporal region refinement. Only this stage can emit positive weak targets.

Projection coverage normally uses observed proposals, NOT shadow ground truth.
It is a model-capacity check, not precision/recall or a calibrated light estimate.
Best coverage and best IoU can select different directions; union coverage is
only a search-space upper bound, never an accepted shadow mask.

Optionally provide independently reviewed image regions with `--review-regions`:

```json
{"clip_id":"scene0002_car01","frame_token":"ACTUAL_CAMERA_TOKEN","region_type":"shadow","mask":"review/shadow.png"}
```

Supported region types are `shadow`, `stable_ground`, `material_boundary` and
`entity`. `reviewed_signals.jsonl` reports finite/missing counts and distributions
of illumination, flow, BEV, entity explanation, complete discrepancy and partial
appearance. Projection evaluation has two separate fields:

- `projection_oracle_capacity` selects the best direction on the review mask.
  This is explicitly an oracle upper bound, never a generalization result.
- `projection_selected_direction` evaluates the direction selected from the
  automatic proposal before reading any review masks. It never reselects a
  direction using the review mask. Its selection policy is recorded as
  `observed_proposal_best_cover_without_review_masks`; this is not a claim that
  the separately ranked physical verifier chose the same direction.

A missing/invalid selected direction has missing metrics, not oracle fallback.
If all valid directions have zero overlap, capacity reports
`status=no_supported_direction`, both selected IDs are null, and numerical
coverage stays zero. It does not select direction zero via `argmax`.
Masks are evaluation-only; they
never seed proposals, tune acceptance or enter training targets. Missing entity
or receiver evidence is reported as missing rather than silently zero-filled.

## Receiver Cache

New `export_nuscenes_motion_pairs.py` exports each camera token once under
`frames/<camera_token>/`, with `observation.json`, `image.png`, and `ground.png`.
Each frame uses its own LiDAR-derived ground plane and anchor, including when
it is the second frame of a pair. Adjacent pairs share the exact observation.
`receiver_frames.jsonl` records the resulting frame-level observations.

The clip grid uses one local plane as a receiver approximation. Per-frame local
ground support remains separate; this is not a full curved-road reconstruction.

Existing pairs can be used for a pilot: duplicate frame references are selected
deterministically, preferring their own pair-start geometry. The protocol records
`canonical_legacy_pair_reference`; it does not claim to repair historical ground
supports. For a clean receiver experiment, re-export pairs and re-run the DIS
analysis against the new pair manifest. The mature BEV cache can be reused when
sample identities, calibrations and feature provenance still match.

## Select Clips

Create a JSONL file containing actual IDs from the pair export, for example:

```json
{"id":"scene0002_car01","scene_name":"scene-0002","camera":"CAM_FRONT","start_sample_token":"ACTUAL_SAMPLE_TOKEN","track_id":"ACTUAL_INSTANCE_TOKEN","num_frames":9}
```

Every clip must be contiguous and have interpolated states for the selected
track throughout. Choose 10-20 diagnostic clips across scenes, including clear
moving shadows, weak shadows, static cases and confounders. Do not synthesize
missing samples or silently shorten a clip.

## Run on HPC

```bash
python -m pytest -q tests/test_shadow_video.py tests/test_entity_memory.py tests/test_receiver_explanations.py
python -u scripts/discover_shadow_video.py \
  --pairs /path/to/pairs/pairs.jsonl \
  --analysis-root /path/to/flow \
  --feature-root /path/to/mature_bev \
  --spatial-feature-root /path/to/current_only_bev \
  --clips /path/to/selected_clips.jsonl \
  --stage signal --bev-role context \
  --output-root outputs/shadow_video_signal
```

This path uses CPU operations and already-exported features/flow; no GPU is
required. `slurm/discover_shadow_video.slurm` exposes the same paths through
`SHADOW_VIDEO_PAIRS`, `SHADOW_VIDEO_FLOW`, `SHADOW_VIDEO_FEATURES`,
`SHADOW_VIDEO_CLIPS`, `SHADOW_VIDEO_OUTPUT`, `SHADOW_VIDEO_STAGE`,
`SHADOW_VIDEO_BEV_ROLE`, and optional `SHADOW_VIDEO_REVIEW_REGIONS`. Submit with the site's CPU
partition, e.g. `sbatch -p AMD slurm/discover_shadow_video.slurm`.
Entity options are also exposed as `SHADOW_VIDEO_ENTITY_SUPPRESSION`,
`SHADOW_VIDEO_ENTITY_CORE_MARGIN_PX`, `SHADOW_VIDEO_ENTITY_MAX_RESIDUAL_PX`,
and `SHADOW_VIDEO_ENTITY_MAX_FB_ERROR_PX`. Single-run domain and owner modes
use `SHADOW_VIDEO_DISCOVERY_DOMAIN` and `SHADOW_VIDEO_OWNER_SCORING`.

## Outputs

- `review.mp4` and per-frame JPEGs: RGB, GT projections, observed/expected flow,
  appearance, aligned BEV change, discrepancy regions, image masks, shadow BEV
  with vehicle trajectory, and flow confidence.
- `frames.jsonl`: region identity, ranking, temporal consistency, proposal source
  bits, full physics decisions, and frame/sample provenance.
- Per-frame NPZ: dense proposal/evidence maps, independent Shadow-BEV masks,
  owner IDs, shadow track IDs, association state, static/moving state, and world
  grid coordinates. Owner `-1` means unresolved; it is not the selected clip ID.
- `pair_evidence.npz`: signed exposure-corrected light changes, raw measured and
  geometric flow, BEV changes, and flow validity/confidence/status maps.
- `caster_memory.jsonl`: identities, geometry, motion, projection history and
  separately retained appearance/BEV descriptors. Per-frame evidence exports
  suppression/occlusion/uncertain masks; pair evidence retains matched motion,
  residual and target visibility. Surface owner indices map to the per-frame
  `entity_surface_index_to_instance_token` list, not the shadow-owner lookup.
- `reviewed_signals.jsonl`: optional independent-region comparisons, not new
  pseudo labels. Actual camera intervals are retained in `protocol.json`;
  reading keyframe pairs does not create a denser camera stream.
- `*_target.png`: image trimap, `1` for physically supported weak foreground and
  `255` for unknown. Rejected candidates do not become negative labels.

Version `entity-shadow-disentangled-video-v7` adds metric current-only BEV/flow
discovery and RGB-refined temporal region representation. V6's separate candidate
and label domains, V5's
motion-matched entity cores and retained caster memory, V4's separation
of box-surface motion from the ground motion reference, V3's complete/partial split and separate
oracle/selected-direction review metrics are retained. Per-frame proposal capacity remains
`projection_capacity`, explicitly marked as not shadow ground truth.

FB inconsistency is not called an occlusion label. Occlusion support is explicitly
the GT-volume ray proxy; unknown causes remain unknown. Flow interpolation is
not performed. Scores and temporal similarities are not calibrated probabilities.

This is a training-free region refiner using existing OpenCV/scipy operators,
not a trained video shadow network. Endpoints without two-sided physics retain
proposals but do not receive new positive training targets. A video sequence can
therefore contain continuous proposals but sparse accepted labels. Clip-level
visualization and independent evaluation must establish actual quality before
training annotation models or claiming BEV/OCC improvements.
