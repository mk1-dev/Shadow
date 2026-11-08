# Track-Centric BEV / Flow Shadow Discovery

This is a new executable path, not a replacement of the frozen A/B/C/D audit.
It does not train a segmenter, VLM, lighting estimator, BEV encoder or OCC model.

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

The default discrepancy is registered illumination change multiplied by one
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

Large flow error receives no shadow bonus:
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
actual memory records used by that search. Descriptors are retained for future
association refinement, not injected as an unvalidated owner score. This path
uses GT identities; it does not silently equate model-predicted IDs with GT IDs.

`--entity-motion-suppression off` disables the explicit motion-core suppression,
not the geometric occlusion exclusion or caster memory. It is a same-input
control, not permission to treat hidden ground as free space. The current GT
receiver already excludes volume rays, so incremental candidate removal can
be zero. `additional_search_removed_cells` reports this honestly. The cached
BEV backbone is unchanged; no FLOPs/speedup or association-quality gain is
claimed without a measured comparison and independent reference.

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
and `SHADOW_VIDEO_ENTITY_MAX_FB_ERROR_PX`.

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

Version `track-centric-appearance-geometry-video-v5` adds motion-matched entity
cores and an independently retained, consumed caster memory. V4's separation
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
