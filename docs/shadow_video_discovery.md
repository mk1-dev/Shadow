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
minus supported entity explanation, only where entity coverage is valid.
`entity_explanation_state` distinguishes `0=missing`, `1=explained` (a supported
nonzero explanation), and `2=unexplained` (a supported prediction with zero fit).
No new acceptance threshold is introduced. When coverage is missing, observed
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
python -m pytest -q tests/test_shadow_video.py tests/test_receiver_explanations.py
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
- `reviewed_signals.jsonl`: optional independent-region comparisons, not new
  pseudo labels. Actual camera intervals are retained in `protocol.json`;
  reading keyframe pairs does not create a denser camera stream.
- `*_target.png`: image trimap, `1` for physically supported weak foreground and
  `255` for unknown. Rejected candidates do not become negative labels.

Version `track-centric-appearance-geometry-video-v3` separates complete and
partial evidence and replaces the ambiguous review `projection_capacity` field
with oracle and selected-direction fields. Per-frame proposal capacity remains
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
