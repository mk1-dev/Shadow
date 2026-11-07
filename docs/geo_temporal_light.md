# Learned Light, GT-Centered Proposals and Longer Observations

## Design

```text
map location + UTC capture time -> solar ephemeris prior in world east/north/up
three RGB frames + registered ground Lab-L differences + exposure validity
                    -> GeoTemporalLightNet
                    -> 96-direction distribution + evidence confidence
GT caster tracks + direction distribution -> swept ground search ROI
observed dark ground + frozen BEV consistency + flow -> candidate components
up to 9 consecutive observations -> signed formation/recovery and track contrast
                    -> entity/material/shadow/unknown competition
```

The geotemporal model is implemented in `shadow_occ/models/illumination.py`.
`scripts/prepare_light_direction_data.py`, `train_light_direction.py` and
`predict_light_direction.py` supply preparation, training and frozen inference.
This is an implemented trainable model, not a claim that weights have already
been trained or that an astronomical lookup is itself learned estimation.

## Location, Time And Direction

nuScenes' official [map API](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_api.py)
documents the four map origins and EPSG:3857 coordinates. The implementation uses
pyproj to convert the anchor ego location to latitude/longitude, then
[pvlib solar position](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.get_solarposition.html)
with microsecond capture timestamps explicitly interpreted as UTC. Capture time
is not the HPC job time. Camera heading is handled by the projection rotation,
not by assuming the car faces north.

pvlib's azimuth is clockwise from north. A direction TO the sun is negated to
obtain light propagation toward the ground. In world coordinates x=east,
y=north, z=up. The three per-frame ephemeris rays rotate the hypothesis bank
slightly around its middle-frame reference. At driving-video time scales sun
motion is small: caster motion/pose and ground geometry generally dominate
shadow-boundary displacement. Light direction alone does not determine that
displacement.

Nighttime does not receive a fake downward sun direction. Unknown location,
diffuse illumination or weak evidence produce uncertain distributions. The
astronomical prior does not measure cloud cover, streetlights or effective
illumination. The current classifier retains the 96 existing directions
(20/35/50/65 degree elevation, 15 degree azimuth spacing); it estimates coarse
direction, not arbitrary continuous illumination or accurate near-horizon/noon
shadow length.

## Learning Without Manual Light Or Shadow Labels

Inputs are a small RGB triplet encoder, a separate encoder for registered ground
lightness/change/validity maps, and candidate direction/solar/camera metadata.
The output is a distribution, not an unqualified single light vector. Confidence
is multiplied by distribution concentration before it can influence matching.

Weak supervision combines a broad astronomical distribution with physical
projection/transition scores pooled over distinct GT tracks. It consumes all
windows and unfiltered physical hypotheses, not only the previous 2 accepted
positive masks. GT supplies object geometry, not shadow or light ground truth.
Confidence targets express evidence quality; they are not calibrated cloud or
sunlight-presence labels.

Official train scenes are divided into train/dev; official val is final eval.
Checkpoint selection uses dev weak loss only. By default two scene-held-out
crossfit models generate priors for training frames, while the final model
predicts dev/eval. Eval images/physical targets never train those checkpoints.
The saved protocol records all manifest hashes and held-out/trained scenes.

Diagnostics compare the frozen network with RGB/receiver inputs removed and
with geographic inputs removed; these are input ablations, not separately
trained baselines. A separate solar-only export supplies a deterministic
baseline. Weak-target agreement is not measured light-direction accuracy or
shadow precision. In particular, a network can learn the metadata prior without
extracting useful additional image information; the input ablations must test it.

## GT-Centered Candidates And Long Windows

`--candidate-source caster` rasterizes the union of GT caster/light footprints
across the central triplet as a search ROI. Newly proposed pixels must still
be observed dark ground with frozen BEV and flow support. The caster-conditioned
pool is built independently from the photometric `shadow_candidate` mask and
then unioned with it. Its default middle-frame proposal ratio is `0.92`; this
ratio controls proposal recall only and is not a final shadow score. Projection
never creates positive pixels by itself. `photometric` keeps the old candidate
pool.

`--temporal-radius 1/2/3/4` requests 3/5/7/9 frames, default 7, by following
consecutive cached pairs from the same scene and camera. Missing observations
are not extrapolated. Extra ground observations require visible receiver support
and the chain of cached receiver-flow support. Each remote observation is
exposure-aligned to the reference using background outside the caster ROI.

For tracks with complete GT states in the available window, remote bright/dark
observations can reveal formation/recovery absent from the central triplet.
Real trajectory fit competes with reversed, cyclically shifted and frozen
trajectories on identical observations. Long-window geometry holds the reference
light direction fixed over these few seconds. Missing complete track states or
invalid exposure supply no invented temporal evidence. Static shadow existence
retains its separate projection branch.

The recorded actual frame count, timestamps, exposure fits and complete-track
count show whether long evidence was really used. GT track states are split into
`seen / complete / partial / inconsistent`, with explicit missing-sample counts;
missing states are never extrapolated. All contributing frame/sample IDs
participate in the independent-light exclusion, including remote frames.
The segmenter/VLM training interfaces remain unchanged; no training is started
by the default refinement audit.

## HPC Workflow

Install the extra packages once in the annotation environment:

```bash
conda activate shadow_annotator
python -m pip install -e '.[lighting]'
```

Reuse the completed 3D audit without new raw-data or mature-model export:

```bash
cd /public/home/eng_zhihongli/project/Shadow
export SHADOW_PROJECT_ROOT="$PWD"
export SHADOW_CONDA_ENV=shadow_annotator
export SHADOW_REBUILD_SOURCE=/public/home/eng_zhihongli/project/Shadow_update/outputs/shadow_3d_flow_audit_20261107_132014
export SHADOW_NUSC_ROOT=/public/home/eng_zhihongli/project/data/nuscenes-full
export SHADOW_CANDIDATE_SOURCE=caster
export SHADOW_TEMPORAL_RADIUS=3
export SHADOW_RECEIVER_AUDIT_ONLY=1
bash scripts/submit_light_annotation_pipeline.sh
```

This submits one GPU light preparation/training/inference job and an afterok CPU
refinement audit. Source caches remain read-only. If SHADOW_REBUILD_SOURCE is
unset, the wrapper first submits raw preparation with SHADOW_LIGHT_BOOTSTRAP_ONLY=1:
it prepares both official splits and physical hypotheses, skipping teacher export
and annotation training even when accepted shadow counts are zero. The existing
UNIAD_* paths/environment are then required as in the main README.

Outputs include light data/model protocols, crossfit checkpoints,
`predictions/light_priors.jsonl`, solar-only priors and refined label/competition
reports. Neither training jobs nor astronomical/model tests were run locally.

Compare full, without_light_prior, photometric_candidates, three_frame,
without_mature_3d and complete_temporal. The expanded pool changes component IDs;
report candidate counts and annotation coverage, not just accepted counts or an
unmatched old/new precision claim. Freeze split/thresholds before final eval.

For the first expanded-pool audit, inspect each unique-window stage separately:
`caster_roi_pixels -> caster_roi_receiver_pixels -> caster_roi_dark_pixels ->
caster_roi_flow_pixels -> caster_roi_bev_pixels -> caster_own_component_pixels`,
then `overlap_original` and `added_observed_pixels`. These fields are written to
`window_audit.jsonl` even when a window produces no components, and aggregated
under `caster_proposals` in the competition summary.

`observed_transition_components` remains conditional on the selected
`best_projection.transitions`. Use `raw_event_components` for the unconditional
central-triplet event count, and `long_window_track_state_counts` to identify
whether a selected caster was complete, partial, inconsistent or unseen.
