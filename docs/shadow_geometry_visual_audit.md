# Frozen Geometry And Observation Audit

This CPU workflow uses the completed transition-stage outputs. It does not change
scores, thresholds, visibility masks, the DIS engine, labels or training. No new
BEV export is needed. It runs a sampling-only audit first and then renders real
RGB/GT/flow examples rather than illustrative images.

```bash
cd /public/home/eng_zhihongli/project/Shadow
export SHADOW_PROJECT_ROOT="$PWD"
export SHADOW_CONDA_ENV=shadow_annotator
export SHADOW_REBUILD_SOURCE="$PWD/outputs/shadow_transition_stages_20261107_211522"
export SHADOW_ANNOTATION_ROOT="$PWD/outputs/shadow_geometry_audit_$(date +%Y%m%d_%H%M%S)"
export SHADOW_TEMPORAL_RADIUS=3
export SHADOW_AUDIT_CASES_PER_TYPE=20
mkdir -p logs
sbatch --export=ALL slurm/audit_shadow_geometry.slurm
```

The job runs regression tests on HPC before auditing. It requests CPU only; no
light estimator, segmenter or VLM is loaded. A fresh output directory is required.

## Case Selection

Each split independently samples up to 20 cases per type, rotating across scenes:

- No visible overlap: unique failing windows.
- Flow-chain clearing: unique failing windows.
- Exposure removed darkening: components with raw darkening and zero normalized
  darkening under a valid fit. Invalid fits are NOT counted as compensation
  removing an event.
- Outside footprint: unique components, not individual light/track comparisons.
- Accepted: actually exported YES components, not merely a winning shadow score.

Categories can overlap but a component is not repeated to fill a category. If
there are three outside-footprint components or one accepted component, render
three or one and report the shortfall. Do not manufacture 20 successes.

## Outputs

`sampling/<split>/summary.json` separates projection/front/FOV failures, sparse
ground support, empty anchor receiver grids, strict binary-map interpolation,
source-image flow support and flow-chain loss. `windows.jsonl` retains per-frame
and per-edge counts, including resampled-versus-cached central visibility.

`visual/<split>/cases.jsonl` indexes case panels. Each category has a contact
sheet. Within each case:

- `panel.jpg`: clean RGB triplet, raw DIS, ground/grid overlay, GT/96-light/dark
  overlay, raw and compensated Lab residuals, and exposure-fit background.
- `dark_gt_light_overlay.jpg`: observed component contour in cyan, GT boxes in
  green, center-track arrows and a heatmap of the full caster/light search bank.
  This heatmap is search coverage, NOT shadow probability.
- `light_bank_96.jpg`: all 96 directions for one explicitly recorded nearest
  caster, including invalid projections. No accepted-best gate hides directions.
  Tiles use a shared component/GT-context crop; coverage is computed on the full
  image, not on the crop, and crop coordinates are recorded.
  `audit.json` records projections for ALL available central GT tracks.
- `receiver_visibility.jpg`: every actual window frame. Green points are
  supported inside the cached receiver; cyan points hit ground support outside
  that receiver; red points have no strict sampled ground support. Ground-mask
  pixels are tinted green. This distinguishes FOV from sparse support and cached
  receiver exclusion.
- `flow_stages.jpg`: raw DIS, raw finite mask, forward/backward validity,
  geometric consistency, texture, receiver visibility, pairwise support and
  chain support. Titles distinguish source-image pixels from the BEV grid.
  `flow_stages.npz` retains these exact arrays; geometric-consistency threshold
  comes from the cached flow protocol rather than a newly chosen audit threshold.
- `event_projection_alignment.jpg`: when a complete focus track exists, show
  observed changes (cyan), predicted footprint differences (amber) and matches
  (green). Its direction maximizes event overlap for diagnosis only; it is NOT
  a learned or calibrated light estimate.
- `exposure.jpg` and `audit.json`: affine scale/offset, residuals, actual fit
  background coordinates, component/background overlap and raw/normalized
  component delta quantiles. Invalid exposure is explicitly marked.
- `observations.npz`: fixed world grid, per-frame visibility, cached central
  visibility, receiver mask, chain support, Lab-L and observed onset/recovery.

`visual/summary.json` reports requested/available/rendered cases and complete
stage accounting, including unstable-exposure and other states. Do not silently
omit a category when reconciling counts.

## Physical Definitions To Check

The bank contains world-frame downward propagation rays, not vectors pointing
toward the sun. Azimuth zero is world +x, positive toward +y; reported angles are
degrees. The current footprint projects the 3D bounding volume ALONG a ray onto
the receiver plane. It is not the vertical box occupancy footprint or the true
vehicle silhouette. RGB projections already use the exported working-resolution
camera matrices with pixel-center correction.

The current cached receiver requires support in the central THREE views. Each
remote frame has its own visibility overlap and intervening flow chain; one
unavailable remote frame does not by itself zero all other remote comparisons.
This audit does not silently replace that rule with an all-window intersection
or a new pairwise policy.

Inspect the overlay and 96-direction atlas first. If no direction covers an
observed component, inspect candidate location, omitted caster types, receiver
plane and timing before fitting a light network. If directions cover the dark
region but not its signed change, inspect complete track states and the per-frame
event/projection panels. None of these audits measures shadow precision without
an independent reference.
