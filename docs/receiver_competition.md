# 2D Appearance versus 3D Entity Explanations

## Active path

The active path now requires decoded mature-model 3D predictions in addition to
BEV features. Upgrade feature-only caches using [3D motion export](mature_3d_motion.md).
GT tracks still supply caster/light hypotheses; predicted boxes provide an
independent entity explanation. The ray/GT-volume calculation below is auxiliary
visibility geometry, not the source of the mature model's 3D output.

```text
nuScenes pairs + cached DIS + frozen UniAD/OpenDriveVLA BEV
  + decoded predicted 3D boxes / tracks / velocity
  -> observed ground receivers in a fixed three-frame BEV
  -> camera rays intersect all-class 3D box volumes
  -> ground flow / predicted 3D surface motion / illumination boundaries
  -> appearance candidates with learned receiver support
  -> real caster/light projections versus shifted-caster controls
  -> entity / material / cast-shadow / unknown explanation scores
  -> observed positive pixels, independent lit negatives, remaining IGNORE
```

"No extra 3D entity" means a locally observed ground receiver without a supported
annotated-volume intersection. It does not mean an empty learned feature, a missed
detection, an unseen LiDAR cell, or a complete dense free-space reconstruction.
The frozen BEV encoder supplies receiver representation consistency, not occupancy
truth. Both static and moving shadows remain candidates.

## Geometry and motion

`physical_ray_volume_support[T,H,W]` tests camera-to-ground segments against convex
GT box volumes, including boxes whose image occlusion extends outside their vertical
BEV footprint. SciPy ConvexHull supplies the volume half-spaces. The closest intersected
volume supplies a front box-surface proxy. Corresponding camera-time boxes map that
surface to the next frame, so its expected flow uses its own depth rather than the
box center or ground depth. Fallback annotation-time boxes exclude receivers but do
not generate motion evidence. Invalid/unavailable flow proxies are NaN.

These box surfaces are an approximation to actual object silhouettes. They are
recorded as diagnostic motion evidence, not assigned as true per-pixel object flow.
Registered ground support still uses actual observed flow and geometric expected
flow, with forward/backward reliability where available.

`receiver_boundary_formation` and `receiver_boundary_recovery` instead track dark
appearance entering/leaving the same world-ground cells. They require comparable
exposure and signed Lab-L change. They are not material flow, caster velocity, or
proof that any flow anomaly is a shadow. Static darkness keeps its separate branch.
Texture cosine uses log-lightness gradients away from visibility/shadow boundaries;
textureless patches remain unobserved, not perfectly consistent.

## Competing explanations

For every real track/light hypothesis, score the observed candidate and a surrounding
2 m receiver domain. Only observed cells contribute to the context terms:

```text
projection_score = clip(0.65 * candidate_coverage
                      +0.20 * boundary_fit
                      +0.15 * observed_dark_fraction
                      -0.35 * observed_lit_conflict, 0, 1)
```

Four control arms translate the projected caster shadow by +/-3 m on either BEV
axis. Each arm searches the same real track/light bank, including runner-up tracks;
the hardest control is compared to the real best. Controls use clipped rasterization,
not wraparound. They are negative EXPLANATIONS, never automatic negative pixels.
Dynamic real-versus-wrong-time trajectory contrast can also support discrimination.

Projection strength scales the real fit by its control advantage. Material support
uses persistent normalized darkness; shadow support combines discriminative
projection strength with registered illumination transitions. Entity support comes
from geometry; unknown support comes from unavailable receiver observations.
The full score contract is recorded in each component's evidence. These are fixed,
uncalibrated engineering scores, not trained probabilities or established quality
thresholds. Freeze/tune them on train-derived dev, never official eval.

The default factorized BEV path enables competition. A shadow winner with sufficient
margin and projection/trajectory discrimination can retain a projection-supported
positive. Otherwise it stays IGNORE, not an invented material/negative label. Near-best
competitive projections replace the old coverage-only hypotheses for positive fusion.
Ownership remains a separate decision: this change does not invent caster association
supervision or outside-BEV targets when the evidence is ambiguous.

## CPU audit first

Run after updating code on HPC. The annotation environment needs SciPy (also declared
in pyproject.toml); the Slurm job executes the new regression tests before processing.

```bash
cd /public/home/eng_zhihongli/project/Shadow_update
export SHADOW_PROJECT_ROOT="$PWD"
export SHADOW_CONDA_ENV=shadow_annotator
export SHADOW_REBUILD_SOURCE="$PWD/outputs/shadow_bev_flow_static_v2_parallel2_20261107_101103"
export SHADOW_ANNOTATION_ROOT="$PWD/outputs/shadow_receiver_audit_$(date +%Y%m%d_%H%M%S)"
export SHADOW_RECEIVER_COMPETITION=on
export SHADOW_RECEIVER_AUDIT_ONLY=1
# Set this to the decoded-object export root from docs/mature_3d_motion.md.
export SHADOW_MATURE_OBJECT_ROOT=/path/to/decoded_objects
mkdir -p logs
sbatch slurm/rebuild_bev_flow_labels.slurm
```

The job reuses raw pairs, DIS and mature BEV caches. It recomputes receiver evidence
and labels, then stops BEFORE training-data export. It does not allocate a GPU,
retrain either annotator, or fail merely because competition accepts zero shadows.
Do not use `--reuse-analysis` on old windows: they lack ray-volume and boundary
evidence. Zero accepted labels are a diagnostic result to inspect, not a reason to
automatically loosen thresholds or proceed to training.

Inspect each split's:

- `bev_flow/windows/*.npz`: separate geometry, flow and boundary evidence.
- `labels/decisions.jsonl`: hypotheses, old baseline decision, competition evidence.
- `competition/components.jsonl`: all candidate scores and export decisions.
- `competition/summary.json`: per-scene coverage and explanation counts.
- `competition/contact_sheet.jpg`, `competition/review/*.jpg`: clean/outlined crops.

The full preparation pipeline enables the same competition by default.
`--receiver-competition score-only` records scores without changing the legacy
projection labels. `off` reproduces coverage-only label selection on the same
new receiver population. Do not directly compare its counts with an old population
that used different geometric exclusions.

## Baselines and evaluation

Each candidate records four baseline scores: local appearance darkness, geometry/flow
supported darkness, best projection coverage, and shadow-versus-other explanation
margin. These are implemented internal baselines, not reproductions of SILT, LISA or
ViShadow. Their checkpoints and the failed local segmenter/VLM are not used as teachers.

### Optional mature shadow baseline

An existing independently pretrained SILT or another shadow detector can supply an
additional appearance prior. Run that model in its own environment on the exported
`pairs/images/*.png`, retaining the exact working image coordinates. Export one NPZ
per frame with a finite HxW float `probability` array in [0,1], and a JSONL manifest:

```json
{"sample_data_token":"camera-token","image_sha256":"exported-image-sha256","model_id":"SILT/checkpoint-identity","probability":"probabilities/camera-token.npz"}
```

Set `SHADOW_BASELINE_MANIFEST=/absolute/path/baseline.jsonl` before CPU rebuild or
full preparation. No external repository/checkpoint is downloaded or silently chosen.
Missing frames remain NaN. Image hash/shape mismatches fail rather than moving scores
to the wrong pixels. Model identity, image and probability fingerprints are retained.

The candidate-versus-context baseline contrast adds at most a 0.2-weight appearance
term, multiplied by discriminative physical projection strength. Thus even a baseline
score of 1 cannot create a positive without receiver/projection evidence. The plain
external model score is also reported as its own baseline; missing scores do not
silently reduce its evaluation population. Use `without_shadow_baseline` or
`SHADOW_APPEARANCE_PRIOR_WEIGHT=0` to remove its auxiliary term. Without supplied
probabilities, all existing comparisons run normally and no external prior is claimed.

After rebuilding a cache, run the fixed-window label ablations:

```bash
python scripts/run_bev_flow_ablations.py \
  --source-root "$SHADOW_ANNOTATION_ROOT" \
  --output-root outputs/shadow_receiver_ablations \
  --variants full projection_coverage competition_score_only geometry_only shuffled \
  --labels-only
```

`full` versus `projection_coverage` isolates the competition decision; score-only
provides the same score population without its label decision. Other feature arms
test learned-BEV support. The audit does not measure precision by comparing with
the same teacher's acceptances.

Optionally supply a fixed independent region reference, JSONL rows with
`id`, `scene_name`, and `shadow: YES|NO`:

```bash
python scripts/analyze_receiver_competition.py \
  --labels-root "$SHADOW_ANNOTATION_ROOT/val/labels" \
  --reference-manifest /path/to/independent_regions.jsonl \
  --output-root outputs/shadow_receiver_reference
```

This compares AUROC/AP on identical matched regions with scene-cluster bootstrap
intervals, reporting unmatched references rather than concealing them. Without an
independent reference, report explanation counts, control margins and coverage only.
These diagnostics do not establish caster accuracy, segmentation IoU, or OCC gains.

When an audit supports a training run, use a NEW root, unset
`SHADOW_RECEIVER_AUDIT_ONLY`, and submit the ordinary rebuild pipeline. Do not resume
the collapsed segmenter or use the previous poorly discriminating adapter to relabel
the new data. Segmentation optimization and VLM output-schema repair remain separate
experiments, not prerequisites for examining this physical teacher.
