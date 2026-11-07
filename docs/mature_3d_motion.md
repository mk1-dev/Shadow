# GT Caster Motion, Image Flow and Mature 3D Predictions

For the completed export's next CPU-only step, use [transition evidence v2](shadow_transition_v2.md).
It reuses the existing 3D audit and does not require exporting this cache again.

## Information Flow

The BEV-feature/image-flow comparison is retained, not replaced by 3D boxes.
`attach_cached_bev` samples frozen spatial features at the same world receivers
in all three frames; `select_bev_flow_receivers` combines their temporal cosine
with observed ground-flow support to select photometric candidates. GT
caster/light explanations then evaluate those candidates. Decoded entity motion
adds a third, complementary explanation. A BEV feature is not an occupied/free
label, so feature/appearance difference alone does not establish a shadow.

The main shadow explanation starts from known nuScenes GT objects, not from a
detector miss:

1. Match the same GT caster across three camera-time states, including ego-motion.
2. Project its 3D box onto observed ground receivers under the same candidate light
   direction at all three times. Search the existing direction bank.
3. Register image observations on those ground receivers with geometry and DIS.
   Compare darkness, signed formation/recovery and persistent appearance.
4. Compare the real trajectory/light explanation against wrong-time, frozen and
   spatially shifted explanations. Keep existence and unique ownership separate.
5. Independently compare decoded mature-model entity motion with observed optical
   flow, to distinguish object surface appearance from receiver illumination.

Steps 1-4 use `physics_labels.py` and `receiver_explanations.py`. The new step 5
is implemented in `mature_objects.py` and connected by
`analyze_bev_flow_discrepancy.py`; it does not replace the GT caster search.

The informative shadow evidence is a ground appearance whose evolution is
explained by a caster/light projection, rather than by a moving solid surface or
a persistent dark material. Optical flow supports correspondence; it is not
required to equal either caster velocity or shadow-boundary displacement.
Static shadows retain their separate projection/persistence branch.

## Decoded 3D Contract

The mature forward pass runs `return_loss=False` in ordered scene replay. Export
reads `result_track.boxes_3d`, `scores_3d`, `labels_3d`, `track_ids` and the decoded
`*_det` branch when available. It does not treat `bev_embed` as occupancy.

Each `<sample>.objects.json` stores model corners, gravity center, confidence,
class, predicted track ID and velocity in world coordinates. Model LiDAR boxes
have bottom-center z; projection uses the model's own corners and gravity center.
The LIDAR_TOP calibrated sensor and its own ego pose provide the world transform.
Camera timestamp offsets are compensated using predicted velocity when available.

For matching predicted tracks, visible box-face homographies map source pixels
to the next camera. A depth buffer selects the nearest predicted face. This is
projected box-surface motion, not exact object-silhouette flow and not the box
center's displacement copied across the object. Occluded/unmatched faces and
unreliable DIS observations have unavailable motion, not zero residual.

At common valid pixels the analysis records:

```text
F_object : expected flow from predicted 3D track surfaces and camera poses
F_ground : expected static-ground flow from geometry and camera poses
F_image  : measured DIS flow
e_object = norm(F_image - F_object)
e_ground = norm(F_image - F_ground)
```

Entity confidence and relative motion fit contribute to the entity explanation.
Lack of predicted boxes alone never creates a shadow positive. Outside the
model's spatial domain, prediction absence is unknown, not free space. The current
stage-1 checkpoint supplies detection/tracking, not a dense occupancy head.
GT tracks remain weak supervisory geometry; they are not shadow masks.

## Upgrade Existing HPC Caches

Feature-only `.npz` files cannot recover decoded heads. Reuse pairs, DIS and BEV
features, but run one ordered mature inference export. No annotation retraining
or raw-data rebuilding is needed for this audit.

```bash
cd /public/home/eng_zhihongli/project/Shadow_update
export SHADOW_PROJECT_ROOT="$PWD"
export SHADOW_CONDA_ENV=shadow_annotator
export UNIAD_CONDA_ENV=drivevla
export UNIAD_ROOT=/public/home/eng_zhihongli/project/OpenDriveVLAre
export UNIAD_CHECKPOINT=/public/home/eng_zhihongli/project/OpenDriveVLA_train/checkpoints/uniad_base_track_map.pth
export UNIAD_TRAIN_ANN_FILE=/actual/path/to/nuscenes_infos_temporal_train.pkl
export UNIAD_VAL_ANN_FILE=/actual/path/to/nuscenes_infos_temporal_val.pkl
export SHADOW_NUSC_ROOT=/public/home/eng_zhihongli/project/data/nuscenes-full
export SHADOW_REBUILD_SOURCE="$PWD/outputs/shadow_bev_flow_20261107_0715"
export SHADOW_MATURE_OBJECT_ROOT="$PWD/outputs/shadow_mature_3d_$(date +%Y%m%d_%H%M%S)"
export SHADOW_ANNOTATION_ROOT="$PWD/outputs/shadow_3d_flow_audit_$(date +%Y%m%d_%H%M%S)"
export SHADOW_MATURE_OBJECT_MODE=required
export SHADOW_RECEIVER_COMPETITION=on
export SHADOW_RECEIVER_AUDIT_ONLY=1
mkdir -p logs
EXPORT_JOB=$(sbatch --parsable slurm/export_mature_objects.slurm)
sbatch --dependency=afterok:"$EXPORT_JOB" slurm/rebuild_bev_flow_labels.slurm
```

Use the same model config/checkpoint as the existing feature cache and the actual
train/val infos paths from the working mature environment. Export uses one GPU;
rebuild runs on CPU and stops before dataset creation or training. The export
script defaults to object confidence >= 0.25 (`SHADOW_MIN_OBJECT_SCORE`).

Full new preparations through `prepare_shadow_annotation.slurm` export features
and decoded objects in the same forward pass. An existing trusted full inference
result cache can instead be converted by `import_uniad_bev_cache.py` with
`--inference-cache`; train-forward GT-matched result caches are not this input.

## Inspect Before Training

Under each rebuilt `<train|val>/bev_flow/` inspect:

- `motion_review/*.jpg`: image, projected entities, predicted surface-flow arrows,
  observed flow at identical pixels, relative flow errors, registered brightness.
- `image_motion/*.npz`: full working-resolution predicted entity depth/confidence,
  expected and observed flow, residuals and motion fit for both frame intervals.
- `windows.jsonl`: per-window object-cache hashes, model provenance and counts.
- `windows/*.npz`: sampled common-ground entity probability, motion residuals and
  inference domain alongside the existing appearance/light/flow arrays.

`<train|val>/competition/` reports per-component explanations, the image-only,
geometry/flow and mature-3D-unexplained baselines, and links to motion evidence.
The `without_mature_3d` ablation removes this new prediction evidence while keeping
the same GT caster, light, DIS and feature inputs. This isolates its increment.

Check projected boxes against images, available tracked-motion coverage, real
versus control trajectory fit, static/moving survival and accepted-label yield.
More visible motion contrast is a hypothesis to measure, not a claimed result.
Unique caster association and downstream annotation quality still need HPC tests.

Legacy GT-only geometry experiments must explicitly use
`--mature-object-mode off` / `SHADOW_MATURE_OBJECT_MODE=off`. Default `required`
prevents feature-only caches from being presented as decoded 3D comparison.

New regression tests live in `tests/test_mature_objects.py`; the CPU rebuild's
HPC test step includes them. They have not been run locally for this update.
