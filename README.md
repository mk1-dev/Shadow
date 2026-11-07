# Shadow-OCC

## Current Research Scope: Physics-Guided Shadow Annotation

The active experiment is **BEV-Flow-Guided Shadow Annotation**. Reuse the mature
UniAD/OpenDriveVLA BEVFormer as a frozen encoder, align its spatial features with
three-frame physical BEV and image flow, and generate receiver-grounded shadow
regions for the existing segmenter and unified VLM. OCC training remains the
next stage. Existing 3D object GT supplies the physical caster tracks. Decoded
mature-model 3D predictions provide a separate entity/motion explanation, not
shadow ground truth and not a replacement for the GT caster trajectories.

The executable BEV path is:

```text
existing UniAD/OpenDriveVLA -> frozen BEV + predicted 3D boxes/tracks/velocity
nuScenes 3D tracks / ground / poses -> physical BEV and geometric expected flow
three camera frames / optical flow -> aligned brightness and observed flow
                                    |
                   common world receiver / fixed BEV grid
                                    |
        BEV components -> GT caster/light trajectory matching
        predicted 3D surface motion <-> measured image optical flow
                                    |
       entity / material / shadow / unknown explanation competition
                                    |
           observed image trimaps -> segmenter + unified VLM
```

Use `scripts/export_mature_bev.py` in the existing OpenDriveVLA environment;
Shadow does not copy or retrain that encoder. The CPU join and label builder run
in the annotation environment. See [BEV redesign and HPC commands](docs/shadow_track_contrast_v2.md#bev-flow-redesign).

The [3D motion workflow](docs/mature_3d_motion.md) connects decoded mature-model
boxes/tracks to image-space motion, alongside GT caster/light trajectory matching.
The [receiver competition workflow](docs/receiver_competition.md) combines these
with ground-flow and illumination-boundary evidence. Feature-only caches need
one mature-model inference export first; then audit on CPU before training.

## Current Audit Priority

Keep physical scores and model training frozen while running two independent
checks: [receiver geometry consistency](docs/receiver_geometry_consistency.md)
on all source windows, and [zero-shot RGB scene observability](docs/shadow_scene_observability.md)
on clean temporal images. The latter prioritizes useful observation conditions;
it does not provide shadow truth, masks, caster IDs or geometry validity.
Cloudy/weak/unknown windows are retained, and static shadows are not excluded.
Use `slurm/tag_shadow_observability.slurm` for the existing audit-case pilot;
it trains no model and compares the already frozen physical results by stratum.

## Cross-Space Discovery

The default candidate path is now `--discovery-mode discrepancy`. First build
ground/GT/decoded-3D motion explanations, then compare projected motion and
receiver appearance with observed flow/RGB on the entire valid receiver grid.
The new branch does not require a local-darkness threshold, a component seed,
or an existing appearance proposal. Its thresholded discrepancy map directly
adds candidates: `appearance UNION discrepancy UNION caster-guided`.
The previous `cross-space` appearance-first mode remains an explicit baseline;
`hard-intersection` reproduces the earlier BEV/flow veto policy.

Frozen BEV cosine measures spatial representation stability. Raw forward/backward
flow validity is stored independently of ground-flow registration support, so a
trusted `observed flow - geometric flow` residual can be measured without first
requiring that residual to be small. Missing flow is unknown, not a large anomaly.
BEV stability and reliable flow residuals now affect new proposal selection,
not just an auxiliary score. Ground motion and available tracked box-surface
motion are compared with measured flow; an entity model must explain that flow
better than the ground model to suppress discrepancy. Unavailable model coverage
is recorded, not treated as observed free space. Strong, geometrically aligned
brightness changes can still propose candidates when DIS is unavailable or when
ground-texture flow matches geometry. Flow error without brightness evidence
does not create proposals. The score does not override physical acceptance.
The hypothesis that BEV is relatively insensitive to shadows remains experimental.

GT/light ROIs can extend candidates using actually observed dark ground, without
requiring learned-BEV or DIS support. Physical matching can use known geometric
correspondence without pretending that failed optical flow was valid. Long-window
events use independently visible reference/remote receiver pairs in this mode;
the original flow-chain support is still reported separately. Existing central
three-frame visibility and ground-cache consistency requirements are unchanged.

The explicit `hard-intersection` arm reproduces the previous candidate policy.
New `discrepancy_*` arrays separate the baseline, active branch and newly added
pixels. `proposal_source_bits` records appearance=1, discrepancy=2, caster=4;
after caster expansion, bits=2 identify exclusively discrepancy-sourced pixels.
`cross_space_review/*.jpg` shows baseline/new/novel proposals and entity-motion
explanations. The competition summary reports the discovery funnel, including
components containing new proposals. That is not novel-pixel precision.
Weak proposals do not lower the existing physical transition brightness threshold.
Lit negative-label rules and caster/light competition scores remain unchanged.

On HPC, first run the focused tests, then the CPU-only audit rebuild using existing
pairs, DIS and mature BEV/3D caches. No image/flow/encoder re-export or training is
needed. The following roots are examples from the existing experiment:

```bash
python -m pytest -q tests/test_discrepancy_proposals.py tests/test_cross_space.py tests/test_cached_bev.py tests/test_bev_flow.py tests/test_bev_flow_ablations.py tests/test_caster_window.py tests/test_rebuild_shadow_analysis.py
export SHADOW_REBUILD_SOURCE=/public/home/eng_zhihongli/project/Shadow/outputs/shadow_bev_flow_20261107_0715
export SHADOW_MATURE_OBJECT_ROOT=/public/home/eng_zhihongli/project/Shadow/outputs/shadow_mature_3d_20261107_124905
export SHADOW_ANNOTATION_ROOT=/public/home/eng_zhihongli/project/Shadow/outputs/shadow_discrepancy_v1
export SHADOW_DISCOVERY_MODE=discrepancy
export SHADOW_RECEIVER_AUDIT_ONLY=1
export SHADOW_REUSE_SOURCE_ANALYSIS=0
sbatch slurm/rebuild_bev_flow_labels.slurm
```

The ablation runner defaults to `hard_intersection`, `appearance_only`,
`discrepancy_proposal`, and `bidirectional_proposal`. Compare `appearance_only`
against `discrepancy_proposal` to change only active discovery, then compare
`discrepancy_proposal` against `bidirectional_proposal` to add GT-guided search.
`cross_space` preserves the previous baseline including caster-guided expansion;
compare it against `full` for a matched caster-enabled discovery control.
`discrepancy_geometry` and `discrepancy_shuffled` test the learned BEV reference.
Three proposal-only settings are exposed: `--discrepancy-minimum-score` (0.35),
`--discrepancy-minimum-light-change` (1.5 Lab-L), and `--discrepancy-flow-scale`
(3 working-image pixels). Freeze them on dev before held-out evaluation.
Cached reselection is supported, but older NPZs without raw flow validity report
that evidence as unavailable. A rebuild from cached pairs/flow recovers it.
Do not use `--reuse-source-analysis` to switch an old hard-intersection cache to
discrepancy: that option intentionally preserves its original candidate policy.

## Frozen A/B/C/D Discovery Audit

Use `configs/discrepancy_abcd.json` for matched experiments. A/B/C/D share the
caster search, geometric correspondence, signed brightness verification,
physical score thresholds, light bank, temporal radius and candidate budget.
A is the old candidate policy with this shared verifier, not a claim to reproduce
every setting of a historical run. `D_no_entity_motion` removes only the GT/predicted
surface-motion inputs to the proposal calculation; geometry exclusions and all
decoded 3D evidence used by later competition remain intact.

`C_bev_shuffled` permutes spatial features. `C_gt_reverse` changes verification
trajectory time order while holding the centre state and candidate discovery fixed;
it is a wrong-time control, not a new set of positive training labels. Static
tracks may be unchanged by this control and should be interpreted separately.
The discrepancy formula and physical scoring formulas are not changed by these
switches. Recipes, code hashes and source-window hashes are written to each arm.

The frozen audit needs source windows rebuilt with `observed_flow_valid_by_interval`.
Old caches missing that field cannot test the entity-motion ablation and are
rejected before starting this particular experiment. Rebuild once from cached
pairs/flow/features using the command above; no encoder/flow re-export is needed.

```bash
export SHADOW_ABLATION_SOURCE="$PWD/outputs/shadow_discrepancy_v1"
export SHADOW_ABLATION_ROOT="$PWD/outputs/shadow_discrepancy_abcd_v1"
export SHADOW_ABLATION_CONFIG=configs/discrepancy_abcd.json
export SHADOW_ABLATION_VARIANTS="A_hard B_cross C_discrepancy D_no_entity_motion C_bev_shuffled C_gt_reverse"
export SHADOW_ABLATION_AUDIT_ONLY=1
sbatch -p AMD slurm/run_bev_flow_ablations.slurm
```

This CPU audit does not build teacher/segmenter/VLM training datasets. Inspect
`comparison/summary.json`, each split's `competition/summary.json`, and
`labels/transition_diagnostics.jsonl`. Source-bit histograms distinguish the seven
appearance/discrepancy/caster combinations. Paired C/D masks report pixels removed
by entity-motion explanations in the branch and cached appearance/discrepancy
union before caster expansion; this does not
prove those removed pixels were false shadows. GT reverse should leave cached
proposal masks unchanged. Over-budget windows are explicitly skipped, not silently
truncated. Positive-window coverage is reported separately from teacher pixel
coverage, which remains unavailable when no teacher dataset is built.

## Default HPC Pipeline

The [geo-temporal light workflow](docs/geo_temporal_light.md) adds a trainable
RGB/ground-change light estimator with geographic UTC sunlight priors,
GT-centered candidate search and up to nine cached temporal observations.
`scripts/submit_light_annotation_pipeline.sh` reuses the completed audit, trains
the light head on one GPU, then performs CPU label refinement without starting
segmenter/VLM training. Model weights still need HPC training and validation.

The [transition-evidence update](docs/shadow_transition_v2.md) addresses the
2/0-positive audit: persistence is no longer automatic material evidence,
single-sided signed changes can support caster matching, and identical-candidate
3D counterfactuals separate scoring effects from entity-motion effects. Its CPU
replay reuses the completed 3D audit without another GPU export.

Internal experiments are available in [BEV-flow ablations](docs/bev_flow_ablations.md):
shared-input feature/flow gates, genuine no-DIS geometry registration, current-only
encoder export, physical-explanation controls, and training-label controls. Reuse
the completed preparation caches; do not regenerate raw nuScenes for each arm.

`slurm/prepare_shadow_annotation.slurm` now builds everything from raw nuScenes;
it does not read `outputs_temporal`, `shadow_labels`, old segmenters or adapters.
Frozen BEV temporal cosine (default >= 0.3) AND registered receiver flow actually
filter the photometric candidates before caster/light matching.

The new `bev-flow-annotation-v2` teacher exports positive observed shadows and
weak negative endpoints and compact persistently lit neighbouring receivers outside
competing shadow footprints. Static dark receivers can support shadow existence
through discriminative projection support even without vehicle motion or reliable exposure compensation;
unique caster ownership still uses motion/time contrast. Other pixels remain IGNORE. Labels from the same physical frame are
merged; contradictory pixels are ignored. Both classes use original camera
images. VLM examples have the same five-view layout for both classes; boundary
frames without two adjacent observations contribute only to segmentation.

Set actual HPC paths and use the existing working mature-model environment:

The [static-shadow rebuild workflow](docs/static_shadow_rebuild.md) fixes the
zero-label failure of job 367774 and reuses existing flow/BEV caches on CPU.

```bash
cd /public/home/eng_zhihongli/project/Shadow_update
export SHADOW_PROJECT_ROOT="$PWD"
export SHADOW_CONDA_ENV=shadow_annotator
export UNIAD_ROOT=/path/to/OpenDriveVLAre
export UNIAD_CONDA_ENV=your_existing_uniad_env
export UNIAD_CHECKPOINT=/path/to/uniad_base_track_map.pth
export UNIAD_TRAIN_ANN_FILE=/path/to/nuscenes_infos_temporal_train.pkl
export UNIAD_VAL_ANN_FILE=/path/to/nuscenes_infos_temporal_val.pkl
export SHADOW_VLM_MODEL=/path/to/Qwen3-VL-8B-Instruct
bash scripts/submit_bev_flow_pipeline.sh
```

Submission order:

```text
prepare raw geometry / flow / mature BEV / teacher / datasets [1 GPU]
  -> segmenter training [2 GPUs] -> segmenter eval [1 GPU]
  -> unified VLM training [1 GPU] -> dev calibration -> frozen eval [1 GPU each]
                                 -> full annotation generation eval [1 GPU]
```

Training branches share data but neither model supplies the other's training
targets. Each run gets a new `outputs/shadow_bev_flow_<timestamp>` root. The
prepare job runs the new regression tests on HPC, then processes train and val
separately; train scenes are further divided into train/dev by the data builder.
Defaults are a pilot of 1200 pairs per official split, at most 24 per scene,
using front/front-left/front-right cameras. Set `SHADOW_PAIRS_PER_SPLIT`,
`SHADOW_PAIRS_PER_SCENE`, `SHADOW_PAIR_ATTEMPTS` and `SHADOW_CAMERAS` to expand it.

Inspect `<root>/<train|val>/bev_flow/summary.json`, `labels/summary.json` and
`teacher/protocol.json` first. These report actual candidate survival, accepted
shadows, lit endpoints and conflict counts. `data/data_check.json` and
`joint/data_check.json` report the data that training will actually consume.
OCC is not part of this run.

## Archived Annotation And OCC Workflows

The VLM is the **GT-constrained shadow annotator**. Its single annotation contains
the shadow region decision and caster association, including a possible
outside-BEV caster when supported. Segmentation and temporal light evidence
provide its observations. The main workflow is
[docs/shadow_joint_annotator_hpc.md](docs/shadow_joint_annotator_hpc.md).

The 2026-11-03 supervision update separates shadow evidence from caster
associations, retains ambiguous regions, adds reviewed segmentation proposals
and scene cross-fit manifests. Rebuild data in a new output root; see the
supervision-policy update and HPC checks in the main workflow document.

Low-level shadow segmentation, temporal evidence and legacy region-verification
details are documented in [docs/shadow_annotation_hpc.md](docs/shadow_annotation_hpc.md).

Runnable nuScenes + Occ3D code for testing whether temporal shadow geometry can
provide a weak 3D cue for BEV occupancy, especially at long range.

The implemented main pipeline is:

```text
nuScenes images + calibration + ego pose + LIDAR_TOP
  -> dark-region candidates
  -> ground projection to a shared BEV grid
  -> ego-motion alignment and temporal tracking
  -> root / tip / direction / length / reliability labels
  -> temporal BEVFormer-style encoder
  -> Shadow Geometry Head
  -> U_shadow-gated fusion
  -> binary and semantic occupancy heads
  -> Occ3D-style evaluation
```

This repository contains a self-contained BEVFormer-style implementation. It
does not require `mmcv`, `mmdet3d`, or an external BEVFormer checkout.

## 1. Data boundary

The main experiment keeps pseudo-label construction separate from benchmark
evaluation:

```text
Shadow pseudo labels: nuScenes images + LIDAR_TOP + calibration + ego pose
OCC supervision:      Occ3D labels
Final evaluation:     Occ3D labels
```

The default `data.geometry_source` is `nuscenes_lidar`. The `occ3d` geometry
source is an oracle/diagnostic mode and should not be used as the main result,
because it uses Occ3D geometry while constructing shadow candidates.

Current implementation status:

| Module | Status |
|---|---|
| Dark regions and multi-camera BEV projection | Implemented |
| Ego-motion BEV alignment and temporal fusion | Implemented |
| Component filtering and basic tracking | Implemented |
| Root, tip, direction, length, reliability | Implemented |
| `P_shadow/P_root/P_tip/P_caster/V_shadow/H_prior/U_shadow` | Implemented |
| U_shadow-gated OCC fusion | Implemented |
| OCC and semantic OCC training | Implemented |
| Temporal caster consistency loss | Connected to training |
| Per-range Occupancy mIoU | Implemented |
| Two-GPU training and evaluation | Implemented |
| SAM2/SAM2Long | Not included |
| GroundingDINO | Not included; caster defaults to a root proxy |
| DINOv2 feature consistency | Not included |
| Official Occ3D leaderboard evaluator parity | Not included |

`rendered_shadow_prior` is an approximate fixed-sun BEV ray-cast prior, not a
full differentiable renderer.

## 2. Repository layout

```text
configs/
  default.yaml                 Full Shadow-OCC configuration
  baseline.yaml                Camera-only OCC baseline
  ablation_no_gate.yaml        Fuse shadow without U_shadow gating
  ablation_no_temporal.yaml    Disable temporal consistency loss
  ablation_no_height.yaml      Remove H_prior from OCC fusion
  ablation_no_render.yaml      Remove rendered-shadow loss
  oracle_caster.yaml           Annotation-based caster diagnostic upper bound

scripts/
  make_shadow_labels.py        Generate pseudo shadow labels
  train_bevformer_occ.py       Training entry point
  evaluate_bevformer_occ.py    OCC evaluation entry point
  evaluate_shadow_height_corr.py  Shadow geometry precheck
  evaluate_shadow_labels.py       Pseudo-label/component statistics
  evaluate_shadow_annotations.py  Optional manual shadow-mask evaluation
  compare_experiments.py           Matched experiment table and deltas
  summarize_seeds.py               Mean/std over repeated seeds

shadow_occ/
  data/                        nuScenes, Occ3D, and LiDAR adapters
  geometry/                    BEV grid, projection, transforms
  shadow/                      Dark masks, labels, tracking, rendering
  models/                      BEVFormer-style model, head, losses
  metrics/                     Occupancy, Semantic, RayIoU, height metrics
  train.py                     Training implementation
  evaluate.py                  Evaluation implementation

tests/                         Synthetic smoke test
```

## Local nuScenes-mini annotation

`run_local_annotation.py` builds a six-camera keyframe manifest from a local
nuScenes-mini directory and launches a dependency-free browser annotation UI:

```bash
python run_local_annotation.py \
  --dataroot /path/to/nuscenes-mini \
  --work-dir /path/to/shadow_mini_annotation
```

The UI supports image decisions, camera filters, bounding boxes, freehand
multi-stroke shapes, zoom/pan, autosave, resume, and latest-record JSONL export.
Use boxes for first-pass screening and freehand shapes for accepted shadows or
ambiguous boundaries. See
[LOCAL_ANNOTATION_README.md](LOCAL_ANNOTATION_README.md) for Windows commands,
output schema, and optional camera/scene/frame controls.

## 3. Data preparation

### nuScenes

Use the full `v1.0-trainval` nuScenes release:

```text
/data/nuscenes/
  samples/
  sweeps/
  maps/
  v1.0-trainval/
```

The metadata must describe the six cameras and `LIDAR_TOP` sensor.

### Occ3D-nuScenes

Occ3D is required for OCC training and final evaluation. The loader supports:

```text
/data/Occ3D-nuScenes/gts/{scene_name}/{sample_token}/labels.npz
/data/Occ3D-nuScenes/{scene_name}/{sample_token}/labels.npz
/data/Occ3D-nuScenes/{sample_token}/labels.npz
/data/Occ3D-nuScenes/labels/{sample_token}.npz
```

Each `labels.npz` must contain one of:

```text
semantics, semantic, labels, voxel_label
```

Optional visibility arrays are supported: `mask_camera` and `mask_lidar`.
The default label convention is semantic classes `0..16`, free `17`, ignore
`255`.

`annotations.json` is not required by the main pipeline. nuScenes metadata and
Occ3D `labels.npz` are sufficient. Sample annotations are disabled by default.

## 4. Ubuntu installation

Install a PyTorch build compatible with the server CUDA driver, then install
this package:

```bash
cd /public/home/USER/project/Shadow
source /public/home/USER/anaconda3/etc/profile.d/conda.sh
conda activate shadow_occ
pip install -e .
```

Verify the environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -m pytest -q
python scripts/train_bevformer_occ.py --smoke
```

For a Slurm job on two RTX 4090 GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
unset PYTORCH_CUDA_ALLOC_CONF
```

Do not use the Windows paths in `default.yaml` on Linux. Pass Linux paths on
the command line as shown below.

## 5. Configuration

Important defaults in `configs/default.yaml`:

```yaml
data:
  version: v1.0-trainval
  geometry_source: nuscenes_lidar
  num_frames: 3
  occ_frame: ego

model:
  use_shadow: true
  use_shadow_fusion: true

train:
  batch_size: 1       # per-GPU batch size under DDP
  num_workers: 0
  epochs: 12
```

The default BEV range is `[-40, 40]m x [-40, 40]m` with `0.4m` resolution,
which gives a `200 x 200` BEV map. The default OCC grid has 16 height bins.

With empty `data.scene_names`, training uses the official nuScenes `train`
scenes and evaluation uses the official `val` scenes. An explicit `--scene`
is useful for debugging, but is not a valid paper split.

## 6. Generate shadow labels

The main path uses LiDAR and images. It does not use Occ3D to create the
pseudo shadow labels.

Small check:

```bash
python scripts/make_shadow_labels.py \
  --config configs/default.yaml \
  --dataroot /data/nuscenes \
  --occ-root /data/Occ3D-nuScenes \
  --geometry-source nuscenes_lidar \
  --output-root outputs_quick \
  --scene scene-0001 --max-samples 100 --num-workers 1
```

Full generation:

```bash
python scripts/make_shadow_labels.py \
  --config configs/default.yaml \
  --dataroot /data/nuscenes \
  --occ-root /data/Occ3D-nuScenes \
  --geometry-source nuscenes_lidar \
  --output-root outputs --num-workers 2
```

The process pool is scene-level. Each worker loads nuScenes tables and images,
so use 1 or 2 workers when memory is limited; 10 or 15 workers can trigger
OOM kills. With multiple workers, the progress bar is sample-level and shows
active scene progress. Per-scene state is also written under
`outputs/label_progress/`, which is useful for diagnosing a slow or failed
scene.

Output:

```text
outputs/shadow_labels/{scene_name}/{sample_token}.npz
outputs/shadow_labels/{scene_name}/{sample_token}.json
outputs/run_config.json
```

The label files contain `shadow_mask`, root/tip/caster heatmaps, direction,
length, reliability, `rendered_shadow_prior`, temporal light confidence, and
component metadata. The default caster is a root-based proxy, not a
GroundingDINO detection.

The label generator now uses the previous frame when available. It warps the
previous camera image to the current image on the calibrated ground plane,
performs robust local exposure compensation, computes Lab `L/a/b` changes,
and combines the temporal candidate with the current HSV dark mask. The
component tracker measures caster/root stability and shadow-tip motion in
global coordinates. Each label JSON also records image-ground round-trip
reprojection statistics.

To summarize these diagnostics after label generation:

```bash
python scripts/evaluate_shadow_projection.py \
  --label-root outputs/shadow_labels \
  --output-root outputs
```

This writes `outputs/metrics/shadow_projection_metrics.json`. Re-run
`make_shadow_labels.py` after enabling or changing these settings; old label
files do not contain temporal-difference fields.

Summarize component and temporal-label statistics:

```bash
python scripts/evaluate_shadow_labels.py \
  --label-root outputs/shadow_labels \
  --output-root outputs
```

This reports nonempty-label ratio, component count, length/elongation,
caster stability, tip motion, temporal selection ratio, and mean light
difference. These are diagnostics for generated pseudo labels, not ground
truth shadow accuracy.

## 7. Shadow geometry precheck

```bash
python scripts/evaluate_shadow_height_corr.py \
  --config configs/default.yaml \
  --label-root outputs/shadow_labels \
  --output-root outputs
```

This writes `outputs/metrics/shadow_height_corr.json` and reports component
shadow-length versus root-neighborhood height correlation for 0-20m, 20-40m,
40-60m, and all ranges.

For LiDAR-generated labels this is a LiDAR pseudo-height precheck. It is not
the final Occ3D height metric. After model training, use `HeightPrior_corr`
to compare predicted `H_prior` with independently loaded Occ3D `H_occ`.

## 8. Fast training and validation

Create a one-epoch debug configuration:

```bash
cp configs/default.yaml configs/quick.yaml
sed -i 's/^  epochs: 12$/  epochs: 1/' configs/quick.yaml
sed -i 's/^  num_workers: 0$/  num_workers: 2/' configs/quick.yaml
```

Train 100 samples on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_bevformer_occ.py \
  --config configs/quick.yaml \
  --dataroot /data/nuscenes \
  --occ-root /data/Occ3D-nuScenes \
  --output-root outputs_quick \
  --shadow-label-root outputs_quick/shadow_labels \
  --scene scene-0001 --max-samples 100 --device cuda
```

The checkpoint is:

```text
outputs_quick/checkpoints/bevformer_occ_shadow_epoch_000.pth
```

Quick functional evaluation on the same scene:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/evaluate_bevformer_occ.py \
  --config configs/quick.yaml \
  --dataroot /data/nuscenes \
  --occ-root /data/Occ3D-nuScenes \
  --output-root outputs_quick \
  --checkpoint outputs_quick/checkpoints/bevformer_occ_shadow_epoch_000.pth \
  --scene scene-0001 --max-samples 20 --device cuda
```

This only verifies the pipeline. It is not a paper result. For a quick
generalization check, remove `--scene` so the official val split is used:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/evaluate_bevformer_occ.py \
  --config configs/quick.yaml \
  --dataroot /data/nuscenes \
  --occ-root /data/Occ3D-nuScenes \
  --output-root outputs_quick \
  --checkpoint outputs_quick/checkpoints/bevformer_occ_shadow_epoch_000.pth \
  --max-samples 200 --device cuda
```

## 9. Formal training and two-GPU evaluation

After the quick run succeeds, generate labels for all required samples and
train with the full configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_bevformer_occ.py \
  --config configs/default.yaml \
  --dataroot /data/nuscenes \
  --occ-root /data/Occ3D-nuScenes \
  --output-root outputs \
  --shadow-label-root outputs/shadow_labels \
  --device cuda
```

Evaluate the official val split on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/evaluate_bevformer_occ.py \
  --config configs/default.yaml \
  --dataroot /data/nuscenes \
  --occ-root /data/Occ3D-nuScenes \
  --output-root outputs \
  --checkpoint outputs/checkpoints/bevformer_occ_shadow_epoch_011.pth \
  --device cuda
```

The evaluator uses `DistributedSampler`, reduces metric state across ranks,
and only rank 0 writes the result file. Do not launch two independent full
evaluators against the same output directory.

Use `--seed` to make repeated runs comparable:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_bevformer_occ.py \
  --config configs/default.yaml --seed 0 \
  --dataroot /data/nuscenes --occ-root /data/Occ3D-nuScenes \
  --output-root outputs/full_seed0 \
  --shadow-label-root outputs/shadow_labels --device cuda
```

## 10. Metrics

Results are written to `outputs/metrics/occ3d_metrics.json`:

```text
Occupancy_mIoU
Occupancy_mIoU_by_distance
Occupied_IoU
Free_IoU
Semantic_mIoU
Semantic_IoU_by_class
RayIoU
RayIoU_by_distance
HeightPrior_corr
HeightPrior_MAE
HeightPrior_corr_by_distance
PseudoShadow_shadow_F1
PseudoShadow_direction_cosine
PseudoShadow_reliability_MAE
samples_per_second
```

The paper-facing report should include overall and 0-20m, 20-40m, 40-60m
Occupancy mIoU, RayIoU, Semantic mIoU, and HeightPrior correlation. The local
RayIoU implementation is an ego-centric ray-space metric; use the official
Occ3D evaluator separately for leaderboard-comparable numbers.

## 11. Baseline and ablations

Recommended experiments:

| Experiment | Config | Purpose |
|---|---|---|
| Camera-only baseline | `configs/baseline.yaml` | Remove shadow fusion and shadow losses |
| Full Shadow-OCC | `configs/default.yaml` | Main method |
| No gate | `configs/ablation_no_gate.yaml` | Fuse shadow without U_shadow |
| No temporal loss | `configs/ablation_no_temporal.yaml` | Test temporal consistency |
| No height prior | `configs/ablation_no_height.yaml` | Test whether H_prior affects OCC |
| No render loss | `configs/ablation_no_render.yaml` | Test OCC-to-shadow auxiliary loss |
| Oracle caster | `configs/oracle_caster.yaml` | Diagnostic upper bound only |

Use separate output roots, for example `outputs/baseline`, `outputs/full`,
`outputs/no_gate`, and `outputs/no_temporal`. The key claim is the difference
between Full Shadow-OCC and the camera-only baseline, especially at 40-60m.

The baseline is strict: `model.use_shadow=false` and every shadow-related
loss weight is zero. Older checkpoints trained before this setting was added
must not be used as the paper baseline. The full model now feeds `H_prior`
into the confidence-gated OCC fusion; retrain the full model after changing
this code.

Compare matched evaluations:

```bash
python scripts/compare_experiments.py \
  --experiment baseline=outputs/baseline/metrics/occ3d_metrics.json \
  --experiment full=outputs/full/metrics/occ3d_metrics.json \
  --experiment no_gate=outputs/no_gate/metrics/occ3d_metrics.json \
  --experiment no_temporal=outputs/no_temporal/metrics/occ3d_metrics.json \
  --experiment no_height=outputs/no_height/metrics/occ3d_metrics.json \
  --output-root outputs/experiment_comparison
```

The first experiment is treated as the baseline and every `delta_*` column is
computed against it. Use the same official train/val split, epoch count,
seed, and checkpoint epoch for all rows.

Summarize three repeated runs:

```bash
python scripts/summarize_seeds.py \
  --experiment baseline=outputs/baseline_s0/metrics/occ3d_metrics.json,outputs/baseline_s1/metrics/occ3d_metrics.json,outputs/baseline_s2/metrics/occ3d_metrics.json \
  --experiment full=outputs/full_s0/metrics/occ3d_metrics.json,outputs/full_s1/metrics/occ3d_metrics.json,outputs/full_s2/metrics/occ3d_metrics.json \
  --output-root outputs/seed_summary
```

For a small manually labelled shadow set, create an `annotations.json` file:

```json
{
  "annotations": [
    {
      "scene_name": "scene-0001",
      "sample_token": "...",
      "mask_path": "masks/sample.png"
    }
  ]
}
```

The mask can be a PNG, NPY array, or inline 2D `mask` list. Evaluate it with:

```bash
python scripts/evaluate_shadow_annotations.py \
  --annotations annotations.json \
  --label-root outputs/shadow_labels \
  --output-root outputs
```

## 12. Architecture and losses

Input:

```text
I_(t-2), I_(t-1), I_t
camera projection matrices
ego-to-global transforms
```

Model:

```text
image CNN
  -> BEV queries and camera-projection cross-attention
  -> per-frame BEV features
  -> ego-motion BEV alignment
  -> temporal self-attention
  -> Shadow Geometry Head
  -> U_shadow-gated [shadow feature + H_prior] fusion
  -> occupancy decoders
```

Shadow outputs:

```text
P_shadow, P_root, P_tip, P_caster
V_shadow, H_prior, U_shadow, F_shadow
```

Training losses:

```text
L_occ + L_semantic + L_shadow_geometry
      + L_temporal_consistency + L_shadow_motion_smooth
      + L_caster_shadow_geometry + L_render_shadow
      + L_uncertainty_gating
```

The temporal constraint keeps caster bases stable after BEV warping; it does
not force shadow tips and boundaries to be identical across frames.

## 13. Troubleshooting

`No module named shadow_occ`: run from the repository root or execute
`pip install -e .`.

`BrokenProcessPool` or OOM during label generation: reduce `--num-workers` to
1 or 2.

`Occ3D label not found`: check the actual layout with:

```bash
find /data/Occ3D-nuScenes -name labels.npz | head
```

`CUDA unknown error`: check `CUDA_VISIBLE_DEVICES`, `nvidia-smi`, and
`torch.cuda.device_count()`. Do not change `CUDA_VISIBLE_DEVICES` after Python
has started. In Slurm, run inside the allocated GPU job.

Without Occ3D, the project can still generate nuScenes-only shadow labels and
run the length/pseudo-height precheck, but it cannot train the current OCC
target or report Occupancy mIoU, RayIoU, and Semantic mIoU.
