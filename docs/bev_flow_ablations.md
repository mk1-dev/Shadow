# BEV-Flow Internal Ablations

## Experimental Definition

The main method is a geometric receiver grid plus frozen temporal UniAD features,
observed DIS registration support, illumination transitions, and factorized physical
shadow/caster decisions. The encoder, image/pair population, split seed, model
initialization and training budget stay fixed within each comparison.

The original dual-BEV table's D and F describe the same method; keep one `full` arm.
Removing the geometric grid is not a receiver-gate ablation: it also removes the
world correspondence and caster projection used by the teacher. A learning-only
or image-only detector needs a separate image-space baseline, not a renamed flag.

Current-only features mean three independently encoded frames, each with UniAD
tracking/BEV history reset before inference. They do NOT mean repeating the middle
frame three times or simply changing the feature cache's name.

## Executable Arms

Recipes are in `configs/bev_flow_ablations.json`.

| Arm | Difference from full |
|---|---|
| `full` | Aligned temporal BEV cosine >= 0.3, DIS, factorized-v3, intersection |
| `geometry_only` | No learned feature gate; geometry, DIS and illumination unchanged |
| `current_only` | Independently exported current-only encoder features; same three-frame gate |
| `shuffled` | Independently permute valid spatial features in each frame before cosine |
| `matched_random` | Randomly retain exactly the aligned arm's number of candidate cells per window |
| `cosine_01`, `cosine_05` | Change only cosine threshold; full is the 0.3 arm |
| `without_final_flow_gate` | Remove final DIS support gate; exposure still uses cached DIS background |
| `geometry_no_dis` | Actually skip DIS; geometry-only correspondence/exposure; no learned feature gate |
| `geometry_no_dis_bev` | Geometry-only/no-DIS evidence plus learned feature gate |
| `physics_fixed` | Fixed light ID 0, strict policy, no time contrast or independent-light gate |
| `physics_bank` | Search all 96 lights, strict policy, no time contrast or independent-light gate |
| `physics_contrast` | Add true-vs-wrong-time contrast to `physics_bank` |
| `physics_shared` | Add independent-track light support to `physics_contrast` |
| `best_projection` | Keep pixels supported by the single best physical explanation |
| `consensus_projection` | Keep pixels supported by >= 2/3 near-best explanations instead of all |

`matched_random` holds final selected-cell count fixed BEFORE connected components
and caster matching. It does not force equal final label counts. All corruption
seeds depend on the experiment seed and window ID, not the processing order.
Repeat random controls with at least three seeds in separate output directories.

The physics ladder deliberately uses `strict-v2`. In `factorized-v3`, trajectory
contrast and independent light support determine ownership, not necessarily YES
acceptance. Removing those gates from full therefore is not a clean test of
whether they improve shadow-existence precision. Compare `physics_shared` with
`full` separately to test strict versus factorized acceptance.

The fixed direction must be predeclared or selected on train/dev and then frozen.
It is not the best direction selected independently for each eval component.
`best_projection` performs that separate per-region best-explanation comparison.

## Run on Existing HPC Caches

Wait for data preparation to finish. No existing labels/checkpoints are overwritten.
The first six arms reuse `pairs`, `flow`, and the stored photometric/learned evidence.
Derived windows keep gates/cosines, not another copy of the large sampled BEV
feature tensor for every arm. The original frozen cache remains the shared source.
The genuine no-DIS arms recompute geometry-only exposure and illumination without
running optical flow, while reusing the same images, GT geometry and BEV encoder.

```bash
cd /public/home/eng_zhihongli/project/Shadow_update
export SHADOW_PROJECT_ROOT="$PWD"
export SHADOW_CONDA_ENV=shadow_annotator
export SHADOW_ABLATION_SOURCE="$PWD/outputs/shadow_bev_flow_20261106_2231"
export SHADOW_ABLATION_ROOT="${SHADOW_ABLATION_SOURCE}_ablations_seed0"
export SHADOW_ABLATION_VARIANTS="full geometry_only shuffled matched_random cosine_01 cosine_05"
mkdir -p logs
sbatch --partition=YOUR_CPU_PARTITION slurm/run_bev_flow_ablations.slurm
```

This job requests no GPUs. Use the cluster's CPU-capable partition. For physics
and flow experiments, change the variant list and use a new output directory:

```bash
export SHADOW_ABLATION_ROOT="${SHADOW_ABLATION_SOURCE}_physics_flow_seed0"
export SHADOW_ABLATION_VARIANTS="full geometry_only geometry_no_dis geometry_no_dis_bev physics_fixed physics_bank physics_contrast physics_shared best_projection consensus_projection"
sbatch --partition=YOUR_CPU_PARTITION slurm/run_bev_flow_ablations.slurm
```

By default this builds per-arm teachers plus `data/` and `joint/`, not models.
Set `SHADOW_ABLATION_LABELS_ONLY=1` to skip the final crop/dataset builders.
Zero-acceptance or missing-class arms are recorded as outcomes, not omitted.

## Export the Current-Only Encoder Control

Run in a one-GPU job using the existing mature encoder environment. Reuse exactly
the same checkpoint, model config and train/val infos used for the source cache.

```bash
conda activate drivevla
SOURCE=/public/home/eng_zhihongli/project/Shadow_update/outputs/shadow_bev_flow_20261106_2231
CURRENT="${SOURCE}_current_only_bev"
for SPLIT in train val; do
  if [[ "$SPLIT" == train ]]; then INFOS="$UNIAD_TRAIN_ANN_FILE"; else INFOS="$UNIAD_VAL_ANN_FILE"; fi
  python scripts/export_mature_bev.py \
    --external-root "$UNIAD_ROOT" --checkpoint "$UNIAD_CHECKPOINT" \
    --ann-file "$INFOS" --dataroot "$SHADOW_NUSC_ROOT" \
    --pairs "$SOURCE/$SPLIT/pairs/pairs.jsonl" \
    --history-mode current-only --output-root "$CURRENT/$SPLIT"
done
```

Then pass `SHADOW_CURRENT_BEV_ROOT="$CURRENT"` and include `current_only` in the CPU
variant list. Replacement caches are checked against the source encoder hashes.

## Positive/Negative Training Controls

These change ONLY segmenter training targets; dev/eval targets and frame membership
stay fixed. This avoids evaluating each label recipe against its own changed answers.

```bash
conda activate shadow_annotator
python scripts/make_shadow_label_ablation.py \
  --data-root "$SHADOW_ABLATION_SOURCE/data" \
  --policy positive-only --output-root "${SHADOW_ABLATION_SOURCE}_labels_positive_only"
```

Other `--policy` values:

- `endpoints`: unchanged positive/lit-endpoint targets.
- `shuffled-negatives`: preserve positives and move the negative budget to randomly
  selected supported ground pixels in the same image; report overlap and actual counts.
- `all-background`: force every nonpositive image pixel to background, intentionally
  including unobserved pixels as the harmful control from the proposed table.

Train with the existing segmenter Slurm script by setting `SHADOW_DATA_ROOT` to the
generated directory and `SHADOW_SEG_ROOT` to a new model directory. Positive-only
also needs `SHADOW_SEG_POSITIVE_ONLY_ABLATION=1`; unset it for other arms. Keep seed,
initialization, epochs, accumulation and loss coefficients equal. Collapsed results
remain results: inspect `last.pth`/`history.json`, not just the presence of `best.pth`.
Segmenter dev and eval reports now include 256-bin precision-recall curves,
quantized average precision, and precision/IoU at the fixed threshold sweep.

Endpoint negatives now exclude projections of ALL annotated endpoint volumes,
including nonvehicles and tracks absent from the three-frame vehicle list. Optional
`--negative-exclusion-root` camera-image masks mark static-shadow risk areas; these
are reprojected into each window's actual receiver grid. Without such evidence,
the teacher is a vehicle-conditioned lit-endpoint labeler, not an exhaustive test
for every tree/building shadow. These exclusions do not change positive acceptance.

## Comparison and Quality

Each arm records its exact recipe, source hashes, per-window gates, photometric
population, selected cells, physical acceptance and association counts. The report
checks equal pair manifests and window IDs. Geometry-only/no-DIS arms are allowed
different exposure-analysis hashes, because that is the intended intervention.

`comparison/summary.json` and `summary.md` report retention/coverage by default.
To measure positive/negative precision, supply one fixed independent reference:

```bash
python scripts/summarize_bev_flow_ablations.py \
  --root "$SHADOW_ABLATION_ROOT" \
  --reference-manifest /path/to/independent_reference.jsonl \
  --output-root "$SHADOW_ABLATION_ROOT/reference_comparison"
```

Reference rows contain `scene_name`, `official_split` (`train` or `val`), `camera`,
`sample_token`, `image` and a same-resolution `target` path with 0/1/255 values.
Use the same reference for every arm. Missing predictions contribute zero coverage
instead of disappearing from the denominator. The report includes positive and
negative precision, known-pixel coverage, positive-reference capture, and 1000
scene-cluster bootstrap intervals. It does not select eval thresholds.

For downstream model comparisons, train each arm on its own derived TRAIN targets
but point every model at the same fixed dev/eval reference. Match source-window and
teacher-label coverage as separate quantities. Do not train/evaluate all arms
against independently regenerated eval pseudo-labels and call that true quality.

The independent-reference pixel report does not measure caster association accuracy;
use the existing unified annotation evaluator with independently checked associations
for that metric. Image-only/DIS-only baselines without geometric correspondence are
separate detector implementations, not provided by the receiver-gate switches.

## HPC Tests

The ablation Slurm script first runs focused feature, teacher, physics and control
tests. Local development does not run training or GPU inference. Model training
and the empirical precision/coverage conclusions are established on HPC.
