# Shadow segmentation and a GT-assisted VLM annotator

For the main VLM workflow, use
[shadow_joint_annotator_hpc.md](shadow_joint_annotator_hpc.md): one GT-constrained
shadow annotation target contains both the shadow decision and caster association.
This document retains low-level segmentation/data preparation and the earlier
binary-verification diagnostic. Exterior casters are annotation fields, not a
separate required model task.

Both branches are retained:

```text
nuScenes object GT + existing image/BEV shadow evidence
  -> caster-associated image targets (shadow / background / ignore)
  -> shared scene-disjoint train/dev/eval manifests
       -> RGB DeepLabV3-MobileNet shadow segmentation
       -> Qwen3-VL single-region LoRA + object-localization auxiliary task
  -> held-out segmentation evaluation + base-vs-LoRA VLM comparison
  -> optional segmentation/VLM agreement masks and BEV mask export
```

Object boxes supervise object localization, not shadow classification. Shadow
training requires explicit image shadow masks or explicitly sourced weak region
labels. The supplied bridge uses existing pseudo masks as weak supervision and
marks them `geometry_weak`. It does not turn nuScenes object GT into shadow GT.
If real public/manual shadow masks become available, the same data pipeline can
mix them with weak data. No manual nuScenes shadow annotation is required to run
the weak-supervision experiment.

The temporal extension follows the same GT caster across frames using
`scene_name + camera + instance_token`. GT accurately identifies the object;
the existence and visibility of its cast shadow remain conditional on light,
occlusion, ground and field of view. Stable paint or wet road can also persist,
so continuity supplies weak evidence rather than a new shadow ground truth.

The RGB segmenter predicts all shadows in an image. For an instance-shadow
dataset, merge instance masks into a full-image shadow mask, or mark other
instances unknown. Reusing one RGB image with conflicting per-instance binary
background targets would teach contradictory labels.
For the ground-shadow VLM task, restrict public masks to ground-receiving
regions with `valid_mask`; shadows on walls or object bodies do not satisfy
this task's YES definition.

## Files to sync

Sync `shadow_occ/annotation/`, the new scripts listed below,
`prompts/shadow_single_region_sft.txt`, `slurm/`, `requirements-annotation.txt`,
and this document. The existing HPC VLM scripts/proposals are independent of
these new entry points and are not replaced.

The existing OCC trainer and model are not changed by this addition.
The annotation modules have separate dependencies and checkpoints.

## Environment

Use a separate environment for annotation work to preserve the working OCC
installation. Install a matching PyTorch/torchvision pair with CUDA support,
then install the package and annotation dependencies:

```bash
cd /public/home/eng_zhihongli/project/Shadow_update
pip install -e .
pip install -r requirements-annotation.txt
mkdir -p logs
```

The VLM entry points use local model files by default. `--allow-download` is an
explicit opt-in. QLoRA needs Linux bitsandbytes. The supplied GPU job uses 4-bit
weights, a micro-batch of one, gradient accumulation and a limited pixel budget.
Actual memory fit must be checked on HPC. The default segmenter initialization
does not download weights. `--backbone-weights imagenet` requests a cached or
downloaded ImageNet backbone, or use `--init-checkpoint` for a previously trained
shadow segmenter.

## Canonical input manifest

`build_shadow_annotation_dataset.py` accepts one or more JSONLs. Relative paths
are resolved against each input manifest. Every row describes one image/crop
and uses masks in that exact image coordinate system:

```json
{
  "id": "unique-image-or-caster-id",
  "scene_name": "scene-0001",
  "official_split": "train",
  "image": "/path/to/image.jpg",
  "shadow_mask": "/path/to/shadow_trimap.png",
  "mask_encoding": "trimap",
  "label_source": "geometry_weak",
  "regions": [
    {"id": "r1", "mask": "/path/to/region.png"}
  ]
}
```

Mask encodings:

- `binary`: zero is known non-shadow; nonzero, including 255, is shadow.
- `trimap`: 0 is known non-shadow, 1 is shadow, 255 is ignore.
- `positive_only`: nonzero is shadow; zeros are unknown and become ignore.

Weak masks default to `positive_only`. GT masks default to `binary`.
`negative_mask` can supply explicit trusted/weak negatives. Optional `valid_mask`
sets unobserved pixels to ignore. NPY is supported; NPZ references use
`{"path": "...npz", "key": "shadow_mask"}`. BEV arrays must first be projected
to the source image; shape-only resizing is prohibited.

Allowed shadow label sources are `manual_shadow_gt`, `public_shadow_gt`,
`synthetic_shadow_gt`, `geometry_weak`, `temporal_weak`, and `model_pseudo`.
They remain separate in evaluation reports. Object auxiliary examples use the
distinct `object_annotation_gt` source.

Optional region fields `shadow_label` (`YES`, `NO`, `IGNORE`) and `label_source`
override pixel-derived targets. Otherwise, targets require at least 80% known
pixels, and shadow fraction >=70% for YES or <=5% for NO. Everything else is
unlabeled for VLM training but retained for scoring.

`group` (`candidate`, `control`, `weak_control`) and `pair_id` are analysis-only
metadata. They never determine the training answer.

`official_split` supports nuScenes `train/val/mini_train/mini_val`, or `external`
for other shadow datasets. Real nuScenes membership is checked. Scene hashing
splits official train into train/dev; official val goes only to eval. For public
datasets, use stable namespaced scene/group IDs and `eval` for an explicitly
reserved test partition. Use one group for related images, not one per frame.
Excluded/tuning scenes can be supplied as a JSON list to `--exclude-scenes`.

Optional `--casters casters.jsonl` joins caster GT with a row's `caster_id`.
The caster ID is the exported record's `id`; metadata must match exactly.
Alternatively, embed the exported caster fields directly in each row, as the
weak bridge does. A clean image and point prompt supervise the object's category
and normalized bounding box; GT height/dimensions are not in shadow prompts.

## Prepare from current nuScenes data

This CPU route exports GT objects, converts the existing BEV pseudo masks into
image weak targets, then builds both training datasets:

```bash
python scripts/export_nuscenes_caster_gt.py \
  --dataroot /public/home/eng_zhihongli/project/data/nuscenes-full \
  --split train --output outputs/shadow_annotation/casters_train.jsonl
python scripts/export_nuscenes_caster_gt.py \
  --dataroot /public/home/eng_zhihongli/project/data/nuscenes-full \
  --split val --output outputs/shadow_annotation/casters_val.jsonl
python scripts/export_caster_shadow_weak_data.py \
  --casters outputs/shadow_annotation/casters_train.jsonl outputs/shadow_annotation/casters_val.jsonl \
  --dataroot /public/home/eng_zhihongli/project/data/nuscenes-full \
  --label-root outputs_temporal/shadow_labels \
  --output-root outputs/shadow_annotation/weak_images
python scripts/add_shadow_temporal_evidence.py \
  --manifest outputs/shadow_annotation/weak_images/manifest.jsonl \
  --output-root outputs/shadow_annotation/temporal_images
python scripts/audit_shadow_temporal.py \
  --manifest outputs/shadow_annotation/temporal_images/manifest.jsonl \
  --output outputs/shadow_annotation/temporal_images/audit_summary.json
python scripts/build_shadow_annotation_dataset.py \
  --manifest outputs/shadow_annotation/temporal_images/manifest.jsonl \
  --output-root outputs/shadow_annotation/data --seed 0
python scripts/check_shadow_annotation_data.py \
  --data-root outputs/shadow_annotation/data
```

Or submit `sbatch slurm/prepare_shadow_annotation.slurm` on `Intel`.
Its default export cap is 1000 samples per official split; change
`SHADOW_EXPORT_SAMPLES` to control the pilot size. Counts can shrink substantially
after visibility/ground/mask checks. Confirm nonempty train/dev/eval and both
positive and negative targets in `data_check.json` before starting models.

The bridge uses near-plane LiDAR observations, a sparse depth buffer, exclusion
of exported object boxes, and bright observed ground for weak negatives. The
ground estimate remains an ego z-plane approximation. Its support dilation and
brightness thresholds are parameters, not a guarantee of semantic ground.
Both mask types remain weak labels. Unknown pixels are preserved as 255.

Outputs include `seg_{train,dev,eval}.jsonl`,
`vlm_{train,dev,eval}.jsonl`, `dataset_protocol.json`, and clean/reference/context
views. No proposal numbers or candidate/control names appear in model prompts.

## Temporal light evidence and static shadow retention

`add_shadow_temporal_evidence.py` uses two separate ground-plane mappings:

- World-ground alignment compensates ego motion and compares the same receiving
  surface for Lab light/chroma changes. Exposure scale and offset are estimated
  from shared observed background, excluding positive shadow labels.
- Caster alignment additionally uses the translation of the same GT vehicle.
  It compares the previous and current shadow support relative to that vehicle.
  It is a translation approximation, not a shadow-shape renderer.

Frames more than 0.75 seconds apart, missing geometry, large caster yaw change
(default 5 degrees), or insufficient shared ground do not impose continuity.
Large unexplained background/exposure changes disable the pair. Static casters
(default <=0.3m/s) receive high consistency weight when aligned light changes
are SMALL. Moving-caster evidence allows darkening at world-fixed surfaces and
uses a lower mask-consistency weight; boundaries are excluded. No minimum
brightness difference is required for an existing static shadow to survive.

Ground support comes from sparse observations in the weak bridge. Unknown
ground, disappeared regions and invalid projections contribute no temporal
constraint. Turning vehicles are skipped because a shadow under fixed global
light does not simply rotate rigidly with the caster.

Audit outputs include counts of usable pairs, static/moving weak-mask retention,
light residuals, and the proportion of repeated shadow pixels with Lab-L change
below 6. These are consistency diagnostics, not measured true shadow recall.
The first 50 pairs have clean/warped review images and weight maps; change
`--review-limit` to control storage. Evidence NPZs retain per-pixel weights and
light changes for the training data.

By default this step records evidence without modifying the weak mask.
`--propagate-positive` can additionally recover UNKNOWN pixels when a previous
positive maps onto observed, currently dark ground under comparable lighting.
It never overwrites known negatives or real shadow GT. New masks are marked
`temporal_weak`. A shadow missing from every input frame cannot be recovered
by propagation alone; it still needs an appearance/geometry seed.

The segmenter adds a detached previous-prediction consistency loss in these
weighted interiors (`--temporal-weight 0.2`). Its model remains RGB-only at
inference. It reports weighted temporal probability MAE alongside pixel metrics.
Use `--temporal-weight 0` to train the matched segmentation ablation.

For valid pairs, VLM regions with enough shared support receive a fourth clean
view: the previous context in caster-aligned current coordinates. The prompt
explains that static shadows need not change brightness. GT or pseudo target
masks, confidence scores and audit answers are not given as visual evidence.
`--no-temporal-context` on the dataset builder supplies a VLM input ablation;
segmentation temporal metadata remains available in that dataset. Default VLM
base-vs-adapter comparison requires identical input signatures. Use the explicit
`--allow-input-ablation` option only when comparing different frame inputs.

If the earlier weak-data export already exists, reuse it and build a new temporal
dataset directory instead of rerunning or overwriting that export:

```bash
python scripts/add_shadow_temporal_evidence.py \
  --manifest outputs/shadow_annotation/weak_images/manifest.jsonl \
  --output-root outputs/shadow_annotation/temporal_v1
python scripts/build_shadow_annotation_dataset.py \
  --manifest outputs/shadow_annotation/temporal_v1/manifest.jsonl \
  --output-root outputs/shadow_annotation/data_temporal_v1 --seed 0
export SHADOW_DATA_ROOT=outputs/shadow_annotation/data_temporal_v1
```

The earlier weak exporter must include `ground_support` for this to create usable
temporal pairs. A pre-extension manifest without ground support is preserved
with `missing_geometry_or_ground` status, rather than inventing visible ground.

## Train the two branches

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_shadow_segmenter.py \
  --train outputs/shadow_annotation/data/seg_train.jsonl \
  --dev outputs/shadow_annotation/data/seg_dev.jsonl \
  --output-root outputs/shadow_annotation/segmenter \
  --epochs 10 --allow-weak-labels --seed 0

CUDA_VISIBLE_DEVICES=0 python scripts/train_shadow_vlm.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --primary-task shadow \
  --train outputs/shadow_annotation/data/vlm_train.jsonl \
  --dev outputs/shadow_annotation/data/vlm_dev.jsonl \
  --output-root outputs/shadow_annotation/vlm_lora \
  --load-in-4bit --allow-weak-labels --caster-ratio 0.25 --seed 0
```

Slurm equivalents:

```bash
export SHADOW_VLM_MODEL=/path/to/Qwen3-VL-8B-Instruct
sbatch slurm/train_shadow_segmenter.slurm
sbatch slurm/train_shadow_vlm.slurm
```

Set `SHADOW_CONDA_ENV`, `SHADOW_DATA_ROOT`, `SHADOW_SEG_ROOT`, `SHADOW_VLM_ROOT`
to override the defaults. Preserve Slurm's GPU assignment; the job files do not
hardcode `CUDA_VISIBLE_DEVICES`. Add only the CUDA library settings already
verified for that HPC environment if they are required.

The segmenter consumes RGB only. It trains BCE+Dice on known pixels and freezes
BatchNorm running statistics for small batches. Both trainers downweight weak
targets using `--weak-weight` (default 0.25). VLM training uses LoRA on language
attention layers, freezes the vision encoder, masks prompt/image tokens out of
the loss, and caps caster auxiliary examples at 25% of shadow examples.
Both YES and NO are required. Object GT alone is rejected as shadow training.

## Held-out evaluation

Run `sbatch slurm/evaluate_shadow_annotators.slurm` after both training jobs
finish successfully. It evaluates segmentation, scores the same VLM inputs with
the original and adapted model, and writes `comparison.json`.

The segmentation report includes pixel IoU, precision, recall and F1 per label
source. The VLM report uses normalized YES/NO sequence likelihood rather than a
generated confidence number, and reports label-source metrics and weak-pair
separation separately. Its score is not calibrated probability. Pair-side YES
uses the maximum region score; candidate/control proposal counts must therefore
remain matched. The comparison requires exactly the same region IDs and reports
paired accuracy changes with scene-cluster bootstrap intervals.

When only `geometry_weak` eval labels exist, these scores measure agreement with
the weak rules. Report them as weak-label consistency, not true shadow accuracy.
`group` separation remains a proxy experiment. Teacher-rule agreement on new
scenes does not alone demonstrate a better annotator.

Eval contexts generated from GT boxes are GT-assisted; replace GT caster hints
with detector predictions before claiming an operational camera-only annotator.
GT and detector context sources are recorded in outputs. Thresholds must be
fixed on dev before eval. Exclude previously tuned/evaluated scenes when a new,
untouched evaluation is required.

## Optional label refinement

Only after independent evaluation supports using the adapted annotator:

```bash
python scripts/refine_shadow_annotations.py \
  --seg-predictions outputs/shadow_annotation/evaluation/segmentation/predictions.jsonl \
  --vlm-votes outputs/shadow_annotation/evaluation/tuned/votes.jsonl \
  --output-root outputs/shadow_annotation/refined_eval
python scripts/export_refined_shadow_bev.py \
  --manifest outputs/shadow_annotation/refined_eval/manifest.jsonl \
  --output-root outputs/shadow_annotation/refined_bev_eval
```

Accepted regions intersect high segmentation probabilities; uncertain regions
remain ignore. Rejected proposals are not automatically made background.
The default thresholds are starting values to freeze or tune on dev, not validated
operating points. Applying a trained annotator to its own training scenes needs
`--allow-train-scenes` on prediction/scoring and is recorded as in-sample annotation.

The BEV output contains `shadow_mask` and `observed_mask`. It is not a complete
replacement for the original OCC shadow-label NPZs: root/tip/direction/height
targets would need reconstruction and unknown pixels would need masking in the
OCC loss. Do not replace the existing `shadow_labels` directory with these files.

## HPC verification commands

Verification is intentionally left to HPC:

```bash
python -m compileall shadow_occ/annotation scripts
python -m pytest tests/test_shadow_annotation_data.py tests/test_shadow_annotation_temporal.py -q
python scripts/check_shadow_annotation_data.py --data-root outputs/shadow_annotation/data
```

Follow these with a short segmenter run and a tiny QLoRA train/dev run before
the full jobs. VLM `--max-steps 2 --accumulation 1` limits the optimizer smoke
run. No model forward pass, data export, test execution or Slurm
submission was performed on the local Windows workspace for this addition.

Implementation references:
[Qwen3-VL processor/model](https://huggingface.co/docs/transformers/model_doc/qwen3_vl),
[PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora),
[Torchvision DeepLabV3-MobileNet](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html).
