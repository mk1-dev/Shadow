# One GT-constrained VLM shadow annotation task

The VLM is trained to annotate shadows using object GT, image evidence and
temporal continuity. Caster association is part of each shadow annotation.
Potential outside-BEV casters are an optional field in that same annotation,
not a separate VLM detection task or an additional model.

```text
nuScenes GT objects/tracks + images + ground/temporal evidence
  -> shadow segmentation and region observations
  -> one GT-constrained shadow annotation target per observation
  -> one Qwen3-VL LoRA annotator
  -> annotated shadow mask references + caster associations
  -> held-out annotation evaluation and optional label refinement
```

## What GT supervises

Object position, category, extent and track identity come from nuScenes GT.
They constrain caster location, region context, temporal matching and the
caster field of the annotation target. These are incorporated into each
annotation rather than trained as separate object-localization questions.

Shadow pixels currently come from explicit real shadow masks when available,
or from the existing geometry/temporal weak pipeline. The code records shadow
label provenance and shadow-caster association provenance separately. An
accurate object box does not automatically turn its nearby dark region into
shadow GT. Independent established associations can be supplied to the boundary
data builder; unique geometry matches stay weak associations.

During inference the VLM receives images/region references, temporal context
when valid, and optional geometry-generated exterior candidate regions.
Target GT identity/coordinates/categories are supervision, not prompt answers.
GT-assisted visible-caster contexts remain a diagnostic setting until replaced
by operational region/context generation. Boundary-shadow observations are
generated without target object GT and checked for direct target appearance.

## Output of the same annotation task

```json
{
  "shadow": "YES",
  "caster": {
    "state": "visible",
    "category": "vehicle.car",
    "bbox_1000_context": [100, 200, 400, 700],
    "outside_candidate_id": null
  }
}
```

`shadow` is YES/NO/UNCERTAIN. `caster.state` is visible, outside_bev,
unresolved or none. A visible caster has a box normalized to IMAGE 3 context;
an outside association chooses one supplied region ID. Unresolved does not mean
empty space. The segmentation mask reference is retained in the annotation
output, so VLM decisions can return to the label pipeline without inventing
free-text masks.

## Main HPC entry points

The main scripts are `build_joint_shadow_annotation_dataset.py`,
`train_shadow_vlm.py --primary-task shadow_annotation`, and
`evaluate_shadow_annotator.py`. The default training task is now
`shadow_annotation`. The default caster auxiliary ratio is zero; separate
caster-task inputs are rejected for this unified task.

```bash
cd /public/home/eng_zhihongli/project/Shadow_update
mkdir -p logs
export SHADOW_VLM_MODEL=/path/to/Qwen3-VL-8B-Instruct

sbatch slurm/prepare_shadow_annotation.slurm
# Check data/data_check.json and joint/data_check.json after preparation.
sbatch slurm/train_shadow_segmenter.slurm
sbatch slurm/train_shadow_vlm.slurm
# After the model jobs finish:
sbatch slurm/evaluate_shadow_annotators.slurm
```

The preparation job builds the ordinary shadow/temporal data, wide-range object
GT and boundary-shadow association data, then merges annotation targets into
`outputs/shadow_annotation/joint/annotation_{train,dev,eval}.jsonl`.
`SHADOW_INCLUDE_EXTERIOR_ASSOCIATIONS=0` omits boundary examples for an annotation
ablation; it does not introduce a different task. Check the number of surviving
exterior associations before expecting the model to learn that output state.

Segmentation uses `SHADOW_DATA_ROOT` (default `outputs/shadow_annotation/data`).
VLM training/evaluation use `SHADOW_ANNOTATION_DATA_ROOT` (default
`outputs/shadow_annotation/joint`). Both can use the annotation environment via
`SHADOW_CONDA_ENV`. Dependencies are in `requirements-annotation.txt`.
New runs require new output roots. Slurm jobs preserve the assigned GPU mask.

## Reuse already exported data

```bash
python scripts/build_joint_shadow_annotation_dataset.py \
  --region-data-root outputs/shadow_annotation/data \
  --offbev-data-root outputs/shadow_offbev/data \
  --output-root outputs/shadow_annotation/joint_v2
python scripts/check_joint_shadow_annotation_data.py \
  --data-root outputs/shadow_annotation/joint_v2
```

The ordinary region data must include `context_crop_xyxy` to supervise a
visible caster's normalized box. Rebuild region manifests with the latest
`build_shadow_annotation_dataset.py` when using older exports without it;
otherwise that annotation correctly falls back to an unresolved caster.
Exterior observations should include `region_mask` for returning pixel labels.
`--balance-shadow` derives YES/NO/UNCERTAIN weights from the training split and
applies them to training and validation loss; the weights are recorded in
`training_protocol.json`. This prevents the writer from selecting a checkpoint
that only predicts the dominant decision.

```bash
python scripts/train_shadow_vlm.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --primary-task shadow_annotation \
  --train outputs/shadow_annotation/joint_v2/annotation_train.jsonl \
  --dev outputs/shadow_annotation/joint_v2/annotation_dev.jsonl \
  --output-root outputs/shadow_annotation/annotator_v2 \
  --load-in-4bit --allow-weak-labels --balance-shadow --seed 0
python scripts/evaluate_shadow_annotator.py \
  --model /path/to/Qwen3-VL-8B-Instruct --load-in-4bit \
  --adapter outputs/shadow_annotation/annotator_v2/adapter \
  --manifest outputs/shadow_annotation/joint_v2/annotation_eval.jsonl \
  --output-root outputs/shadow_annotation/eval_v2
```

## Evaluation and refinement

The original and adapted VLM produce the same annotation schema on identical
observations. `compare_shadow_annotations.py` checks input signatures and target
consistency, then reports annotation agreement and caster association metrics
with scene bootstrap for shadow decision changes. GT and weak agreement remain
separate. Exterior candidate-bank misses remain in known-positive exterior
association denominators. These results do not create occupancy GT outside
the benchmark volume.

Outputs are `annotations.jsonl` (raw responses, parsed annotations, mask
references and association metadata) and `metrics.json`. A YES annotation can
select its existing segmentation/proposal mask; uncertainty is retained rather
than forced into a positive/negative label.

```bash
python scripts/refine_shadow_annotations.py \
  --seg-predictions outputs/shadow_annotation/evaluation/segmentation/predictions.jsonl \
  --annotation-results outputs/shadow_annotation/evaluation/tuned/annotations.jsonl \
  --output-root outputs/shadow_annotation/refined
```

This joins ordinary region annotations by segmentation parent ID. Boundary
annotations already carry their source-image mask reference and require a
sample/camera join for later BEV geometry-label integration. Neither output
should be substituted for complete OCC geometry labels without rebuilding the
root/tip/direction targets and preserving ignore masks.

## HPC validation

```bash
python -m compileall shadow_occ/annotation scripts
python -m pytest tests -q
python scripts/check_joint_shadow_annotation_data.py --data-root outputs/shadow_annotation/joint
```

Validated on 2026-11-03 with `shadow_annotator` (`torch 2.6.0+cu124`,
`transformers 4.57.6`, `peft 0.21.2`, `bitsandbytes 0.50.2`): all tests passed,
a 1000-sample nuScenes pilot produced non-empty train/dev/eval splits, and
Qwen3-VL-8B completed 4-bit LoRA training/evaluation on one 24 GiB RTX 4090.
Peak observed device memory was 13.12 GiB. The pilot contained zero
`outside_bev` training examples, so increase the pilot or use boundary-focused
sampling before claiming that state is trainable.

The first unweighted pilot exposed a majority-class failure: its adapter parsed
all held-out examples but predicted `NO/none` for all 67, giving 86.6% raw
agreement and 50% balanced accuracy. A one-epoch `--balance-shadow` diagnostic
raised visible-caster IoU50 recall from 0% to 77.8% on 9 positives, but produced
43 false positives out of 58 negatives and only 51.8% balanced accuracy. These
pilot results validate the execution path, not production annotation quality.
Use balanced accuracy, positive recall and negative specificity alongside raw
agreement; do not select an annotator by weighted validation loss alone.

The batch partition was pending under scheduler priority during validation;
the same training entry point ran through `srun` on an explicitly selected
RTX 4090. Use `sbatch` for managed production runs when the partition grants
the request immediately.

The earlier binary verifier and standalone exterior selector remain legacy
diagnostic entry points; they are not the recommended main training workflow.
