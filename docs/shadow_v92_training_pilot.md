# v9.2 Multi-Scene Training Pilot

Date: 2026-11-10. Source: local working-tree snapshot based on `587a6cf`,
not a newly published Git commit. No production worktree was overwritten.

## Implementation

- Ground-warped FSD semantic state changes are paired with independent RGB
  illumination changes. Joint onset/recovery observations enter the existing
  caster competition; semantic state changes alone cannot create positives.
- Dense receiver confidence, inherited pre-FSD cores and confidence-weighted
  extensions retain the v9.1 separation. M3 thresholds remain unchanged.
- Cached-scene planning, GPU prior inference, parallel CPU annotation, CPU
  trimap export and GPU student training now have connected entry points.
- Both supervision arms export the exact same selected clip IDs. Extra
  reference windows cannot silently contribute labels to the baseline.

## Data And Execution

Planned: 8 train / 3 dev / 3 eval scenes, 42 clips, 191 unique camera frames.
Raw image counts: 108 / 38 / 45. After excluding unsupervised images and
paired alignment, training/dev contain 35 / 10 usable images from 8 / 2 scenes.
Evaluation retains 45 images from 3 unseen scenes.

New train supervision: 33,841 inherited core pixels, 640 expanded pixels,
2,323 negative pixels. The baseline has 33,841 positive and 4,716 negative
pixels. Both classes survive resize. Approximately 99.91% of raw training
pixels remain IGNORE; expanding scene counts did not produce dense supervision.

Jobs: final regression `372633`, CPU dataset `372634`, two-student training
and evaluation `372635`. All completed with exit `0:0`. Regression:
563 passed, 2 skipped, 3 distributed tests deselected.

Both students: DeepLabV3-MobileNetV3, ImageNet backbone, 384x704, 10 epochs,
batch 2, accumulation 4, seed 0, balanced BCE + 0.5 Dice. Extensions use the
existing warmup/ramp and absolute confidence weighting. Both yielded
noncollapsed checkpoints. Selected epochs are zero-based 9 and 3, respectively.

## Held-Out Results

These are agreements with one frozen weak teacher on known pixels, not
accuracy against independent shadow ground truth. The operating threshold
is 0.5; no eval threshold was selected.

| Metric | v8 core student | Motion-illumination student |
| --- | ---: | ---: |
| Balanced accuracy | 66.88% | 73.25% |
| FPR | 62.53% | 33.91% |
| Positive recall | 96.29% | 80.40% |
| Positive IoU | 87.31% | 76.16% |
| AUROC | 0.8970 | 0.8451 |
| Binned AP | 0.9825 | 0.9716 |
| Recall at 1% FPR, descriptive test curve | 67.08% | 54.30% |

The new student reduces false positives at 0.5 but loses recall and ranking
quality. Its lower FPR is not evidence of improved overall discrimination.
The all-foreground teacher IoU is already 85.88%, so foreground IoU alone is
misleading. One scene supplies 28,287 of 34,030 evaluated known pixels.
Boundary scores are unavailable on these sparse trimaps, not zero-quality scores.

The new student's selected dev balanced accuracy is 93.93%, versus 73.25%
held-out, indicating a substantial scene-generalization gap. New positive
supervision is only 1.89% of the inherited core count, while negative coverage
also changes. This run does not isolate an extension-only causal effect.

## Next Mainline Work

Keep the connected pipeline. Expand usable, scene-diverse positive and negative
observations, not just raw scene/image counts. Existing inherited labels remain
weak labels, not automatically validated cores. Do not infer a need for more
epochs or another network from this run. Both checkpoints remain unapproved
for self-relabeling, and OCC integration is not part of this pilot.

HPC root:
`/public/home/eng_zhihongli/project/.shadow_auto_runs/v92_training_8_3_3_20261110`

Key results: `training_result_summary.json`, `dataset/dataset_summary.json`,
`training/metrics/teacher_agreement.csv`, `training/metrics/cross_scene_summary.json`,
and each student's `best.pth`, `history.json`, and `readiness.json`.
