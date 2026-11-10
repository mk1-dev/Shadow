# Scene-Diverse Student Training

The physical teacher, FSD, core acceptance and M3 thresholds are unchanged.
Expansion changes observation coverage, not the meaning of a positive label.

## Prepare

Use a new output directory. Keep dev/eval scenes and exact clips from the
previous plan, and evaluate against the previous frozen teacher trimaps.
The training stage is cumulative: stage 50 reads both stage 25 and stage 50
batches, while references come from the cumulative experiment manifest.
Only scenes with completed pre-FSD core runs are eligible; cache presence alone
does not make a scene ready. No downloading nuScenes again is required.

```bash
python scripts/run_shadow_completion_training.py prepare \
  --source-plan "$SOURCE_PLAN" --output-root "$NEW_RUN" \
  --train-scenes 32 --dev-scenes 3 --eval-scenes 3 \
  --clips-per-scene 6 --train-stage 25 \
  --heldout-plan "$OLD_RUN/plan.json" \
  --frozen-eval-manifest "$OLD_RUN/dataset/motion_illumination/seg_eval.jsonl" \
  --negative-maximum-cells 256 --negative-maximum-fraction 0.25

python scripts/run_shadow_completion_training.py run \
  --plan "$NEW_RUN/plan.json" --assets "$FSD_ASSETS" \
  --annotation-workers 4
```

Camera round-robin selection prefers distinct caster tracks within each camera.
Scene ordering is deterministic and independent of teacher scores. Increasing
the number of clips exposes more core observations; no artificial dilation or
student-generated positives are introduced. Actual usable training scene counts
must be read from `suite/suite.json`, not inferred from requested scene counts.

## Negative Coverage

The configurable budget applies only after existing lit-ground, temporal,
caster-projection, occlusion, FSD and proposal exclusions. Spatial tile
round-robin sampling distributes retained observations instead of selecting
only the brightest patch. Raising the budget never turns IGNORE into background
without observed lit evidence. Both student arms use the same budget policy.

`dataset/class_distribution.json` reports per-scene core/expanded pixels and
resized positive/negative counts. `dataset_summary.json` includes positive and
negative scene coverage, the largest scene's positive share, and negative budget.

## Full-Image Predictions

Every suite evaluation now saves `full_image_summary.json`, per-frame full-image
coverage in `predictions.jsonl`, and up to three RGB/prediction/teacher panels per
scene in `full_image_review/`. Full-resolution masks and probabilities are still
saved for every image, including frames with no labelled pixels.

The report separates known-pixel agreement from predicted area on IGNORE pixels.
The latter is coverage, not accuracy. Inspect large predicted road regions and
scene-level variation alongside positive/negative recall, balanced accuracy and
AUROC. Dev selects checkpoints; eval does not choose thresholds or training data.

Student inputs remain single RGB frames. Explicit temporal and receiver-pair
loss weights remain zero. Neither new labels nor full-image diagnostics are
automatically approved for self-training or OCC.
