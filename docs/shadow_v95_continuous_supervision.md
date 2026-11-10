# v9.5: Continuous Supervision and Paired Reproducibility

No teacher candidate, mask, physical acceptance threshold, scene split or
evaluation label is changed. Dynamic evidence can strengthen confidence; its
absence does not revoke inherited positives' right to participate in Dice.

## Objective

`--supervision-mode continuous` includes all known positive/negative pixels in
class-normalized weighted BCE and balanced weighted Dice. Confirmed positives
have coefficient 1, inherited positives 0.7, and extensions retain their saved
confidence capped at 0.25. Previously inferred interiors keep their lower saved
confidence. Existing conflicts/unknowns remain 255 with zero coefficient.

BCE and Dice use confidence-weighted statistics. Class normalization intentionally
preserves positive/negative loss scale: uniformly changing every positive's weight
does not multiply the entire class loss. Weights distinguish relative evidence
within a class. This differs deliberately from v9.4's absolute auxiliary BCE.
Extensions participate from epoch zero under the continuous objective. The
historical extension warmup remains unchanged for v9.3/v9.4 modes.

## Paired A/B/C

```bash
python scripts/prepare_shadow_loss_comparison.py \
  --baseline-suite "$V93/suite/suite.json" \
  --layered-suite "$V94/suite/suite.json" \
  --output-root "$V95/plan" --seeds 0 1 2

python scripts/run_shadow_segmenter_suite.py \
  --plan "$V95/plan/suite.json" --output-root "$V95/training"
```

- A uses original v9.3 sidecars and its BCE/Dice plus extension auxiliary loss.
- B uses v9.4 sidecars and the inherited-weak auxiliary loss.
- C uses B's exact RGB/targets/sidecars with continuous confidence weighting.

The builder verifies identical binary targets, source RGB, scene roles, IDs,
row reliability and common eval reference. It never rebuilds teacher labels.
The runner creates one common ImageNet-backbone/random-head initialization per
seed, verifies each arm loaded the same tensor state, and checks each epoch's
sample-order hash. All arms use the same budget, sampler, dev selection metric
and workers. No threshold is selected on eval.

Default paired execution is FP32, strict deterministic PyTorch/CUDA algorithms,
TF32 disabled and an explicit DataLoader generator. Determinism errors fail the
run rather than silently reverting. CPU/GPU, PyTorch/CUDA/cuDNN versions, model
initialization hash and loader seed are stored in `reproducibility.json`.
History records order hashes, actual updates and AMP skips. Checkpoint file
hashes are in `suite_status.json`. FP32 nonfinite gradients abort before update.

`paired_seed_summary.json` reports each seed, mean, sample standard deviation
and range. A single seed has no estimated standard deviation. Full-image and
scene-level diagnostics remain separate from known-teacher-pixel accuracy.

## Re-evaluate Existing Checkpoints

```bash
python scripts/reevaluate_shadow_checkpoints.py \
  --manifest "$FROZEN_EVAL" --output-root "$V95/saved_checkpoint_review" \
  --checkpoints old_baseline="$V93/training/v8_core/best.pth" \
                new_baseline="$V94/training/v8_core/best.pth"
```

Both models run sequentially inside one Python inference process, using the
same code, manifest, threshold and FP32 inference path. Their hashes and metrics
are saved in `reevaluation_summary.json`. If the gap persists, evaluation
configuration alone cannot explain it; training trajectories and tiny-dev
checkpoint selection still need the paired repeat. This does not assert that
CUDA nondeterminism is already proven to be the cause.

These experiments diagnose the student objective. The subsequent M2/M4 work
remains explanation-driven illumination discovery and dense visible shape
completion, not more hard AND gates or another backbone.
