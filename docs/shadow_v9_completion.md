# Shadow_OCC v9: Discrepancy-Guided Shadow Completion

## Implemented Information Flow

```
RGB + calibrated receiver + GT tracks + frozen current-only/context BEV + DIS
  -> unchanged M1 entity suppression and Caster Memory
  -> unchanged anchor-first M2 appearance / BEV / flow discrepancy proposals
RGB -> pinned official FSD -> independent semantic proposals (source bit 16)
  -> proposal UNION, not multiplication or intersection
  -> unchanged M3 caster/light competition and strict BEV core/owner
  -> near-feasible rendered caster/light envelope in calibrated image pixels
  -> image-space illumination correspondence + contrast-sensitive Graph Cut
  -> visible candidate / strict core / supported expansion / IGNORE
  -> scene-isolated, deduplicated 1/0/255 dataset + confidence/tier sidecar
  -> core BCE + Dice; curriculum-weighted expansion BCE
```

`--shadow-prior-manifest` on `scripts/discover_shadow_video.py` enables v9.
Without it, v8+ discovery and targets retain the existing path. No OCC/VLM
training, SAM2 or ViShadow inference is added.

The cross-space branch remains independent: a low FSD response cannot remove
its seed. A high FSD response alone creates a candidate, not a training label
or a caster ID. Flow residual magnitude is not shadow probability.

## Completion And Supervision

- M3 acceptance, trajectory competition, light bank and owner thresholds are
  unchanged. The BEV target and owner remain strict physical outputs.
- The plausible image envelope is the union of the selected and near-feasible
  M3 hypotheses, rendered from their actual box/light geometry. There is no
  arbitrary mask dilation or new box uncertainty distribution.
- Native calibrated RGB pixels are registered by the ground homography.
  Valid existing exposure fits, chroma agreement and high-pass log-luminance
  texture agreement support darkening/recovery. Missing exposure is missing
  evidence, not an event. These observations do not require valid boundary flow.
- Graph Cut uses the union of semantic, cross-space and native illumination responses for unary
  costs, Lab contrast-sensitive Potts edges, fixed strict-core positive seeds,
  optional explicit negative seeds, and the visible receiver domain. It does
  not render shadow shape from a box. Only seeded connected cut components survive.
- Expansions require a completed visible region, near-feasible physical support,
  independently measured appearance, and semantic or cross-space support (or
  strong observed illumination within a component connected to a physical core).
  Their weight is at most 0.25. Unsupported completed candidates remain IGNORE.
- `shadow_amodal_hypothesis_bev` stores historical regions projected by caster
  geometry onto currently entity-occluded ground. It is an inferred hidden-state
  hypothesis, not a current image observation, training positive, or new owner.
- This version does not implement a learned amodal shape model or 12 Hz sweeps.
  It consumes the actual camera frames in the existing pair/clip cache.

Image NPZ adds `shadow_core_image`, `shadow_expanded_image`,
`shadow_visible_completed_image`, `shadow_confidence_image`,
`shadow_amodal_hypothesis_image` (visible candidates plus inferred hidden area),
`shadow_tier_image`, `shadow_prior_response_image`,
`shadow_plausible_projection_image`, `shadow_native_illumination_image`,
`shadow_native_correspondence_valid_image`, and source bits. Tiers are
`0=unknown, 1=core, 2=expansion, 3=explicit negative`. Expanded image labels are
not silently promoted into strict BEV/owner labels.

The existing dataset exporter merges overlapping clips, preserves conflicts
as IGNORE, records core/expanded pixel counts, and writes `pixel_supervision`
NPZ references with SHA256. The dataset loader resizes and flips confidence and
tiers with the label. Reliable negatives retain the existing independent lit
receiver rule, excluding completed proposals and FSD response >= 0.25.

Training first uses core/negative supervision for two epochs. Expansion BCE
then ramps over three epochs; `--expansion-warmup-epochs` and
`--expansion-ramp-epochs` control this. Expansion is excluded from core Dice.
Its BCE is normalized by expansion pixel count, not confidence mass, so a
uniform 0.25 confidence is not accidentally cancelled. Both reductions support
DDP, including ranks with no expanded pixels. Existing non-layered datasets
have no expansion term and keep the previous loss behavior.

## Data And Weights

| Resource | Required Now | Source / Reuse |
| --- | --- | --- |
| nuScenes images, metadata, poses, GT tracks, existing ground/LiDAR cache | Yes | Reuse existing official data; no duplicate download |
| Pair/receiver, DIS and frozen UniAD BEV caches | Yes | Reuse current clip inputs |
| FSD source | Yes | Official IMU-Group/FSD, commit `b22927c5a2ec25c878a15ecf76344dde474fe503` |
| `FSD_best.ckpt` | Yes | 209,842,750 bytes; SHA256 below |
| EfficientNet ImageNet checkpoint | No | Full FSD checkpoint loads strictly; no backbone download |
| FSD training dataset | No | Frozen inference only |
| SAM2 / ViShadow weights | No | Not part of this implementation |
| nuScenes camera sweeps | Later | Not used by this v9 completion path |

Official FSD SHA256:
`a408c3e9f6cb4c6b304c5cfe742346e97f6434c73e9116b78a362a7180cd58f2`.

FSD `binary_mask` is a nonnegative ReLU response, not a logit or a calibrated
probability. The adapter stores the raw response and uses the official 0.5
threshold. Clipping to [0,1] for completion is a bounded response, not
probability calibration. Its `logit` field is already sigmoid; it is saved
separately without applying sigmoid twice. RGB [0,1] is passed to the model;
ImageNet normalization is internal. Checkpoints load with `strict=True`.

Weights and external source live outside the Shadow Git tree. On the current
HPC they are under:
`/public/home/eng_zhihongli/project/models/shadow_priors/FSD_b22927c/`
with `source/`, `FSD_best.ckpt`, and isolated `python_deps/`.

Reproducible download on another machine:

```bash
python scripts/fetch_shadow_prior_assets.py --output-root /path/outside/Shadow/fsd_assets
python -m pip install --no-deps --target /path/outside/Shadow/fsd_assets/python_deps \
  efficientnet-pytorch==0.7.1 PyMaxflow==1.3.2
```

The full external checkpoint is about 200 MiB. A three-field float32 response
cache has an uncompressed upper bound of `12 * H * W` bytes per unique image
(about 4.12 MiB at 450x800), before NPZ compression. Cache by camera token once,
not once per overlapping clip. Existing raw nuScenes and BEV caches are not
duplicated. Dense completion/confidence outputs need additional space measured
from the pilot; `summary.json` records actual prior cache bytes.

## HPC Execution

Use a new code snapshot and new output roots. Leave the current v8+ training
jobs and frozen results untouched. Set these paths to one existing clip batch:

```bash
export SHADOW_PROJECT_ROOT=/path/to/new/code_snapshot
export SHADOW_FSD_ASSETS=/public/home/eng_zhihongli/project/models/shadow_priors/FSD_b22927c
export SHADOW_PAIRS=/path/to/existing/pairs.jsonl
export SHADOW_CLIPS=/path/to/existing/clips.jsonl
export SHADOW_PRIOR_OUTPUT=/path/to/new/fsd_prior_cache
mkdir -p "$SHADOW_PROJECT_ROOT/logs"
sbatch --chdir="$SHADOW_PROJECT_ROOT" slurm/export_fsd_shadow_priors.slurm
```

After prior export completes, run the existing discovery command with its same
pair, analysis, feature and clip inputs, adding:

```bash
export PYTHONPATH="$SHADOW_FSD_ASSETS/python_deps:${PYTHONPATH:-}"
python scripts/discover_shadow_video.py \
  --pairs "$SHADOW_PAIRS" --clips "$SHADOW_CLIPS" \
  --analysis-root /path/to/existing/flow_analysis \
  --feature-root /path/to/existing/context_bev \
  --spatial-feature-root /path/to/existing/current_only_bev \
  --shadow-prior-manifest "$SHADOW_PRIOR_OUTPUT/manifest.jsonl" \
  --output-root /path/to/new/v9_discovery --stage full
```

Create a new runs JSONL with `method`, `split`, `run_root`, and `pairs`, using
the same scene-isolated train/dev/eval assignment. Then use the existing
`build_shadow_mainline_experiments.py export --auto-negatives`, `suite`, and
`run_shadow_segmenter_suite.py` commands. Do not pass an old v8+ freeze lock as
authorization for v9; the prior and completion protocol are new teacher inputs.

## Literature And Scope

- [FSD, NeurIPS 2025](https://github.com/IMU-Group/FSD): official pretrained
  shadow/opacity responses provide an external semantic and shape prior.
- [Learning Shadow Correspondence, ECCV 2022](https://arxiv.org/abs/2208.00150):
  shadow correspondence motivates separating material flow from shadow changes;
  this code retains geometric correspondence rather than importing its network.
- [ViShadow, TIP 2024](https://github.com/HarryHsing/Video-Instance-Shadow-Detection):
  instance/temporal association motivates separating current visibility from
  hidden state; ViShadow weights and its tracking model are not used here.
- [PyMaxflow](https://pmneila.github.io/PyMaxflow/): existing graph-cut solver
  for the explicitly defined binary completion energy.

The resulting protocol is **no target-domain manual shadow annotation, with
an externally supervised pretrained shadow prior**. It is not shadow-supervision
free across all source datasets. Model responses, projection fits, and held-out
teacher agreement are not independent shadow GT.
