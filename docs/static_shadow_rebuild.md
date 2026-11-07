# Static Shadow Existence and Cache Rebuild

## Structural Fix

The previous factorized policy split existence and ownership only at the final
selector. Upstream code still required both onset/recovery events and >= 0.2 m
motion per interval before even testing projections. Its combined score was
`0.4 * cover + 0.3 * onset_fit + 0.3 * recovery_fit`, so a perfect static projection
could score at most 0.4 and never reach the 0.65 threshold.

The new default factorized path separates these quantities throughout the pipeline:

| Evidence | Shadow existence | Unique caster ownership |
|---|---|---|
| Observed dark receiver, geometry, flow and learned BEV support | Required | Required |
| Static GT caster's projected box | Allowed | Insufficient on its own |
| Per-frame normalized darkness persists over three observations | Supports static branch | Does not determine ownership |
| Reliable exposure and onset/recovery | Supports dynamic branch; not mandatory for static | Required by the dynamic association gate |
| Real-vs-permuted/frozen trajectory and independent light support | Not a veto on existence | Required by default |

Static/dynamic existence uses projection coverage as its own score (default >= 0.65).
The trajectory score remains separate and still controls association. Static
projection evidence produces `YES / unresolved`, not an invented caster assignment.

An invalid exposure fit is no longer applied to the appearance arrays or used as a
window-wide veto. Each image's own local Lab-L ratio supplies persistent darkness;
onset/recovery is disabled when exposure is not comparable. Physical label export
retains those windows for the static branch. `strict-v2` and explicit
`--existence-mode motion` preserve the old dynamic-only experiment.

## Positive and Negative Labels

Factorized-v3 now defaults to `--positive-fusion consensus` (2/3 of near-best
explanations), rather than requiring intersection across every ambiguous direction.
The observed dark mask remains the source of positive pixels; projections cannot
create positives outside that observation.

The `bev-flow-annotation-v2` teacher uses two negative sources:

1. Existing lit endpoints where the same receiver darkens/recovers.
2. A compact lit receiver patch in the same window, within 12 m of an accepted
   positive, normalized brightness ratio >= 0.97 in all three images, and middle
   brightness at least 6 Lab-L units above the positive median. The patch is outside
   competing shadow projections and other selected dark candidates. Its BEV area
   is capped at twice the accepted positive area.

The second source needs neither moving vehicles nor onset/recovery. It produces
`observed_lit_receiver` NO regions with the same native-image, five-view layout as
YES regions. All-class annotated volumes and optional risk masks remain exclusions.
Positive, negative and conflicting observations are merged per physical image;
all unassigned pixels remain IGNORE. V1 datasets remain readable.

## Rebuild Existing HPC Data Without GPU Export

Use the cache root from the failed preparation, for example:

```bash
cd /public/home/eng_zhihongli/project/Shadow_update
export SHADOW_PROJECT_ROOT="$PWD"
export SHADOW_CONDA_ENV=shadow_annotator
export SHADOW_REBUILD_SOURCE="$PWD/outputs/shadow_bev_flow_20261107_0715"
export SHADOW_ANNOTATION_ROOT="$PWD/outputs/shadow_bev_flow_static_v2"
sbatch slurm/rebuild_bev_flow_labels.slurm
```

This CPU job runs regression tests, then:

```text
existing pairs + DIS evidence + mature BEV cache
-> recompute receiver appearance/BEV gates with static support
-> static/dynamic projection labels
-> v2 positive + endpoint/lit-receiver teacher
-> shared segmentation and joint VLM datasets + checks
```

It does not rerun nuScenes export, DIS, or UniAD inference. It does not read old
labels. The new root links the original immutable input directories and stores
only new receiver/label/dataset results. Source and output roots must differ.

Rerunning ONLY `build_physics_shadow_labels.py` is insufficient: old BEV windows
already dropped appearance candidates when exposure failed. The receiver evidence
must also be rebuilt, which the script above does.

To submit rebuild plus downstream training/evaluation in one dependency graph,
set the same environment variables and the local Qwen model path, then use this
INSTEAD OF the standalone `sbatch` command:

```bash
export SHADOW_VLM_MODEL=/path/to/Qwen3-VL-8B-Instruct
bash scripts/submit_bev_flow_pipeline.sh
```

With `SHADOW_REBUILD_SOURCE` set, the submitter selects the CPU rebuild job and does
not require a UniAD environment or checkpoint again. Segmenter/VLM jobs retain
their own afterok dependencies. Use a new output root for every submission. Unset
`SHADOW_REBUILD_SOURCE` to return to full raw-data preparation.

## Check the New Counts

Inspect these separately before interpreting model loss:

- `train/bev_flow/summary.json`: persistent dark receivers and selected persistent dark cells.
- `train/labels/summary.json`: attempted/processed windows, unreliable-exposure windows
  retained for static analysis, absolute projection fits, accepted static/moving
  projections and unique associations.
- `train/teacher/protocol.json`: YES regions, lit-endpoint regions, lit-receiver
  regions and known positive/negative pixels.
- `train/teacher/negative_audit.jsonl`: per-window negative support, brightness,
  contrast and competing-projection counts, including failed negative proposals.
- `data/dataset_protocol.json` and `joint/protocol.json`: per-split scene/class counts.

Do not expect static positives to create nonzero trajectory gain or unique
association counts. Those are intentionally separate outputs now. The earlier
offline count of 41 coverage-supported components is diagnostic evidence for this
fix, not a promised new acceptance count: consensus masks and image support are
still evaluated on the actual cached dataset.

## Regression Tests

`tests/test_static_shadow_teacher.py` covers stationary projection ranking without
events, unresolved static ownership, compact lit negatives without motion, and an
end-to-end synthetic window with invalid exposure through physical labels, teacher
trimaps and both YES/NO five-view training examples. BEV tests separately verify
that failed exposure fits retain static darkness without fabricating transitions.
Tests and empirical validation run in the HPC annotation environment.
