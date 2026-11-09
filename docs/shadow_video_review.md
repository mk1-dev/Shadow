# Frozen Video Change Review

For clean-video blind review followed by result reveal, complete mainline MP4
exports and frozen new-scene replay, see [shadow_video_release.md](shadow_video_release.md).

This tool reads two completed video runs. It does not run M1-M4, change masks,
rescore hypotheses, train models, or turn reviews into pseudo-labels. Geometry
calculations only display frozen grid cells and already-selected caster/light
hypotheses, using saved world grids, caster corners and light directions.

## Run on CPU

Activate `shadow_annotator`, then set the actual Geometry-Anchored and Anchor-first
run roots and the original camera pair manifest:

```bash
python -u scripts/review_shadow_video_changes.py \
  --before-root "$SHADOW_REVIEW_BEFORE" \
  --after-root "$SHADOW_REVIEW_AFTER" \
  --pairs "$SHADOW_VIDEO_PAIRS" \
  --output-root outputs/review_anchor_first_v8plus \
  --context-radius 1
```

The Anchor-first run reported for job 370440 is:

```text
/public/home/eng_zhihongli/project/.shadow_video_runs/mainline_v8plus_anchor_first_20261109_001900
```

Use the actual previous Geometry-Anchored run as `SHADOW_REVIEW_BEFORE`; do not
substitute a different discovery experiment. Both inputs must contain the same
scene/camera/frame windows and receiver grid. Run-local track IDs and numeric
owner IDs may differ. Relocated run directories are supported: local frame NPZ
filenames take precedence over old absolute artifact paths in `frames.jsonl`.
The original RGB paths in the pair manifest must still be accessible.

For clips from more than one manifest, supply both:

```bash
python -u scripts/review_shadow_video_changes.py \
  --before-root "$SHADOW_REVIEW_BEFORE" \
  --after-root "$SHADOW_REVIEW_AFTER" \
  --pairs "$SHADOW_VIDEO_PAIRS" "$SHADOW_VIDEO_VAL_PAIRS" \
  --output-root outputs/review_anchor_first_v8plus
```

No GPU, feature export, dataset rebuild or model inference is needed. Inputs
remain read-only. Choose a new output directory outside both frozen input runs;
existing review material is not overwritten.

For Slurm, set `SHADOW_REVIEW_BEFORE`, `SHADOW_REVIEW_AFTER`,
`SHADOW_REVIEW_OUTPUT`, and `SHADOW_VIDEO_PAIRS`. Optionally set
`SHADOW_VIDEO_VAL_PAIRS`, `SHADOW_REVIEW_CONTEXT_RADIUS` (0..3),
`SHADOW_PROJECT_ROOT`, or `SHADOW_CONDA_ENV`:

```bash
mkdir -p logs
sbatch -p AMD slurm/review_shadow_video_changes.slurm
```

The script requests 4 CPUs and 8 GB memory, with no GPU request. Select an
available CPU partition on your cluster instead of `AMD` when necessary.

## Change Review

Open the generated `README.md` and `contact_sheets/page_*.jpg` first.
`review_manifest.jsonl` groups changes inside full affected connected proposal
regions. A group may contain several changed grid cells or several owners; a
grid cell is not an independent shadow instance. Added, removed and retained
positive cells are counted separately, as are the net positive change and owner
changes. Owner-change categories can overlap positive additions/removals.

Each `cases/<case_id>/` contains:

- `clean_crop.jpg` and `clean_context.jpg`: unpainted original RGB, retaining
  context around the affected region and visible selected caster geometry.
- `review.jpg`: clean RGB, saved before/after weak image masks, grid changes,
  observed appearance, anchor support, seed provenance and selected M3 geometry.
- `owner_changes.jpg`: before/after owner maps with colors keyed by the actual
  caster token, not the run-local owner index. Owner additions are green, losses
  blue and replacements orange. The main change map prioritizes positive
  additions in cyan and removals in red.
- `temporal_context.jpg`: clean and saved-mask views of neighboring actual
  frames. No mask is synthesized for a missing observation.
- `evidence.json`: per-cell signal statistics, original region physics decisions
  and ranking information, missing-field lists and explicit projection sources.
- `evidence.npz`: exact cropped grid masks, owner/track indices, world points,
  available frame evidence, and adjacent-interval evidence including Lab-L.
  `grid_box_xyxy` locates the crop in the original receiver grid.

Unavailable old-version evidence stays missing, not zero. Source-bit histograms
describe provenance, not causal attribution. In particular, bit 8 without other
source bits does not mean the cell had no appearance evidence. Existing
appearance and seed-color support are reported independently.

The selected projection is the saved selected/associated caster's box projected
with the saved light direction. It is not the requested track's generic best
coverage bank and is not a true shadow silhouette. Missing caster geometry or
light directions are recorded rather than replaced with a guessed hypothesis.

## Owner and Continuity Review

`owner_region_transitions.jsonl` builds a frame-local bipartite overlap graph
between positive regions in the two runs. It retains one-to-one links, splits,
merges, many-to-many groups, new-only and removed-only regions. Owner comparison
uses caster tokens. Same numeric IDs are never evidence of the same caster, and
same track IDs across runs are never assumed to identify the same region.
Each transition links to relevant review cases. `same_owner_set` does not imply
the association is correct; it only means the saved token set did not change.
`resolved_new_region` is separate from `became_resolved` on a linked existing
region; discovering another resolved region is not evidence of disambiguating
an old one. Removed regions are also separated from lost resolution on retained regions.
`mask_derived_regions_with_resolved_cells` reconstructs region counts from saved
positive cells and track grids; it is not silently equated to the original
reported region count, which may use different overlap accounting.

`observation_timeline.jsonl`, `track_observations.jsonl`, and
`continuity_manifest.jsonl` cover all frames, including unchanged clips:

- `observed_image_mask`: a current region and its emitted image proposal exist.
- `observed_bev_only`: a current grid proposal exists but RGB refinement has no
  emitted image mask.
- `observed_region_without_mask`: a region record exists without either mask.
- `retained_memory_only`: remembered state exists without a current region/mask.
- `orphan_mask`: a saved mask lacks a current region record; inspect it instead
  of silently treating it as an observation.

Track summaries report actual observed image frames, longest consecutive
observed-mask streak, memory-only frames and orphan frames separately. A track
is counted as observed throughout a clip only if an actual image proposal exists
on every frame. Neither a remembered existence score nor an occlusion state
counts as an observed mask. These are weak observed proposals, not manual GT.
Missing memory traces are explicitly reported as unavailable.

## Independent Judgments

`reviews_template.jsonl` initializes every case as `unreviewed`; added-positive
and removed-positive judgments are separate. Suggested shadow judgments are
`shadow`, `material`, `reflection`, `entity`, `uncertain`, or `mixed`. Suggested
caster judgments are `correct`, `incorrect`, `ambiguous`, or `uncertain`. Record
per-region caster details in `notes` when a case includes multiple regions.
No judgment is automatically approved or fed back into training.

`summary.json` contains behavior counts, not precision/recall or label-quality
claims. `input_fingerprints.json` retains hashes of the original manifests,
protocols, frame/image-mask NPZs, memory traces and saved light banks.

```bash
python -m pytest -q tests/test_shadow_video_review.py
```
