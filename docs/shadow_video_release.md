# Frozen v8+ Review and New-Scene Runs

This workflow leaves M1-M4, all scores, thresholds and light hypotheses unchanged.
It adds clean-video review, qualitative mainline videos and a frozen new-scene
runner. It does not train Segmenter, VLM or OCC, or convert judgments to labels.

## 1. Prepare Blind Review

Use the existing 19-case change review as `SHADOW_CHANGE_REVIEW`. The frozen
Anchor-first run is `SHADOW_VIDEO_REFERENCE`. Pair manifests must still resolve
the original calibrated RGB files; both train and val manifests can be supplied.

```bash
python -u scripts/prepare_shadow_video_release.py blind \
  --review-root "$SHADOW_CHANGE_REVIEW" \
  --run-root "$SHADOW_VIDEO_REFERENCE" \
  --pairs "$SHADOW_VIDEO_PAIRS" "$SHADOW_VIDEO_VAL_PAIRS" \
  --output-root outputs/v8plus_blind_pack
```

Known reference from job 370440:

```bash
export SHADOW_VIDEO_REFERENCE=/public/home/eng_zhihongli/project/.shadow_video_runs/mainline_v8plus_anchor_first_20261109_001900
```

Distribute **only `outputs/v8plus_blind_pack/blind/`** to the reviewer. The
operator retains `private/` and the previous review directory. Directory
separation is not access control; reviewers must not already inspect the model
overlays if an independent blind review is intended.

The blind folder contains full, unpainted RGB videos with opaque filenames,
camera-time playback, `videos.json` and `judgments.jsonl`. It contains no masks,
source-bit counts, owner tokens, physics scores or change-category hints.
`focus_frame_slot` locates the case's time, not a model-projected spatial ROI.
Multiple changed regions can share a clip/frame: Stage 1 records an independent
scene observation; Stage 2 evaluates each proposed region separately.

In `judgments.jsonl`, fill:

- `shadow_visibility`: `clear`, `weak`, `not_observed`, `unjudgeable`.
- `shadow_types`: any of `vehicle`, `non_vehicle`, `ambiguous`.
- `caster_visibility`: `visible`, `outside_view`, `multiple_plausible`, `unjudgeable`.
- `caster_description` and `notes`: visual evidence without model identities.

`not_observed` is not a true negative label. `multiple_plausible` and
`unjudgeable` are legitimate judgments, not annotation failures.

## 2. Record Judgments, Then Reveal

```bash
python -u scripts/prepare_shadow_video_release.py reveal \
  --pack-root outputs/v8plus_blind_pack \
  --judgments outputs/v8plus_blind_pack/blind/judgments.jsonl \
  --reviewer reviewer_01 \
  --output-root outputs/v8plus_revealed_review
```

Incomplete judgments, altered task references and changed clean videos are
rejected before revealing results. The submitted blind judgments are copied
and hashed before model review images are published into this new directory.
This records the workflow order; it cannot prove that a person never saw model
results elsewhere.

The reveal folder contains the saved case panels/evidence and a separate
`overlay_judgments.jsonl`. Record region quality using:

- `shadow_core`, `boundary_extension`, `material_expansion`, `entity_leakage`,
  `non_vehicle_shadow`, `uncertain`, `mixed`.
- Separate `added_positive_quality` and `removed_positive_quality` judgments.
- Owner judgment: `unique_correct`, `incorrect`, `multiple_plausible`, `unjudgeable`.
- `correct_caster_tokens`: only independently established identities; do not
  copy the model's predicted token as the answer.

Original source-bit provenance remains unchanged, including missing provenance.
Neither stage supplies pixel GT or automatic precision/recall measurements.

## 3. Export Complete Mainline Videos

Export all clips first, retaining failures and uncertain examples:

```bash
python -u scripts/prepare_shadow_video_release.py showcase \
  --run-root "$SHADOW_VIDEO_REFERENCE" \
  --pairs "$SHADOW_VIDEO_PAIRS" "$SHADOW_VIDEO_VAL_PAIRS" \
  --output-root outputs/v8plus_complete_videos
```

For a separate qualitative set of 3-5 reviewed examples, add
`--clip-ids CLIP_ID_1 CLIP_ID_2 CLIP_ID_3` and choose another output directory.
The manifest explicitly marks operator-selected examples rather than implying
that they represent overall quality.

Each clip exports a clean video and a nine-panel mainline video:

1. Clean RGB and camera timestamp.
2. Saved entity-core suppression overlay.
3. Same-world BEV anchor consistency.
4. Raw observed flow versus ground-geometric expected flow.
5. Observed appearance evidence.
6. Saved RGB-refined continuous proposal.
7. Saved caster trajectories/corner samples and selected light footprints.
8. Saved weak image target, including its boundary.
9. Separate Shadow-BEV proposal and positive layer, not occupancy.

The caster geometry is reprojected for visualization only, without reranking.
Historical caster centers are projected into the current camera to display
their world-space track; they are not past-image optical flow. The flow panel
shows raw observations, not a claim that all vectors are reliable. The last
frame explicitly has no outgoing flow. Missing maps remain missing/black.
`showcase_manifest.jsonl` retains the observed-versus-memory-only timeline and
input artifact hashes. No mask is painted from memory-only retained state.

Videos use the actual camera intervals, duplicating frames when needed on the
constant-rate MP4 timeline. The image targets remain saved weak targets, not
new segmentations created by the export tool.

## 4. Freeze the Current Algorithm

Create the lock once, on the code version intended for the next run:

```bash
mkdir -p outputs/v8plus_freeze logs
python scripts/run_frozen_shadow_video.py freeze \
  --reference-root "$SHADOW_VIDEO_REFERENCE" \
  --development-roots "$SHADOW_OTHER_DEVELOPMENT_RUN" \
  --output outputs/v8plus_freeze/lock.json
```

Omit `--development-roots` if none are needed, or pass all earlier development
run roots. The reference scenes are always excluded. Include other historical
tuning scenes as well: excluding only the last 12 clips does not undo earlier
scene exposure.

The lock records the current Git commit, algorithm-file hashes, reference
protocol hashes, M1-M4 configuration, receiver grid, model/checkpoint/config
hashes and excluded development scenes. Fixed internal defaults must match the
reference; they are not silently substituted. The operator must verify that
the chosen reference was produced by the intended source: older protocols do
not themselves prove historical source identity.

## 5. Run Unseen Scenes Without Retuning

Export pairs, DIS analysis, temporal/context BEV and current-only BEV caches for
the new scenes using the existing preparation/export pipeline. This runner
does not create missing caches or reuse the old scene's observations.

Create `new_scene_batches.jsonl`, one row per prepared input set:

```json
{"id":"heldout_val","pairs":"/data/new/pairs/pairs.jsonl","analysis_root":"/data/new/flow","feature_root":"/data/new/bev_temporal","spatial_feature_root":"/data/new/bev_current_only","clips":"/data/new/clips.jsonl"}
```

Each clip uses the original discovery schema:

```json
{"id":"new_clip_001","scene_name":"scene-XXXX","camera":"CAM_FRONT","start_sample_token":"SAMPLE_TOKEN","track_id":"INSTANCE_TOKEN","num_frames":7}
```

Paths can be absolute or relative to the batch manifest. New-scene selection
must be decided before inspecting this run's outputs. Scene-disjoint selection
is enforced; representativeness still depends on the submitted selection.

```bash
python -u scripts/run_frozen_shadow_video.py run \
  --lock outputs/v8plus_freeze/lock.json \
  --batches /data/new/new_scene_batches.jsonl \
  --output-root outputs/v8plus_unseen_scenes
```

Or submit the CPU replay against existing caches:

```bash
export SHADOW_VIDEO_FREEZE_LOCK="$PWD/outputs/v8plus_freeze/lock.json"
export SHADOW_VIDEO_NEW_BATCHES=/data/new/new_scene_batches.jsonl
export SHADOW_VIDEO_NEW_OUTPUT="$PWD/outputs/v8plus_unseen_scenes"
sbatch -p AMD slurm/run_frozen_shadow_video.slurm
```

The runner checks development-scene overlap, source identity and selected BEV
cache metadata before invoking the unchanged `discover_shadow_video.py`. It
records the exact command and manifests in `frozen_run.json`, and checks actual
exported protocols, clip identities and model signatures afterwards. Discovery
continues verifying referenced RGB/ground/flow/feature hashes as usual. Exit 0
without matching frozen settings is recorded as `verification_failed`, not a
successful frozen replication. Existing output folders are never overwritten.

## Local Verification

```bash
python -m pytest -q tests/test_shadow_video_release.py tests/test_shadow_video_review.py
```

Tests use synthetic frozen outputs and exercise real OpenCV video encoding and
decoding. New-scene batch execution is mocked in unit tests; HPC replay and
real-image human judgments remain separate work.
