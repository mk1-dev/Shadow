# Receiver Geometry Consistency Audit

Freeze light directions, scoring, label gates and all model training. This tool
reads existing pair JSON, ground masks and BEV NPZ files; it does not re-export
nuScenes, estimate optical flow, rebuild BEV or generate labels. Existing visual
audits remain the source for the outside-footprint case review.

## Quick Hash Index

This mode reads metadata only, without opening images or BEV arrays:

```bash
cd /public/home/eng_zhihongli/project/Shadow
conda activate shadow_annotator
python -u scripts/audit_receiver_consistency.py \
  --source-root outputs/shadow_transition_stages_20261107_211522 \
  --output-root outputs/receiver_hash_index_20261107 \
  --metadata-only
```

`frame_observations.jsonl` groups by scene, camera and camera token and retains
every pair/slot occurrence. It records ground hash, image hash, timestamp,
pair-local ground-model signature, world anchor, LiDAR sample and exported
support/occlusion statistics. A different model signature is not automatically a
different visible mask. A different file hash is not automatically different
binary pixels. No canonical observation is chosen by this audit.

## Pixel And Grid Comparisons

Run the full file-reading audit inside a CPU allocation, or use this CPU-only
Slurm entry. There is no GPU or model stage:

```bash
cd /public/home/eng_zhihongli/project/Shadow
export SHADOW_PROJECT_ROOT="$PWD"
export SHADOW_CONDA_ENV=shadow_annotator
export SHADOW_REBUILD_SOURCE="$PWD/outputs/shadow_transition_stages_20261107_211522"
export SHADOW_EXISTING_SAMPLING_ROOT="$PWD/outputs/shadow_geometry_audit_20261107_224244/sampling"
export SHADOW_ANNOTATION_ROOT="$PWD/outputs/receiver_consistency_$(date +%Y%m%d_%H%M%S)"
export SHADOW_TEMPORAL_RADIUS=3
mkdir -p logs
sbatch --export=ALL slurm/audit_receiver_consistency.slurm
```

The full Python entry also accepts `--sampling-root` to read already completed
sampling JSON rather than recomputing DIS or replaying all illumination events.
The source root is the frozen pair/BEV preparation, NOT the visual-audit directory.

## Output Interpretation

- `ground_mask_comparisons.jsonl`: one comparison per distinct ground-hash pair
  within a camera token. Reports image/timestamp/projection identity, LiDAR sample
  identity, plane normal/height difference, mask intersection, union, symmetric
  difference, IoU and mismatch ratio. Both-empty masks have undefined IoU rather
  than being treated as perfect visible observations.
- `window_consistency.jsonl`: separates original float64 reconstruction versus
  cache, precision-only differences on the SAME original support map,
  original-versus-long-window support selection on the SAME stored grid, combined
  differences and differences restricted to the cached receiver mask. Conflicting
  image/projection identity is flagged, not attributed to ground support alone.
- `window_ground_variants.jsonl`: compares selected versus alternative pair-local
  support for every actual long-window time, not just the central +1 frame.
- `summary.json`: metrics grouped by relative frame, including both mean and
  union-weighted mismatch ratios, empty receiver reasons and existing flow
  failure counts. Window-conditioned comparisons can repeat a camera token; they
  are not independent examples.
- `review/` and `ground_difference_contact_sheet.jpg`: largest nonzero source-mask
  disagreements, showing RGB, mask A, mask B, A-only red and B-only cyan.

The float64 world grid is reconstructed with the original pair basis, anchor and
grid. This is a reconstruction from recorded geometry, not a previously stored
float64 visibility artifact. Old BEV NPZ files do not embed independent ground
hashes; original support provenance is inferred from their verified source pair
manifest. The report makes this limitation explicit.

The `.95` soft-support comparison is a diagnostic counterfactual against the
unchanged `.999` production rule. It is NOT adopted for labels, and an increase in
support is not proof of correctness. Boundary-localized disagreements are useful
to inspect but alone do not prove floating-point error: a changed ground model
can move boundaries too.

Empty receiver classification separates an empty source reference mask, no
reference-grid support, a missing central view, mutually disjoint supported views
and exclusions after a nonempty visibility intersection. None means the RGB has
no road. Raw DIS finite-rate is not guessed from a downstream receiver mask; the
existing visual audit's raw flow NPZ is available for that inspection.

## Before A Frame-Level Cache Migration

Do not silently select one of conflicting masks. Confirm whether the dominant
effect is pair-local ground-model change, coordinate precision, source support
or central-view intersection. A future frame observation should have stable
camera/time/calibration identity and versioned ground/occlusion provenance while
retaining local slope. Image-to-BEV sampling must also key the query grid/anchor.

Flow validity belongs to an EDGE with source frame, target frame, direction and
flow-engine identity; it cannot be a single target-independent flag stored on a
frame. Keep frame observations and pair-flow observations separate. This audit
does not alter either representation or migrate existing caches.
