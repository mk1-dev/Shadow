# Caster-Conditioned Transition Evidence

## Why Change The Competition

The HPC audit exported 431 train and 529 val components, but competition retained
only 2/0 positives. Almost every winner was `ground_material`. The v1 material
score was persistent darkness times one minus projection strength. This confuses
a static shadow with a dark material: both can stay unchanged at the same world
receiver. Decoded 3D entity motion mostly provides exclusion and is rarely present
on the already ground-filtered candidates.

The update keeps the BEV feature, DIS, geometric receiver and GT caster inputs.
It changes how available evidence enters the explanation, not the default
projection/association acceptance thresholds.
The score distributions do change. Previous calibration or precision estimates
are not transferred to this scorer merely because numerical thresholds are unchanged.

## Research Connections

- [ViShadow, TIP 2024](https://arxiv.org/abs/2211.12827): uses shadow/object pairing
  across frames and association-cycle training. We take the separation of
  instance association from mask detection; our matcher is GT-trajectory/light
  projection, not ViShadow's trained contrastive model or cycle loss.
- [MetaShadow, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_MetaShadow_Object-Centered_Shadow_Detection_Removal_and_Synthesis_CVPR_2025_paper.html):
  motivates object-conditioned shadow reasoning. Here existing 3D tracks condition
  the search; no MetaShadow weights or synthesis network are bundled.
- [Cast and Attached Shadow Detection, ECCV 2026](https://github.com/cvlab-stonybrook/ShadowDetectionAttached):
  separates cast and attached shadows and iterates geometry/light reasoning. We
  keep entity-surface explanations separate from cast shadows on ground receivers.
  Our direction bank is not their learned light estimator.
- [SILT, ICCV 2023](https://github.com/hanyangclarence/SILT): addresses noisy
  shadow annotations through iterative tuning. Optional exported shadow
  probabilities remain an appearance prior, not replacements for geometry or
  automatic self-labeling truth.

These are design connections, not claims that the papers use our scoring rules.

## Implemented Changes

1. **Shared persistence, explicit nonresponse.** Stable darkness alone contributes
   neither a material label nor a shadow positive. Material evidence requires a
   valid predicted illumination transition with no observed response. Nonresponse
   is limited to exposed, visible receivers and remains an explanation score, not
   an exported NO label. Ambiguous static appearances remain unknown.
2. **Available signed transitions.** Factorized-v3 defaults to
   `--temporal-evidence available`. The transition fit averages observed onset and
   recovery sides, instead of assigning a missing side zero fit. The real order
   still competes against every wrong-time permutation and a frozen-middle
   trajectory with exactly the same observed sides. No events means no temporal
   gain. `--temporal-evidence complete` retains the complete-event comparison.
3. **Per-interval exposure.** Valid exposure in one interval survives a failure in
   the other. Old caches reconstruct these flags from their pair metrics. Invalid
   intervals supply neither boundary evidence nor material nonresponse.
4. **Continuous caster motion.** A moving hypothesis's temporal gain is visible to
   competition before complete association approval. Signed observed boundaries
   are matched to its three projected footprints, with one BEV-cell tolerance.
   Projection selectivity and matched transitions jointly rank caster/light
   explanations. Optical flow still supports receiver correspondence; it is not
   shadow velocity.
5. **Existence versus ownership.** Static geometric alternatives do not by
   themselves erase a temporally unique owner that passed the association test.
   Competing temporal owners and independent-light requirements are retained.
6. **Same-population attribution.** Each component records the winner and support
   decision with the mature entity-motion contribution removed. Audit totals show
   whether that contribution actually changes rejection, rather than attributing
   the entire old-projection-to-new-competition difference to the 3D head.

`receiver-explanations-v1` remains the geometry-cache layout. New decisions carry
`receiver-competition-v2-transition-evidence`; label protocols also record the
temporal mode and source hashes. Both old and new decisions are readable by the
analysis script.

## CPU Replay On The Completed Audit

No mature-model export, DIS recomputation, new checkpoint or GPU is required.
Use the completed audit as source, not the older feature-only preparation root:

```bash
cd /public/home/eng_zhihongli/project/Shadow
export SHADOW_PROJECT_ROOT="$PWD"
export SHADOW_CONDA_ENV=shadow_annotator
export SHADOW_REBUILD_SOURCE=/public/home/eng_zhihongli/project/Shadow_update/outputs/shadow_3d_flow_audit_20261107_132014
export SHADOW_ANNOTATION_ROOT="$PWD/outputs/shadow_transition_v2_$(date +%Y%m%d_%H%M%S)"
export SHADOW_REUSE_SOURCE_ANALYSIS=1
export SHADOW_REUSE_ANALYSIS=0
export SHADOW_MATURE_OBJECT_MODE=required
export SHADOW_RECEIVER_COMPETITION=on
export SHADOW_RECEIVER_AUDIT_ONLY=1
export SHADOW_CANDIDATE_SOURCE=photometric
export SHADOW_TEMPORAL_RADIUS=1
mkdir -p logs
sbatch slurm/rebuild_bev_flow_labels.slurm
```

`--reuse-source-analysis` links frozen BEV windows into a NEW output and rebuilds
only label decisions and competition reports. It preserves the candidate
population. It must not be combined with `--reuse-analysis`, which retries an
existing output. The Slurm job runs regression tests on HPC before replay.

Compare these fields under `<train|val>/competition/summary.json`:

- `winner_counts`: fewer unjustified material decisions is not automatically
  more correct shadows; inspect unknown and shadow outcomes separately.
- `observed_transition_components` / `matched_transition_components`: whether
  actual signed changes provide usable caster-conditioned support.
- `mature_3d_counterfactual`: identical-candidate changes due only to mature
  entity-motion evidence.
- `baseline_accepted`, `competition_supported`, `exported_positive`: survival
  before competition, after competition, and after mask fusion respectively.
- Label summary `unique_weak_association`: actual exported association support,
  not merely the existence of GT object tracks.

Per-component `best_projection_transitions` includes observed/expected pixel
counts, match fractions and nonresponse. `without_mature_3d` and
`complete_temporal` are separate ablation arms in `configs/bev_flow_ablations.json`.

With the explicit photometric/three-frame settings above, this replay does not
expand the source candidate pool. The newer [geo-temporal light workflow](geo_temporal_light.md)
adds explicit caster-driven candidate expansion and longer observations; evaluate
those as separate arms, not as an identical-population comparison.
No improved precision or training readiness is claimed before HPC results.
Local execution of model/tests was not performed for this update.
