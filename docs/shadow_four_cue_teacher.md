# Four-Cue Shadow Supervision v9.4

The mainline is 3D entity explanation, geometry-anchored BEV/flow correspondence,
independent RGB illumination changes and pretrained FSD coarse segmentation.
No new backbone, light network or VLM training is introduced. M3 physical
acceptance thresholds, scene selection and the frozen eval reference are unchanged.

## Actual Information Flow

1. 3D tracks and visibility separate entities/occlusions from ground receivers.
   Caster identity and motion remain available for physical association.
2. Same-world BEV anchors provide spatial identity. Flow measures correspondence
   quality; unavailable flow can use geometry, while a reliable contradictory
   observation lowers correspondence quality. Flow residual is not shadow probability.
3. Registered, exposure-compensated RGB/Lab supplies independent illumination
   changes. FSD state changes identify joint onset/recovery evidence.
4. Caster-light reasoning bounds the feasible physical projection envelope.
5. FSD/GraphCut shape and joint events produce direct and connected interior
   labels. Four-cue-supported direct pixels can become confirmed cores without
   overlapping any inherited v8+ core.

## Supervision Tiers

| Tier | Evidence | Loss |
| --- | --- | --- |
| 0 | Unknown | IGNORE |
| 1 | Physical envelope, visible receiver, BEV anchor/correspondence, qualified RGB illumination and joint FSD state event | Strong BCE/Dice |
| 2 | Direct extension or bounded event-anchored interior | Existing low-confidence BCE and warmup |
| 3 | Observed lit-ground negative | Existing negative supervision |
| 4 | Inherited physical weak core without full current corroboration | Absolute confidence-weighted BCE, no strong Dice |

Inherited static shadows remain positive, not background or automatically
discarded. Their default weight is 0.25. Independent spatial observation,
FSD support and physical-envelope agreement raise it to 0.5, but do not claim
a temporal illumination event. Joint-event-confirmed cores have weight 1.
FSD alone cannot create a confirmed core. Missing flow is not itself evidence
against a shadow. Caster uncertainty remains separate from shadow existence.

`shadow_inherited_core_image`, `shadow_weak_core_image` and
`shadow_core_evidence_status_image` retain provenance. Status codes are
0=not inherited/confirmed, 1=inherited weak, 2=static spatially supported,
3=joint-event confirmed (including newly discovered cores).
Dataset export preserves tier 4 through overlapping-clip merges; a confirmed
observation takes precedence. The student logs `inherited_weak_loss` separately.
Its denominator is the inherited pixel count, not confidence mass, so the
confidence coefficient cannot cancel itself. It is active from epoch zero;
extension warmup remains independent.

## Frozen Comparison

Run v9.4 in a new directory with exactly the previous 32 training scenes and
selected clips. Reuse the previous 210-clip raw FSD manifest; recompute physical
annotation, trimaps and students, not RGB, BEV or flow caches. Continue using
the original v9.3 eval trimaps. Never resume v9.3 annotations as v9.4 labels.

The preceding `b13acfc` expanded run completed on 30 usable training scenes /
186 images. Its motion-illumination student had teacher balanced accuracy
0.8175, negative recall 0.6455 and AUROC 0.9831. These are frozen teacher
agreements, not independent shadow accuracy. Whole-image predicted fractions
were approximately 36%, 48% and 61% across the three eval scenes, so known-pixel
scores alone cannot establish full-image segmentation quality.

The v9.4 change is a new supervision experiment, not a demonstrated quality
improvement. Single-frame student inputs and zero explicit temporal/pair-loss
weights remain unchanged. Offline GT/geometry are teacher-only inputs, not
deployment-time student inputs.
