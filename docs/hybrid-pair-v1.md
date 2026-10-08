# Hybrid principal couple: frozen broad end-to-end detection diagnostic

Frozen2026-10-08 before any SAM3.1 inference. This is **detection, principal
pair selection and full original segmentation**, not a4D reconstruction run.
No challenge GT, matched FORM-HOI, source calibration or multiview inputs.

## Scientific question and population

Does automatic candidate geometry plus candidate-ID reasoning resolve the
direct8B-coordinate grounding failure without episode-specific corrections?
Keep original regression9/1/14/7. Add8remaining episodes by the first8ordered
SHA256 digests of `world_reward.hybrid_pair_v1/{episode:06d}`:28/13/3/11/18/10/22/6.
All12 are fixed before execution, not guaranteed historically unseen or held-out.
Failures remain in the denominator and visuals; no replacement/reroll.

## Pipeline

1. Official SAM3.1 multiplex generates concept-level **person** and official
   object-description candidate banks over every original frame, independently.
   Record query/native IDs, source RGB hashes, all final masks, original boxes,
   visibility and capacity-drop diagnostics. Convert only actual final mask bounds
   to exclusive pixel boxes; don't change mask geometry or fill occlusion.
2. Qwen3-VL8B-Instruct receives the official action/description as quoted data,
   16uniform original full-clip frames and untinted automatic per-track appearance
   context/crops from its max-visible original frame. Select a **joint** actor–object
   pair of existing namespaced IDs, or abstain. It cannot produce geometry.
3. Up to3bounded uncertainty rounds focus only on uncertain tracks' actual
   visibility intervals. If there are too many candidates for one prompt, screen
   every page once; no top-k/last-page deletion. Unresolved capacity, malformed
   JSON, unknown IDs or empty role banks result in explicit abstention.
4. Retain the selected native ID masks on the entire original timeline.
   A fragmented/failed track is visible as missing segmentation, not silently
   joined to another physical identity. This first experiment does not claim
   native identity fragmentation is solved.

SAM3.1: September source2345a4ad109ac29c569da749c91d84f10dc08c40;
HFdaa63191845a41281374e725f4c9e51c7a824460, customSAMLicense, grantedHEAD
verified with existing credential held in RAM; no checkpoint read by that probe.
External checkpoint training/challenge overlap remains unverified.
Keep native temporal heuristics initially; capacity64, multiplex16, BF16 SDPA,
no FP8FA3 or compilation. Correct audited source transport defects through a
general tested adapter, not per-episode adjustments: PIL originalRGB input with
native normalization, compatible session initialization arguments, and explicit
native final-batch flushing (the public base wrapper drops `is_last_batch=True`).

Qwen unchanged model revision0c351dd01ed87e9c1b53cbc748cba10e6187ff3b,
nativeBF16 SDPA/processor resolution/output1536, deterministic greedy. Explicit
multi-image row/grid/pixel binding differs from the prior one-image-only helper.
Keep singleton conversations: prior batch8 showed numerical decision variation.

## Efficiency and predeclared gates

All inference/data/rendering Azure-only; baseline read-only, fresh immutable
committed snapshots. One loaded SAM predictor across cohort, reset/reuse video
between concepts; decode CPU prefetch bounded. Persist full banks once; one
loaded Qwen across cohort. No duplicate H100 job or model download.
Inclusive SAM3600s, per-episode600s, Qwen1200s, CPUQA300s. Do not lower inference
resolution or silently discard objects to meet a speed budget. Measure actual
init/hash/decode, all concepts, selection, persistence and preview costs.
Any provenance/normalization/output-grid/oracle violation blocks scientific
execution; native candidate cap drops are recorded failures, not high recall.

QA: all12overview includes frame0 and max-visible selected object frame,
all30original first frames for each clip, plus16uniform full-clip views.
Actual selected mask contours, not VL-generated rectangles. No-success-only
selection. Tiny JPEG≤180k each, private Azure SHA/ETag-bound publication;
load lightweight cohort overview first, detailed panels only when useful.
No source videos/models/full renders transferred to tethered laptop.

This cohort measures automatic coverage/cost and supports qualitative rejection,
**not verified identity accuracy, 3D improvement or victory over CARI4D**.
External annotated validation remains necessary before selecting thresholds or
claiming generalization. Existing V-COCO endpoint annotations cannot alone
validate unique principal actor selection or dense temporal identities.

Primary inspiration, not exact code reproduction:
[AgentRVOS](https://arxiv.org/html/2603.23489v1),
[InterRVOS](https://arxiv.org/abs/2506.02356),
[SAM3.1release](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/RELEASE_SAM3p1.md).

## Execution decisions

- Initial producer2fffd7 failed before preparation/inference: its immutable
  transport closure omitted the referenced SAM Dockerfile. No candidate bank,
  GPU inference or model download occurred. Fix general Dockerfile closure,
  test final-batch adapter, and relaunch a fresh namespace with the same frozen
  cohort/settings. This is an infrastructure repair, not a quality ablation.
