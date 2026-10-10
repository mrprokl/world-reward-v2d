# Temporal depth and contact: literature, isolated tests, decision

2026-10-10. Primary arXiv papers and author repositories, independent depth and
contact audits. Research through September 2026 plus explicitly identified
Open-CHOIR v4 update of October 7. No claim that newest means best for Track 1.

## 1. Depth coherence, not merely smooth trajectories

| Primary resource | Mechanism relevant to the observed bottleneck | Decision |
|---|---|---|
| [SCOPE, SIGGRAPH 2026; June 19](https://arxiv.org/abs/2606.21300), [code](https://github.com/zhengzhang01/SCOPE) | Sequence-shared affine point-map gauge, temporal decoder and multi-timescale derivative supervision preserving true changes. | Highest-priority 2026 geometry challenger; not a metric-depth drop-in. |
| [Video Depth Anything, CVPR 2025 Highlight](https://arxiv.org/abs/2501.12375), [code](https://github.com/DepthAnything/Video-Depth-Anything) | Temporal head and long-context/keyframe processing; separate relative and metric checkpoints. | Efficient metric-video reference for the next controlled upstream comparison. |
| [Learning a Depth Covariance Function, CVPR 2023](https://arxiv.org/abs/2303.12157) | Depth correlations matter in geometric optimization; independent sample weighting overstates information. | Supports covariance-aware evidence, not direct copying of the learned GP. |
| [ICDepth, ECCV 2026; July 2](https://arxiv.org/abs/2607.01677), [code](https://github.com/alexhe101/ICDepth) | Video diffusion with RGB conditioning kept distinct from the depth output. | Secondary: relative disparity, heavier inference and checkpoint restrictions. |

SCOPE supervises predicted temporal derivatives against actual derivatives;
this is different from suppressing every change. Its paper reports 300 frames
at 378x672 in 7.68 s using 76.53 GB on H20/FP16, **not our H100 measurement**.
Default public inference uses chunking, so do not assume this paper configuration
or runtime applies unchanged. Shared affine geometry must be tied to one allowed
clip gauge, never per-frame alignment or challenge calibration.

Current hypothesis: frame-correlated inferred Z makes the fitter overly
confident in a changing common error. Residual correlation is not ground-truth
depth error. Temporal prediction can improve the evidence itself; covariance
can only improve how uncertain evidence is counted.

## 2. Contact coherence requires correct relative placement

| Primary resource | Mechanism relevant to the observed bottleneck | Decision |
|---|---|---|
| [Open-CHOIR, May 2026; v4 October 7](https://arxiv.org/abs/2605.20992v4), [code](https://github.com/hxwork/CHOIR) | Correct main hand-object ray-depth ambiguity first; jointly optimize with soft barycentric anchors, normals and periodically rebuilt contact memory. | Closest 2026 method to the actual problem. Adapt automatic-contact components, not its manual mask seeding. |
| [Real2Sim in HOI, May 2026](https://arxiv.org/abs/2605.14462) | Human-anchored fitting, semantic surface/SDF contacts and physics-aware refinement. | Structural cross-check; only use signed distances after verifying solid topology/orientation. |
| [Physical Interaction, SIGGRAPH Asia 2022](https://arxiv.org/abs/2209.10833) | Confidence/forces distinguish static from sliding contacts; fixing a wrong initial contact is harmful. | Conceptual support only: RGB-D sensor/known object assumptions differ from Track 1. |
| [CARI4D, December 2025; revised April 2026](https://arxiv.org/abs/2512.11988v3) | Automatic contact-aware coupling and temporal refinement. | Preserve as comparator; low acceleration alone is not interaction fidelity. |

Nearest-vertex switching is not itself a bug: real sliding on dense geometry
changes vertices. The structural risk is unstable or wrongly attached evidence
coupled to independently misplaced human/object geometry. Open-CHOIR separates
placement correction from contact fitting and retains image evidence. It
explicitly excludes re-grasping/nonrigid objects from its claimed scope; it does
not establish a complete general-purpose solution for every challenge motion.

**RIFE is not this fix.** [ECCV 2022 paper](https://arxiv.org/abs/2011.06294)
estimates intermediate RGB flow to synthesize frames, not independent depth,
contact or pose measurements. Invented frames cannot increase geometric
confidence. Keep the original indices/FPS/full timeline; interpolation may
initialize unknown latent poses or improve presentation, never masquerade as
observations or evidence of recovered motion.

## 3. Implemented isolated hypotheses, baseline unchanged

Producer `fdd3d5f34057651363706c358438a7e79356aebe`:

- `depth_covariance.py`: analytic within-frame rank-one covariance
  `Sigma = sigma_iid^2 I + sigma_common^2 11^T`; O(M) symmetric whitening,
  actual supported counts, no dense matrix or generated observations. This is
  neither DepthCov's learned spatial GP nor a temporal denoiser. Across-frame
  AR(1) noise is not modeled. Sigma-common=0 is exactly the old route.
- `contact_patch.py`: bounded anatomical-ID pool per active interval and soft
  existential surface-distance residual. Points remain frame-varying; no
  tangential lock, zero velocity or manually selected grasp. This is **not full
  Open-CHOIR**: no relative-depth rectifier, barycentric object-anchor memory,
  normal compatibility, signed penetration or joint human refinement yet.
- Both factors are explicit opt-ins, default disabled. Geometry, K, scale,
  activations and original T stay fixed. No full4D/model rerun before gates.

## 4. Actual Azure numerical stress and fail-fast decision

Frozen `sequence_evidence_stress_v1.json`: 12 cases, independent DEV/RESERVED
noise seeds, two variants per case = **48 fits**; 24 frames/case at 30 FPS.
One shared observation bank per comparison, two CPU workers; no sweep or
challenge/model/data mounts. CPU container completed in **48.40 s**; every fit
converged and owned container removal was verified. Transport latency is extra.

| Candidate | DEV target point-error ratio | RESERVED target point-error ratio | Frozen gates | Decision |
|---|---:|---:|---|---|
| Within-frame covariance | 0.3409 | 0.6196 | Fail both splits | Do not adopt fixed common uncertainty globally. |
| Soft anatomical pool | 0.7865 | 0.8625 | Pass this assay both splits | Keep isolated candidate, not real-data/production validation. |

Covariance fails the IID control: point error **1.726x DEV / 1.592x RESERVED**;
reserved combined local-outlier case is **1.250x** baseline. Correct correlated
uncertainty helps the intended regime but mis-specified uncertainty discards
useful absolute-depth information. Do not select its coefficient or enablement
from challenge episodes. Upstream coherent video depth and external uncertainty
calibration are needed before another production candidate.

Soft contact target aggregate improves **21.35% DEV / 13.75% RESERVED** in this
numeric assay; absent-contact output is exactly unchanged. Summed contact fit
time was **75.21 s vs 8.68 s** over the two splits (parallel wall differs), an
**8.67x** penalty requiring profiling before adoption. No precision/sampling
reduction is justified by these results.

### Limitations found independently, not hidden behind passing gates

- Known authored mesh/K, exact authored first-frame gauge anchor, and noisy
  truth-derived initial priors are given equally to both methods. These are
  legitimate manufactured controls, **not truth-free RGB inference**. No
  challenge ground truth is used. Points are random interior cube points, not
  physically visible/rendered surface measurements.
- `rolling_contact` is actually two-axis planar sliding; regrasp is an instant
  face change separated by two inactive frames. All eight candidates mostly
  touch one face, with J=pool=8; pool coverage and changing membership are not
  realistically stressed. Existential minimum gap does not certify all contacts
  or signed penetration.
- The frozen gate named `all_moving_amplitudes_retained` checks **endpoint
  displacement only**. Removing the true 3 Hz oscillation entirely retains
  0.9023 endpoint displacement and passes, while path retention is only 0.6598.
  Reserved covariance `common_fast` path retention is 0.8636 despite an endpoint
  of 1.0394. Motion quality cannot be claimed from this gate.
- Both seeds use the same simple geometry/noise family; RESERVED is held-out
  numerical noise, not a held-out real video benchmark. Improvements are not a
  verified win over CARI4D or a challenge score.

The sealed Azure report is preserved without rewriting its metadata or numeric
gates: 29,032 bytes, SHA256
`9224c159625c4e68d9d999a7eb39ef8973ed994d1051644f6a439b985e9d780f`.
Path `/srv/scenesmith/world-reward/results/sequence-evidence-stress-fdd3d5f34057651363706c358438a7e79356aebe/report.json`.
Later source clarifies authored-anchor disclosures and relabels the endpoint
check without changing the frozen thresholds. Counterexample regression tests
make the limitation explicit. Historical run identity remains unchanged.

## 5. Selected architecture and next acceptance boundary

1. **Video-coherent depth evidence**: benchmark metric Video Depth Anything
   against SCOPE after checkpoint/source/license/overlap checks; one RGB-only
   clip gauge, no per-frame scale/shift or reference calibration. Evaluate axial
   error and temporal error on external RGB-D only after sealing predictions.
2. **Rectify relative human-object placement before contact association**, using
   allowed image/geometry evidence; then soft surface anchors with normal
   compatibility, explicit visibility and controlled memory. Jointly retain
   human/object image evidence so contact cannot drag the object away.
3. **Independent acceptance**: visible surface observations, true edge/curved
   rolling, mixed contact/noncontact anatomy, gradual release/regrasp and reduced
   pools. Freeze full-path, rapid amplitude/phase, rotational and contact plus
   visual-fidelity gates before any new held-out evaluation. No retuning on v1
   RESERVED or challenge outputs. Inspect tiny Azure-served QA only after
   numerical/integrity gates; full renders/models remain remote.

Licenses are separate for code, checkpoints and third-party models. Author repos
identify SCOPE MIT with third-party notices; VDA Small Apache-2.0 and Base/Large
CC-BY-NC-4.0; CHOIR authored code MIT with third-party terms; ICDepth code
Apache-2.0 but weights CC-BY-NC. Exact checkpoint provenance/overlap remains
unverified, so none was downloaded/adopted in this experiment. Our two analytic
primitives are independently implemented, not vendored upstream code.

## 6. Verified execution optimization, no quality retuning

Producer `05b439d32daad252b1a8aff450118da5d29e5bc6` groups points sharing the
**identical conservative face set** within bounded 32-point windows. The
continuous triangle computation, surface, point order and support do not change;
no candidate union, approximate distance or persistent optimizer cache. Default
legacy execution remains single-point, only the opt-in patch route batches.
Independent read-only code audit found no critical correctness issue; disparate
face sets may incur grouping overhead, so the gain is not universal.

Actual Azure repeat: same 48 fits, config, seeds and inputs, **23.68 s** wall,
versus 48.40 s before. Summed soft-contact fitter time **75.21 -> 28.72 s**,
**2.618x speedup**. All per-case metrics and convergence flags are **bit-identical
in this pinned runtime**; all four gate decisions unchanged. Both mechanisms
retain their original quality decisions; no new tuning or model run.

Optimized report: 29,647 bytes, SHA256
`cd942c608178c29803117bfe32c61672716fb010acbb0787db6fde2f11d1e764`.
Path `/srv/scenesmith/world-reward/results/sequence-evidence-stress-05b439d32daad252b1a8aff450118da5d29e5bc6/report.json`.
Owned container absence verified, exit 0, GPU unused; no heavy transfer locally.

**342 targeted tests pass in 6.99 s** after disclosure/batching changes; whole
repository suite is not claimed green (historical failures recorded in saved
gain audit). Performance evidence cannot strengthen the quality/generalization
claim. Full production 4D baseline and submission remain unchanged.
