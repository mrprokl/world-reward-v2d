# Frozen 4D: causal diagnosis, not parameter hunting

User QA, 2026-10-08: an apparently resting object jitters; episode14 appears
to contain the wrong/bizarre reconstructed object. These are failure reports,
not manual labels or permission to change individual test predictions.
Preserve the complete random diagnostic, its failed episode7, and its hashes.

## Mechanisms established by code (before measurement)

- SAM3D Objects generates one mesh from **frame0**, seed0. Qwen/SAM2 may choose
  a later seed, but generation still uses the propagated frame0 mask. Geometry
  fidelity certification preserves that proposal, not its semantic correctness.
- Full-T ICP reinitializes translation from the median segmented monocular
  depth. Only the previous rotation survives as an extra candidate. Estimated
  depth, changing visible surface and occlusion can therefore move a static
  object even with a correct mask. This is a mechanism, not yet measured cause.
- Viterbi selects the supplied candidates using full silhouette IoU and a
  first-order speed penalty. It has no caller-supplied confidence, persistent
  material correspondences or contact/support state, and does not optimize
  continuous translations. Whole rendered silhouette versus visible mask is
  occlusion-confounded; depth ambiguity is not solved by silhouette alone.
- CoCoNet uses96-frame windows with first-occurrence assembly. Window-boundary
  artifacts are a hypothesis to measure, not an assumed cause.
- The native full refinement moves object translation and human body rotation
  controls; object rotation, hands, roots, identity, geometry and K stay fixed.
  A wrongly oriented or wrongly shaped object is therefore not repaired by it.
- The viewer's floor is a fixed display-only plane inferred from humanY. It
  neither modifies geometry nor supplies physical friction/ground constraints.
  Do not snap the output to this plane, or interpret it as measured ground.

## Read-only measurement, frozen before execution

`infra/full4d_diagnose.py` runs on Azure CPU only, offline, with original data
and predictions read-only. It keeps all original frames of episodes9,1,14:

1. Compare original/aligned initializer, native forward, native refined and
   final export. Hash-bind trusted native deserialization; compare identical
   re-expressed vertices to disambiguate object-frame changes from real motion.
2. Measure centroid and complete-vertex rigid RMS steps, SO(3) rotation steps
   and accelerations using30fps; report whole-clip distributions and fixed
   two-second blocks, **not human-labelled static intervals**. Equal-vertex
   RMS is explicitly not an area-weighted challenge metric.
3. Report actual first-occurrence boundaries, mask area, low-resolution
   common-visible-region RGB flow and uniformly sampled estimated depth.
   Optical flow, monocular depth and masks remain uncertain proxies.
4. Produce9 uniformly spaced rows: originalRGB, automatic object mask, exact
   final mesh overlay. No crop chosen manually, mesh replacement or pose fit.
5. Verify original bytes after reading. Reject export/representation mismatch;
   separate that bug class from inference error. No confidence/accuracy PASS.

## Primary research available by 2026-09-30

| Source/version | Mechanism relevant here | Boundary |
|---|---|---|
| [CARI4D v3, 2026-04-19](https://arxiv.org/html/2512.11988v3), §3.1–3.4, §8 | Explicit frame0-visibility assumption; RGB+RGBD pose hypotheses, human-occlusion-aware silhouettes, forward/backward reacquisition, then temporal/contact optimization | Our ICP is not FoundationPose. Large pose flips can survive CoCoNet. Existing native losses do include temporal regularization; not a friction simulator. NVIDIA research code/weights, dependencies separately licensed. |
| [SAM3D v2, 2026-06-02](https://arxiv.org/html/2511.16624v2) | Strong single-image generative shape/layout prior | A plausible proposal does not establish observed geometry or temporal stability. Custom SAMLicense; checkpoint/challenge overlap unverified. |
| [CHOIR v3, 2026-05-28](https://arxiv.org/html/2605.20992v3), §7.1/7.3 | Shape/scale-latent-fixed pose follow, sequence fitting, phases from point motion/masks | Not a full-body drop-in. Published angular thresholds, seed retries/interpolations are not adopted. Its quick-start manual masks are prohibited here. [Author code, pre-cutoff commit](https://github.com/hxwork/CHOIR/tree/7a0612b434ad2cb2edc186683fbd5b3f6241b573) MIT does not clear all weights/dependencies. October7 paper revision excluded from September claims. |
| [MV-SAM3D v2, 2026-04-09](https://arxiv.org/html/2603.11633v2) | Visibility/entropy-weighted evidence over a shared object latent | Needs poses in a common rigid-object frame. Frames of a moving object are not static scene views; DA3 camera-scene poses alone are wrong. No challenge multiview assets. SAMLicense/dependency rights and overlap need separate audit. |
| [RHINO v1, 2026-05-16](https://arxiv.org/html/2605.17014v1) | Background camera/scene inference separated from foreground motion, common reconstruction frame | Not a validated object–ground support solver. Motion-implies-contact heuristics fail for holding/leaning on stationary objects. Apache author code does not clear dependency weights. |

[4DAnyone v1, 2026-08-20](https://arxiv.org/html/2608.20335v1) is deferred:
human-centric generated views can inherit a wrong body estimate (§H); they are
not new measurements of object geometry or metric contacts. See the existing
release/license audit, not a new model acquisition.

## Minimum general correction, with causal ablations

**First identify whether the mask, shape or pose is wrong.** A correct mask plus
wrong mesh calls for shape inference; a wrong mask requires general identity
association. Temporal smoothing must not hide either failure.

- Shape experiment: automatic visibility/coverage/sharpness selection of a
  bounded set of source frames; generate alternative clip-constant proposals;
  assess unseen-in-fit frames of the **same monocular** video using visible
  contours/depth/appearance. This is self-supervised consistency, not held-out
  performance. Do not assume largest mask is unoccluded or correct.
- Pose experiment: compare CARI-style RGB/RGBD multi-hypothesis tracking and a
  persistent-RGB-correspondence SE(3) graph under the **same frozen mesh, K,
  gauge, masks and timeline**. Include observation uncertainty; optimize all
  poses together rather than resetting translation to noisy depth each frame.
- Support experiment only after pose/geometry: infer camera-relative static
  evidence and conditional support/contact states automatically. A resting
  prior must permit observed sliding, lifting and fast motion. Physical ground
  must be estimated from permittedRGB evidence, not the visualizer plane.

Choose numerical settings on a separate qualified nonchallenge cohort. Test
observations-only, temporal coupling, then conditional support separately.
Reject lower acceleration that erases true motion, loses frames, shrinks the
mesh or worsens independent3D/contact errors. Reuse frozen upstream artifacts,
batch independent hypotheses/views on H100; no full pipeline rerun to inspect
an unchanged result. A future visual improvement is not a verified leaderboard
gain or permission for per-episode repairs.


## Actual saved-only result, 2026-10-08

CPU diagnostic `5adcb031d5d5a09c4b06d962a03c5f1fe6b906cb` completed all
415/668/442 original frames in about61s of analysis, with zero models, zero
optimizers and no prediction writes. Offline read-only sources were rehashed.
The representation-equivalence and exact refined/export equality gates pass
for all three clips: exported poses and display-only ground are not the source
of the numeric jitter. Native forward/refinement can still change wrong poses.
Two earlier diagnostic infrastructure failures remain failures (host scientific
import, then historical writable frontend/pin-schema assumptions); neither
changed inference. Numeric settings and population were not retuned.

| Fixed first2s, not a manually labelled static segment | Initializer | Forward | Refined/export | RGB flow median, 320px-wide proxy |
|---|---:|---:|---:|---:|
| ep9 centroid step median | 18.35mm/frame | 20.46mm/frame | 9.22mm/frame | 0.005875px/frame |
| ep14 centroid step median | 40.94mm/frame | 37.69mm/frame | 16.36mm/frame | 0.002232px/frame |

This shows the mismatch already exists before the learned/refinement stages;
refinement attenuates translation but does not identify the true resting state.
Flow under the saved mask is not ground truth or proof of static world motion.
In ep14 the forward network additionally introduces a median5.372deg/frame
rotation over that same first2s, versus zero initializer rotation; refinement
freezes rotation exactly, so the error cannot be corrected there. Whole-clip
medians are not jitter estimates because they include real manipulation.
The ep9 boundary192 step rises from46.93mm initializer to79.98mm forward then
68.76mm refined; this alone does not prove window-seam causality.

**ep14 is an upstream identity failure, not just poor shape.** The permitted
metadata requests a black frying pan picked up from a table. All nine original
Qwen output records contain exactly the same person and object boxes despite
the source actor moving. The saved SAM2 mask remains at approximately
x632..731/y663..695 in every uniform view, while the pan moves with the actor.
The additional automatic-box viewport shows the true pan **below the original
Qwen box** at frame0; SAM2 segments background between tripod legs, not the
pan or tabletop. Thus initialization is wrong already, not merely later drift.
Its persistent background region is consistent with the supplied wrong seed.
The multi-image response repetition and wrong frame0 location must be isolated
from image/coordinate transport; a SAM3 upgrade is a hypothesis, not a diagnosis.
The code parses raw per-view records without a box-copy fallback, then uses
only the first joint seed to initialize SAM2. Later Qwen observations never
correct or independently validate propagation. Therefore a plausible first
box/nonempty full-T mask was incorrectly treated as sufficient handoff evidence.
The generated frame0 mesh has extents approximately0.748/0.429/0.649 inferred
metres and poor initializer silhouette IoU0.156..0.206 across uniform views;
these are additional uncertain shape/pose diagnostics, not true dimensions.
Final full-mask IoU0..0.053 is occlusion-confounded, not a calibrated score.
Replacing SAM3D alone would keep generating/tracking the wrong observations.

**Revised priority:** independently ground per-view observations with one
unchanged task-conditioned algorithm; first verify original decoded-image and
encoded image/frame-ID binding (the repeated JSON is an observed failure, not
proof of a universal Qwen copying mechanism). Use appearance/point correspondence and
bidirectional instance association to check identity/reacquire through occlusion.
Do not declare stationary targets wrong from action text, copy manual boxes,
force motion/contact, or silently accept missing evidence. Assess target
association separately from pose/shape before spending on a full4D rerun.
The existing single-call Qwen/SAM2 baseline stays frozen as the A branch.
Persistent observations and uncertainty-aware global SE(3) pose estimation
form the separate motion B branch; conditional support comes only after this.
A shape-anchor experiment is conditional on legitimate identity, not the first
repair. Earlier rejected multiview/LK tests are not revived or reclassified as
success; new evidence/operator needs a new preregistered independent gate.

Actual compact result/identity records are in
`results/audits/full4d_causal_diagnostic_v1_actual.json`, its `_detail` and
`full4d_causal_grounding14_v1_actual.json`. Tiny original/mask/final-mesh QA
sheets were generated on Azure; only three SHA-verified JPEGs total462418B crossed
locally, including the167410B automatic zoom. Its new camera/crop is strictly
a display viewport: no predictions are refitted, no per-frame evaluation
alignment and no manual box selection. The CPU focus completes all27views;
its host publication fails on an unintended NumPy metadata import. One stdlib
metadata-only continuation reuses exact JPEG bytes (no render/model retry),
retains the original FAIL/partial file, and records each private PUT/ETag.
A focused regression fixes host metadata without changing the running scientific
producer. Full image/focus publication receipts remain distinct. Native inference, masks, depth arrays, meshes and full videos stayed
remote. Human QA remains REJECT for reconstruction quality, not label creation.


### September-eligible ownership mechanisms

[SAM3 v2](https://arxiv.org/html/2511.16719v2), §3/§C.3, separates an independent
per-frame concept detector from the video tracker: instance association, new
masklets and detector-guided re-prompting address drift that a one-seed tracker
cannot detect independently. SAM3.1 release March27,2026 is before the cutoff.
Text is a simple object noun phrase, not a reliable arbitrary action/relational
query; keep all instances and associate the actual interaction automatically.
Derive concepts from permitted metadata/the same general model, never a human
QA hint. Source/model custom SAM terms and checkpoint overlap remain unaudited
for adoption. No acquisition or inference is inferred from this literature.

[Qwen3-VL v2](https://arxiv.org/html/2511.21631v2), §3.2.4/§5.7/§5.9, supports
normalized coordinate grounding and multiple-image/video tasks; those benchmark
results do not guarantee instance tracking or per-frame visible boxes. The
minimal causal control is the same9views/question in nine independent
single-image conversations, batching with native left padding. Prior boxes
must not leak between queries; explicit missing observations remain missing.
Compare automatic observations before any mesh regeneration/full4D rerun.

Focused local pure/adapter tests: **225 PASS** (no local inference/render).
