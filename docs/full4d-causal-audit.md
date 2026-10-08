# Frozen 4D: causal diagnosis, not parameter hunting

User QA, 2026-10-08: an apparently resting object jitters; episode14 appears
to contain the wrong/bizarre reconstructed object. These are failure reports,
not manual labels or permission to change individual test predictions.
Preserve the complete random diagnostic, its failed episode7, and its hashes.

## Established code mechanisms versus unmeasured attribution

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

- Shape experiment: automatic visibility/coverage/nettement selection of a
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
