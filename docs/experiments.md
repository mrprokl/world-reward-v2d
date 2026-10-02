# Research protocol and decision ledger

## Validation ladder (no challenge GT)

1. Contracts: data provenance, nonoracle flags, units/frames, rotations, full-frame
   coverage, clip identity, official sample row equivalence, accessible code commit.
2. Synthetic geometry/motion: deterministic seeds, known metric scale/contact,
   thin/concave/symmetric objects, injected occlusion/outliers. Use official scorer
   mathematics and confirm single first-frame alignment and gap handling.
3. External non-overlapping labeled benchmark (e.g. BEHAVE/InterCap only after
   license and overlap check); disjoint sequence split and motion/occlusion strata.
   Hyperparameters selected here, not from challenge leaderboard probing.
4. Challenge RGB-only checks: reprojection, visible silhouette, temporal feature
   residuals, scale/rigidity, observed contact, full trajectory and rendered QA on
   remote machine. **Proxy quality does not establish held-out improvement.**
5. Freeze artifact/code/config/weights hash and submit to the five official metrics.
   Record complete receipt and compare same artifact vs official CARI4D scores.

## Prioritized hypotheses / stop conditions

| ID | Hypothesis | Small first test | Continue gate / abandon reason |
|---|---|---|---|
| H0 | Correct RGB-only baseline is executable and MHR-exportable | One simple block/roll episode, masks→mesh→depth→human→pose→CARI4D→conversion | Full coverage, valid provenance/roundtrip/render; fail before batching if units/access broken |
| H1 | Object initialization/mesh scale dominates CD-O; geometry is not solved by CARI4D itself | Best visible RGB frames; generated vs video-fitted primitive for block/roll/hoop; joint silhouette/depth scale | Improve held-out-frame silhouette/reprojection without scale/contact contradictions; reject aesthetically plausible wrong shape |
| H2 | Confidence-aware reinitialization and symmetry-aware temporal pose beat blind tracking through occlusion | Compare official track branch vs multihypothesis registration on occluded external clips | ≥10% external pose-error gain, no visible reprojection regression; reject identity flips/drift |
| H3 | Joint clip identity plus mild motion-aware smoothing improves human reconstruction, unlike fixed high smoothing | Initialization/CoCoNet/postopt ablation with same input/mesh; dynamic motion strata | External Pareto CD-H/ACC-H improvement; never validate on acceleration-to-zero only |
| H4 | MoGe-3 improves thin parts and depth scale over default MoGe-2 | Same masked crops, same calibration-from-RGB protocol | Measured depth/pose gains on non-overlap validation; reject if self-consistency alone or worse edges |
| H5 | Image-anchored hand/contact refinement fixes real interaction without altering geometry | Unlock fingers only with 2D keypoints + video evidence and nonpenetration | External hand/contact gains, body/CD-O preserved; reject detached hands or exploit-only PEN gains |

Thresholds are provisional preregistration, to finalize **before** measured experiments.
No experiments have yet established performance gains. Start minimal episode smoke
test; parallelize independent hypotheses only after shared contracts are sound.

## Decisions

- 2026-10-02 D01: Track 1 input firewall; forbid cross-track/source-MV assets even
  when accessible. Full dataset downloads through pinned Track 1 file whitelist.
- 2026-10-02 D02: Azure-first, remote-resident heavy artifacts; no local videos,
  checkpoints, decoded frames, or meshes. User explicitly confirmed this constraint.
- 2026-10-02 D03: Reproduce commercial MHR baseline wrapper rather than assume
  paper SMPL-H code is submission-compatible; mesh reconstruction and identity
  conversion are separate validated stages.
- 2026-10-02 D04: Use pretrained models only after access/license/provenance audit;
  paper novelty or demo images alone do not establish a deployable SOTA method.
- 2026-10-02 D05: Full Kaggle Foundational 4.b forbids manual test labeling.
  Automatic object-prompt grounding + SAM propagation replaces manual SAM2 GUI.
  Procedural shapes must be selected/fitted algorithmically from RGB, not hand-labeled.

## Results

- 2026-10-02 R01 (synthetic only): multi-hypothesis SO(3)/translation Viterbi with
  image confidence and explicit true mesh symmetries implemented. Generated
  occlusion scene: greedy mean translation error 0.342857 m, selected path 0 m.
  Tests additionally check exhaustive global optimum, false 180° flip rejection,
  real fast 170°/1 m motion with strong evidence, exact-zero speed, missing states,
  invalid rotations, overflow and time/shape contracts. Full suite 68 passed.
  This verifies selection logic, **not** real tracking or challenge performance.
- 2026-10-02 R02 (synthetic/schema only): exact Track 1 row-key slicing, full-video
  coverage, constant identity, padding and frozen Parquet roundtrip tested against
  official helper contracts. Converter validation refuses nonfinite input and
  non-native formats; it does not invent a CARI4D-PCA mapping. Automatic detector
  seed selection rejects invalid/ambiguous boxes and emits the official SAM2 JSON,
  without manual annotations. Full lightweight suite: 190 passed. Actual GPU
  model forward, masks and reconstruction quality remain separate gates.
- 2026-10-02 D06: source/model licenses separately audited. SAM/custom
  FoundationPose source eligibility and commercial CARI legacy header scope need
  organizer clarification; do not claim an award-eligible final stack yet.
- 2026-10-02 R03 (H100 synthetic model gate): official Apache MHR forward and
  unmodified official converter on 3 generated frames: mean vertex residual
  0.000051495 mm, worst frame mean 0.000065821 mm; gate 0.01 mm passed in 14.32 s.
  Independent second model forward confirmed frame/unit/shared-identity fidelity.
  Converter SHA-256 `c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e`;
  Torch 2.5.1+cu124. Synthetic only, not fitted challenge predictions or a score.
- 2026-10-02 R04 (automatic segmentation initialization): predeclared episode 15,
  16 sparse frames, confidence 0.3 / ambiguity margin 0.05. All seed frames rejected
  due to near-tied person detections; SAM2 propagation did not run. Missing standard
  detector NMS is a likely cause, not established by counts alone. Next test adds
  generic class-wise IoU NMS 0.7, while retaining distinct-instance ambiguity
  rejection and all previous thresholds. No manual per-frame annotations.
- 2026-10-02 R05: generic NMS 0.7 did not resolve R04; all 16 frames still rejected.
  This falsifies the simple high-IoU duplicate explanation at that threshold.
  Stop blind retries or threshold relaxation. Inspect only automatic detector
  labels/scores/box geometry on three remote RGB frames to distinguish genuine
  distractors from preprocessing/model-query failure, before changing the method.
  Full suite now 231 passed locally and on Azure with official-kit parity enabled.
- 2026-10-02 R06 (RGB-derived detector diagnostics): on frames 0/250/500, each
  person query returns three spatially disjoint boxes of similar confidence,
  while the object query returns one box. Thus ambiguity is genuine multiple
  detected persons, not duplicated queries. No human instance labeling performed.
  Next method ranks automatically associated person tracks by video-wide proximity
  to the detected interaction object, requiring sufficient observations and a
  clear winning margin; retain failure rather than guess in ties/crossings.
- 2026-10-02 D07: implement object-affinity actor selection as a generic automatic
  initializer, not a person-ID label for a challenge episode. Gated one-to-one IoU
  association, median normalized object-to-person-box distance (center distance
  only for equal outside distance), ≥3 observations/≥50% support, margin >0.05.
  Ambiguous associations or affinities fail. Synthetic actor+distractor/crossing/
  missing tests pass; full suite 265 passed. Real smoke retry remains to verify.
- 2026-10-02 R07: initial object-affinity actor gate rejected the real smoke for
  ambiguous person-track association. Its documented global rejection means an
  unrelated distractor crossing can reject the entire clip; do not relax identity
  margins. Next localizes ambiguity to association components and propagates
  contamination, allowing only an independent unambiguous winner. Contaminated
  close tracks must still compete; they cannot be deleted to manufacture a winner.
- 2026-10-02 R08: component-aware association also rejected the winning actor
  (ambiguities at sparse frames 333/366; no identity fix accepted). Full-clip sparse
  bbox identity is an unnecessarily strong prerequisite for a segmentation seed.
  Next initializer uses a fixed first-three-sparse-observation prefix with the
  same actor-affinity/support/margin rules, then SAM2 handles full-frame temporal
  propagation. No sparse association is emitted as final human/object motion.
  This is a general architecture change, not a manual actor label or threshold
  relaxation. Final actor tests: 272 passed locally; latest remote suite 271 passed
  before the additional expired-uncertainty test was mirrored.
- 2026-10-02 R09 (H100 engineering smoke, not accuracy validation): the fixed-prefix
  automatic initializer chose actor track 0 with 3/3 object observations, no
  identity contamination; object-affinity runner-up distance 1.23 vs winner 0.
  SAM2 then produced person and object masks for all 501 original frames of
  episode 15; no missing/empty masks, median areas 27,797/4,154 pixels. End-to-end
  automatic masks 52.00 s, without GT/manual prompts/oracles. This proves runtime
  and coverage, **not** correct segmentation, mesh reconstruction or CARI4D victory.
  Validate multi-frame silhouette/rigidity/contact and independent detector agreement
  before trusting masks. Latest full suite 272 passed locally and on Azure.
- 2026-10-02 D08: prevent mutable-job source races using read-only committed
  job snapshots, rather than editing a running launcher. Official runtime image
  export and independent CARI Torch CUDA/import smoke passed; no reconstruction
  claim. Add strict streaming mask geometry/temporal diagnostics with 53 synthetic
  tests (full local suite 325 passed), reporting occlusion/motion/overlap without
  arbitrary accuracy thresholds or fixes to individual test masks.
- 2026-10-02 R10: actual SAM3D kernel gate caught a CPU-only PyTorch3D build
  (`Not compiled with GPU support`), despite successful imports. Kaolin Chamfer,
  nvdiffrast CUDA raster and FlashAttention passed. EGL context creation failed
  with 12291; root cause unverified. Source callgraph confirms the official minimal
  Objects generation and CARI inference use PyTorch3D/CUDA raster, not pyrender EGL.
  Keep EGL failure as optional visualization evidence, not a core-model blocker.
  Rebuild only a derivative PyTorch3D layer, FORCE_CUDA=1/SM9.0 with v0.7.9 resolved
  commit `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`; require actual GPU KNN to pass.
- 2026-10-02 R11: DINO auxiliary acquisition exited 0 with pinned clean DINOv2/v3
  source, validated official vitb/vits SHA-256 and recorded first-observed HTTPS
  reg4 hashes. No FoundationPose acquired; no complete-baseline readiness claim.
- 2026-10-02 R12: mask proxy pass on all 501 frames; largest-component median 1
  for both entities, maximum image-diagonal centroid jumps 0.001079/0.004910.
  Three sparse independent detector bbox agreements ~0.963–0.967 human,
  ~0.887–0.893 object. Masks overlap on 377 frames: report, do not strip object
  pixels or label contact from this alone. These values are not segmentation GT
  accuracy, temporal held-out reconstruction quality or a score.
- 2026-10-02 R13: corrected SAM3D derivative actual core kernel smoke passed;
  EGL still failed and recorded as optional. Body direct smoke failed before the
  forward because bootstrap had excluded six source Python files named `data/`.
  Diagnose/fix the infrastructure whitelist, not model hyperparameters. Original
  images are preserved; small source-only derivative will be independently tested.
- 2026-10-02 D09 (external validation rights audit): BEHAVE and InterCap data/source
  are non-commercial research-only; InterCap also requires account approval.
  Do not advertise commercially unrestricted validation data. CARI paper trained
  on BEHAVE, commercial MHR repeatedly validated on 79 BEHAVE clips, and overlap of
  all SAM/DINO/MoGe components is unverified. Use independent procedural MHR/rigid
  geometry with BSD PyTorch3D for numerical/development validation first, explicitly
  not a real-world SOTA proof. Real InterCap comparison requires lawful access and
  rights clarification. Separate identities/objects/recordings, never random frames.
  Sources: https://virtualhumans.mpi-inf.mpg.de/behave/license.html ;
  https://intercap.is.tue.mpg.de/license.html ;
  https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE .
- 2026-10-02 R14: object model import failed on missing `libusb-1.0.so.0`.
  Install just libusb in a derivative image, then pre-GPU Open3D/Objects import
  passes. Real offline fixed frame 0/seed 0 generation subsequently passed in
  45.90 s: 751,586 vertices, 1,503,164 faces, watertight and consistent winding.
  Transform scale 0.07596754 is an inferred generative estimate, **not calibrated
  physical scale**. Geometry must be decimated/validated and metrified before
  CARI4D; no final submission/reconstruction quality claim.
- 2026-10-02 R15: Body source repair passed imports; strict checkpoint gate then
  rejected state missing from the network checkpoint but constructed from MHR
  assets, plus DINO mask token and unused hand PCA copies. Do not ignore arbitrary
  missing weights. Bind exact MHR submodule state to an independent load of the
  explicit pinned SAM asset, require learned/head state, audit genuinely unused
  inference state separately. Added 98 small Body guard tests including retained
  buffer alias regression; corrected comparison snapshot clone. Full suite 423
  passed. Body model forward remains unverified until these gates pass.
- 2026-10-02 R16: direct Body RGB forward passed on fixed frames 0/250/500 in
  15.64 s. Learned and frozen head state loaded strictly; omitted rig/corrective
  state matches an independent load of the exact SAM MHR asset. Deterministic I54
  hand copies and zero DINO mask token are audited exceptions; non-null DINO token
  masks rejected. Native MHR reforward and keypoint reprojection gates passed.
  This verifies sparse model geometry/units, not full trajectory/shared identity
  or reconstruction accuracy. Latest local suite 456 passed. Official shared-
  identity converter is the next separate real-output gate.
- 2026-10-02 R17: official converter on the real sparse Body output completed in
  22.55 s, with shared shape/scales and zero expression. Independent reforward
  mean residual 0.3304 mm; per-frame means 0.2721/0.4512/0.2679 mm. This is the
  conversion loss relative to predicted meshes, **not error against challenge GT**.
  Native input forward max residual ~2.5e-7 m; keypoint reprojection <0.000066 px.
  Next extend the verified initializer to all 501 original frames, preserving
  full indices; do not present framewise estimates as CARI temporal refinement.
- 2026-10-02 R18: full 501-frame Body initialization passed in 172.99 s, then
  full official conversion with one shared identity passed in 106.61 s: 0.9719 mm
  mean / 1.4861 mm worst-frame mean residual to predicted meshes, no invalid input
  frames. Frame count/pinhole/native model checks cover the entire clip. Still
  framewise initialization, not a final CARI refinement or GT benchmark.
- 2026-10-02 R19: three fixed MoGe2 depth predictions passed in 8.06 s with only
  RGB-size FOV prior. Added equal-frame robust single depth-scale and partial
  rigid ICP with 39/49 synthetic tests; no learned scale truth claim. Analytic CUDA
  camera/depth raster gate must pass before human/object scale consistency smoke.
  Full local suite 597 passed.

- 2026-10-02 D20: audit caught a gauge error before running object alignment: SAM
  Object inferred pose/scale from MoGe1, whereas the human anchor fits MoGe2.
  One cannot multiply the independent MoGe1 pose by the MoGe2 scalar, nor change
  its intrinsic matrix as if this were an exact rigid correction. Split depth
  alignment from object generation: verify MoGe2 K, Z and pixel-centre rays; fit
  one clip scalar to predicted human surfaces, export already-aligned pointmaps
  and regenerate the object with those explicit pointmaps and the same K. No
  second scale application, manual labeling, GT or score optimization.
- 2026-10-02 R20 (H100 analytic camera gate): actual PyTorch3D CUDA raster passed
  in 0.93 s: tilted triangle maximum Z error 6.34e-7 m, occlusion and closed-box
  error 2.38e-7 m; closed-box silhouette IoU 1, projection error 0 px. Camera Z
  and perspective/rectangular-image conventions now independently checked.
  This is numerical geometry validation, not reconstruction accuracy.

- 2026-10-02 R21 (H100 predicted-depth consistency): human-anchored MoGe2
  pointmap export passed on original frames 0/250/500. Camera K agrees within
  4.58e-5 pixels, points.Z equals depth exactly and ray reprojection error is
  <0.000279 pixels. Human rendered-mask IoU: 0.8052/0.8044/0.8432; valid human
  depth correspondences: 24,212/23,347/25,804. Diagnostic per-frame median
  depth ratios 1.0891/1.0900/1.0873 are consistent; only one clip scalar is
  applied. Residuals to predicted human surfaces ~0.0031–0.0038 relative, not
  independent metric accuracy. Frozen pointmaps remain Azure-only.
- 2026-10-02 D21: test fixed-shape, fixed-scale rigid object pose using 24 generic
  orientation hypotheses (not declared object symmetries), automatic observed
  depth and silhouette. Keep exact official budget arrays; reject lost closed
  oriented volume. Accept an ICP step only if trimmed depth RMSE and automatic
  mask IoU do not worsen; report rejected underconstrained/near-plane candidates.
  Three-frame hypotheses are an engineering smoke, not a full trajectory or
  score. No depth-only residual masquerades as unseen-surface pose accuracy.

- 2026-10-02 R22 (H100 grounded object generation): explicit human-anchored
  MoGe2 frame-zero pointmap produced a new SAM Object in 46.99 s: 753,476
  vertices / 1,506,944 faces, watertight. Pose now belongs to the same predicted
  human gauge (translation Z 7.3492, uniform canonical scale 0.23643665), with
  explicit K preserved. This illustrates why the independent MoGe1 output
  (translation Z 1.0680, scale 0.07596754) could not receive the MoGe2 scalar.
  Geometry/scale accuracy and challenge performance remain unverified. The
  upstream reestimated internal K is not used by the disabled layout optimizer;
  only the verified caller K is used for output projection diagnostics.
- 2026-10-02 D22: extend already-verified MoGe2/K/ray inference to every original
  frame. Save only camera-Z depth, validity and K for full videos: dense XYZ is
  redundant under the strict +0.5 pixel-centre pinhole gate. Reconstruct rays
  downstream on Azure; do not transit heavy arrays locally or duplicate 3×XYZ
  storage. Each decoded frame and output retains SHA-256 provenance.

- 2026-10-02 R23 (fail fast): rigid pose smoke stopped before pose fitting: the
  official-budget object failed closed/oriented/positive-volume geometry gate.
  No pose result or quality claim emitted. Diagnose original vs budget raw vs
  processed mesh edge incidences and signed volume before changing the method.
  Do not loosen the physical gate or exploit an open/inverted submitted mesh.

- 2026-10-02 D23: original Body raw blocks are retained for the native CARI
  initializer: body133 re-encoded with official body260 helper; root Euler ZYX
  converted by official interleaved-root6D helper (different from body6D layout);
  hand108/shape45/scale28 unchanged and camera translation applied once. Require
  native MHR decoder roundtrip on every frame (vertices, joints, 70 keypoints
  <1e-5 m), exact topology and fixed-K reprojection <0.05 px. Ninety-six tiny
  adapter regressions pass, full suite 740; real GPU validation is separate.
  No 204→PCA pseudo-inverse, invented weights or license-eligibility claim.

- 2026-10-02 R24: topology diagnosis localized the budget failure: original
  closed mesh has two correctly oriented components (outer positive, cavity
  negative); budget keeps them and adds a near-zero-volume third component
  causing one non-manifold edge, with no boundary edges. Preserve the cavity;
  measure triangle area before removing anything. A tiny component volume alone
  is not permission to delete meaningful geometry or raise thresholds.
- 2026-10-02 R25 (H100 native CARI human adapter): all 501 frames decoded and
  matched original SAM Body vertices/joints/70 keypoints, maximum errors
  1.1161e-6 / 9.5554e-7 / 9.8483e-7 m. Fixed-K projection maximum 8.6317e-5 px;
  8.42 s. This proves canonical input compatibility, not CoCoNet inference or
  challenge accuracy. Shared-identity conversion remains separately verified.

- 2026-10-02 R26: complete MoGe2 inference passed on all 501 original frames
  in 258.30 s; full camera/Z/ray gates and per-frame hashes recorded. Dense
  outputs remain Azure-only and store depth/K/validity without redundant XYZ.
- 2026-10-02 R27: machine-level collapsed-triangle exclusion did not restore
  budget topology; second pose run correctly stopped before fitting. This
  falsifies a simple collapsed-area explanation. Do not relax machine thresholds;
  inspect the specific non-manifold component and simplification implementation.
