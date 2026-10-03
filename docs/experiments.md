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

- 2026-10-02 R28: detailed budget diagnostics found two opposite copies of one
  positive-area triangle (area 7.48e-6 model-units² each), not collapsed triangles.
  Do not remove real surfaces based on near-zero component volume. Regenerate a
  topology-valid approximation using fixed generic simplification budgets, with
  componentwise signed-shell preservation as fallback. Validate each candidate
  and the official repacked result, without filling holes or inverting cavities.

- 2026-10-02 R29 (H100 sparse rigid pose consistency): global target 4096
  retained the known non-manifold defect; fixed generic next target 4080 passed
  closure, both signed shells/Euler, and official 4096 repacking. No positive-area
  artifact was manually deleted, cavity inverted or hole filled. Fixed extent
  in the predicted human gauge: 0.2324×0.2349×0.2332 m. Three-frame rigid
  hypotheses passed in 30.49 s; automatic-mask IoU 0.8249/0.7982/0.8707 and
  inferred-depth residual 0.0064/0.0339/0.0290 m. Image/depth consistency only,
  not verified physical shape/pose accuracy.
- 2026-10-02 D24: extend fixed geometry and scale to all 501 original frames.
  Use 24 generic orientation seeds plus the previous image-selected rotation,
  deterministic partial-depth ICP with non-worse image gate, then select a full
  Viterbi path without asserting object symmetries or averaging rotations.
  Predeclared costs: 1−automatic-mask-IoU; translation weight 1, rotation weight
  0.1, original-frame time units, no speed bounds. This is an initializer
  hypothesis, not an accuracy/acceleration claim or hidden metric tuning.

- 2026-10-02 D25: prepare native CARI inputs only from hashed original RGB,
  automatic full-frame masks, native-roundtripped Body parameters and our own
  fixed-scale object trajectory. Native oriented mesh frame A is compensated
  by P@inv(A), with camera-space geometry equality checked across the clip.
  Preserve fixed RGB-size K, shared human-anchored MoGe2 depth scalar, no shift,
  and strict quantized-depth saturation checks. Official H5 export is explicitly
  marked complete and independently decoded. Source video ABI uses same-disk
  hardlink, not a resolved symlink or another local download.
- 2026-10-02 D26: native CoCoNet forward uses our own object-pose pickle via the
  historical --foundationpose-file ABI, never the FoundationPose executable or
  oracle modes. Strict pinned checkpoint/config, local pinned DINO Hub loader,
  offline supervision contract, CUDA/network-none and all original frames.
  Runner stores GT={}, checked before accepting output. Checkpoint load and
  actual full network forward are distinct gates. Pure builder tests119 pass,
  total908; native execution is not yet established by those tests.

- 2026-10-02 D27: source audit caught the cross-bind-mount hardlink EXDEV
  failure before native input preparation. Stop only the still-waiting old
  preparation unit (no output created); corrected v2 uses SHA-verified copy
  entirely within Azure and waits for the unchanged live object producer.
  Full-forward dependency now names that exact v2 unit. Checkpoint state loading
  is an independent fail-fast gate requiring no prepared episode inputs, not a
  substitute for actual network forward. Native output poses must be proper
  SO(3) rigid transforms; reject scale/shear/reflections, never repair silently.
- 2026-10-02 D28: Body/depth initializers generalized to episodes 0..29 with
  unchanged episode15 defaults, exact selected-video/mask SHA and original
  timeline binding. New automatic-mask entrypoint uses existing immutable
  source and offline image, without builds/downloads. Full local suite999 passes;
  tests use tiny synthetic inputs, not extra challenge labels or heavy assets.

- 2026-10-02 R30 (native gate fail-fast): checkpoint-only job exited1 before
  loading CoCoNet because Body's top-level tools package won parent sys.path.
  Azure confirms pinned native tools/__init__.py and entrypoint both exist.
  Fix import ordering to the already-tested native-first PYTHONPATH and assert
  resolved entrypoint identity. No vendor code, model or live object job changed;
  no checkpoint-forward success claimed from this failed gate.
- 2026-10-02 D29: throughput branch batches independent raster poses at the
  unchanged 1536×1152 camera/grid, with exactly scalar topology/near-plane gates.
  Require exact silhouette and <=1e-5m camera-Z parity on our own procedural
  anisotropic mesh; median of3 synchronized alternating trials at batch8,
  accept optimization only if speedup>=1.3. This does not modify the live
  object trajectory job or tune to challenge labels. Full local suite1104 passes.
  Automatic masks now also hash both public metadata files and reject ambiguous
  episode records before reading prompts; immutable offline launch unchanged.

- 2026-10-02 D30: after actual native full forward, decode canonical parameters
  with exactly the original Body MHR decoder, then official full-video shared
  identity fitting. Never concatenate native6D/body260/hand108/scales28 into
  kit136/68. Engineering representation-fidelity limit2mm per-frame mean,
  chosen consistently with the preceding Body conversion, not tuned to GT;
  verify independently through the Apache reference forward. Restore original
  packed metric mesh pairing by P_source=P_aligned@A, no second gauge scaling.
  Final conversion stays unverified until its actual remote run passes.
- 2026-10-02 D31: full code archive grew above100KB control payload; launcher
  correctly refused before Azure execution. Send only deterministic committed
  shell/Python-import closure plus the small project package/config, preserving
  Git metadata. Distinct entrypoint directories beneath jobs/<commit> prevent
  same-commit closure collisions; SHA verification remains mandatory. No large
  data/model transfer or arbitrary payload-limit increase.

- 2026-10-02 D32: test native SAM Body full-mode hand proposals on the same
  predeclared sparse engineering frames, automatic human masks only. Distinct
  outputs preserve the existing body initializer; no shared-identity or hand
  accuracy claim. Original133/108/3/28/45/72 parameter blocks must re-forward to
  predicted vertices/joints/keypoints, not full-mode zeroed266 raw logits.
  Full mode changes per-frame identity; it is a proposal source, not final
  challenge reconstruction. Source/weight terms remain independently unresolved
  for release eligibility. Second-pass SOTA priorities and falsification tests
  documented in literature.md; no external metric experiment established yet.

- 2026-10-02 D33: source audit found full-mode joint rotations remain cached
  before hand fusion upstream. Re-decode fresh rotations from original blocks;
  keep stale source rotations explicitly audit-only, never hand constraints.
  Full hand geometry/projection source ABI otherwise checked. First compact O1
  proposal is a shared5DOF trace-free symmetric log-stretch: SPD determinant1,
  fixed pivot, faces/order/cavity preserved, principal stretch bounded[2/3,1.5],
  no clipped parameters or uniform metric-scale fitting. Analytic volume and
  invalid-input tests establish mathematics only, not shape-fit improvement.
  Exact duplicate padding uses deterministic einsum. Full local suite1251 passes.

- 2026-10-02 R31 (Azure native checkpoint gate): corrected native-first imports
  allowed strict pinned CoCoNet load, finite model state and offline DINO passes
  in7.12s. This gate performed **no network forward or episode inference**.
- 2026-10-02 R32 (Azure hands engineering): native full-mode proposals passed
  vertices/joints/keypoints/control/projection roundtrips on original0/250/500
  in34.38s. Fresh decoded joint rotations replace stale pre-fusion cached values,
  retained audit-only. No shared-identity fitting or hand accuracy measured.
- 2026-10-02 R33 (raster throughput fail-fast): batch gate failed before parity,
  because pinned PerspectiveCameras has no extend() API. Correctly failed with
  no speed claim; construct full OpenCV camera tensors at batch size through
  the same official projection utility. Keep failed report and use a new
  report path/job for correction. Unchanged live object producer reached300/501.

- 2026-10-02 D34: minimal O1 fitter optimizes one volume-preserving5DOF shape
  against permitted visible depth points with supplied poses fixed. Only
  observed-to-deformed-surface nearest neighbours, equal frame weights, robust
  coordinate loss10mm in raw metric units, deterministic<=2048 samples/frame,
  prior.01 and conservative log-stretch bounds; <=100 evaluations. No hidden
  surface attraction, R/t/metric-scale fitting or automatic caller mesh mutation.
  Nonconvergence/raw residual worsening returns an unaccepted proposal. Tiny
  noiseless procedural partial-view test verifies implementation (>85% fitting
  residual gain), not RGB-inferred pose robustness or challenge shape accuracy.
  Full local suite1292 passes. Synthetic rendered/occluded tests still required.

- 2026-10-02 D35: shared depth gauge, grounded object generation and rigid
  trajectory initializers generalized to episodes0..29, default15 preserved.
  Sparse frame indices now derive from each original clip length; producing
  masks/body/depth/shape reports must bind the selected episode and video hash.
  No camera/gauge, generic simplification budget, pose hypotheses, ICP objective
  or Viterbi hyperparameter changes. Wrapper passes selected episode explicitly.
  Local suite1334 passes; actual other-episode execution is a separate gate.

- 2026-10-02 D36: predeclare O1 rendered procedural test: asymmetric/thin/
  unchanged-symmetric shapes ×0/25/50% own rectangular occlusion ×known and
  deterministically perturbed fixed poses (.02rad/.005m),18 conditions. Fit
  original synthetic frames0/2/4, hold out1/3/5; native full-size K/ray+.5,
  shape samples8192seed0, observations<=2048, unchanged fitter defaults. Measure
  physical two-sided CD without registration, held-out visible-mask IoU and
  volume/topology; require5% deformed-case CD gain and<=5% heldout-IoU loss,
  zero-control CD nonregression. Known/perturbed poses are explicitly derived
  from synthetic truth: this isolates shape behavior, **not RGB pose inference**.
  Fail any condition=>keep experiment, no challenge adoption or victory claim.

- 2026-10-02 R34 (Azure raster batch gate): corrected camera construction
  passed exact silhouette and camera-Z parity against scalar/singleton renders.
  Three synchronized alternating trials, batch8 at1536×1152 on our own642-vertex
  anisotropic mesh: scalar median.062400s, batch median.034993s, speedup1.7832,
  peak638118912bytes. Predeclared>=1.3 gate passes. This measures rasterization
  only, not full ICP throughput or reconstruction accuracy; live producer unchanged.
- 2026-10-02 D37: native initializer adapter, preparation, full forward and
  converter now route episodes0..29 with default15 preserved. Reject explicit
  wrong episode fields; historical reports remain bound by selected paths/SHA.
  Checkpoint-only gate stays independent of prepared inputs. Full timeline,
  units, source assets and fidelity gates unchanged. Local suite1443 passes;
  generalized wrappers and actual other-episode execution remain separate gates.

- 2026-10-02 R35 (Azure O1 falsification, d3b87cb): all18 rendered conditions
  executed in27.45s. Twelve asymmetric/thin cases passed the predeclared5% CD
  improvement/held-out silhouette gate, including pose bias and50% occlusion.
  **All6 unchanged symmetric controls failed nonregression**: exact-pose CD
  increased from0 to.1203/.2575/.1558mm; biased-pose CD5.5220mm increased to
  5.5627/5.6220/5.6396mm. Lower visible fitting loss did not imply true shape
  improvement. Reject adoption of this fitter; keep only its frozen experiment
  and reusable mathematical primitive. Do not loosen gates or tune to these
  controls. Discrete nearest-surface sampling and pose/shape confounding are
  hypotheses for investigation, not established causal diagnoses.
- 2026-10-02 D38: native wrappers now forward strict selected episodes and
  optional exact producer units. Default15 dependency names preserved;
  non15 never implicitly waits for an episode15 job. Missing/failed producers
  or missing selected reports fail; waits bounded12h. Checkpoint-only reads
  Body assets/report but never waits for preparation. Tiny fake-shell tests
  exercise routing without Docker, Azure or data. Active snapshots unchanged.
- 2026-10-02 D39: H1 next minimal experiment transfers only54 decoded finger
  controls into final officially converted136D pose. Keep root, wrists, body,
  shared identity and object unchanged; use full-mode original204D controls
  verified against the source decoder, not raw266 or pre-fusion rotations.
  No reconversion or automatic adoption. Sparse ABI/projection/visible-image
  falsification precedes full-video testing; no hand-accuracy claim.

- 2026-10-02 R36 (episode0 routing fail-fast): the generic automatic-mask job
  completed mask generation but failed final provenance reporting: the detector
  BatchEncoding shadowed the validated input dictionary, causing KeyError
  video_sha256. No passing report means these masks cannot feed downstream jobs.
  Rename the detector-local variable and add a regression on provenance binding;
  preserve the verified failure cause, quarantine incomplete output before a
  uniquely named retry. No manual masks, thresholds or actor choices changed.

- 2026-10-02 D40: test observed-to-continuous-triangle CUDA distance before
  writing a replacement shape fitter. Direct one-sided PyTorch3D primitive,
  min_triangle_area0 on nondegenerate10mm triangle; analytic interior, boundary,
  outside vertex **and outside hypotenuse**, then gradient/rigid/duplicate gates.
  Source audit found a fixed1e-8 barycentric denominator may misclassify tiny
  triangles. Keep original metric fixture and1e-9m² tolerance; no rescaling,
  vendor patch or loose tolerance to hide a failure. Not adopted before GPU gate.

- 2026-10-02 R37 (Azure continuous-distance fail-fast, c344eb3): PyTorch3D
  point-face primitive failed analytic small-triangle containment in.816s.
  Outside-edge squared distance expected.000412499982m², actual.000399920042m²,
  error1.25799e-5m² vs frozen1e-9 gate. This is an actual kernel-scale failure,
  not evidence against all continuous-distance fitting. Do not build a fitter
  on this backend or rescale geometry to conceal it. Keep failed report.
- 2026-10-02 R38 (Azure automatic episode0, c344eb3): after the detector-variable
  fix, full mask provenance/coverage report passed in72.67s; both entities have
  zero empty frames, median23584.5human/2859.5object pixels. Frozen failed output
  remains quarantined; no threshold/manual-label change. This is general routing
  and execution evidence, not segmentation accuracy or challenge validation.
- 2026-10-02 R39 (Azure full object producer, ca6b234):501original frames passed
  in3929.56s with fixed packed geometry/proper rigid poses and final Viterbi path.
  Greedy mask-IoU median.86233 is only a fitting diagnostic, not the final path's
  score or a GT metric. Native preparation automatically started afterward;
  actual CoCoNet inference/conversion still awaits its validated input report.

- 2026-10-02 R40 (Kaolin infrastructure fail-fast,58825d2): standard import
  failed before numerical tests in2.03s: Warp tried writing `/.cache` under the
  nonroot offline container user. This says nothing about distance correctness.
  Preserve report; retry only that exact source-bound PermissionError with
  task-isolated writable HOME/cache and a distinct report. Frozen10mm triangle,
 1e-9m²/1e-6 gradient gates and unresolved import-license status unchanged.
- 2026-10-02 D41: predeclare whole-candidate batch8 scheduling test: our own
 642v/1280f anisotropic mesh,3full-size rendered views,24generic (not truth)
  orientation seeds,8192surface/2048visible samples, unchanged ICP and selection.
  Exact complete candidate/rejection/best JSON parity, then3alternating CUDA-
  synchronized trials; median>=1.3×, elapsed<=120s. Independently test canonical
  KDtree ICP on72own noisy partial-view hypotheses; synthetic-oracle seeds are
  explicitly disclosed. Status/inlier/iteration and NN/trim indices must match
  exactly, pose/residual<=1e-6, median>=1.3× and<=60s. Failures retain partial
  diagnostics; neither test measures challenge accuracy or authorizes adoption.

- 2026-10-02 R41 (Azure throughput fail-fast,dc6d5d2): cached canonical-tree
  ICP passed all72exact status/inlier/iteration and NN/trim comparisons, then
  exceeded frozen60s total at60.013s during timing trial1 (34alignments into
  original mode). Whole candidate scalar/batch8 passed all3complete JSON/
  rejection/selection comparisons, then exceeded120s at125.486s in timing.
  Neither completed all3alternating trials or established useful median speedup;
  preserve partial timings, do not claim failure of parity or proven acceleration.
  Both remain unadopted; no increased budget retry of these frozen tests.
- 2026-10-02 R42 (Azure Kaolin analytic kernel,dc6d5d2): infrastructure-only
  cache retry passed original fixture/contracts in2.874s, distance error
 1.28988e-11m², point gradient0, triangle gradient4.04e-9 and rigid error
 2.53e-9m². Standard import noncommercial closure and release binary identity
  remain unverified. Numerical pass alone does not authorize final stack/fitter.
- 2026-10-02 D42: independent own continuous triangle primitive uses float64
  signed cross-product containment and nearest closed segments, no epsilon,
  geometry rescaling or vendor code. Chunk64no-grad nearest-face search then
  selected-face autograd; float64 squared metre output on original device.
  Predeclare same original10mm analytic/gradient gates, exact chunk/duplicate
  checks, thin nondegenerate triangle, then2048points against5120own faces,
 3finite forward/backward trials<=3s each and total<=30s. Tiny local tests are
  mathematics/schema only; no GPU/fitter adoption until actual remote gate.

- 2026-10-02 R43 (Azure own continuous geometry,f5feba2): all original analytic
  distance/gradient, rigid, chunk, duplicate/lowest-face and thin-triangle gates
  passed in1.554s. Distance error5.421e-20m²; point/interior triangle gradient
  error0. Three2048point/5120face forward/backward trials.06399/.05682/.05666s,
 90,553,856bytes peak allocation. No vendor metric primitive used. This establishes
  numerical/feasibility evidence, not shape estimation or challenge improvement.
- 2026-10-02 D43: independent fixed-pose continuous/isotropic shape proposal
  isolates a new measurement-model hypothesis. Same5DOF shared determinant-one
  model, conservative box, prior.01,10mm robust transition and hard<=100calls;
  unlike the old fitter, continuous face distance and isotropic pseudo-Huber.
  Supplied poses remain fixed and uncertainty unmodeled. Training convergence/
  residual improvement only marks a proposal; no mesh application/adoption.
  New untouched rendered falsification is required; no retuning old18controls.
- 2026-10-02 D44: clean-episode serial initializers bundle preflights all seven
  outputs absent, waits only explicit selected mask producer, and stops on first
  failed original stage. Episode0 queued under6b0020c; no individual duplicate GPU
  jobs, no generic report reuse or overwrite. Declarative12stage full-route planner
  and final native-conversion loader now test provenance/schema only; they do not
  replace original stage-specific numerical verification or license clearance.

- 2026-10-02 D45: new whole-candidate cached-ICP scheduling gate evaluates only
  one frozen procedural view (24generic, non-oracle seeds). Same candidate
  bytecode/private globals except solver binding, scalar rendering unchanged.
  Require exact image/slot/rejection/decision/status/inlier/iteration/best-index
  parity; pose/residual absolute error<=1e-6, not exact numerical JSON. Three
  alternating synchronized full-candidate timings,>=1.3× median and<=60s total.
  This is a new end-to-end experiment, not a relaxed repeat of the failed72-
  alignment budget. Full-video image parity remains a separate adoption gate.

- 2026-10-02 D46: freeze eight new continuous/isotropic fixed-pose conditions:
  ellipsoid(.33,.19,.27) and closed box(.52,.26,.38), correct/deformedSPD5,
  known/biased fixed poses, bottom35% horizontal occlusion. Six new camera paths,
  fit0/2/4, held1/3/5,8192coupled samples seed19, original1536×1152 camera rays.
  Correct canonical CD must remain<=1e-12m; deformed canonical CD gain>=5%,
  held-out IoU relative loss<=5%, topology/volume preserved. All8conditions
  execute within120s or retain partial failure. Both pose conditions disclose
  synthetic oracle initialization: this is measurement-model falsification,
  not non-oracle RGB or joint pose/shape validation. Two changed loss components
  prohibit a causal claim about discretization alone. Old18cases stay failed.

- 2026-10-02 R44 (Azure cached full-candidate gate,8ee3029): all candidate/image/
  branch/best-index comparisons passed; maximum pose/residual error1.11e-16.
  Three synchronized trials original6.313/6.162/6.186s, cached4.835/4.833/4.924s;
  median speedup1.2794 below frozen1.3 useful threshold, total45.84s. Reject
  adoption as this scheduling experiment; do not lower the threshold. Numerical
  parity is not bit-identical JSON and has not been tested on complete videos.
- 2026-10-02 R45 (Azure continuous shape falsification,5d45a5c): all8new
  conditions executed15.117s. Exact-pose correct controls abstained and retained
  zero canonical CD. Both **biased-pose correct controls failed**: canonical CD
  increased0→1.4681mm ellipsoid and0→1.2592mm box, despite improved held-out IoU.
  Four deformed cases improved canonical CD from6.963/7.437mm to4.041–4.559mm
  and passed. Reject fixed-pose continuous fitter adoption. Continuous geometry
  did not remove pose/shape confounding in these cases; this is not proof of one
  universal causal mechanism. Next shape work requires pose nuisance handling,
  identifiability and default-abstention selection, not another relaxed threshold.

- 2026-10-02 R46 (Azure general routing,6b0020c): episode0 full initializers
  completed at12:11UTC: body790frames302.89s, depth790frames407.27s, native
  adapter8.97s. Decoder errors vertices1.3345e-6m/joints9.8708e-7m/keypoints
  1.1071e-6m; projection8.6317e-5px. All seven serial stages passed, no duplicate
  GPU jobs or changed thresholds. No full object poses/native forward yet on0.
- 2026-10-02 D47: runtime archive now retains only transitive static/literal
  imports of the selected committed shell entrypoint, package initializers,
  configs and pyproject. All29wrapper closures validated without Azure launch;
  initializer estimated91.8→43.9KB encoded. Dynamic computed own imports fail
  where detectable; arbitrary unknown dynamic infra plugins remain unsupported.
  Existing live source snapshots are not changed.
- 2026-10-02 D48: experimental M0/M1 conservative shape-selection and automatic
  angular-view helpers are pure proposal/schema policies only. One paired-SE and
  normalized Schur conditioning are uncalibrated heuristics; default abstention,
  no statistical-identification/adoption claim. Views use normalize(-R.T@t),
  anchor quality then greedy max-min angles, invalid/degenerate poses never
  repaired. Actual model/geometry tests must precede use in the final pipeline.

- 2026-10-02 R47 (Azure budget generalization,b6b27d0): episode0 object producer
  stopped before trajectory fitting because all eight fixed global/componentwise
  simplification attempts failed topology/orientation/Euler checks. Globals created
  boundary/nonmanifold edges; componentwise changed sign/Euler. This is a true
  production failure, not permission to fill holes/delete shells/relax gates.
  Original grounded mesh and full initializers remain frozen; investigate a
  link-condition/topology-preserving simplifier on own controls first.
- 2026-10-02 D49: clean-episode actual shell route composes masks, seven existing
  initializer stages, original full object tracker, native preparation/forward/
  conversion. Requires explicit0..29, preflight all twelve targets absent before
  GPU use, no resume/retry/overwrite. Native wrappers accept explicit --no-wait
  for serial report-only dependencies, mutually exclusive with unit wait and
  checkpoint-only; actual Python provenance validation remains mandatory.
  Frozen historical episode15 job snapshots/defaults unchanged.
- 2026-10-02 D50: independent nested M0 pose-only/M1 sharedSPD5+per-view rigid
  nuisance gate: new ellipsoid(.31,.17,.29)/box(.48,.30,.34), three new camera
  paths, correct/deformed(.035,-.02,.009,-.011,.006), exact/biased supplied poses.
  Own synthetic pose-oracle disclosure; top-quarter occlusion and disjoint
  checkerboard128train/128holdout points per view. Same independent pose starts,
  50objective calls/model, <=100including diagnostics,120s total. Local data
  Schur uses vector continuous closest-face/region residual FD in normalized
  shape.05log/rotation.01rad/translation.01m, no priors/damping/pseudoinverse.
  Fixed-feature Jacobian is not exact nonsmooth/statistical information. Unknown
  global pose modes force selected M0; raw M1 separately faces original correct
  canonical0/nonregression and deformed>=5% gates. Default safety cannot conceal
  raw shape regression; held-out pixels are not independent temporal validation.

- 2026-10-02 R48 (Azure actualnative15,5d4f5db→414aac2): complete preparation
  passed501frames3582.24s, mesh/pose coordinate-change error9.02e-16m; exhaustive
  original PNG/H5/RGB/mask checks completed. Native CoCoNet then processed all
  six96frame windows including terminal overlap in68.85s. Official shared
  identity conversion passed501frames127.96s, worst per-frame mean1.78963mm
  to native predictions (frozen2mm fidelity gate), object frame error9.02e-16m.
  This is actual full-network/conversion evidence, NOT challenge accuracy.
- 2026-10-02 R49 (Azure nestedpose/shape,19a9a86): new8conditions all executed
 55.78s; overall hypothesis rejected. Some hard50-call optimizers failed
 convergence and kept their exact initial controls, including deformed box
 cases canonicalCD5.894mm unchanged. Full per-condition diagnostics remain
 remote; these failures do not establish that nuisance-pose shape fitting is
 impossible. No increased-budget rerun/selection claim; global modes remain
 unverified and conservative selectedM0 is not an efficacy win.

- 2026-10-02 R50 (H1reportformat fail-fast): the finger-transfer consumer
  stopped before output because the frozen actual-forward report omitted its
  top-level network field. The producer did enforce a Linux loopback-only guard
  and immutable Docker network-none wrapper. Do not rewrite it or treat all
  missing fields as offline: separately verify exact legacy source/wrapper,
  SHA-bound report/bundle; original transient systemd ExecStart is no longer
  available, so do not claim launch binding. Sidecar is only the source-bound
  mandatory loopback-guard contract, never security namespace attestation. Future forward
  reports explicitly record the enforced field; invalid present fields still fail.

- 2026-10-02 R51 (finalarchive gate fail-fast,9d489bf): consumer found that the
  original hard-wired episode15 converter report omitted episode_index. The
  conversion itself passed all501frames; this is format compatibility, not failed
  native geometry. Accept absence only for the exact producing414aac2 source
  SHA d2642f98 and requested15, retaining every source/input/artifact hash check.
  Any present malformed/wrong field or unknown source still fails. Preserve
  original report bytes; record explicit legacy identity basis in loader manifest.

- 2026-10-02 R52 (Azure CPU topology bootstrap,d64f0f6): BuildKit treated the
  bare local `sha256:<imageID>` FROM argument as `docker.io/library/sha256` and
  failed source resolution before the pinned wheel or geometry test. No passing
  build manifest was created. This is an infrastructure naming failure, not a
  topology result. Retry only with a content-addressed local tag verified against
  the exact original image ID before/after build; no new base image or apt repair.
- 2026-10-02 R53 (Azure MV source,d64f0f6): pinned public sparse acquisition
  `abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd` passed in5.47s. Frozen inventory:
  115 source/license/readme files,929905 bytes; clean Git and read-only tree.
  No extra model, challenge asset or local large transfer. This establishes
  source acquisition only; native preprocessing/conditioner/dynamics untested.
- 2026-10-02 R54 (Azure CPU topology build-v2,3d87d50): content-addressed local
  base reference passed before/after image-ID binding. Exact wheel imported and
  native tetra QEM/self-intersection smoke passed. Frozen image:
  `sha256:a34cdf72b862f97a7177be0b920842d637e812e0fe5b98c56fe940ffd8d67d81`.
  This is bootstrap only; larger sphere/torus/cavity/budget gate still required.
- 2026-10-02 R55 (Azure final integrity,3d87d50): episode15 full501-frame final
  archive/schema and upstream SHA-chain passed0.242s. Exact legacy conversion
  format retained without rewriting reports. Separate forward source-contract
  audit passed0.304s; launch binding and network-security attestation remain
  explicitly false. Neither result is challenge accuracy or licence eligibility.
- 2026-10-02 R56 (Azure MV preprocessing,3d87d50): failed before imports because
  root-created tempfile0700 became0500 when write bits were removed; non-root
  inference could not traverse the public source tree. No model/numerical test
  ran. Preserve the failed report; verify all original content hashes and change
  only public source access modes to0555 directories/0444 files before a unique
  v2 report. Future acquisition must not retain tempfile-private permissions.

- 2026-10-02 R57 (Azure mesh-link,cb3865c): sphere and torus each passed two
  deterministic global4096 QEM runs. Thin hollow control failed on its first
  run at strict convex-support/containment validation, after topology, volume,
  sampled-distance and self-intersection checks. The combined error does not
  establish shell escape specifically. Reject this backend configuration for
  production; no threshold relaxation or repeat on the old cohort. Original
  episode0 failure remains frozen. Test a constrained-placement alternative on
  new controls before touching production meshes.
- 2026-10-02 R58 (Azure MV preprocessing-v2,cb3865c): source-only access repair
  passed without content/manifest changes. Native checkpoint preprocessing
  passed10.054s; exact single XY flip error0m, SSI inverse max1.39e-7m, actual
  SS pointmap/RGB/mask518x518 grids registered. No conditioner/dynamics or shape
  accuracy claim. Proceed to the frozen partial SS embedding/fusion gate only.
- 2026-10-02 R59 (Azure MHR metadata,cb3865c): CPU inventory passed1.510s,
  no forward. Existing official696MB TorchScript already exposes127 joint names,
  249 parameter names,889x249 transform and127 parents. No200MB metadata download
  needed. Its MHRDemo graph is distinct from generic upstream MHR:204 controls
  plus45 identity names, with separate facial-expression module. Audit actual
  exported methods/graph; do not assume generic321 columns or117 padding.
- 2026-10-02 R60 (Azure partial SS,cb3865c): actual native six-modality
  conditioner and Stage1 fixed-latent dynamics passed33.941s, including replay,
  single-view/duplicate fusion, arithmetic mean and nonanchor permutation.
  Existing pinned weights only; no SLAT, entropy verification or decoded shape.
- 2026-10-02 R61 (Azure sparse H1,4ea44ea): finger-only proposal passed1.833s
  reference-forward phase for0/250/500, frozen136-pose/shared68+45 identity.
  Native decoded hand controls54 retained, other82 pose columns unchanged.
  Vertex/joint displacement is diagnostic, not accuracy; semantic invariance
  and independent synthetic RGB accuracy remain required before adoption.
- 2026-10-02 D51: new independent endpoint-QEM hypothesis fixes placement to
  surviving source edge endpoints, same4096/CD1%/volume5%/two-run/120s gates.
  Four new radial-star/torus/offcentre-hollow/disconnected controls. Global
  intersection selection plus outer-only float64 winding of all inward vertices
  tests containment, not convexity. Sampled boundary distance is explicitly NOT
  continuous minimum shell separation. Old convex-control rejection remains
  frozen; no production simplifier or challenge mesh is changed.
- 2026-10-02 R62 (Azure independent endpoint-QEM,b977218): four new controls,
  two deterministic runs each, passed26.505s.4096faces,2048–2052vertices;
  sampled CD/source-diagonal0.00352–0.00723<0.01; net-volume errors0.00069–
  0.00414<0.05. All three hollow cases retained outer-only winding containment
  and zero detected pair intersections. Endpoint placement configuration may
  now be tested as a separate real-mesh budget proposal; no challenge accuracy
  or universal exact embedding guarantee, no production-route adoption yet.
- 2026-10-02 D52: existing SHA-bound MHRDemo/Body checkpoint only; own neutral
  204 controls,45 identity and72 zero expressions,54 finger columns. Validate
  actual127 names/parents/889x249 transform and Body side indices. Six native
  forwards: corrective off/on, each neutral,216 central perturbations and exact
  replay. Frozen±0.001/0.002rad, finite nonzero vertex/orientation derivatives,
  excluded-joint/scales invariance,300s total/120s forward phase. No challenge
  inputs or full H1 adoption; neutral semantics is not RGB or hand accuracy.
- 2026-10-02 D53: separate CPU real-object endpoint budget proposal, explicit
  episode0..29,900s/4CPU/16GiB. Exact position welding preserves every oriented
  triangle before one QEM call; no approximate merging, face cleanup or repairs.
  Canonical GLB export then exact official weld/padding fidelity check, topology,
  intersections, source deviation and once-baked grounding scale. Full-resolution
  source cavity containment remains unverified; final two-shell containment is
  checked where applicable. No pose/production route or frozen output changes.
- 2026-10-02 D54: native full-constructor/decoded-shape integration gate only.
  Existing six exact Objects checkpoints/seven configs, no depth model or compile
  warmup; actual source/image SS prerequisite required. Own rendered ellipsoid
  RGB/masks/OpenCV pointmaps in three views; synthetic oracle observations are
  disclosed, generating mesh/poses never passed to inference. Single anchor and
  unweighted three-view, seed42,SS50/SLAT25, Gaussian+mesh decoding, no texture,
  geometry/layout repair or accuracy/adoption selection.300s total/180s generation/
  80GiB peak; distinct frozen Azure proposals, no local renders/checkpoints.
- 2026-10-02 R63 (Azure full MV bootstrap,9421d97): constructor stopped24.553s
  before complete model initialization because DINO called Torch Hub with keyword
  `repo_or_dir`, while our offline shim named its required argument `repo`.
  Preserve the failed report; support the actual positional/keyword API without
  changing native models/fixtures/budgets. V2 has distinct report/proposal paths;
  this is infrastructure compatibility, not a multiview numerical result.
- 2026-10-02 R64 (Azure finger semantics,9421d97): actual named joint/parameter/
  checkpoint mapping passed phase A. Native216-row forward then failed4.969s:
  MHRDemo concatenates model controls with zero identity rows and does not
  broadcast1x45 identity, unlike generic upstream MHR. V2 supplies216 identical
  zero identity rows explicitly, unchanged controls/gates/budgets; old failure
  report remains frozen. Neutral forward had completed, no derivative claim.
- 2026-10-02 R65 (Azure real-object endpoint budget,9421d97): exact source
  welding/manifold checks and source embedding passed. QEM attained2138vertices/
  4096faces with46 retained components, but the output embedding check failed.
  Reject real-mesh proposal; no component/cavity deletion, threshold relaxation,
  retry or production-route adoption. Procedural endpoint success does not
  generalize universally. Investigate constrained placement or fresh generated
  geometry, not mesh repair to evade penetration.
- 2026-10-02 D55: independent six-case hand RGB synthesis, prerequisite actual
  named-semantics V2 pass. Fixed neutral/left/right/bimanual/occlusion/crop cases,
  actual named controls/limits, shared zero identity/scales, diffuse plain materials.
  Known generating rig/poses/camera/LBS/visibility stay in a private evaluation
  directory; inference public manifest contains only six RGB names/hash/size.
  No generated mask, bbox or calibration given to inference.120s render budget,
  two-case anatomical visibility preflight; failure is not real-data accuracy.
- 2026-10-02 R66 (Azure finger semantics-v2,f4e64c5): phase A again passed;
  explicit-batch neutral/perturbation/replay executed, but exact geometry replay
  failed after three calls5.496s. No derivative or phase-B claim, no renderer
  launch. Diagnose native CUDA determinism with fixed seeds and deterministic
  algorithms before changing the numerical contract; preserve this failure.
- 2026-10-02 D56: synthetic mask/inference stages mount only public RGB and
  automatic masks, never the evaluation directory or challenge inputs. Frozen
  detector/SAM2 thresholds, no GT/full-image bbox fallback; actual installed
  SAM2 VCS and source inventory recorded, not falsely called pre-pinned. Six
  body/full paired inputs, fresh native204 controls/rotations, identical pixels,
  120s masks/180s inference budgets. Pending synthesis/semantic gates; no quality
  result or candidate adoption. New writable output directories owned by runtime
  user only; no broad permission changes.
- 2026-10-02 R67 (Azure full MV-v2,f4e64c5): actual native full constructor,
  single and unweighted three-view SS/SLAT/mesh decoding passed78.868s;
  generation49.280s, peak allocated17.641GB. Single370648vertices/741288faces;
  three-view357424vertices/714840faces, both watertight. These are integration
  proposals from disclosed synthetic oracle observations, not accuracy evidence.
  Native decoder-to-GLB axis rotation and anchor pose need a verified gauge
  bridge before any scoring/real-video use; neither mesh is adopted.
- 2026-10-02 D57: diagnose the frozen failed216-row reference fixture without
  loosening replay/support tolerances.20 forwards maximum: default CUDA, strict
  CUDA and strict CPU, both corrective modes with three repeats. Actual scripted
  source hashed; kernel identity/causal mechanism remain unverified.120s budget.
- 2026-10-02 R68 (Azure reference determinism,aa2ba21): PASS10.319s/20calls.
  Default CUDA changed vertices up to4.57764e-5 model-cm and skeleton elements
  up to3.81470e-6; strict CUDA and strict CPU each replayed raw-bit identically
  in both corrective modes. This supports a strict execution route, not hand
  accuracy, CPU/GPU cross-device bit equality or a proven kernel cause. V3 uses
  deterministic algorithms, CUBLAS workspace, disabled TF32 and fixed seeds;
  frozen V1/V2 failures and numerical thresholds are unchanged.
- 2026-10-02 D58: reserve/chown only the new renderer dataset, not the shared
  validation parent, and mount that dataset only. Separate private CPU evaluation
  can read truth only after frozen public inference/official proposals; report
  absolute hand PVE/finger MPJPE and distinct wrist-relative diagnostics over all
  six cases. Frozen>=5% mean nonneutral PVE gain and<=0.01mm neutral-side
  regression are synthetic-only hypothesis gates, never adoption or V2D scores.
- 2026-10-02 R69 (Azure strict semantics-v3,95d1294): FAIL5.594s after3calls
  on exact replay despite R68 strict diagnosis passing. Phase A passed again,
  phase B did not. R68 reused a model after default executions; its replay
  evidence did not certify cold TorchScript execution. Preserve failure; test
  four separate fresh processes, neutral1→perturb216→replay216, optimized on/off
  and both correctives, without warmup or tolerance relaxation. Cause unknown.
- 2026-10-02 D59: new independent object RGB cohort: two analytically embedded
  asymmetric surfaces, six physical views each, fixed diffuse textures/cameras.
  Only training RGB0/2/4 reaches automatic border-color segmentation and MoGe2;
  apply_mask=False retains genuine full-grid predictions, invalid geometry fails
  rather than filling. Private meshes/poses/visibility never enter inference.
  Single/three-view native generation uses equal50/25 steps, seed42 and original
  predicted depth; actual raw decoder arrays/export/native-camera transform are
  checked and frozen.300s generation budget. Private anchor-camera surface CD
  has no GT alignment/fitting:>=5% median gain,<=5% per-object regression and
  closed oriented positive volume required; this is not full Track1 scoring,
  embedding proof, novel-view pose validation, photorealism or adoption.
- 2026-10-02 R70 (Azure fresh reference replay,b70ee6a): PASS17.546s/12calls,
  four distinct worker processes, no warmup. Both optimized fresh conditions
  changed geometry (max4.57764e-5 model-cm vertices,3.81470e-6 skeleton elements);
  both unoptimized conditions replayed bit-identically. This supports disabling
  TorchScript optimization for reference validation, not a proven causal graph
  pass/kernel diagnosis. V4 preserves strict algorithms and every prior tolerance.
- 2026-10-02 R71 (Azure semantics-v4,0b5ee29): PASS8.801s/6calls. Actual
 54 named finger controls passed both corrective conditions, exact replay,
 excluded-joint/scales invariance and two-step central orientation derivatives;
 corrective skeletons are identical. Disabling JIT optimization follows R70,
 not a numerical threshold change. This verifies the own neutral reference
 rig semantics, not RGB anatomy/quality, SAM/reference shape parity or adoption.
- 2026-10-02 R72 (Azure hand RGB pipeline,d0b9d27): renderer completed all six
  cases with anatomical visibility preflight (first neutral hands each886visible
  pixels, left-bend840/886). Automatic six-person masks passed10.010s. Prediction
  stopped3.275s before inference: installed Body module parent was missing from
  PYTHONPATH. Original failure remains frozen; V2 adds the already installed
  audited package path only and uses a distinct predictions-v2 output. RGB,
  automatic masks, models and numerical settings are unchanged; no quality claim.
- 2026-10-02 R73 (Azure independent hand quality,bb81cf8 continuation): RGB
  renderer passed5.274s, automatic masks10.010s, paired Body/full inference
  passed26.304s/12calls, official shared-identity proposals passed27.336s and
  private evaluation completed0.234s over all six cases. Mean nonneutral hand
  PVE baseline133.978mm→candidate132.875mm:0.8233% gain, below frozen5% gate.
  Neutral-side changes-0.497/-0.201mm; synthetic hypothesis rejected, no adoption
  or CARI4D win claim. Absolute errors include wrist/body camera error; private
  wrist-relative and per-case metrics are retained for diagnosis. This diffuse
  procedural cohort is not photorealistic/external real-data validation.
- 2026-10-02 D60: H1 diagnostic separates pivot and articulation. Across observed
  cases, wrist-position errors75–203mm dominate camera hand PVE; wrist-relative
  baseline errors mostly7–11mm and finger transfer often worsens them. FULL also
  re-predicts body/IK wrists/shape, so local wrist-angle copying under a different
  parent is not a justified fix. Deprioritize wider hand-transfer scaffolding;
  investigate global/body evidence after the independent object experiment.
- 2026-10-02 R74 (Azure RGB object cohort,e4c30ec): render-only12RGB passed
  3.537s. Observations stopped0.007s before model load: audited HF repository
  SHA blob itself links to the task's global deduplicated Xet blob. Actual target
  is regular1,323,815,904bytes, content SHA280741fd…cc1a01 exactly matches pinned
  primary metadata; Xet identifier9f4c4857…db37c is storage, not content SHA.
  Accept only these exact audited targets and mount the one global blob read-only;
  content SHA/size/model/FOV gates unchanged. Original observation failure frozen,
  V2 distinct directory. No reconstruction or quality result yet.
- 2026-10-02 R75 (Azure object observations-v2,1c68365): six automatic RGB
  masks and genuine MoGe2 full-grid pointmaps passed9.427s. Native first single
  mesh decoded and raw/export/camera parity reached artifact publication, but
  the scalar parity tolerance remained NumPyfloat32 and JSON reporting failed.
  Preserve decoded artifact and partial failed receipt; cast this diagnostic
  scalar to Pythonfloat only. Distinct proposals-v2 repeats unchanged inputs,
  native steps/seed/gates; no model, observation or quality retuning.
- 2026-10-02 R76 (Azure independent object quality,9cc6863): all four native
  single/three-view proposals passed139.118s, peak17.715GB; actual raw/export/
  imported-camera parity max4.499e-7m. Private no-alignment evaluation passed
  3.024s. Camera CD: radial275.158→278.349cm, ring766.140→764.904cm;
  median relative gain-0.18784%, below frozen5% gate, per-object regression
  +1.1600%/-0.1613%. All four closed/oriented/positive-volume; embedding and
  final budget remain unverified. Reject this RGB-only fusion hypothesis, no
  adoption or CARI4D win. Camera CD combines shape/pose/scale; large absolute
  errors warrant separating predicted depth/centers from shape before another
  fitter. Source-coordinate parity is not metric calibration or fusion quality.
- 2026-10-02 D61: CPU-only H1 gauge diagnosis after frozen predictions. Fit
  one positive human Sim3 on baseline case0, reuse it unchanged for both modes
  and all six cases. No per-case/wrist alignment, input correction, new acceptance
  policy or rescoring of the rejected H1 gate. Report raw versus shared-aligned
  hand/wrist errors; if public kit role assets cannot be source-pinned, clearly
  label nonhand-vertex proxy rather than official Track1 alignment. H1/O2 are
  separate camera cohorts and cannot establish relative interaction accuracy.
- 2026-10-02 R77 (O2 frozen-factor diagnosis,CPU only): predicted anchor
  foreground median MoGe2 depth3.171m versus own rendered1.414m (radial),
  5.606m versus1.571m (ring). Single predicted centers Z3.145/5.742m versus
  truth1.600m; inferred XY extents also much larger. No GT-derived scale or pose
  correction applied. These diffuse object-only RGBs lack human/scene scale
  cues; their raw metric failure cannot isolate fusion shape quality. Preserve
  R76 rejection and prioritize supported shared gauge / motion measurements,
  not adjusting the failed fixture to force a better metric.
- 2026-10-02 D62: new CPU libigl intersection-blocking QEM, not retrying
  endpoint-QEM or deleting its intersecting faces. Pin libigl2.6.0 commit
  40e7900ccbd767f1f360e0eb10f0f1a6432e0993 and Eigen3.4.0 headers; compile
 600s deadline on Azure, no apt/dependency fallback. Native qslim target4096,
  block_intersections=True rejects collisions before collapse. Two new close-
  shell/disconnected nonconvex procedural controls,180s total; independent
  topology, embedding,<=1% sampled diagonal CD and<=5% per-shell/net volume.
  Match original shells by native birthface maps, not sorted-volume proximity.
  Floating predicates and skipped one-ring checks are not universal proof;
  independent final embedding is mandatory. Only if controls pass, propose one
  separate episode0 budgeted mesh, with unchanged metric scale/poses and no
  component/cavity removal or numerical threshold relaxation.
- 2026-10-02 R78 (Azure H1 common-gauge proxy,a30e5b3): CPU diagnosis
  passed0.319s. One baseline-case0 nonhand Sim3 (scale0.973535), reused for
  both modes/all cases, changes cases1..5 mean hand PVE baseline133.978→63.852mm,
  candidate132.875→63.065mm. Heldcase aligned wrist errors28.686–96.811mm
  remain; a common gauge explains part, not all, of raw error. Private LBS
  nonhand complement is explicitly not official role_alignment. No inference
  correction, object interaction evaluation, transformed export or new selection;
  original H1 rejection remains frozen. Full tiny suite3353PASS34.46s.
- 2026-10-02 R79 (Azure guarded QEM build/controls,c1d461d→7d0e1d0):
  source+compiler+native binary build passed13.585s; image20dd08fe…69d26f,
  binary SHA3bd9ba61…b3dff, source pins/no apt preserved. New two-control
  geometry gate passed2.107s. Each2052vertices/4096faces, independent0intersections;
  sampled diagonal CD0.7134%/0.3757%, net volume0.0655%/0.1243%, max matched
  shell-volume0.1551%/0.2690%, close-shell containment passed. Native birthmaps
  retain original shells; no repairs or scale fitting. This permits one separate
  episode0 proposal, not universal simplifier/embedding proof or adoption.
- 2026-10-02 D63: new adjacent-frame RGB motion cohort (three objects×eight
  frames), two fresh textured surfaces and a weak-texture control. Only public
  RGB/automatic masks/MoGe reach inference; meshes, R/t, depth and visibility stay
  private. Stateless own OpenCV LK→mesh-PnP tests fixed shape/scale, forward/back
  consistency, disjoint spatial reprojection tracks and explicit abstention.
  No new checkpoints or challenge labels; moderate and fast-step motion frozen
  before render. Inherited anchor depth/shape error is separated from relative
  tracking diagnosis, never canceled using GT during inference.
- 2026-10-02 R80 (Azure episode0 guarded proposal,8cdad20): native global
  intersection-blocking QEM reached2138vertices/4096faces with independent zero
  intersections. Proposal failed6.252s at unchanged1% sampled diagonal CD /5%
  net and birthface-matched shell-volume gate; reject, no export/adoption.
  Numerical measurements were lost by a raise-before-assignment reporting flaw,
  so the exact failing axis remains unknown. Future calls retain metrics before
  rejecting; original report stays frozen and no threshold is changed.
  This improves embedding over endpoint-QEM's six intersecting faces, but is
  not sufficient fidelity. Do not claim success or starvation of small shells
  without the missing measured evidence.
- 2026-10-02 R81 (Azure new public motion cohort,3698a35): render-only24RGB
  passed4.163s; automatic masks and unchanged MoGe2 observations passed13.174s,
  all24original frames. No private GT reaches inference, no downloaded CAD or
  new checkpoint. Separate fixed-anchor generation/tracking/private evaluation
  begins at ec60dd9; spatial reserved-track reprojection is not held-out temporal
  accuracy. Full tiny suite3423PASS33.10s before final tracking tests.
- 2026-10-02 R82 (Azure motion anchors/tracking,ec60dd9): three actual
  RGB-derived fixed anchors passed93.342s, peak18.448GB. Tracking stopped8.549s
  on object0 frame5: depth ICP proposed a whole-mesh camera crossing, caught
  before raster clipping. Earlier four temporal frames abstained on sparse
  spatial LK support; no completed trajectory or quality result. Preserve failed
  tracking/report.json and reuse frozen anchors for tracking-v2. Fix initialization
  about camera-coordinate anchor pivot (t=current_center-R*anchor_center), reject
  invalid fitted ICP, retain valid measured-center alternative; thresholds and
  masks unchanged. Also fix private evaluator loop indentation found by source
  review, with a tiny eight-frame actual-file regression test before any evaluation.
  No failed method gate is reclassified; full suite3428PASS34.27s before fixes.
- 2026-10-02 R83 (Azure frozen mesh replay diagnostic,b6dbabe): measurement
  completed6.049s, same source/binary/configuration and available topology/
  intersection signature as R80. Sampled diagonal CD0.287707%, net volume
 0.173878% pass;39of46 birthface-matched shells exceed5% volume error, worst
 14.0343% on inward component10. Historical candidate arrays were lost, so
  bitwise identity is not proved. Replay arrays/maps now hashed and retained
  remotely; original rejection/controls untouched, no prediction export/adoption.
  This attributes the replay's failure to cavity volume fidelity, not global CD
  or embedding; motivates one volume-constrained contraction test, not deleting
  cavities or relaxing gates.
- 2026-10-02 R84 (Azure all-frame motion quality,b6dbabe): tracking-v2
  passed22.398s, all24frames;11PnP proposals,10explicit abstentions to measured
  depth baseline. Actual predicted raster attachment parity passed for each
  anchor; sparse counts30/65/59. Private evaluation passed44.716s. Mean camera
  CD per object288.093→288.093,946.319→950.868,629.788→628.960cm; median gain
 0.131468%<5%, per-object regression0/+0.480712%/−0.131468%, fast-step gate
  passes. Relative-motion surface diagnostic33.329→33.329,68.726→63.931,
 56.967→54.840cm; not actual-GT shape CD and never inference correction. Reject
  this fixed-anchor direct LK/PnP hypothesis: large inherited metric bias remains,
  so more tracking confidence is not itself reconstruction accuracy. No adoption,
  final mesh/embedding guarantee or CARI4D win. Full tiny suite3450PASS32.89s.
- 2026-10-02 D64: isolate one mesh-fidelity mechanism after R83. Retain native
  QSlim cost/placement and one global intersection tree; add pre-collapse
  cumulative signed source-shell volume guard at the unchanged5% limit. Commit
  local volume delta only after successful contraction; no cost normalization,
  per-shell independent decimation, component deletion or repair. Build from
  already pinned libigl/Eigen sources on Azure, no new acquisition. New thin-
  cavity/multiscale controls first, independent embedding/CD/net/per-shell volume
  and birth mapping; one distinct episode0 proposal only if controls pass.
  If budget/fidelity fails, abandon this decoded source rather than repeatedly
  broadening tolerances. Controls/previous failures remain immutable.
- 2026-10-02 D65 (real validation rights screen): CORE4D metadata conflict
  prevents claiming commercial clearance: instructions96b9084 sayCC-BY4.0,
  HF81bb2bb cardMIT, website dataset JSON-LD atd3203b6 saysCC-BY-NC4.0.
  Bundled SMPL-X model/software terms also cover derived meshes/animations;
  reading supplied arrays with NumPy does not resolve their rights. No dataset
  acquired or restricted source imported. Ask owners for written scope and two
  hashed single-view validation minibundles; do not fetch~104GB RGB shards plus
  ~40GBmotion batches for two clips or assert no pretraining/challenge overlap.
- 2026-10-02 R85 (Azure volume QEM build,64a7a06): offline inherited-source
  compilation/ABI gate passed13.048s, imagec8fb1632…e21137, binary SHAeb606feb…055e7.
  Native placement/cost and collision blocking unchanged; fixed cumulative
  per-source-shell signed volume limit5%. No new package/source acquisition,
  GPU, challenge geometry or accuracy gate during build. Full tiny suite
 3464PASS/1optional-trimesh-testSKIP33.15s; actual procedural geometry checks
  follow on Azure and are not inferred from source-contract tests.
- 2026-10-02 R86 (Azure volume controls/episode0,5e0a16f): two new controls
  passed2.968s. Thin cavity CD0.743042%, net-volume0.002443%, max shell0.162438%,
  zero volume vetoes; six multiscale cavities CD0.548030%, net0.043602%, max
  shell4.990999%, one native veto. Independent intersections zero and every
  cavity contained. Separate episode0 proposal passed7.589s: CD0.287107%,
  net0.483433%, max matched-shell4.999369%, all original46shells retained,
  final4096faces with independent embedding/float32 export/official packing
  checks passed. Frozen metric scale baked exactly once, poses unchanged;
  geometry SHA90579ac5…73380. Engineering-valid mesh proposal, not accuracy
  validation/adoption or CARI4D victory. Proceed to original full-video pose
  and native forward chain, not relaxed geometry tolerances.
- 2026-10-02 D66: consume the frozen qualified volume mesh in the existing
  episode0 full-video/native chain. CPU receipt/GLB/NPZ hashes, exact metric
  oriented triangles, topology and upstream embedding checks are bound; no
  GPU resimplification, component repair or second grounding scale. Use a
  compact nonprocessing render view, retaining the official packed arrays.
  Strengthen full-body video provenance and automatic object-mask coverage;
  keep the original pose hypotheses, image/ICP/Viterbi and downstream stages.
  Full tiny suite3484PASS/1optional-trimeshSKIP30.12s. Actual initializers
  cover790frames; no existing full-object/native targets will be overwritten.
- 2026-10-02 D67: after native episode0 dispatch, prepare one new independent
  joint-human/object RGB grounding cohort, three clips×three frames. Actual
  reference MHR/own asymmetric bottle rendering, true focal1280/960/1600
  private, inference prior fixed1280; automatic detector/SAM masks, Body and
  MoGe2 only. Compare raw object observations against one human-anchored
  positive clip scale. Freeze predictions before private visible-surface CD;
  median clip gain≥5%, no clip regression>5%, full nine-frame coverage.
  Human-first-frame shared Sim3, depth bias and permuted-alpha are diagnostics
  only, never inference corrections or main-score alignment. No additional
  generative object model until this cheaper grounding hypothesis survives.
- 2026-10-02 R87 (native episode0 preflight,3fe357c): launch stopped before
  any computation because the legacy full-pose failure left an **empty** output
  directory. No new predictions or receipts were written. Remote inspection
  confirmed no files, the old failure log remains; remove only with `rmdir`
  (refuses nonempty directories), then use a fresh unit/log. This is cleanup of
  disposable failed-run noise, not replacement of frozen predictions or a
  change to geometry thresholds. J1 research can queue behind the new unit to
  avoid overlapping H100 jobs; predecessor outputs are not J1 inputs.
  Empty legacy target removed; original failure log SHA67404776…035d4 retained.
  J1 source integration full tiny suite3567PASS/1optional-trimeshSKIP33.75s;
  this validates contracts/metric implementations, not actual model outcomes.
- 2026-10-02 D68: source-backed CPU throughput hypothesis only: compare exact
  native depth writer framewise against synchronous batches8 on new16-frame
  procedural1536×1152 depth arrays, same level9 PNG encoding/metadata, complete
  canonical-byte validation and two paired orders. Require byte/order/metadata
  parity and≥1.25× median write throughput within240s CPU budget. Temporary H5
  files remain remote and are removed after retained digest/timing decisions.
  No active-chain changes, geometry/accuracy claim, compression/gate relaxation
  or external acquisition. Record concurrent GPU/CPU load if benchmark executes.
  Native source SHA14297609…5b0c5; source/runtime contracts full tiny suite
 3577PASS/1optional-trimeshSKIP33.33s. Benchmark will use CPU4/16g with no
  GPU visibility; active pose process remains unchanged. Shared-host throughput
  is explicitly not isolated whole-pipeline timing.
- 2026-10-02 R88 (Azure CPU batching,772ae1c): whole240s gate failed
  (242.702s recorded during executor teardown; wrapper124). First balanced
  pair completed exact encoded byte/order/metadata parity and exhaustive
  validation: single write80.213s vs batch8 20.057s (~4.0×), validation
 20.193s/20.216s. Reverse pair batch8 write20.053s/validation20.224s;
  final single writer did not complete before deadline. Temporary H5 directories
  confirmed removed, partial receipts retained. **Not** a complete balanced
  throughput gate/adoption, no median reported, deadline unchanged and no rerun
  to relabel failure. Mechanism supported by first pair, but active native
  preparation stays frozen. Future production batch change still needs the
  predeclared complete throughput/parity evidence on a new protocol.
- 2026-10-02 D69: audit critique confirme l'étape native de raffinement final
  absente de nos précédentes chaînes. Ne pas présenter l'export501frames déjà
  validé comme la baseline complète: c'est CoCoNet-forward + conversion propre.
  Ajouter la vraie optimisation publique `smplh_parity`, full-clip/batch0,
  300steps demandés/301updates natifs, hyperparamètres par défaut. Body rotations
  et objet translation seuls optimisés; root/mains/identité/échelle/caméra,
  translations internes et objet rotation fixes. Pas GT ni contact manuel,
  pas nouvelle FoundationPose, pas moyenner silencieusement identité native.
  Deux petits assets officiels hashés directement sur Azure, nouveaux outputs
  stricts séparés; vérifier couverture/ABI/paramètres fixes bit-identiques et
  history/finaldiagnostics finis avant conversion officielle2mm inchangée.
  Budget exploratoire dur7200s sans retry automatique; timing/qualité inconnus.
  Préparer épisode15 déjà gelé, queue GPU derrière J1 indépendant, aucune
  modification/reprise du long job épisode0 actuellement450/790 à16:58:51UTC.
  J1 source audit confirme unités/masques/K corrects; renforcer seulement son
  reçu masques avec booléens noGT/nohandlabels explicites. Le job J1 gelé ne
  sera pas remplacé et ses reçus producteurs contiennent déjà ces booléens.
- 2026-10-02 R89 (native refinement CPU preflight,0b55102): acquisition réelle
  des deux assets officiels PASS, total173282bytes directementAzure; receipt
  SHA5f7816b2e47773a1320370bffbc41cb04d055f0edf79af42ccd231769d01c41a.
  CPU import gate échoue honnêtement avant réservation de sortie/GPU: Kaolin
  importe Warp, qui initialise son cache dans `/.cache` non writable pour le
  UID scenesmith. Aucun résultat refined ni conversion n'a été produit. Corriger
  seulement HOME/XDG_CACHE_HOME vers `/tmp` dans les deux containers, sans
  modification vendor/dependencies/pertes/budgets. Nouvelle unité/log et source
  immuables, assets hashés réutilisables; ancien échec conservé. Premier full
  suite intégration3689PASS/1optional-trimeshSKIP33.32s, pas preuve GPU/qualité.
  Pose épisode0 toujours active600/790 à17:08:40UTC; J1 demeure en queue.
- 2026-10-02 R90 (native refinement preflight-v2,9f2a4ec): vraie importation
  CPU offline du module pin7c0d, PyTorch3D, nvdiffrast, transformers, Kaolin et
  Utils **PASS**. Configuration native complète vérifiée:300steps/batch0,
  lr.001, contact200, silhouette.002, penetration2, temporal100,
  object-translation-prior100; aucun contexte CUDA initialisé. L'avertissement
  Warp «GPU unavailable» est normal dans ce container sans accèsGPU, pas une
  validation des kernels. Caches éphémères `/tmp`, vendor toujours readonly.
  Deux assets réutilisés avec hashes exacts. Unité
  `world-reward-cari-native-final-refinement-v2` active et en attente du job
  J1 à17:12:33UTC, aucun répertoire de raffinement GPU encore réservé.
  Pose épisode0 active650/790, sans changement à sa source. Hypothèse J1 et
  raffinement restent **non mesurés**, ne pas annoncer de gain/qualité/soumission.
  Tests ciblés cache/refine65PASS2.26s et consommateurs529PASS22.81s;
  précédent full-suite3689PASS/1optional-trimeshSKIP33.32s. Aucune adoption,
  dérogation licence ou règle Kaggle acceptée par ce lancement.
- 2026-10-02 D70: nouvelle hypothèse caméra J2, fondée sur l'ambiguïté
  focal/Z-shift des modèles, pas sur résultats privés J1 encore inconnus.
  Même9RGB/mêmes masques automatiques, sans nouveau rendering/modèle/GT.
  MoGe2(None) neuf fois → medianf trois frames par clip → nouveau MoGe2 avec
  FOV fixé et nouveau Body avec exactement le même K. Une alpha humaine
  commune par clip appliquée une seule fois àXYZ, humain inchangé. Aucun
  rescaling de traduction sauvegardée ou changement de K sans recalcul natif
  Zshift. Budget600s, nouveau namespace `predictions_camera_v1`; ancien J1
  fixe1280 immuable. Instrumenter seulement le solveur natif hashé pour refuser
  fallback<2pixels nearest64 avant solve; appeler l'original inchangé et
  restaurer enfin, vérifier18calls natifs et9Body. K centré/squarepixel positif,
  pas clamp/GT/calibration promise. Comparaison principale: CD caméra brut
  **aligned-learned contre aligned-fixed**, seuils J1 inchangés median gain≥5%
  et aucun clip régression>5%, tous9frames y compris contrôles960/1600.
  Les deux bundles/masques/RGB sont vérifiés avant ouverture du GT privé;
  diagnostiques focal et Sim3 ne sélectionnent rien. Aucun score de cette
  fixture n'est la métrique complète V2D. Préparer queue derrière raffinement
  épisode15, sans changer/reprendre jobs actuellement actifs.
- 2026-10-02 R91 (17:39:33UTC): épisode0 full-object pose **PASS790frames**,
  3712.869s, producteurbaba81b; son reçu SHA8a0c8cef…436bb et bundle
  SHA942d1286…995 conservés sur Azure. La chaîne native continue en CPU
  préparation depth100/790, pas encore forward/conversion/schema. IoU greedy
  median.390973 est diagnostique image, pas score GT ni trajectoire Viterbi
  finale. J1, raffinement15 et J2 toujours en queue GPU séquentielle. Nouveau
  producteur J2 `20714cf21436259ab8e6abc1993a90f97da1c2d5`, fermeture15files/
  54864bytes encodés; full-suite3732PASS/1optional-trimeshSKIP37.92s. Rien ne
  prouve encore un gain caméra, une baseline raffinée complète ou une victoire.
- 2026-10-02 D71: tester un **nouveau protocole CPU compact distinct**, pas
  requalifier l'échec R88:8frames/une batch8 complète, même1536×1152,
  nouveau seed1711, ellipse/occludeur et vrais zéros invalides; valid_count
  égale exactement count(depth>0), scale.83/shift0. Garder240s, warmup2frames
  chacun, deux paires AB/BA, source/encodeur PNG natif exact, validation
  exhaustive et égalité payload-byte/ordre/métadonnées; seuil median≥1.25×.
  Namespace `depth-batch-compact-v1`, ancien16frames/seed1709/default/wrapper
  et échec immuables. Scope write/encode/close uniquement, pas débit complet
  préparation, pas validation réduite, pas adoption active; tailbatch2frames
  couvert au warmup mais pas chronométré. CPU4/16g/noGPU/networknone, charge
  CPU concurrente possible pendant préparation0: ne pas annoncer mesure
  isolée. Tests18PASS.24s; tester réellement sur Azure après gel du source.
- 2026-10-02 R92 (Azure compactCPU,96542a7): **PASS56.838s**, deux paires
  AB/BA complètes, payload-byte/ordre/métadonnées identiques et validation
  exhaustive. Median single13.357s, batch8 3.370s, **3.963827× write-only**;
  validations3.369–3.385s chacune. Zéros/valid_count conservés, fichiersH5
  temporaires supprimés vérifiés. CPU préparation0 concurrente, donc pas débit
  isolé du pipeline ni adoption dans son lecteur gelé. R88 ancien16frames
  garde son statut FAIL. Full suite3740PASS/1optional-trimeshSKIP37.47s.
  Reçu Body15 distant confirme MHRmodèle SHA352e271a…7377bc, exactement le
  hash natif du hand-spec; identité assets n'est plus une inconnue d'ABI.
- 2026-10-02 D72: validation **réelle objet-seul externe TUD-L**, distincte
  de challengeGT et d'une métrique humaine/HOI complète. Trois scènes IDs1/2/3,
  neuf RGB préchoisis par noms triés first/median-index100/last avant valeurs
  privées:0/4074/8227;3/4013/7710;4/4028/7969. Trois ZIP HFpin6527f7d,
  total374952356bytes, hashes exacts; fulltest14.9GB/train jamais acquis.
  CC-BY-SA4 publisher et HF concordants, attribution/evidence conservées;
  sources privées/médias restent Azure, aucune transmission locale. Extraction
  publique RGB640×480 et SHA seulement, privé depth/K/masks/allinstances/
  meshes séparé700; ZIP/nonselected nettoyés, subset9 conservé reproductible.
  Comparer MoGe2 fixedf800 contre9focals RGB→median3/scene→nouveau9fixedFOV;
  total27 appels natifs, fallback<2sampled64 refusé, même checkpoint/source,
  aucune scale/GTcalibration/masque privé/Body à l'inférence. Budget600s32g4CPU
  GPU serial derrière J2, acquisition séparéeCPU600s; évaluationCPU90s.
  Tous9RGB/predictions/paramètres/counters validés **avant** lecture privée.
  GTsensor Z×depth_scale(parimage)/1000 et rayons entiers BOP; XYZprédits
  +.5 restent intacts avec leur propreK, jamais GTK pour reprojeter prédiction.
  Main CDhalf visibleobjet sur mêmes pixels et mêmes8192samples max; threshold
  median scene gain≥5%, aucune scène régression>5%, eachmethod coverage≥95%
  et learned≥fixed−1point parframe. Pas dropping, alignement, oracle-ray,
  sélection privée ou fauxscore humain. DepthAbsRel/MAE/Zbias/focalerror
  diagnostiques seulement. TUD-L absent des24datasets train/10eval MoGe2
  publiés, mais training_overlap_excludedFalse reste obligatoire: pas preuve
  entièrement unseen, victoire CARI ou clairance du stack SAM/CARI entier.
- 2026-10-02 R93 (Azure TUD-L acquisition,2a18859): **PASS15.318s**,
  374952356bytes acquis directementAzure,9RGB/noms préchoisis conservés,
  41fichiers privés hashés, licences primaires concordantes; ZIP temporaires
  supprimés. Reçu SHA d096f1eddbca90c8d037ba85e323aa826974c97ba2fd658b0c9cdd76de2dd6ea,
  public manifest SHA171e89b563520bb9311220b0ddbce068a63d43f843815edf139cb74b61352ecc.
  `world-reward-tudl-real-camera` source2a18859,11files25580bytes, queue
  derrière J2; **aucune inférence réelle TUD-L ou qualité observée**. Fullsuite
  3856PASS/1optional-trimeshSKIP37.36s. Épisode0préparation450/790 à18:22UTC,
  source gelée intacte. Quota Azure vérifié: SKU NCC40adsH100 famille
  StandardNCCads2023Family40/80cores (NCads/H100 autre famille0/0), pas nouvelle
  VM créée; ne pas confondre les noms SKU pour un futur secondGPU.
- 2026-10-02 D73: avant toute adoption CPUbatch8, un **seul gate sur les
  vraies profondeurs prédites gelées** épisode15, frames0–8 (8+tail1).
  Aucun nouveau RGB/label/GT, ni changement de prepare0 actif. Binder original
  PASS501prep producteur5d4f5db/SHA579498b, depthreceipt611d52a, align788ac61,
  H5a9458af; champs legacy manquants acceptés uniquement par ces identités
  exactes/source ép15, pas fallback générique. Regénérer raw=np.where(valid,
  depth,0), aligned=raw*scale, valid_count=valid.sum **même si depth=0 valide**.
  Source/indices/float32/clipscale/nativeJSONidentity intacts. Warmup2 chacun
  et paires AB/BA9frames, tous nouveaux H5 validés exhaustivement: PNG exacts/
  noms/metadata doivent égaler le **subset original H5**, pas seulement eux.
 240s4CPU16g/noGPU/networknone, medianwritegain≥1.25×, hashes entrées vérifiés
  après, H5 temporaires supprimés. Pas replay501payloads pendant240s: reçu
  original501exhaustive lié au H5SHA entier reste source. Scope neuf frames
  write-only, no whole-prep timing/adoption/qualité. Tests48PASS1.03s;
  anciens protocols/defaults et prédictions restent gelés.
- 2026-10-02 R94 (actual-depth CPU, f15c0a1): **FAIL.056s integrity**,
  avant lecture H5/calcul/timing, aucune trial ni résultat de batching.
  Source historique depth660e197 n'émettait pas `input_dataset_revision`,
  pourtant nouvelle vérification le demandait; script primaire à5d4f importe
  `_validate_inputs` du Bodyhelper qui vérifie le datasetpin5f68335 et fixe15.
  Ne pas réécrire le vieux receipt ni inventer son champ. Accepter absence
  **uniquement** pour depthreceiptSHA611d52a/script660e197 exacts, avec
  source historique audité; champ présent erroné reste FAIL. Namespacev2/unit
  neufs, budgets/parité/frames/valid_count/gates inchangés. Reçu échec SHA
  bdcf72b2739aabe0030b13506d221732f6b73b5baf6961a192d5db9806034a00 conservé.
  Épisode0préparation550/790 à18:32:46UTC; aucune relance du lecteur actif.
- 2026-10-02 R95 (actual-depth-v2 dispatch,f10b1a7): correction ABI seule,
  même9frames/240s/parité/protocole, fermeture7files16496bytes. Unité active
  à18:38:04UTC: warmups passés et single première paire56.011s, validation
  19.108s; batch8 en cours. Reçu par défaut `status=fail` sans exception ni
  finalphase **n'est pas encore un échec terminal**. Pas median/balanced gate
  ni adoption avant fin des deux paires. Source héritée n'a pas été réécrite.
  Fullsuite3889PASS/1optional-trimeshSKIP41.40s avec caches bytecode désactivés;
  anciens caches locaux jetables nettoyés, code/receipts/predictions intacts.
  Prepare0 depth600/790 observée; tous GPU jobs qualité restent séquentiels.
- 2026-10-02 R96 (actual-depth-v2,f10b1a7): **FAIL240s deadline**,
  242.711s lors teardown, wrapper137. Première paire complète égale au H5
  original: single56.011s/validation19.108s, batch8+tail1 18.669s/19.057s.
  Reverse batch18.641s/validation19.174s, single55.924s mais sa dernière
  validation n'achève pas avant deadline. Pas full balanced gate, pas median
  validé/adoption et aucun seuil/budget relâché pour forcer PASS. Tous nouveaux
  H5 temporaires supprimés vérifiés; échec SHA69f17613cf476e39b8bff170e124f8cf8c1e1dd99f6f7eebdb8f21a768fb8c8e
  et parités partielles utiles conservés. **Écarter l'adoption active** plutôt
  que répéter une troisième fois le même budget. Le compact R92PASS prouve
  le mécanisme CPU, pas la procédure complète sur ces profondeurs réelles.
  Native prepare0 reste active600/790 à18:40:51UTC, aucun job GPU relancé.
- 2026-10-02 D74: optional validated Azure target in immutable launcher,
  default argv/snapshot unchanged, job name cannot choose VM or remote root.
  Invalid/injection/abbreviated fields fail before Git/Azure;63tests PASS.04s,
  full3915PASS/1optional-trimeshSKIP39.39s. Not a compute allocation or experiment.
- 2026-10-02 D75: independent second H100 for quality research, setup budget≤1h
  conditional on capacity/import/runtime. Exact confidential VMI and fresh
  Docker, no user disk/OS/live-state cloning. New resource group/NIC/NSG,
  temporary private-only SSH VM01→VM02 with host-key verification and revocation;
  allowlisted image/MoGe/independent-validation TAR transfers Azure-only. Existing
  queue/readers/output namespaces untouched. Fail allocation/CUDA/import gates
  without claiming GPU results; cleanup only newly identified task resources.
- 2026-10-02 R97/D75: VM02 actual creation dispatched19:04 after direct user
  authorization of image terms. Initial blank-EULA prompt failures happened
  before deployment and created no VM; new NSG/NIC only. Creating NCC40adsH100
  zone1/private10.0.0.9, noPIP, fresh OS128GB and no shared-disk clone. Explicit
  private containerd/Docker bootstrap and whitelist14public/56withprivate
  transfer helpers56tinytests PASS22.48s. CPU/H100/image checks still pending;
  no secondGPU measurement/adoption. Export uses SHA-bound pinned CARI image,
  exact MoGe chain and independent TUD-L subset only, not active containers.
- 2026-10-02 R98: VM02 provisioning/H100/CC ON/HTTPS egress **PASS**.
  First runtime dispatch FAIL209/STDOUT before Python/shell: missing new
  `results` directory in launcher-assumed fresh root. No Docker/config/GPU ran;
  fix scoped fresh root1000/results only and new v2 unit, retain v1 failure.
  This is bootstrap integration evidence, not research-model inference.
- 2026-10-02 D76: isolate DA3METRIC-LARGE against frozen MoGe2 fixed camera
  on existing nine TUD-L RGB, no duplicate MoGe GPU run. Same originalK800,
  native canonicalZ×processedmeanf/300 once then declared bilinearZ/.5XYZ.
  GPU600s32g4CPU/newVM02/newoutput, strict406F32 fullstate inclbuffers and
  native sky correction/seed, no confidence drops/private K/body/calibration.
  Primary source/weightsApache2; GPLAPI excluded via27filedepth closure.
  Pair only after baseline frozen and SHA-checked Azure transfer; keep≥5%median
  scene gain/no>5%regression/coverage gates, no challenge victory inferred.
  GeoCalib adaptation rights/newcamera-solver cost defer that alternative.
- 2026-10-02 R99: actualVM02 private runtime **PASS12s**, source0365d05,
  zeroimages/containers, ownsocket/overlay2/containerd namespaces/NVIDIA runtime.
  Task-only export **PASS**, source671d10c: 1337763840byteassets9b876f95…e6027
  +14565534720byteimage203af62c…21b4e, pinnedimageIDb47e4450…380a7.
  Private strict-host-key SSH transfer dispatched Azure01→02; no local data
  transit. VM01 GPU forward active89%/4342MiB at19:14, no validatedfinalchain.
  No new model inference or GPU-smoke claim onVM02 before import gates.
- 2026-10-02 R100: original queue ends, no TUD migration/duplicate inference.
  Episode0 fullforward finishes but official converter per-frame mean>2mm gate
  FAIL; do not relax fidelity or emit invalid submission. Empty conversion output
  and useful failure trace retained. J1 shared-human scale quality PASSmedian
  clipgain30.2626%, clips+84.9365/+30.2626/−3.0973%; independent syntheticonly,
  noadopt/CARIclaim. J2 completes; retrieve paired camera comparison separately.
  TUD-L actual27callsPASS13.539s/evalPASS1.919s: fixedCDhalf81.2699/93.4758/
  74.2677cm versuslearned15.7115/4.2393/6.2498cm, mediangain91.5848%, allcoverage
  gatesPASS. Actualcamera hypothesis supported on independent object-onlyreal
  cohort, not human/V2D score. Transfer these frozen predictions Azure01→02,
  don't rerunMoGe. Native501refinement FAILbeforeGPU 'predecessor unit not loaded'
  after completed transientJ1 collected; no refinement outputs produced. Explicit
  continuation newunit/newoutputs rechecks source/assets without reacquisition;
  all other GPU readers already finished, no restart of old unit or gate waiver.
- 2026-10-02 R101: pairedJ2 camera comparison **REJECT**median−22.8774%,
  clips−541.0395/−22.8774/+39.8150% againstfixed-K sharedscale; J2 vsownraw
  depth gains51.0897% do NOT prove it beats J1. Opposite TUD-L outcome exposes
  domain/camera-estimator bias; no universal learned-K adoption or cherry-pick.
  J1qualitySHA03d7445b…209ef;J2qualitye1b6fcd9…b05bd;
  TUDqualitye3cb111e…2d2c;predictiona1879591…682a. TUDlearnedf546.1106/
  527.2065/542.1047px are RGB-only, not inferred from privateK. Frozen prediction
  TAR68085760bytes SHA81e3226c43c6311f98c7a43f704b80a9dc69ce4bb0d233d971ffc6c88f281881
  transferredAzure01→02 withnoMoGerun. DA3 acquisition actualPASS35.157s,
  allpins/source/1.337GBweightverified; source_manifestSHA3faa74f9b1b22fe076fd37833f1227da94c27256a06e1e8ebc27ce8b2367ede5.
  Import image/GPU/runtime gates still pending, not inference. Explicit native
  continuationv3 source2286fbd dispatched no active01GPU predecessor; oldfailed
  unit/readers untouched and original sourceoptimizer protocol unchanged.

- 2026-10-02 R102: nativeepisode15 actualfull501refinement **PASS258.872s**,
  300requestedsteps/301nativeupdates, forward/assets unchanged. Officialconverter
  and fullfinalschema501PASS afterwards; quality againstGT remains unverified.
  Episode0forward790conversion2mmFAIL remains unresolved; new sealednumerical
  diagnostic predeclares first/lower-median-error/worst probes, F64pose-only
  LM60/fd1e-6/tol1e-5 then exactF32replay gate2mm. No identityfit/adoption,
  raworiginalarchive/errorreceipt saved even onFAIL;40tinyNumPytestsPASS only.
- 2026-10-02 D77: VM02 import failure identifies OCI-index vs classicconfig-ID
  representation mismatch, not broken image contents. Bind exactsealed14.6GBTAR
  SHA and OCIgraph/index→amd64platform→config7ebfff18…c6d3, then fullorderedrootfs
  diff-IDs; no arbitraryimageIDwaiver/rebuild/reimport/oldunitrestart. Explicit
  continuation repeats extractedtaskassetinventory/SHA, realGPUtorchsmoke only
  aftergraphPASS. DA3CPUimport alias binds that newimportreceipt; no modelresult
  before gates. TinyownOCIgraph11testsPASS, noactualimportproofyet.

- 2026-10-02 R103: importv2 actualCUDA**PASS**, sealedTAR→OCIindex→AMD64platform
  →classicconfig and44rootfs identical. Native15converter**PASS132.557s**,
  independentmean.826410mm/worst1.787483397thframe, referencepointmax10.1643mm
  diagnosticnotgate; schema501PASS. Noheld-outquality/CARIvictoryclaim.
  Frozenreportsrefine22b3c20a…891c/conversion63d7d549…236b/schema18d0b18e…8d8f.
  TemporarySSHauthorization+VM01privatekey+newallowrule revoked afteralltransfers;
  onlynewdeny-ingressrule remains. DA3pairedchain9735385 dispatchedunderD76;
  readonlyaudit112testsPASS, fulltinysuite4148PASS/1optionalSKIP61.97s before
  diagnosticruntimeintegration, no numericalGPUprooffromtests.

- 2026-10-02 R104: DA3chainv1 **FAILCPUimport** beforeGPU/baselineextract/private
  truth: realCARIimage lacksaddict. No randomfallback/fakeDict/installallrequirements.
  Minimalrepair pinnedaddict2.4.0 actualPyPIwheel3832bytes SHA
  249bb56bbfd3cdc2a004ea0ff4c2b6ddc84d53bc2194761636eb314d5cfa5dfc,
  embeddedMITlicenseSHAca488d33c512d0b226142090af90e89ae266a901a293f89fd642dfec931e22c1,
  exact7fileZIP/noRequires-Dist. DirectAzureacquisition, read-onlyzipimportpath;
  unchangedbaseimage/DA3source, noresolver/install/rebuild. NewCPUgateoutputv2,
  oldv1reservedfailure retained; sameD76camera/seeds/depth/quality gates.
  Fulltiny4171PASS/1optionalSKIP60.53s beforeminimalwheelrepair;79focusedPASS.

- 2026-10-02 R105/D76: DA3Metric sameK800 **PASS9actualcalls11.290s**,
  strict406FP32states nativepreprocess392×518/canonicalfactoractual once;
  CPUpairedquality**PASS2.339s**. MoGefixedCDhalfcm[81.26989554,93.47578099,
  74.26773488] vsDA3[70.57089420,75.32652457,58.95176918], gains
  [13.1648%,19.4160%,20.6226%], median19.4160%, allcoveragePASS/no regressions.
  Supports a real-object depth improvement at the SAME prior camera, not
  correctcalibration/human/V2Dvictory. Errorsremainlarge andlearnedcameraMoGe
  previouslymuchbetterbuthumanregressed, so don'tgloballyreplacefrontend.
  Newpriority: independentlytestaffineZhuman-grounding andDA3human/newRGBcohort,
  notfitGTK800orclaimcalibrationfromthisobject-onlycohort. Onlyactualpinned
  addictMIT3832bytezipimportwasadded; baseimage/DA3sourceunchanged.

- 2026-10-02 D78 (predeclared, not executed): fresh J3 human/object RGB cohort,
  threeclips×sixframes, newclipconstantidentities/bottlemeshes/hiddenfocals and
  depth-arm/non-contact/occlusion controls; onlyautomaticperson+bottlemasks.
  ReusepinnedBody/MoGe fixedK1280 (notprivateK) eighteenactualcalls each, save
  rawZ/XYZ/predhumanrender/masks before any fit/privateevaluation. TestOWN
  camera-ray-preserving Z'=αclipZ+βframe; no framewise objectmesh scale,
  humangeometrychange/contactattraction/perframeGTalignment. IdentifyαONLY
  within-framehumanZvariation; checkerboard8×8train/holdoutregions separated,
  Huber10IRLS/equalframeweights, ≥64pairs/frame/≥32eachsplit, pooledwithin-frame
  SD/meanZ≥.005. Underconditionedabstainsretainα-onlybaseline; invalidinputfatal.
  Affine selected onlypublicpredhuman-heldoutmedianrelativeerrorgain≥5% with
  no>5%frameregression, otherwise α-only, never privateGTselection. Allframes
  finalprivatecameraCDhalf/relativehuman-objecterror/coverage reported;
  medianclipgain≥5%/no>5%clipregression/coverage≥95% gate, noCARIclaim. Newcohort
  ratherthanrepeatedhyperparametertuningoldJ1/J2; rawsource/masks/mesh fixed.
  LiteraturemotivationDo-as-I-Do2606.19333/MoGe22507.02546, independentformula
  avoidsHaWoR/MANO and off-rayXYZtranslations. GeoCalibsource-separation and
  discrete3meshSAMselection lowerprioritiespendinggaugevalidation/rights.

- 2026-10-02 R106: native episode0 refinement **PASS415.248s/790frames** but
  refined official conversion still **FAIL** unchanged2mm per-frame gate. No
  final schema/submission produced for0; do not weaken fidelity threshold.
  Next bounded sealed original+F64 pose-only diagnostic is numerical research,
  not GT fitting or submission adoption.
- 2026-10-02 D78 implementation audit: independent reviews identified missing
  copied-mask equality and coefficient-to-pointmap lineage proof. Public consumer
  now verifies exact PNG masks, native18focal-call telemetry, producer/helper/
  model pins and no prior fit; evaluator replays declared alpha/beta exactly on
  all18frames BEFORE first private read. Tiny analytic18-array firewall tests
  reject changed rays/Z/support/source and metadata without touching GT.
  Full suite4297PASS/1optionaltrimeshSKIP66.16s, not GPU/accuracy proof. Split
  immutable prepare(render/masks) and validation(infer/fit/quality) bundles to
  keep code-only control payloads under100KB; no larger local data exception.

- 2026-10-02 R107/D78 actual **REJECT**: freshJ3 publicBody18+MoGe18 calls
  PASS40.756s, publicaffinefitPASS15.268s, privatequalityPASS5.967s. All18frames
  and95%coveragegatePASS, but gains[+.91334%,-14.97561%,-6.60420%], median
  -6.60420%. Noadoption/nohyperparameter-retune onthiscohort; predictedhuman
  heldoutconsistency didnotgeneralizetoobjectquality atwrongfocal/depthcontext.
  Source2867ab1; predictionreceipt4fe3b535f7c3c83f4ede99096a6d9fbabe0934ef1aeb957925604d78c679c2ed,
  fitc1c69492b4c9cc572bca6714c1163023d30a5d4bfdba6c787a49426651854a68,
  qualityd8405043595a716efcf52c4d85e4cd60664cb8ea32e7e633b2e0fbd96e0302e0.
  Inference/fit/publicfreeze/privatefirewall areengineeringPASS, distinctfrom
  hypothesisFAIL. Preservefrozenresults; investigatebetterdepth/camera/human
  anchoring onnewvalidation, notreselectbetausingprivategroundtruth.

- 2026-10-02 R108: native refined0 sealedconverterdiagnostic **PASS311.155s**
  runtime/provenance, **fidelityFAIL**. ActualF64LM60 fd1e-6 withunchanged
  sharedidentity onprobes[0,535,1], then separateexactF32replays: beforemm
  [6.09732674,1.91778015,6.28012083]→[6.09670672,1.91576035,6.27935987].
  Worstnativeofficialframe1=6.28013468mm. Thisdoesnotrepair2mm; nofullpose-only
  rerun/adoption or gatewaiver. Fulloriginalerrors+parameters sealedbefore
  anygate; runtimeactual1LM+2replays, source/model/inputs unchanged.
  Diagnosticreceipt dbcb2177c20c213ee68cf026a89b8f493dc4851930907af0d6753992a5ccdce2.
  Nextcandidateonlyifjustified: predeclaredsmall constrainedsharedidentity
  inversefit plus reservedframefidelity, never averageperframeidentities or
  altertargetgeometry toclaimconversionPASS. GTaccuracyremainunverified.

- 2026-10-02 D79 (predeclared): refined0identityprobe aftersealedF64pose-only
  failure. Nativeidentitiesvary maxrange28scale3.1790/45shape6.1944, but variation
  alone isnot a geometriclowerbound. FitexactofficialF64LMjoint40 tol1e-5
  fd1e-6 allvertices/prior0 onuniformframes[0,197,394,592,789], ONE shared
 113identity (scales68+shape45). Reserveidentityframes[98,296,493,690]+original
  worst1; fitONLYtheir136poseLM60 with trainidentityfixed. Independentlyreplay
  original/proposedexactF32 controls/reference onall9/10probes, everymean≤2mm
  plusnoindividualregression>1e-4mm orREJECT/no790run. Reservedposesusetargets,
  so theseare identity-reserved representationprobes NOTheldoutaccuracy.
  Reusefrozenoriginalconverter/parameters; full790native targetregeneratedand
  exactcanonicalSHAchecked, NOcoldconverterduplication/GT/modeltargetwarp.
  Wholebudget900s/32GB/GPU01alone; sourceimage/codebound. Callbackjointchunk64/
  frames_per_chunk1, pose_batch4, replaychunk16 are disclosedmemorychoices.
  88focusedcontracttestsPASS, full4345PASS/1optionalSKIP65.16s BEFOREexecution;
  no submission/adoption evenifprobePASS, fulltrajectoryfidelitystillrequired.

- 2026-10-02 D79 actual **FAIL13.489s before solvers**: full790native target
  replay has different canonicalbytes from sealedoldSHA8856755a…25fdeb, despite
  identicalparams/assets/source/decoder/chunk16. Failure report
  05a558bc7dee69553d6b34ddc83ab9ff0f3361b7c648d99a75d9b942836602c3.
  Nojoint/pose/referencecalls, nohashwaiver/adoption. Theoldtargetarray wasnot
  saved, so noold/newnumericalgeometrycomparison is possible.
  Nextbounded standalone reseal: TWO freshstrict nativeworkers (seed0, TF32off,
  deterministicCUBLAS, unoptimizedTorchScript) must decodebitexact; savecanonical
  full790F32 .npy onAzure. Oneunchangedhistoricalfull790referenceF32replay must
  agreewitholdperframeerrors at existingrtol1e-5/atol1e-4mm BEFOREanyjointsolve.
  Exactofficial targetunitconversion follows pinnedsource (F64castbefore×1000).
  Newtargetrequires explicitnewlineage; oldFAIL/oldSHAremain. If either gatefails,
  STOP withoutcoldconverterretry or loosenedtolerance. Noaccuracy/submissionclaim.

- 2026-10-02 D80 actual strictreseal **REJECT37.154565s**: twofreshPython
  nativeworkers completedstrictseed0/TF32off/CUBLASdeterministic/nooptimizedJIT
  full790/chunk16, but rawtargets differ7182scalars (~.0164%), meanpointL2
  3.3861546e-8mm/max.00101896mm. Firsttargetcanonicaldf8cfc73aeea41e114825205bc822003d07b471ce976ee458ef12ed229f12397,
  second d17a842e61ae2634a95fab7b042b127fc076acffa948cdbec63ece2a8c3395b8.
  Failure receipt314f80e8bbb19e9883c75ba262c13417cd34754c0eba364516f1060073c9e825;
  code e2e446a35b2a6258ec52074cd9783964046b179c/63356Bcontrol.
  Nohistoryreference/solver; bothsavedtargets+reports retainedAzure. DoNOTrepeat
  samesettings orcallitbitexactPASS. CustomLBS/index_addis acandidateoperation,
  notcausallyestablished (PyTorch hasdeterministicalternatives).
  Firsttargetselectedbyprotocolorder BEFOREanyerrorselection maydefineaNEW
  frozeninverseproblem, neveroldtargetrecovery. Nextstandalonehistoricalcontrol
  reference usesunchangedrtol1e-5/atol1e-4mm, nooldSHAwaiver/noaccuracyadoption.

- 2026-10-02 D81 (predeclared): immutablefirstD80savedtarget (worker0byprotocol
  order, notlowesterror) definesnewinverseproblem. No redecodingorconversion;
  original controls/referenceF32exact source and targetF64mm arithmetic onceon
  all790. Historicalagreementrtol1e-5/atol1e-4mm unchanged;180s/offline budget.
  OldD79/D80FAILtargetSHA remain; strictreproducibility/historicaltargetrecovery
  False evenifthiswarmstartgatepasses. ActualD80maxperframemeandifference
  2.8553522e-7mm; all790poses/residuals/fitaccuracy stillseparatecontracts.
  141focusedtinytestsPASS, nosolver/adoption orquota.

- 2026-10-02 D82 (predeclared): independent GeoCalib frontend engineering
  gate on VM02, not a camera/accuracy experiment. Acquire only pinned Apache
  standalone MSCAN/Hamburger module and native four-class slice, retain source,
  licenses and CC-BY-4.0 publisher weight statement; public 116,074,121-byte
  checkpoint stays Azure. No PerspectiveFields/full package initializer/LM or
  challenge inputs. Release supplies no independent digest, record this limit.
  Strict checkpoint['model'] full parameter/BN-buffer schema, native zero-overlap
  second-component mapping only; no missing/unexpected-key filtering. Two new
  own procedural RGB arrays (320x416), native RGB normalization, seeded same-input
  replay must be byte-exact with finite, correctly-shaped/bounded native fields.
  Acquisition300s CPU-only; frontend180s H100/offline/32GB, immutable sourceimage.
  PASS only authorizes designing independent calibration validation, never
  camera correctness, general eligibility or submission adoption. 136 combined
  frontend/frozen-target/runtime-bundle tiny tests PASS before execution;
  independent source/API audit PASS, full4445PASS/1optionaltrimeshSKIP64.06s.

- 2026-10-02 D81 actual **PASS8.499485s**: one unchanged official-reference
  replay on all790 first-worker frozen targets, historical-error agreement at
  unchanged rtol1e-5/atol1e-4mm; maximum absolute drift4.2438507e-5mm.
  No native decoding/solver/cold conversion. Source e1e8e92a63c690174f5bc40733939235fe46d04a;
  receipt2c0269f5eb02f07e8d9791b059527059cb8d592b521a0abd9edf3cfae8432280,
  errors50b9595422ab2c596e317e6aa6996e79189939c742ea428b28c1df04a5851910,
  targetfilefcb86e264525ca7c6b93a6878061c3523a71af1385d5a05aef8bd017bc76f7d1.
  This validates a numerical warmstart for the NEW inverse target only;
  D79/D80FAIL and historical recovery/strict reproducibility False remain.
  Next bounded shared-identity probe keeps original indices, solvers and gates.

- 2026-10-02 D82 acquisition **PASS2.650749s**, VM02 CPU-only. All exact
  source bytes/licenses retained, disposable streams removed; weight SHA256
  86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac,
  receipt77ed2ed2e65f7011eb930425f37020f57f226c25d1bca5be381119bcbf79b83c.
  Publisher-independent weight digest/header remain unverified at acquisition.

- 2026-10-02 D82 nativefrontend **PASS2.863132s** on VM02 H100: strict889
  stateentries (748parameters/141buffers), no mapping/missing/unexpected keys;
  28,951,604 stateelements finite. Two seeded batched calls on new own RGB
  arrays bit-exact, native field shapes/bounds/unit checks PASS, peakCUDA
  684,662,784bytes. Receipt056f2d84882b301a4e8410e3672ffa5b5c646e5f3b030d9d1621f4b9a3f6eb30.
  Asset rehash PASS; no camera solver/calibration/GT or accuracy/adoption claim.
  Low procedural latitude confidence is not calibrated uncertainty and motivates
  explicit effective-support/conditioning gates, not a correctness claim.

- 2026-10-02 D83 (predeclared): constrained shared-identity probe using ONLY
  D81's exact first-worker frozen target and verified historical warmstart.
  Pinned D81 receipt/source/script/full790 error archive are revalidated before
  Torch; original error vector still selects worst1, never substitute new errors.
  Same D79 five-fit/four-reserved-plus-worst frames, joint40/F64 FD1e-6/prior0,
  reserved pose60, two independent F32 replays, every probe≤2mm and no individual
  regression>1e-4mm; total900s/32GB/offline on VM01. No new decode/cold conversion,
  old D79/D80 failures unmodified. Runtime completion and representation verdict
  recorded separately. Even PASS does not authorize fulltrajectory fidelity,
  strict determinism/historical recovery, GT accuracy or adoption. 203 focused
  tiny contract tests PASS before execution.

- 2026-10-02 full30 route audit (source-only): current generic episode chain
  is forward-only; final path must include native refine→refined conversion→
  refined schema and all30/16,563frames packing/official preflight. Keep whole
  episodes on one VM; per-VM one sequential GPU lane plus a bounded CPU native
  prepare worker can overlap without changing prediction algorithms. VM02 needs
  Azure→Azure SHA-verified immutable Track1/assets/images, not local transfers.
  Volume mesh wrapper is currently episode0-only; generic meshes still need
  topology/budget gates. No claimed30episode availability/accuracy or speedup;
  representation fidelity and source-license eligibility remain distinct gates.

- 2026-10-02 D84 camera-solver contracts: own SciPy pinhole/gravity math only,
  fixed geometric principal center and exact integer-grid resize/crop transport;
  no restricted GeoCalib/PF/LM optimizer imports. Three focal starts,100nfev,
  soft-L1 2degrees, confidence ESS≥256/cellESS≥8/mean≥1e-6 on train/holdout/two
  train subgrids; rank3/condition≤1e5, heldout up≤10°/latitude≤5°, focal within
  .15..4×image diagonal, half-grid focal disagreement≤10%. Otherwise explicit
  image-diagonal fallback and no estimated gravity. 35 own analytic contract
  tests PASS (including anisotropic resize/crop, unidentifiable fields and
  concentrated/infinitesimal confidence); not actual camera accuracy. Fresh
  room/MHR RGB cohort and later fresh complete HOI validation remain required.

- 2026-10-02 D83 actual runtime **PASS12.929974s**, representation **REJECT**.
  Exactly one joint40, one reserved-pose60 and two F32 replays, D81 bindings
  unchanged. Probe frames[0,197,394,592,789,98,296,493,690,1]; beforemm
  [6.09733,1.20092,1.31454,2.50279,1.65158,1.38154,2.09534,1.97096,1.61542,6.28012],
  after[5.62512,1.43520,.57670,2.94117,1.84514,1.32795,2.89512,2.47170,1.83885,5.81397].
  Five still exceed2mm and six regress beyond1e-4mm; shared-identity updateL2
  3.07041. Receipt8ddca5be3de229cd1364b97cb4aee07715e72c6275d128d6d6ac79ab361cd333.
  No full790 joint retry/tolerance change/adoption. Need a structurally faithful
  clipconstant identity source, not further retuning this failed inverse probe.
  Full tiny4518PASS/1optionaltrimeshSKIP63.48s at source37dbd16.

- 2026-10-02 D84 fresh camera-only RGB hypothesis (predeclared): six newly
  manufactured open-front textured rooms, MHR actors/bottles, three focals×two
  pitch/roll pairs; three weak uniform-background controls. Public nine RGBs only;
  camera/gravity calibration private until all native predictions are frozen.
  No claim of photorealism/independent real-world generalization. Exact audited
  GeoCalib frontend nine seeded calls + frozen own CPU field solver. Reproduce
  all public field-fit/camera results before private read. Strong acceptance≥5/6,
  median focal error≤5%/worst≤15%, gravity median≤5°/worst≤10°; weak acceptance0/3;
  all9 focal errors include explicit abstention1280 fallback and require≥5%
  median relative gain over fixed diagonal. PASS only supports this synthetic
  camera hypothesis, never full HOI or adoption. Render120s VM01; only the13
  exact tiny cohort/semantic files transfer Azure→Azure, SHA/exclusive/private
  permissions checked, no models/challenge/predictions. Infer180s H100 VM02,
  private quality60s CPU/offline. No image-ID substitution: renderer OCI index
  and frontend config-ID are bound to their own separate stage receipts.
  Full4633PASS/1optionaltrimeshSKIP67.93s before five extra transfer source/closure
  tests (22transferPASS); private archive0600 at creation, stdlib-only transfer,
  current renderer/semantic source and model hashes checked before extraction.

- 2026-10-02 D84 render **PASS5.268301s** on VM01: exactly9 new RGBs and
  two official MHR forwards. Reportbaf515907f21001516979a52ef8d6694037b45f9228f74cabea1ceb9e9bcabfe,
  publicmanifest4faefffb9bd416b5255437e27a05a805a6e90cf82bbb6dcaf30c571a37b83370,
  privatecalibration94feac21eef88413222c61e09eaba5458cd728f0d174915abc33423ad127deb1.
  Exportv1 failed before archive creation: root-owned transfer parent0700 denied
  UID1000 traversal despite cohort0700 ownership1000. No copy/inference/private
  evaluation occurred. Preserve failedv1 log/emptyreceipt; explicit new exportv2
  uses a new directory and traverse-only parent0711, private children unchanged.
  Import checks UID1000 readability before writes. Temporary Azure-only SSH key,
  source-restricted NSG allow and parent permissions are revoked/restored after
  transfer; no local data/model transit and no implicit retry/overwrite.

- 2026-10-02 D85 (predeclared): NEW native frame0 identity by protocol order,
  shape45/scalePCA28 fixed across12 rounded linspace indices
  [0,72,143,215,287,359,430,502,574,646,717,789], original poses/root/hands
  unchanged. Twelve original checkpoint direct-head calls return204 controls;
  verify expanded68 scales bit-identical, freeze targetF32 and controls read-only
  BEFORE one native12batch and one pinned official-reference12batch replay.
  Native maxpoint≤.01mm; official every-frame mean≤2mm, maxpoint diagnostic only.
  Strict seed0/F32/noTF32/noJIToptimized/offline180s/32GB on VM01. No LM, cold
  conversion, oldtarget read, error-selected identity, alignment or GT. D79/D80
  failure receipts preserved. PASS authorizes only this new12frame ABI fidelity,
  not full790, identity accuracy, historical recovery, determinism or adoption.
  Independent source/unit audit READY;23focused and63independent tiny tests PASS.
  Final source suite4662PASS/1optionaltrimeshSKIP63.54s. Both Azure transfer
  parents' existing direct children independently checked0700 before use0711;
  no unrelated readable transfer data is exposed.

- D84 runtime control optimization: audited single fail-fast CPUimport→H100infer→
  CPUprivatequality wrapper; each child's original isolated mounts and namespace
  remain unchanged. SHA validated before any work, no extra predictions/calls;
  105 associated tests PASS, independent2wrappertests PASS. This avoids serial
  control-plane wait overhead, not an asserted inference speedup.

- 2026-10-02 D84 actual: Azure-only665,600byte cohort transfer/import13files
  **PASS**, SHA55a21e5046a969fefedf0adebedc51a0de8e19414e727f5028c587ac9040a198;
  importreceipt190490176a15d8a10886eb767d9bf6c274c8a4af3cf59ed44500d19c923e2e13.
  Nine frontend calls **PASS4.324657s**, peakCUDA414,652,416bytes, receipt
  bb3211c94ea7931676299fd2a42ab11ab72cbe0774bee4618b1ee55a687ec270.
  Exact public CPU replay/privatequality **PASS.250362s**, receipt
  07d83f65a881209bb4cb70ee29b7ea7b5213c264b2cf85d634275dd65f31943e;
  camera hypothesis **REJECT**: all9 abstain from insufficient effective support,
  strong0/6 and weak0/3, median focal error25% unchanged, gain0. No threshold
  retuning/adoption. Strong train upESS~39–52/latitude~60–92 in shown cases,
  below256; means~.025/.002 do not establish reliable spatial evidence.
  Source audit confirms RGB/BGR/strict889 state; our declared Torch antialias
  differs from official Kornia Gaussian-preblur resize. Thus frontend weights/
  architecture are native, preprocessing is NOT pixel-identical; causal link
  to abstention unproven. Preserve rejected cohort and use a fresh validation
  with audited native preprocessing only if that new hypothesis merits priority.
  Temporary SSH sourcekey removed, destination authorizedline removed preserving
  unrelated keys, NSG allow deleted (deny4096 remains), both parents restored700.
  Verified destination transientTAR removed; originalRGB/truth/fields/receipts kept.
  Final source suite4664PASS/1optionaltrimeshSKIP65.16s at0c0e547.

- 2026-10-02 D85 actual **PASS14.369049s** on VM01, source0c0e547db14c6f82717754c2146cbc19adbd3ddf,
  receipt d22d052a8a631dda57e3b60a0790d4eaf963f808496a806326e3a3e68a14ec67.
  Exactly12direct-head+1native12batch+1official12batch; native maxpoint
  .001930491mm and official worstframe mean.000214798mm/maxpoint.001583769mm.
  All bound gates PASS, frame0expanded68 bitconstant, artifacts frozen before
  independent replays. This resolves a NEW sharedidentity ABI structurally;
  it does not validate identity accuracy or restore the historical target.
  Next standalone790 ABI pass and fresh15RGB paired identity-quality test use
  separate new namespaces; no new inverse LM/adoption or GT-derived identity.

- 2026-10-02 D84 cleanup completed: source665,600byte transientTAR also removed
  after independent completed13fileimport/quality/SSHrevocation proof and exact
  SHA/byte check; total1,331,200 transientbytes removed across Azure. All frozen
  RGB/truth/fields/rig/source receipts preserved; no data downloaded to Mac.

- D86 (predeclared after D85PASS): full790 new frame0identity direct controls,
  original522 native source blocks/poses untouched. Fiftychunk16+tail6 direct
  head calls, freeze ENTIRE F32target+pose136/scale68/shape45 before fifty native
  and fifty official-reference calls. All790 native maxpoint≤.01mm and official
  every-frame mean≤2mm; fixed300s/32GB/offline VM01. SourceD85 receipt/script/
  settings/model identity checked beforeTorch; preserve D79/D80 failures.
  Independent source audit36tests PASS; no inversefit/GT/accuracy/adoption claim.

- D87 (predeclared): three fresh own MHR identities×five moving/bottle-occlusion
  RGBs, shape45/all68scale controls verified against actual249 getter bounds.
  Manufacture varied private focals1160/1480/1720, never expose them to inference.
  Render120s then automatic person./bottle. masks180s, separate offline mounts.
  Native Body15/MoGe15 at fixed RGB-sizeK1280; freeze all raw blocks/observations
  before frame0sharedshape45/scale28 branch, original pose/root/hands unchanged.
  Freeze both15predictions before one official-reference15batch fidelity replay;
  each≤2mm, no solver. GPU600s/32GB then CPUprivatequality120s/8GB. All15 cases
  scored rawcamera PVE without GTalignment, with same visible-object-depth proxy
  and one baseline-human-derived sharedalpha perclip. Require all5frames human/
  object support; no fill/drop/GTcontact/privateobjectmesh inference. Human median
  pairedclip improvement≥5%, no human clipregression>5%; EACH hand's clipmean
  relative-visible-object vector error increase≤1e-4cm (numeric allowance), not
  an average that hides one hand. Weak/missing/zero-baseline human gains reject.
  These proxy/identity metrics do not verify rigid object/contact/penetration,
  full temporal HOI, photorealism or real-domain victory. No adoption from ABI
  alone; reject candidate if quality gates fail, no same-cohort thresholdretune.

- D86/D87 source freeze: final full tiny suite **4818PASS/1optionaltrimeshSKIP
  64.14s**, all three shell entrypoints syntax PASS and whitespace audit PASS.
  Independent native204/rootflip/reference and renderer/mask audits READY.
  Public producer freezes all30 raw/paired artifacts before reference/private
  evaluation; original model/source/semantic/LBS bindings rechecked CPU-only.
  Quality uses own minimal equivalent pixel-center unprojection and fixedseed
  8192-point sampling (data-free parity PASS), not an unused prior evaluator
  dependency. Small immutable source closures only; no images/models reach Mac.

- 2026-10-02 D86 actual **PASS18.744309s** on VM01, source
  f82594ac1f7aa651fee2b7bc0a8de931735e49fc, report
  13d29416518cea15f932ea4f3508774d663a038966a86ccdc5e43018fec09308.
  All790 frames: 50direct+50native+50official chunk16/tail6 calls. Frozen full
  target/controls before replay; native worstmean.001026345mm/maxpoint
  .002883442mm, official worstmean.000254210mm/maxpoint.001831117mm.
  Unchanged.01mm native-point/2mm official-mean gates PASS, source bindings
  checked, unit completed0. This establishes full NEW sharedidentity ABI
  fidelity only; old failures remain, no accuracy/historical recovery/adoption.
  Exact committed source payloads: D86 69,912B/18files, D87prepare54,000B/15,
  D87validation95,740B/23, all beneath100KB. D87 preparation dispatched only
  after D86 completion; no simultaneous VM01 GPU or heavy local transit.

- D87preparev1 engineering **FAIL2.503965s** BEFORE any forward/RGB/mask: all68
  scale scalar pattern violates seven native getter-locked zero controls.
  Receipt e69236dd756883e80a730198073e4d9403ba3c30e9976cee944a5df347de6daf,
  sourcef82594a; actualreferencecalls0. Preserve failedv1; no accuracy result
  or threshold tuning. Independent bounded CPU model-getter diagnostic confirms
  eyes136–138, hipheight/depth147–148, kneeknock151 and ankleheight152 locked0;
  remaining native scale bounds symmetric with minimum+.1/-.1, shape204/205±10.
  New engineeringv2 namespace: locked[0,0]→exactzero, every remaining scale
  same predeclared(-.04,.03,.07); reject other illegal bounds, retain full249
  animated/neutral guards. No clipping, name-selected exception, pose/shape/K/
  scene/gate/budget change or post-render repair. No challenge/validation truth
  consulted for this model ABI fix. Inference/mask/privatequality namespacev2
  consistent; producing revisions/helper source and hashes bound independently.
  Independent v2 audit125focusedPASS, finalfull4825PASS/1optionaltrimeshSKIP
  64.88s, shellsyntax PASS. Producer receipt records all3x68 actual vectors and
  locked native indices before first forward; privatequality recreates exactrig.

- D87preparev2 actual **PASS** source0236b0bd99d4c15054e7bdd88e8839cd0591e02a:
  render7.127171s/two official15moving+3neutralcalls, all15RGB distinct, receipt
  bf21e238e23a386334d0cfb48ae606635625d38db4c2ea54ee7349dde5b2af62.
  Public manifest2c584ea633a958c737520d53c68c12b1429b8358f182c07bf46e53627b8f8267.
  Automaticmasks16.283815s:30DINO+15SAMencodes+30SAMpredictions, all30immutable
  masks, receiptaa1c8346cfd7609a58d055f71762060aca238c216100bd8aa990ca1de79ea909.
  Model-locked scales zero exactly; full249legal, no synthetic private fields
  visible to automaticmasks. Source-only prepare54,432B/15files, validation
  96,216B/23files; no heavy local transit. Native paired inference dispatched
  AFTER both terminalPASS; accuracy still unmeasured.

- D87validatev2 engineering **FAIL.008237s** before model/GPU calls: consumer
  required truncated{sha,bytes} asset records but original automaticmask producer
  correctly returns{path,sha,bytes}. Failedreceipt
  177c40d1bd11b00daf8146632f2fff1f2fa2f7b52bff58985d6b4322896dd27a;
  Body/MoGe/shared/referencecalls all0, no private read. Fix exact canonical
  path+all9pinnedSHA/bytes asset inventory (not ignoreextras/dropintegrity),
  producer-shaped tiny fixture prevents recurrence. Preserve original rendered
  RGB/masks and failedpredictions_v1; newpredictions_v2/quality_v2 exclusively.
  No new synthesis or mask inference required, no measured accuracy/gate change.
  Independent133focusedPASS, shellsyntax PASS, finalfull4833PASS/1optional
  trimeshSKIP65.00s. Exactproducer-shaped receipt contract audited independently.

- D87validatev3 actual runtime **PASS**, hypothesis **REJECT**, source
  c1b75f9b5e8d494aff4f6fec985d51ce8b21098e, VM01. Publicinfer45.303485s,
  15Body+15MoGe+15shared-head+1official15batch; all30predictions immutable
  before reference/private scoring. Worstofficialmean.000306584mm; receipt
  f01674dd2e196b16a7ec0657dbf033414424a324dc6abd5ada62817fc2188df4.
  CPUquality5.945505s all15retained, noGTalignment; receipt
  6d598bc7e6f4d0063189f652bef6241ec91a0caa4be281a6c2cc74cee8ee3883.
  Human relative gains[.0001895584,.0006064028,.0000133744], median
  **.01895584%** <5%; human nonregression PASS, perhand nonregression FAIL
  (clip0hand increases[.00578453,.13495487]cm >.0001cm allowance).
  First-frame identity quality unsupported. No same-cohort retune/medoidretry
  or adoption; structuralD85/D86passes do not alter this accuracy decision.
  D87reuse for NEWdepth hypothesis is no longer independent/preregistered
  after this privatepeek: require a freshD88cohort before DA3vsMoGe private
  comparison. Prioritize depth (independentD76signal19.416% atsameK800) over
  speculative identitymedoid; that external result is not fullHOI superiority.
  Alljobs terminal0/no concurrentGPU; tiny source payload96,336B/23files.

- D87 scale-of-error diagnostic (after frozen quality decision, no retune): raw
  human cameraPVE clips[96.397263,90.780927,182.034193]cm; sharedidentity gains
  only[.018273,.055050,.002435]cm. Baselinehuman-derived objectdepth α
  [1.082365,1.058942,.960445] all5frames supported. Thus this synthetic test
  does not establish native human camera/global accuracy; camera-prior/model
  domain bias are plausible, not proved causes. Do not let tiny identity ABI
  residuals conceal large reconstruction error or claim improved fullHOI.

- D88 predeclared before newRGB/predictions/private evaluation: dense metric
  camera-Z comparison MoGe2 vs DA3METRIC-LARGE, **not** human identity/HOI.
  Three fresh own identities×five new poses/bottle occlusions, privatefocals
  [1200,1500,1800], shape2[(-.21,-.09),(.26,.11),(.43,-.12)], legalfree scales
  [-.03,.025,.06] with model-locked0; separatedepth_rgb_v1 namespace.
  Protocol helper frozen; oldD87/code/receipts remain unchanged. Fixedpublic
  K1280 bothbackends,15calls each, noBody/masks/shared/shape fitting.
  NativeDA3 processed392×518 withf647.5/653.333, metricfactor(actualfmean/300)
  exactlyonce thenbilinearZ; MoGe originalnativevalidity retained. Allpublic
  source/model/arrays hashed/frozen BEFORE private CPU scorer. FullGT-visible
  object support>64everyframe, EVERYnativevalidpositivepixel required: missing
  support yields None+explicitinvalidcounts+wholecomparisonREJECT, never
  intersectiondrop/penalty/clipping. Primary uncapped meanabsrelativeZ over
  allfixedprivatevisibleobjectpixels, equal5frame/clip means; medianpaired
  clipgain≥5%, no clipregression>5%, all15complete, zero-perfectbaseline
  relativegainundefinedreject. DiagnosticonlyhumanZ (surfaceCD deferred), noalignment,
  alpha/beta, GTintrinsics fitting, interaction/contact/rigidmesh/realwin claims.
  Stop if weak support/geometryinvalid/no measuredgain; no samecohortretune.
  AllstagesVM01 serial, nativeDA3publicpinned1,336,734,448B model acquisition
  reusesexistingremote-onlyverifiedsource/model/wheel driver; no localweights
  norSSH/privatecopy needed. Image remains actualoriginalINDEXb47; no rebuild
  or config-ID waiver. Render120s, optionalCPUimportgate90s, eachRGBbackend180s
  thenprivatequality120s, source-only closures<100KB beforedispatch.
  D88 sourcefreeze independent139focusedPASS, newprimary/privatequality66PASS,
  finalfull4972PASS/1optionaltrimeshSKIP66.06s; fiveentrypointbashsyntax PASS.
  Removed irrelevantBody dependencies from NEWproducer only with standalone
  MoGefocal-support helper: callable AST exactly originaljointproducer and
  sameupstreamgeometrySHA; thirteen support/restore/ABI tests PASS. OldD87/
  modeldrivers unchanged. Initial101.4KBsourceclosure too large rejected before
  dispatch; newclosure~79.7KB, no100KBbudget relaxation or data inclusion.

- D88assets+prepare actual **PASS**, VM01 sourced691578c89db3f218afd4b1293f1e8f6957ca18b:
  pinnedDA3acquisition30.956223s, original1,336,734,448B/model/sourceverified,
  receipt68fccf6a856d1809c146f75fcab1fb58b2fb07dbca024ba3b5f9d0bf82636a9b.
  Addict3832Bunchangedzip/noinstall, receipt
  b3c7cc468641137485d6ca3a99a1665444a56a630b5d87eb16303d93465cd6b9.
  Fresh15RGB render7.099678s/twoofficialcalls, receipt
  929cd1ad072324f67ab85cc8015f737a96959810523a65c5cc1c3b76c7d3a166;
  publicmanifest0329fa8616e6a19b2bc3f945b097720eacc93384e4ae3987ae72cfa7c12ed2b4.
  SeparateCPU/networkacquisition overlaps GPUrender only, notbackendGPU.
  NoheavyMac/SSH transfer, sourcepayloadassets12,632B/7files, prepare48,168B/14,
  validation80,112B/24. Bothterminal0 before serialMoGe→DA3→privateCPUdispatch.
  All15newtruths remainprivate; no depthaccuracyresult yet.

- D88 actual runtime **PASS**, general depth replacement **REJECT**,
  sourced691578c89db3f218afd4b1293f1e8f6957ca18b onVM01. MoGe15calls17.974782s,
  receipt40aa813e60f5e517172d907bed01b9e01b25cf3c77eec865f1679f7389e23ee5;
  DA315calls17.353809s, receipt
  75286c4aab27347d17fb1b9b3f556307f220625c9d49f19149a95f50840c8dce.
  All30immutableoutputs +sources/assets checkedbeforeprivate; noα/fitting.
  CPUquality6.235516s, all15objectsupport100%complete, receipt
  718c10517fbba50a7af1c8b065382443c112079a21e4521b8560dd667a36ab8c.
  ClipmeanabsrelativeZ MoGe[.0364503,.1058318,.2477178],
  DA3[.1794567,.1213736,.2824282]; gains[-3.9233216,-.1468538,-.1401205],
  median **−14.68538%**, all3regress (worst392.332%). Primarygain and
  nonregression gates FAIL, coverage PASS. KeepMoGe baseline; D76TUD-Lgain
  19.416% is domain-specific evidence, not permission for generalreplacement.
  No gate adjustment, depthrescaling, GTcamera adaptation, candidate adoption
  or human/HOI superiority claim. Freshcohort+allsource receipts preserved;
  no networkauth/NSG modifications or heavy local transit, unitterminal0.

- D89 is a **posthoc CPU diagnostic**, not a new quality experiment: preserve
  D87 rejection and hard-pinned original quality/prediction receipts, replay
  historical metrics/decision unchanged in memory after full public SHA audit,
  then decompose all18439human correspondences on all15frames in float64.
  Report camera PVE, centroidXYZ/norm, cameraRMS and centeredRMS with exact
  RMS²=centroidnorm²+centeredRMS². No alignment, human scale/K fitting,
  prediction correction/export, new selection or adoption. One global initial
  Sim3 in the official judge means D87 absolute cameraPVE is **not Track1 CD-H**;
  this diagnostic can suggest global versus articulated error, not establish
  cause or a score. Reuse frozen assets RO on VM01, separate new diagnostic
  receipt RW, CPU-only120s; no RGB/models/data downloaded locally.

  D89v1 engineeringFAIL5.881702s, source d2fc62852e9a481ca8abd618db30cee8c2a9d355,
  receipt4c77b1653f9cf2bb0feaa43130541be263ce19b369d4186bd021f745ce09e20b:
  actual depthsupport dataclass returns tuples, JSON historical receipt lists.
  Originaldecision/frame/clip/hash comparisons passed; tuple/list structural
  mismatch stopped before the new decomposition. Fix by exact JSON-roundtrip
  comparison, no floating tolerance or historical rewrite. Real depth-support
  tuple regression test also rejects numerical1e−12change; preservefailedv1,
  use exclusive diagnostic_centroid_v2. No model runs or prediction changes.

  D89v2 actualPASS11.376439s/source15382d53a28476aee72c3f93a8609ccb0b876469,
  receiptddaf0a690c94197f33c9b1148308709bf3cf6d39c463928070210f1856642fdf.
  Original D87 decision/metrics/SHA replay exact in JSON value domain; all15
  frozen frames unchanged. Raw cameraPVE[96.397263,90.780927,182.034193]cm;
  meanframecenteredRMS[4.948759,5.279616,5.982611]cm. Meanframecentroid-squared
  fractions[.997212944,.996489146,.998887970]: global centroid error dominates
  **this synthetic raw-camera measure**, not a demonstrated Track1 failure or
  proof of constant gauge over time. Sharedidentity centeredRMS
  [4.973102,5.303134,5.969669]cm; originalrejection stands. No Sim3, corrections,
  causalcamera claim or handdominance assumption. CPU-only/modelcalls0,
  bundle24files98,636B; no heavy local transfer. Finalsource suite5012PASS/
  1optionaltrimeshSKIP71.91s, diagnostics40PASS including all15NPZ integrity.

- D90 public-only identity-consensus gate predeclared before run: ep000000
  existing790Body predictions,12rounded uniform temporal anchors,11disjoint
  integer midpoint frames. Decode12identities in the SAME native zero-pose,
  zero-translation frame; choose the geometric all-vertex RMS medoid, retain
  the exact raw shape45/PCA28 of that single anchor. No coefficient averaging
  or selection by silhouette. Hash-freeze identity before all paired renders;
  native raw redecoding must reproduce existing geometry/controls≤1e−5.
  Raw/candidate keep root/body/hands/camera/translation frozen, candidate only
  identity changes. Same raster/grid and automatic masks, comparison region
  is predetermined complement of automatic object mask for both modes and
  human target; every11midpoint has positive supportedhuman pixels, no missing
  frames or candidate-dependent support. Gates eachmidpointΔIoU≥−1e−4 and
  meanmidpointΔIoU≥0, finitegeometry/fullcoverage; anchors diagnostic only.
  Failure stops this policy without trying other medoids on the same outputs.
  Passing permits a structural pipeline pilot BEFORE neutralheight/cache,
  not accuracy adoption, fullHOI validation or verified CARI4D victory.

  D90 actualruntimePASS20.187367s, publicconsensus **REJECT**, source
  c72ea35b0746ff03864ea40cf037ea0d23b27e34. Geometricmedoidanchor72 frozen
  before image scoring, identityreceipt
  6da447c5be53b44f5dd685ac1439edc72cfd64eb44fa42a1f75dcc9de6fb5bba;
  report059d775819caaac11a5ed9944496d2840d8b1cb32b5264dee8cab798e25e586b.
  Rawreplayroot8.302e−7m/camera1.139e−6m/controls1.49e−8, allparityPASS;
  3nativeheads/46samegridrasters, all23frame/mask/sourcechecksPASS.
  Eleven midpointmeanΔIoU−.010760618 (−1.07606percentagepoints), worst
  −.024725697 (−2.47257points), eight regressions. Bothpredeclared mean/perframe
  gatesFAIL. No same-output anchor policy retune, constrainedinitializer/cache/
  CoCoNet followup, adoption or submission. Stable identity with fixed original
  pose/camera is not supported by this consistency test; coupled pose/camera/
  identity estimation is a distinct future hypothesis requiring its own protocol.
  Fullsource suite5084PASS/1optionaltrimeshSKIP73.02s, 72newfocusedPASS;
  immutablebundle8files34,284B, noheavydata onMac or activeGPUafter terminal0.

- D91 next capability gate, **not a candidate-quality experiment**: verify a
  differentiable PyTorch3D0.7.9 soft silhouette in the same pinned Azure image
  and OpenCV camera convention as the existing hard renderer. Own closed cube,
  manufactured translated reference, no video/models/challenge/private labels.
  Fixed256×192 grid, scaled K, sigma/gamma1e−4 and faces_per_pixel8; preserve
  all vertices/faces, metric extent and positive depth. Check hard/soft camera
  projection parity, finite nonzero translationXYZ gradient and a single fixed
  small negative-gradient step decreasing the SAME loss. No tuning on failure,
  alignment, morphology shrink or predicted interaction exported. This only
  enables future shared-identity/pose/camera fitting; it proves no accuracy.
  Offline Azure GPU180s maximum, scalar/source receipt only on the Mac.

  D91actualFAIL2.171658s/source35ce1a6c39fe8655a0a4ca46d406b7435d49f82e,
  receipt1ba7780830bcc96269c615e1b5dd03e20e1e0dddc0f4dc2a6ade5ed7cb6995a3.
  HardmaskbyteparityPASS, projectionmax3.231e−6px and Kscaleerror0. NativeCUDA
  raster backward explicitly lacks a deterministic implementation and rejects
  torch.use_deterministic_algorithms(True): 4rasters, completedbackwards0,
  translationsteps0. Strictautogradcapability remainsFAIL, no globaldeterminism
  waiver or completedgradient claim. New numerical-reproducibility or alternate
  derivative contract would require separate preregistration/namespace; not
  silently rerun this failedv1 with settings changed. No challenge data/models.
  Fullsource suite5120PASS/1optionaltrimeshSKIP73.63s, 36probeCPUtestsPASS;
  actualimagePython3.11.10/Torch2.5.1+cu124. ORTabsent verifiedCPUimportgate,
  so futureDWPose requires pinned isolated runtime acquisition, not an assumed
  installedGPUprovider. No model checkpoint/keypoint inference acquired yet.

- D92 separately predeclared **numerical**, not bit-deterministic, CUDA
  backward capability: immutableD91FAIL/source/receipt retained. Two fresh
  Python processes, same fixture/seed/K/shader/target, each exactly5rasters/
  1backward/1fixed0.5mmtranslation. Temporarily permit the known atomic-add
  kernel only inside each backward, synchronize then restore deterministic
  enabled/warn-only states in finally; all forward settings remain strict.
  Inputs/source/target/lossbefore identical, each finite nonzeroXYZ gradient
  individually produces strict same-loss decrease and preserves metricextent.
  Per-component symmetricgradient agreement rtol1e−5/atol1e−7; lossafter
  agreement rtol1e−5/atol1e−7. No averaging, branch selection, retries or later
  tolerance relaxation. Two independentworkerreceipts + newparent manifest,
  all in exclusiveAzure namespace, total180s (not180perworker). PASS means
  numericalreproducibility on this H100 fixture only, never exactgradient,
  geometricidentifiability/HOIquality/adoption. No video/models/GT acquisition.

  D92actualPASS6.177863s/sourcecf27c8343b047c34cb19a9faeda4637a8ea0f751,
  parentreceipta1de8261ed2e2898c141b985f3b52656c7411036af0e15141fe5e51a19429dec.
  Freshworkers2.285162s/2.163099s, receipts
  3414a1d8e28e09a90d46bb2c041dea441fb6966d748e947027c3bb4a6c4364be /
  bb93d418f9dacc275893bc5659f96879691724917b0409b5b98c085e73f51408.
  Both strictforwardlossbefore .0011723724892362952, afteronefixed0.5mmstep
  .0010569767327979207 (samebits), gradientsXYZ approximately
  [-.154606506,.177536920,.006768564], maxabsolutereplaydifference
  9.313226e−10 <predeclaredbound. Both5rasters/1backward/1step; originalflags
  andbackwardhookrestored, metricextentretained. NumericalcapabilityPASS;
  bitdeterminismFALSE, originalD91FAILpreserved. No reconstruction/GT/quality
  adoption. Fullsource suite5173PASS/1optionaltrimeshSKIP73.46s,53newfocused
  PASS, source-onlybundle8files23,776B, terminal0/noGPUjob active.

- D93 preregistered **actual public-video coupled XY consistency**, distinct
  from rejected D90 fixed-pose identity substitution. Episode15 has501frames;
  twelve rint-uniform anchors and eleven disjoint floor-midpoints, sorted23.
  M0=firstRGB identity; M1=full-neutral-native-vertex RMSmedoid among actual
  anchors, first-protocol tie, no silhouette choice/averaging. Both decoded
  geometries/masks/identities frozen BEFORE optimization. All original pose,
  rootrotation, hands, expression, K=RGBdiagonal1920, Z and object fixed. Each
  branch learns ONLY per-frame metricXY: w=0, delta=.05*tanh(w/.05),30Adam
  steps lr=.005m, orderM0thenM1, no best-checkpoint selection or relaunch.
  This changes projected/physical human-object XY relation; no contact claim.

  Soft256x192 edge-coordinateK/6, sigma/gamma1e-4,FPP8 asD92. Human6x6
  blocks require>=19/36pixels; exclude a block if ANY automaticobjectpixel.
  Fixed8x8low/48x48full checkerboard even=train/odd=heldout, same exclusions
  for both branches. Equal-frame train MSEalpha plus .001*mean((delta/.02m)^2);
  no reservedpixel enters loss. Every23frame needs>32observedhumanpixels on
  both full train/heldout and positive lowtrain support; no frame omission.
  Only each native backward permits CUDAatomic-add, synchronized and original
  deterministic flags/hook restored immediately asD92; all forward strict.
  Numeric/bit gradient reproducibility on this video is NOT verified.

  Freeze BOTH finalparameters before full-resolution hard scoring: raw23,
  M0initial23,M1initial23,M0final23,M1final23=115hardrasters; initial/final
  baselines diagnostic, no selection. Gates coverage/finite/±.05mXYbounds,
  each branch finaltrainobjective<=initial; paired finalheldout M1minusM0
  mean>=.005, median>=.01, worst>=-.01 across all23. Spatial checkerboard
  is correlated, not independent 3D or generalization validation. PASSonly
  authorizes a new hypothesis to pursue, no adoption/export/fullHOI/score.
  FAIL preserves output and stops this policy without gate/medoid retuning.
  Azure-only offline600s total,32GiB peak allocated/reserved GPU limit and
  immutable own outputs/episode_000015/identity_xy_fit_v1. Historical0644
  Body/masks remain unchanged read-only; exact original receipts, assets
  and current audited manifest/source establish provenance, without inventing
  dataset-revision/forward-basis fields omitted in legacy receipts.

- D94 preregistered **minimal independent2D acquisition only** while D93 uses
  theGPU: no image inference, install, fitting, provider fallback or new global
  environment. Azure-only acquire exact pinned DWPose low-level133joint ONNX,
  standalone source/Apache notice, publishercard; ONNXRuntime1.30.0CPU wheel
  and Flatbuffers25.12.19wheel plus primary notices at cutoff-safe revisions.
  Validate SHA/bytes, reject invalid ZIP paths/symlinks, audit wheelMETADATA
  dependencies and retain embedded third-party textual notices. BoundedHTTPS
  streaming, no credentials, exclusive own namespace, no mutable resolver or
  heavy Mac transit. Failure leaves immutablereceipt and stops, no same-path
  overwrite/retry. PASS=source/model/wheel integrity only: training/teacher/
  COCO/UBody rights/overlap, actualONNX channelgraph/runtimeABI, CPU cost and
  2D accuracy remain unverified. Future isolated offline installation and
  actual133-output image smoke need their own gates, not implied by acquisition.

  D93 actualruntimePASS42.067650s/sourceb0850dc4da2d403477ddbdb7f5b67df7006b534e,
  receipt`d0c5efa69988a87cd148a3abbe7f5d9542aed6b107de6c122da8de76326ae6bb`.
  Medoidanchor136, actual4nativehead/64soft/60backward/115hardcalls; rawparity
  root7.7804e-7m/camera1.0341e-6m/controls2.9802e-8, integrity/flagsPASS.
  PeakGPUallocated1,134,425,088B/reserved1,237,319,680B wellinside32GiB.
  M0train .011020621→.010981362;M1 .010601408→.010547599, bothnonincreasing.
  Max|XY| M0 .01707443m/M1 .01611209m. Pairedfinalheldout M1minusM0 mean
  .025384998, median .019786718, worst **-.012062284** below frozen-.01: 
  hypothesis **REJECT**, no sameepisode threshold/medoid/optimizer retune.
  Full23meanheldout IoU raw .85693245,M0initial .83704839,M1initial .85333777,
  M0final .82377277,M1final .84915777. Fitafterminusbefore M0mean-.01327562,
  worst-.04438353,22/23regress;M1mean-.00418001,worst-.03544947,15/23regress.
  Thus the small softtrain decrease does NOT demonstrate hardheldout gain;
  M1betterthanM0 is not sufficient for adoption, 3D/HOI accuracy or victory.
  No causal attribution to blur/camera/optimizer without a separate experiment.
  Identityreceipt`8e18a269ddf62990762cbbdc808bebbdc8b3f3afb7f482098ab14da067eecffe`,
  inputgeometry/masks`1c807e9380b7cfcf6044c88d92a2eff5e39d9cdeabec8f6555be97f594dffd8a`,
  finalXY`0ec010e51f363e901596540f04d6a88a15ea7d37f26e074ba489d0749b601df5`
  all immutableAzure-only, no export/adoption. Sourcebundle11files49,792B;
  suite5231PASS/1optionaltrimeshSKIP71.84s,183independentfocusedPASS,
  terminal0/GPUinactive confirmed. Next independentRGB2D observations need
  actualruntime/ABI gates before another coupled identity/pose hypothesis.

  D94 sourcefreeze gates:62focusedCPUfixturesPASS (synthetic HTTP/ZIP, no
  networkmodelcalls), full5293PASS/1optionaltrimeshSKIP72.34s andbashsyntaxPASS.
  Exactly9assets158,360,974B planned: ONNX134,399,116B, CPUORT23,561,046B,
  flatbuffers26,661B, standalone11,608B, publisher28B, andprimarylicenses/
  ORTThirdPartyNotices338,088B; heavy bytes stayAzure. Own source closure
  remains below100KB, credential-freeenv-i hostPython acquisition only.

  D94actualengineeringFAIL2.513977s/sourceb067acf9dbb8e463884dc58461d6f3dd03e77975,
  receipt`9cff44a6fa41d7f9c54dac0c212b74057d4852686276e9cc740298e6b6bc3f82`.
  All9files exact158,360,974B downloaded/published, failure during wheel audit
  before any returnedwheel_audits, source execution/install/inference/GPU0.
  OriginalFAILreceipt/assets preserved immutable; no download retry/overwrite.
  Metadata-only Azure ZIP inspection now diagnoses which strict metadata/tag
  contract differs; do not claim successful runtime, licences or source clear.
  Any repair requires a new read-only consumer/audit namespace, not modifying
  original acquisition or downloading identical158MB again. Terminalfailed1/
  noactiveGPU confirmed; no heavy Mac transit.

  ActualZIPdiagnosis: ORT353members, version/tags/dependencies all exact,
  embeddedLICENSE/ThirdPartyNotices present. Flatbuffers14members, exact
  version25.12.19/tags py2-none-any+py3-none-any/noRequiresDist but **noembedded
  licence**: our blanket embedded-licence guard correctly failed its contract.
  This is a packaging mismatch, not proof of missing Apache rights; matching
  pinned upstreamLICENSE was already acquired separately. PreserveD94FAIL.

- D94v2 preregistered read-only **separate wheel audit**, no downloads/install:
  bind originalfailedreceipt/exact9assets and oldsource; audit both safeZIPs/
  exactMETADATA/WHEEL/deps. Require ORTembeddedLICENSE+notices; for ONLY exact
  Flatbuffers25.12.19 wheelSHA/version allow absence of embeddedlicence and
  retain separately pinned matching-source ApacheLICENSE with explicit
  `embedded_license_present=False`, not inventing one or relabellingwheel.
  Otherpaths/types/CRC/size/version/tag/dependency guards unchanged; new
  immutable results/dwpose-wheel-audit-v2 namespace, oldassets/receipt RO,
  own retainedtexts+primarylicence bindings rehashed, AzureCPU/offline120s.
  PASSonlyintegrity/noticeretention, no runtime/eligibility/trainingclearance.

  D94v2actualengineeringFAIL .004679s/sourcee6237186b80ac1eefceceb5843f84aa5869cf2b8,
  receipt`de8486168bbbe7bb8e27fb5be44bd3d6bb61e1f02093bb33a107b4eff91ea21e`.
  OriginalD94 `wheel_audits` assignment failed while building its list, so the
  key is **omitted**, not an explicitly emptylist. Diagnostic printing used
  default[] and our consumer incorrectly required[]; stoppedbeforeassetread/
  wheel audit/install/inference. Preservefailedv2. Separatev3namespace must
  require REALkeyomission under originalreceiptSHA, not synthesize empty
  evidence; same exactall9asset/source/license/ZIPguards and120scontract.
  Add regression fixture matching realomission and reject invented emptylist.
  No metadata, licence exception, numericalgate or sourcepin relaxation.

  D94v3 actualPASS .230968s/sourceb04cfe24e11b5469ca7d12a2747ed6c51643044e,
  receipt`e7fa6fce0397654ec5d1d2c07c49bd6f2a50655cda185d4297bfb8f0f4aae3e2`.
  OriginalD94 omission validated exactly; both oldFAILreceipts/all9assets/
  originalsource rehashedbefore+after. ORT353members embeddedlicenceTRUE;
  Flatbuffers14members embeddedFALSE, externalmatchingprimaryTRUE; safeZIP/
  exactMETADATA/tags/dependencies/retainedtextCRC+SHA PASS. No redownload,
  upstreamexecution/install/inference/GPU or runtime/eligibility/accuracyclaim.
  Sourcebundle6files14,108B, full5326PASS/1optionaltrimeshSKIP73.19s,
  independent95focusedPASS. Failedv1/v2evidence remains intact. Terminal0/
  GPUinactive; all158MBheavyassetsAzure-only. Next: isolatedCPU runtimeABI
  and two automatic-mask-cropped publicRGBs, not another policy-fit retune.

- D95 preregistered **actual independentRGB2D runtime/ABI**: two original
  syntheticpublicimages clip_00_frame_000/clip_01_frame_000, automaticSAM2
  humanmask bbox minxy/max+1, exactD87publicmanifest/maskreceipts. Oldcohort
  already privatelyevaluated: this is not fresh quality or D87retuning.
  No privategeometry/depth/K/challenge/modelassets exposed. Require D94v3
  receipt e7fa6fce0397654ec5d1d2c07c49bd6f2a50655cda185d4297bfb8f0f4aae3e2
  and all9pinnedassets/notices intact. Installonly pinnedCPUORT1.30.0 and
  Flatbuffers25.12.19 wheels into isolatedtemporary/tmp prefix, offline
  pip--no-index--no-deps--target, no resolver/globalenvironment modification.
  Runtimeversions/importorigins/dependencies/providers verified explicitly.

  Use unmodified low-level onnxpose.py pinned16fb69ab...291a2, RGBuint8
  and one automaticallyderivedbox perimage, no fullimagefallback/ControlNet/
  syntheticneck. Nativepreprocess float64normalized listNCHW feed retained
  unchanged on FIRSTtest; record actual suppliedtype/dtype/shape andORT
  acceptance, never inventing nativefloat32cast or a channelgraph proof.
  TwofreshCPUsessions, intra4/inter1/sequential/CPUExecutionProvideronly,
  eachtwoimages=4nativecalls. Actualmodel inputoneRGBcrop[1,3,384,288],
  twoSimCC[1,133,576]and[1,133,768], allfinite native133keypoints/scores,
  rawscoresunclamped, invalidityfromscore<=0 (sentinel transformed toimage).
  Compare perimage nativeoutputs/rawSimCChashes byte-identical betweensessions;
  preserve all133outputs, no confidence/quality thresholds or sample dropping.
  Success only means ABI/CPUreplay, not meaningful2Daccuracy/generalization.

  Failfast180s INCLUDINGisolatedinstall, CPU4/8GiB, Azureoffline/noGPU,
  exclusive validation/dwpose_smoke_v1. Freezeinputs/source priorrun and
  rehashafter, temporaryprefixremoved, no heavyMac transit. NativefeedFAIL
  stops unchanged; any explicitcast requires a separatelydeclaredcontract,
  not silent same-job fallback. No3Dfit/export/accuracy/adoption/eligibility
  clearance claim; source/training/teacher/overlap rights remain unverified.

  D95sourcefreeze5407PASS/1optionaltrimeshSKIP73.07s,81newfocused and176
  combinedfrontend testsPASS, independentreview/no-blocker. Actualcode
  sourceee91a530398f2f8bc0aced813f94717c114f5ee7, bundle7files22,840B;
  runtime driver never casts nativefloat64list, checksactualprovider/options,
  all4predictionNPZ hashes/inventory and temporaryprefix cleanup evenfailure.
  FakeORT fixture covers actualdriverflow only, not realcheckpoint evidence.
  AzureCPUoffline dispatch started; no parallelGPU or heavyMac transfer.

  D95v1 actualengineeringFAIL1.116425s/sourceee91a530398f2f8bc0aced813f94717c114f5ee7,
  receipt`f95a4dbdb03bd02e9bd65a216bec233de2cb469bf78e71dd4162568bae41ca41`.
  Exactisolated wheels/imports succeeded; graphmetadata guard failed before
  any inference/prediction, temporaryprefixremovedTRUE, GPU0. Preservev1.
  SeparateAzureCPU metadata-only query of exact724f4ff2...1843 checkpoint:
  input `input: tensor(float) [batch,3,384,288]`; outputs `simcc_x:
  tensor(float) [batch,MatMulsimcc_x_dim_1,MatMulsimcc_x_dim_2]` and `simcc_y:
  tensor(float) [batch,MatMulsimcc_y_dim_1,MatMulsimcc_y_dim_2]`; custommetadata
  empty, CPUprovideronly, isolatedprefixremoved. Export metadata is symbolic;
  it does not prove actual output dimensions or successful nativefeed.

- D95v2 preregistered structuralmetadata repair ONLY: new immutable
  validation/dwpose_smoke_v2, bind preservedv1 FAILreceipt/SHA/source read-only.
  Accept only exact observed symbolic input/output names/types/dimensiontokens
  for the already pinned checkpoint, persist actual metadata before validating
  so early failures remain diagnostic. Keep ACTUALfloat32 SimCC arrays exactly
  [1,133,576]/[1,133,768], finite133nativecoordinates/scores, unchangedfloat64
  nativefeed, fourcalls/twofreshCPUsessions/byte-replay, exactsame twoRGBinputs,
  licenses/assets/source rehash and180sCPUbudget. No cast/channel/preprocessing
  repair, confidence/quality threshold or semantic/3Daccuracy claim. No same-job
  fallback; failedv1 remains unchanged, installprefix always disposable.

  D95v2sourcefreeze5428PASS/1optionaltrimeshSKIP72.53s,197independentfocused
  testsPASS andbashsyntax/diffcheckPASS. Actualsource9b0a4e19ac8a13b50417089e351913b2aacf0892,
  bundle7files23,328B. Exactsymbolicmetadata means no arbitrary wildcard;
  rawmetadata savedbeforeguard; actualnumeric133contracts unchanged. AzureCPU
  dispatch started withv1failure mountedRO, no GPUjobs/heavyMac transfer.

  D95v2actualPASS1.708719s/source9b0a4e19ac8a13b50417089e351913b2aacf0892,
  receipt`d96eb5c8c4030bf2e16924093aef03a9636f12cc2f3d3a8b7475731fe49d9a79`,
  scriptSHA`ea0beb43dd698261a5b2accdbd0432dcc8de82b56958b0e93cec8062089ce72c`.
  ActualfourCPUcalls .061496/.060066/.060202/.059888s; nativefloat64CHWlist
  accepted unchanged byORT1.30.0 with graphfloattensor contract, but internal
  conversion mechanism NOTobserved/proven. ActualSimCCfloat32 dimensions133
  anddecodedfloat64coordinates/rawfloat32scores/validity byte-identical across
  twofreshsessions. Bothimages133positive scores, not semanticdetectionproof.
  PerimageNPZSHA`f29821b51a120ccb1b8171cf8792c2aac885be155dd598332eb181fdccb7cb33`
  and`e102d74ae5181cc825e5618afe950f87dbf256545f2436789a84e9e9097f8b29`,
  identicalsessionreplay; all4artifacts/source/assets/notices/publicinputs/
  priorv1FAILreceipt rehashed. Temporaryinstall removed, globalimageunchanged,
  terminal0/MainPID0/GPU0%105MiBdriveronly. ABI/replayonly, no quality/3Dfit/
  trainingrights/eligibility/adoption claim and no heavyMac transit.

- D96 preregistered **fresh causal human-root refit in full human/object
  synthetic scenes**, not photorealistic/fullHOI-method/Track1 validation.
  New validation/keypoint_rgb_v1, schema world-reward-keypoint-rgb-v1, original
  1024x768,3clips×5frames, bothentitiesvisible≥64pixels. Generate new fixed
  scenes BEFORE seeing outputs; no renderer retry, threshold/recipe retune or
  image/frame selection. Publicmanifest contains onlyschema/images/file/SHA/
  width/height. Manufacturinggeometry/depth/camera/controls remaineval_private.
  TrueK1280 onallclips coincides with prechosen genericRGBdiagonalprior1280
  in BOTH inferencebranches; no privatecalibration transmitted. This isolates
  a known-focal synthetic scenario and does not establish real-camera quality.

  Frozen generation: shape45firsttwo ((-.33,.09),(.21,-.18),(.38,.14)), others0;
  scale68freecontrols(-.025,.045,.060), actualzero-lockedcontrols0, strictnative
  bounds/no clipping. Interactionside r,l,r respectively. Forframef0..4:
  activeuparm_ry=.21+.040f, elbow_bend=.30+.050f, wrist_ry=-.06+.025f,
  oppositeuparm_ry=.018(f-2), activeindex/middle/ring/pinky1_rz=.14+.025f;
  otherarticulations/rootcontrols0. ActorcameraYyaw(-.12,.19,-.21)+.035(f-2),
  actorXYZincrement(.012(f-2),.007sin(.8f),.035(f-2)); neutralcombined3identity
  bboxframing uses1280 and29%grid margin plus.4m, one clipconstantdistance.
  Bottleownclosedmesh×1.08, rotationvector(.03(f-2),.07f,-.04(f-2)); position
  activehandbboxcenter + (.040*(-1forl/1forr),.018,zoffset[f]),
  zoffset=(-.09,-.035,.025,.07,.115). SkinRGB(.73,.69,.65), garmentband.23..81
  exclhandregions with(.27+.03sin(13x),.33+.02cos(11y),.39+.025sin(8z));
  bottle(.19,.38,.43),cap(.25,.27,.29); existingdata-independentroombackground
  unchanged, no semanticperclip colorcodes. 120sCUDA/32GiB/4CPU render.
  Then separateRGB-only Grounding/SAM2automaticperson./bottle. masks, same
  existing thresholds/NMS/ambiguity policy,30detector/30SAM/15encoders/all15
  frames, nonemptyoriginalgridmask no fallback,180sCUDA. Source/models/image
  pinned/rehashed, no privateassets exposed to mask/inference containers.

  After successfulD95ABI/preparation, one fixed baseline: SAMBody15 +MoGe15
  withgenericK1280 andautomaticmasks, nativeMHR firstRGBshape45/scale28 shared
  unchanged inbothbranches. Freezeallbaseline/identity/raw133DWPose15 and a
  common visibleobjectpoint proxy scaled frombaselinehuman BEFORE refit.
  FitCOCO[5,6,7,8,11,12,13,14,15,16]→nativeMHR[5..14]; reserveface/wrists
  [0,1,2,3,4,9,10] solelydiagnostic. Rawpositivefinite scores→equalbinaryweights;
  ≥6train/frame andfull15coverage required; degenerategeometry/Jacobian fails
  entiretrial ratherthan droppingframes. No semanticconfidence qualityclaim.
  Trainonly6nativeparameters: three nativeEulerZYXrootcontrols (no assumed
  externalSE3 conversion) andcameraXYZtranslation. Nativeglobal_trans0, meters
  YZfliponce+translationonce. Body/hand/expression, K/shape/scales and common
  objectproxy fixed. No silhouette/depth residual or optimizerGTinitialization.
  Proposed60evaluations/frame/180sfit, robust5pxHuber+priorsσXY=.15m,
  σlogZ=.15/σrotation=.15rad, boundsXY±.30m/logZ±ln1.25/rotationincrementnorm≤.30rad.
  Exact optimizer/Jacobian numerical guards must be frozen before any fresh
  inference/refit; prepare-only run cannot imply these unimplemented gatesPASS.
  Finalall15candidates frozen BEFORE separateprivate evaluation: medianperclip
  cameraabsolutePVE gain≥5%, noclipregression>5%, bothperhand/clip relative
  objectvectorerrorregression≤5% (fixedcommonobjectproxy), zero-baselinecase
  explicit; reportcentroidZ/centeredPVE/heldout2D without Procrustes/alignment.
  Rejectrecipe onanygate; no samecohortretune/realadoption, objectrigidmesh/
  contact/penetration accuracy or challengewin claim.

  D96prepare sourcefreeze5475PASS/1optionaltrimeshSKIP72.88s,47own/161combined
  focusedPASS and47independentreviewPASS, bashsyntax/diffcheckPASS. Source
  42f457fc46fbeb814befb48b8104403b06922f63, bundle16files58,056B; existing
  primitiveforeground_truth already gatesbothentities≥64, no duplicatechanged
  algorithm. ActualAzure prepare-only dispatch120srender+180smasks started;
  no privateevaluation, Body/DWPose inference/refit or qualitydecision yet.

  D96prepareactualPASS source42f457fc46fbeb814befb48b8104403b06922f63:
  render7.189764s/twonativeMHRcalls/all15freshRGB+privatecases, receipt
  `2407c53871f4b7f8e16d1e3420416a65b93bffcb0ea22cafb14b963f4206e158`;
  automaticmasks16.371879s/30detector30SAM15encoders/all15nonemptyperson+bottle,
  receipt`d9cae7c4c7131b599ebcafc58a415b9f763286806f80d4a6a46ee114249d8568`.
  Manifest2199B/SHA`082549b5a1f8a4a7d687bc17ec6847f3628d6d4230186053951caa2454d2979d`,
  nativecontrols/fullprivate-publicinventory/model/sourceintegrityPASS.
  Humanmaskareas27,989..31,597pixels/object2,114..2,937; these are predicted
  support, not accuracy. OldD95v1 preserved; no privatequality evaluation,
  nativehumanDWPose/reconstruction/refit yet. Terminal0/MainPID0/GPUidle93MiB,
  no heavyMac transit. Torchvisionread-onlyNumPywarning observed; no evidence
  bytesmutated (posthashPASS), avoidcausalclaim or samecohortmodelretune.

  D96 next **freeze public observations/baseline ONLY**: consume exactfresh
  manifest+maskreceipt above, all15orderedRGB/masks, no privatefolder mounted.
  Produce15SAMBody and15MoGe predictions underfixedgenericK1280, then15native
  sharedfirstRGBidentity decodes. Retain raw blocks/geometry/depth/points,
  clipconstantshape45/PCA28/expanded68, native308keypoints/127joints/full18439
  vertices andhandregions. UseexactexistingnativeMHR/source/assets, YZflip
  once+cameraTonce; verify sharedgeometryagainst officialreference≤2mmmean
  all15. Rawpredictions freeze BEFORE identitydecode, allshared freeze BEFORE
  independentreference verification; no fit/evaluation. 600sCUDA32GiB4CPU.
  Separatelyfreeze15nativeDWPose133RGBcroppredictions withautomaticmaskbbox,
  sameactualD95v2CPU1.30.0/Flatbuffers/nativefloat64-list/SHA/exportsignature
  contracts; onefreshCPUsession15calls/180s, rawcoords/scores/validity/SimCC
  hashes retained. BindactualD95v2PASSreceipt/source+oldFAILread-only, retain
  all15cases/no thresholding or confidence-basedsample selection. Prefix
  removed, allinputs/source/assets/outputs rehashed. No freshprivategeometry,
  fitoptimizer/contact or qualitygate yet; abstentionnevermasqueradesasPASS.

  D96 optimizercontract frozen BEFORE freshpublicBody/DWPose observations:
  for eachframe fixedbody/hand/firstRGBidentity nativeMHR10trainingkeypoints,
  exactly60evaluatedstates/59Adamupdates lr.01/betas(.9,.999)/eps1e-8, select
  LOWESTtotalobjective EVALUATEDstate (firsttie), neverunevaluatedlaststep.
  Sixdimensionlesslatentzeros: Δxy=.30tanh(u[:2]); ΔlogZ=ln1.25tanh(u[2]);
  Tz=Tz0exp(ΔlogZ); ΔnativeEuler_i=(.30/sqrt3)tanh(u_i), i3..5, giving
  norm≤.30rad andfinitegradientat0, fixedaxis-boxsubsetofrotationball.
  This source-only simplification is BEFORE anyfreshpublicBody/DWPose output,
  avoidsradial0/0 and is not observation-drivenretuning. Allnativeglobal_trans0.
  DataL=mean_valid h(||reproject−DWPosepixel||2/5px), h(q)=.5q² ifq≤1 elseq−.5;
  priorL=.5[(Δx/.15m)²+(Δy/.15m)²+(ΔlogZ/.15)²+||ΔEuler/.15rad||²], coefficient1.
  Equalbinaryfinitepositive observations; no prior multiplied bylandmarkcount,
  no learningrate/weights/camera/shape adaptation or silhouette/depthloss.
  Require≥6validtrainingpoints/frame with observedxyextent≥32px EACHaxis;
  initialunregularized2DresidualJacobian fromsameinitialnativeforward wrt6
  latentcoords mustfinite/fullrank withsmin/smax≥1e-5. Priorrows never enter
  ranktest; otherwisefailentiretrial/allcases retained, no bestframepicking.
  Initialnativegeometry/keypoints parity againstfrozenbaseline≤1e-5m, allfinite
  positivecameraZ/no clipping, nativecontrols/shape/scales/boundschecked.
  TrainstrictCUDAfloat32/noTF32/autogradno contextdetach or suppressedkernel
  errors; anyruntimefailure stops beforequality, not silent relaxation.
  Finalbesttotalobjective≤initial+1e-6/frame, onefinalnativeexport/frame;
  exactly900objectiveforwards+15finalheads, initialJacobian20orlessresidual
  reversegradrows reusessameinitialforward, 180sCUDAfit/32GiB4CPU inclmodel
  load/integrity. Freezeall15candidateblocks/controls/V/KP/J/rotations+source
  beforeprivate; numericalreplay/qualitydecision separate, not predictedwin.

  Data-freepolicy implemented aa57daf20testsPASS, independentmathreviewPASS
  after two concrete fixes: rejectMaskedArray targets BEFORE np.asarray,
  andfiniteoutput arithmetic/physicalbound guards preventoverflow fromfinite
  extremes. No clipping/loss repair. NativeTorchautograd/Jacobian/runtime not
  yetverified; policytests do not establish3D/HOI gain or privacy-stage completion.

  Publicproducer sourcefreeze5572PASS/1optionaltrimeshSKIP72.05s,37baseline/
  40DWPose/20puremath focusedPASS plusindependentreview/no sourceblocker.
  Bashsyntax/diffcheckPASS; actual sourcef5c78f7abfec3c6234bbbee7f7dad70736836ae7.
  Baselineclosure22files86,428B actualdispatch, DWPose9files~27KBplanned,
  allnewownfiles/oldhelpers untouched. Rawkeypoint308+nativeSO3rotations
  additionallyretained;45explicitheadcalls15rawparity+15rawkeypoint+15shared
  with15Body/15MoGe/1officialreference. Azure baseline-only started600s;
  no privateevaluation/fitting or quality/adoption claim.

  D96actualPUBLIC baselinePASS44.827999s/sourcef5c78f7abfec3c6234bbbee7f7dad70736836ae7,
  receipt`8ebfce153ea5ff708578c155d60adb97e737b672e2c2df56df38c635fbbc80c3`,
  scriptSHA`4950979a0dfbeb6759f0a7149a9d79ca61fdb13c96cf09c5349fca1f6f73459b`.
  Actual15Body/15MoGe/45explicitnativehead/1officialreference, all15raw+15
  shared artifacts frozen inorderedstages; native308keypoints/127joints/proper
  rotations/full18439V andshape45/PCA28/68byteconst verified. All15official
  meanfidelity .000146754..000252711mm wellbelow2mm; not reconstructionerror.
  Source/assets/publicbytes rechecked; private_truth_read/ground_truth_used/
  accuracy_verifiedFALSE. No failure or fallback.
  D96actualPUBLIC DWPosePASS2.855874s/source776e479c0a3492078313b93b8bf92e7443b274cf,
  receipt`949bdf218514335cedc2745b081f2a79f93d4372bc66a0894a1bda7749e29c49`,
  scriptSHA`f6fbf3e578e2a3c603a9644f341401576cc6360694ec52cf909907ba00771121`.
  ExactlyoneCPUfreshsession15nativeunmodifiedcalls/all15NPZ,133positivescores
  eachframe (not correctness), actualSimCC133finite/nativecoords/scores/validity
  integrityPASS. D95v2PASS/oldv1FAIL/allassets/sources/publicoutputs rehashed,
  temporaryprefixremoved, no image/globalenvironment change/GT/fit. Actual
  DWPosebundle9files27,368B. Bothterminal0/MainPID0/GPU0%87MiB; no heavyMac
  transfer. Baseline+independentobservations NOWfrozen; freshprivatequality
  remains UNREAD. Next run ONLYalready-frozen sixparamroot policy, not retune
  optimizer after viewingobservations or declare these ABIchecks a3Dgain.

  D96finalfit prereg implementation details BEFORE fitting/privatescore:
  commonobjectproxy uses15 SHAREDfirstRGBidentity baseline human hardrasters
  atK1280, not originalperframeidentity geometry. Visiblehuman=sharedsilhouette
  &automaticperson &~automaticobject &MoGevalidity. Exactlyoneequal-frame
  robustpositive depthscale perclip/all5frames,≥32validcorrespondences/frame,
  all5supported (no optionalframe rejection). Fixedautomaticobject&MoGevalid
  points×thatclipscale,≥32points/frame, max8192 uniformlysampledseed0/sorted
  indices inoriginalpixelorder; serializeall15proxyarrays/bindings read-only
  BEFORE firstoptimizerforward. No proxychange/rerender/rescale/recenter afterfit.
  Exactly15baselineproxyhardrasters plus915nativefitheads; 180stotal budget
  includinginput/modelintegrity/load/proxy/Jacobian/fit/finalfreeze, no retries.
  Entireinput/producer/NPZ/SHA/identity/frame/source audit beforefirstforward
  andafterfinal15exports. Newroot_fit_v1 namespace; no privatefolder mounted.
  SeparateCPU120squalityreadsprivate ONLY afterfitcomplete/frozenaudit, compares
  sharedbaselinevsrefit (originalraw additionallydiagnostic). PerframePVEmean
  vertices→equal5framemean/clip, median3clipgain≥5%, noclipregression>5%;
  perhandrelativevector usesPREDproxy median minuspredLBS handregionmean vs
  TRUEvisibleobjectmedian minustruecorrespondinghandregionmean, normcm,
  equal5framemean/clip,≤5%regression foreachhand+clip, zero-baseline exactrule.
  NativewristsnotavailableinrenderGT: do NOTinventgroundtruth308/127landmarks;
  handregionvertexPVE maybediagnostic, labelledasregionnotjointaccuracy.
  Qualityprotocolsuccess distinctfromhypothesisacceptance; anygateFAILrejects
  recipe withoutsamecohortretune/realadoption. No alignment/contact/Track1win.

  D96 fit/quality implementation freeze: public-only native CUDA fitter and
  separate CPU quality driver share a lightweight artifact consumer. Historical
  producer sources are mounted at their canonical immutable paths, read-only
  for hashing only (not imported or executed), avoiding unnecessary GPU/renderer
  imports and nonexistent child mounts inside the read-only CPU source bundle.
  Recheck all nine DWPose assets, notices, historical source texts and capability
  receipts, Body/MoGe assets and all original public arrays before/after fitting.
  Native249 bounds and rootEuler/control correspondence are checked at every
  head forward; the frozen consumer also checks the exported correspondence.
  The quality firewall verifies all15 evaluated schedules, initial native parity,
  observation-only singular values/rank and actual training validity/indices
  before its first private read. True visible-object median uses ALL foreground
  object pixels at pixel centers (+.5); the common predicted proxy remains the
  independently frozen, bounded8192-point sample. Reserved face/wrist RGB errors
  and hand-region vertex PVE are diagnostics, never unavailable GT joint labels.
  Independent reviews found and fixed integration/source-closure and evidence
  guards before execution; optimizer, data, time limits and quality gates unchanged.
  Final independent source audit PASS;121 combined policy/fit/quality and64
  bundling tests PASS. Actual native autograd,180s budget and held-out quality
  are not established by these lightweight tests; no GPU run or GT quality read
  has happened at this source freeze.
  Full source suite5673PASS/1optionaltrimeshSKIP71.10s; bashsyntax/diffcheck
  PASS. Clear disposable local bytecode/test caches; models/data remain Azure.

  D96actual strict fit **FAIL**27.683201s, source52f35c0b0df9a1448adf16c1cff796007fb41198,
  script`d5146d1f89a0fbbd0e4614abd8896a9e13679bd4307535a3cea0b2f46a84583b`,
  receipt`aacb82e7c48b5b28ec26f1eb3bb88194c76fbb3c0ce07cc54d6032adb25412d2`.
  Actual Git closures fit25files98,016B and quality25files99,952B encoded.
  All15 common proxies frozen BEFORE the first optimizer head; native249 bounds
  or rootEuler correspondence guard failed on clip_00_frame_000 at latent zero.
  One native head returned before the guard, but counters increment AFTER it:
  recorded0 objective heads means0 fully validated objective forwards, NOT zero
  model calls. No Jacobian, loss evaluation, Adam update or final candidate;
  no private truth read, no quality namespace/run, no silent bounds clipping,
  bypass or repeat. TerminalExecMainStatus1/MainPID0/GPUidle81MiB.
  This is an execution/initial-feasibility failure, not a scored3D rejection.
  Preserve the failed report/proxies; a NEW read-only CPU metadata audit of all15
  frozen raw/shared249 controls against exact native bounds will distinguish
  scale/shape/articulation limits from root-control mapping, without model forward,
  fitting, repair or private geometry. Only after that evidence define a future
  fresh protocol; do not relax the already-failed D96 hypothesis on this cohort.
  Post-failure read-only receipt check: baseline and DWPose SHA unchanged,
  quality_v1 absent, all15 proxy artifacts retained, empty candidates. Three
  frozen shared consistency scales1.0195142344/1.0058856855/.9935995941 are
  diagnostic estimates, not ground-truth metric gains. First frame had10 valid
  training observations; failure preceded their Jacobian/loss evaluation.
  Bounds audit source frozen separately: CPU60s/8GiB/4threads, exact reference
  asset696,110,248B and49 immutable baseline/failure/proxy/source/manifest files;
  only `get_parameter_limits/get_parameter_names`, zero reference/model forward,
  zero optimizer or private read. All15 raw/shared control blocks retained and
  compared against actual249 limits; violations are recorded, not repaired.
  28 own/112 combined policy/bundling tests PASS; five-file source closure~9.5KB,
  no imports of unused Body/MoGe/DWPose/fit/evaluation drivers. Diagnostic execution
  PASS is distinct from prediction feasibility or reconstruction quality.
  Full suite5701PASS/1optionaltrimeshSKIP74.41s and independent audit PASS;
  bashsyntax/diffcheck PASS before the separate CPU metadata dispatch.
  ActualCPU auditv1 **FAIL**.114352s/source d232abb7bc306860bdda254d82a768a257ba2fc0,
  script`a6b0542eea2d7b8d717ef9fe4eb0f5610f37a5d0c81c36d45c958476d204ebd8`,
  receipt`1a566a9c1e8ac31cce6654372b79ee2dad00fb47a92d24134cafa915c886244f`.
  All49 baseline/failure/proxy/source inputs passed, but reference-file immutable
  host-mode check failed before Torch/model metadata load. A Docker read-only
  bind does not erase host write bits: preserve this packaging failure unchanged.
  Explicit newv2 engineering-only namespace will require exact assetSHA/size and
  canonical regular path plus actual `/proc/self/mountinfo` read-only-file-mount
  proof for this reference ONLY; public predictions/sources/reports still require
  no write bits. Record asset mode and mount evidence, do not chmod/change asset,
  prediction, physiological bounds, D96 fit policy or private-read status.
  V2 source audit43tests/127combined PASS; exact VFS file mount must be uniquely
  read-only (a read-only parent alone is insufficient; underlying superblock
  may remain rw). Recheck same mount and exact model bytes after metadata read;
  both previous failures and all public artifacts retain strict immutable modes.
  Independent review and Bash syntax PASS; no prediction/model forward introduced.
  V2 full suite5716PASS/1optionaltrimeshSKIP74.00s before immutable CPU dispatch.
  Actual bounds auditv2 **PASS**2.361279s, source3cdef2da0c2e935080410562c826752a1833fcef,
  script`8fce37c4d9aed75328e302f54c7f092e532ab2253d47c7db4c50eecba7130003`,
  receipt`8e5e725f244a5a122fa5a154ce0ce49334619494d1f5df9a45c58e6b7bdbd21b`.
  Five-file actual dispatch10,560B encoded; CPU getters only,0model forwards/
  optimizer updates/private reads, all originals rehashed. Exact model VFS mount
  ro/relatime, host write bits128 (owner write), backing superblock rw: explicit
  read-only bind proof succeeded without modifying original model.
  All15 raw and15 shared cases each have36 exact249 bound violations:
  32pose+4scale perframe,0root/0shape; rootEuler byte correspondence correct
  onall15 inbothbranches. Perbranch540 violations total,510 on metadata[0,0]
  controls,30 on finite one-sided clavicle bounds; maximum excess.409344733.
  Examples flexible spine/foot/body-length controls, clavicles, hip-height/depth,
  knee-knock and ankle-height scale controls; some last-scale excesses are tiny
  numerical residuals, others are substantial, so not a blanket float tolerance.
  The root-only bounded-D96 trial was infeasible at the frozen initial remainder;
  it did not test 2D→3D quality or independent depth improvement. Raw/shared
  sameness of these counts is not shape/interaction correctness. Clarify native
  [0,0]/soft-limit semantics before proposing a new method; no clamp or waiver of
  the failed contract, no samecohort optimization. OriginalfitFAIL and auditv1FAIL
  SHA preserved, quality_v1 absent; terminal0/MainPID0/GPUidle81MiB.
  Important semantic boundary: these are exact violations of the dense returned
  bounds under the preregistered D96 interpretation. Public exporter/getter source
  has not established whether every[0,0] means a genuine equality constraint or
  a sentinel for an absent sparse limit. Do not call all SAM predictions physically
  invalid from this audit. Next cheap decisive check is CPU-only JIT getter code/
  graph and minmax metadata introspection, still without forward/GT. No new
  fitting protocol is frozen until this distinction is resolved.

  Separate getter/state inspection source frozen BEFORE execution: CPU60s/
  8GiB/4threads, exact native reference SHA/size and uniquely read-only VFS
  file mount; only the unchanged bounds-audit-v2 PASS receipt is an input.
  Preserve actual TorchScript `get_parameter_limits` code/graph, all249 names/
  dense limits and bounded limit/minmax buffer metadata. Repeat snapshots and
  rehash model/source/prior receipt; zero model forwards, optimizer or private
  geometry reads. New `results/mhr-limits-semantics-v1` namespace, five-file
  closure7,680B encoded;23 dedicated/66 combined tests and independent audit
  PASS. Getter delegation/cached tensors alone may leave zero-zero semantics
  unresolved; inspection PASS does not authorize bounds changes or adoption.

  Actual isolated getter/state inspection **PASS**1.994215s, source
  f4e8a0073f8722991c7f51cf69a60a9994db9180, script
  `82ce61931d781d5d54bcc4da116ad7090f579b1135cc6195dc3003d942ce2d93`,
  receipt`c78b5ec438f1ff91685bd0cf53b76658d718504e4ac25e36c26597ef3d7c713a`
  (44,313B/0444). Actual dispatch5files7,864B; full5739PASS/1optionaltrimesh
  SKIP75.82s before execution. Getter code/graph returns `self.parameter_limits`.
  All37 dense[0,0] columns occur in the actual198 sparse minmax parameter-index
  entries; all198 sparse bounds exactly match their dense returned rows. Thus
  these37 are NOT missing-sparse-limit sentinels. This establishes metadata
  membership/values, not a hard forward constraint or physiological/accuracy
  verdict. Source/model/prior receipt snapshots rehashed; zero native forwards,
  optimizer/private reads; terminal0/MainPID0. Original D96 FAIL unchanged.

- **H97 — exact external camera-translation follow-up (preregistered).**
  Use the same15 public RGB/DWPose/shared-body observations and the15 object
  proxies already frozen before the unscored D96 failure. This cohort's private
  geometry remains unread; this is a methodological follow-up, NOT a fresh
  independent replication. No D96 rerun, native-bounds interpretation change,
  clipping, root rotation, articulation, shape, scale, K or object-proxy change.
  Source algebra establishes Vnew=Vshared+(Tnew-Toriginal), identically for
  native308 keypoints/127 joints; native controls and global rotations remain
  byte-identical. Multiplying individual vertex depths is explicitly forbidden.
  Three zero-initialized latent variables: XY=.30tanh(u) metres and camera
  Tz=Tz0*exp(log(1.25)*tanh(u_z)). Same binary-positive equal-weight10 training
  landmarks,7 heldout RGB diagnostics,≥6 supported points/32px each-axis extent,
  radial Huber/5 and physical-coordinate prior sigma.15. Exact float64 NumPy
  pinhole/latent Jacobian2n×3 at zero, no prior rows, ratio≥1e-5; reject singular
  or nonfinite/behind-camera states, no repair. Manual CPU Adam LR.01,betas.9/
  .999,epsilon1e-8;60 EVALUATED states/59updates, first lowest total objective.
  All60 latents/losses/59updates replayed from frozen public inputs BEFORE any
  private read; all15 candidates serialized F32 with unchanged native identity
  and one additive F64 deltaT (F32 translation-field rounding recorded).
  Fit CPU60s/8GiB/4threads/network-none, zero Torch/model forwards or new rasters.
  Separate CPU120s quality only after complete public/source/candidate/proxy
  audit: shared-body camera-absolute PVE equal5-frame means perclip, median3
  improvement≥5%, no clip regression>5%, no hand/clip relative-object-vector
  regression>5%; original raw branch diagnostic. No alignment, fabricated GT
  keypoints, frame removal or objective-only adoption. Evaluate ONCE; rejection
  ends this recipe on these labels. Positive requires an untouched new cohort
  for confirmation, then real-domain validation before production adoption.
  Independent analytic math audit PASS (43 combined tests;30 additional
  randomized finite-difference checks, worst3.67e-9). This freezes methodology,
  not observed accuracy; implementation/integration checks still precede dispatch.
  Engineering-only code-transfer ceiling explicitly changed100→128KB to retain
  the complete ordinary import closure (~103KB evaluator) instead of introducing
  opaque dynamic imports/duplicating validators to save3KB. No data/model/render
  bytes transit locally and no experimental/challenge threshold is changed.
  Implementation audit frozen:101 dedicated policy/fit/public/private tests
  PASS, including a15-frame producer-shaped tiny fixture executing900 actual
  NumPy objectives/885 Adam updates and full replay before its absent private
  truth gate. Source/GT/schedule/rank/candidate/proxy/inventory mutation tests
  refuse private reads;7 pure private-metric functions AST-identical to D96.
  Independent source review/Bash syntax PASS. Complete closures27files each,
  ~99.3KB fitter/~103.0KB evaluator, ordinary imports and historical source-only
  canonical mounts. Lightweight fixtures are not a native or efficacy result.
  Full5840PASS/1optionaltrimeshSKIP79.14s before execution. Producing source
  2799e8d2c468aff9230b422beea33779e640f325; actual code-only closures fitter
  99,760B/evaluator103,496B. Actual H97 public fit **PASS**19.228492s CPU,
  script`4248a26be67b89267f0780bd9565e60eab1c80f4e5fbefd800d901060f3f74bf`,
  receipt`a9108a7409912783926eb4fffe6103a97a4c304d3cb57c8a21d72e1b2664a812`
  (133,374B/0444): all15 candidates/900 evaluated states/885updates/300
  observation-Jacobian rows, zero model/raster/private reads; all input/source
  hashes unchanged. All15 rank ratios .40105..40873 and objective decreases
  establish numerical RGB consistency only, NOT independent3D improvement.
  Actual H97 quality-v1 **FAIL**6.018066s at first private topology validation,
  script`b45e34d071c36724eb5acedf61441e9ba5df3a19924febfb26674a886c67bba0`,
  receipt`733b7ba34bf8703782a8aa703a6fdc5fed379372b582807962287aeabedca38b`
  (35,320B/0444). All15 candidates/proxies/sources and900-state replay audited
  before the first private read, but `Truth topology differs`: NO frame/clip
  quality metrics or decision were computed. Private topology had been opened;
  no longer describe this cohort as entirely unread. Original failure remains.
  Read-only source + stdlib ZIP/NPY-header/face-index inspection established
  actual truth human_faces int32[36874,3], baseline int64[36874,3], EXACT same
  integer indices (0..18438), truth typed-face SHA
  `f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6`.
  Object faces remain int64[384,3],0..193/nondegenerate; all coordinate payloads
  were excluded from this diagnostic. First host query lacked NumPy and read
  nothing; the successful inspection used stdlib only. Renderer preserves native
  face dtype whereas the Body loader casts baseline faces to int64. The quality
  validator had an incorrect packaging assumption, not different geometry.
  NEW quality-v2 may accept ONLY actual int32 truth human faces with identical
  pinned indices; no cast/reindex/mesh/candidate/optimizer/gate modification.
  Reuse the exact2799 frozen fit and preserve v1FAIL SHA, source/proxy/candidate
  hashes; no refitting. This is the first quality scoring attempt after an ABI
  correction, not a performance-driven second evaluation or new independent set.
  V2 source audit: exact int32 human-face bytes SHA required on every frame,
  exact integer indices equal the immutable int64 baseline; all other truth
  checks unchanged (whole-validator AST parity after that explicit dtype delta
  and added guards). Reuse v1 metrics/aggregation/private producer checks without
  calling its failed run; original fit2799 revision is distinct from new v2
  evaluator revision. Pin/recheck original fitPASS and quality-v1FAIL, all source
  consumers/policy/fit hashes and complete900/885 replay before private scoring.
  Independent audit PASS;28-file closure~104.6KB, no data/model transfer.
  Final v2 focused47tests and148 combined H97tests PASS; previous whole suite
  5880PASS/1optionaltrimeshSKIP82.73s (additional final test cases verified in
  the focused run). Actual28-file code-only dispatch105,008B, source
  85782e18ee45aa071c9475d5e5c9492c2a81496d; exact prior fit2799 unchanged.
  Actual H97 quality-v2 execution **PASS**6.638334s, all15 frames scored ONCE;
  script`197541442e8fe02fe1393a358a35b95c0503d8105f1b5694a739507810de1a49`,
  receipt`c8c06b237327c1ffb01746c4cc2b75d67661b8642b54a3944be62f64e71f4897`
  (59,906B/0444). Native int32 face ABI verified without cast/reindex; all
  public/private/source/proxy/candidate inputs rehashed, fitPASSa910 and old
  quality-v1FAIL733b SHA unchanged; no optimizer/native rerun or gate change.
  **H97 hypothesis REJECTED, not adopted.** Camera PVE clip means (cm):
  shared baseline17.671374/9.026262/5.399058 → fitted23.115441/11.937983/
  8.474327, relative gains−30.8073%/−32.2583%/−56.9594%; median−32.2583%.
  Both human5% gates fail, though every hand/clip relative-object vector improves
  and its nonregression gate passes. Reserved RGB error also worsens:
  equal-frame mean2.182218→2.343398pixels. Training-objective decreases and
  observation Jacobian rank3 did NOT imply useful3D identification.
  Preserve frozen outputs/reports; no XY/Z/weights/LR retuning or new quality
  read on this now-scored cohort. Retain unmodified shared-body baseline and
  existing metric-depth procedure. Evidence supports studying visual bias and
  depth-gauge/orientation separately on a new untouched cohort, not accepting
  a hand-relative proxy gain that masks worse body reconstruction. No fullHOI,
  real-domain superiority, acceleration/PEN or CARI4D victory is established.
  All units terminal; GPUidle0%/81MiB, no heavy Mac transfer.
  Next prioritized hypothesis (NOT frozen/executed): one new-cohort fixed-depth
  XY+native-rootEuler pilot. Hold metric camera Z/native remainder/object proxy
  fixed; test whether orientation-aware visual correction helps without using
  global depth to explain biased2D/body-shape residuals. Require native parity
  and a fresh preregistration, automatic silhouette safeguard and unchanged
  camera-PVE/all-hand gates. Independent measurements can be biased; do not
  assume five DOF or confidence weighting solves H97. GEM-X deferred: temporal
  SOMA69-scale→MHR68 inverse/export and new~4GB assets are higher-risk before
  a discriminating body experiment. No new images/weights/quality run yet.
  Final complete source suite5888PASS/1optionaltrimeshSKIP83.96s; Bash syntax/
  diffcheck PASS. Remove disposable local test/bytecode caches; immutable Azure
  failures, successful predictions and scalar receipts preserved for reproduction.

- **H98 — fixed-depth native XY/root-Euler feasibility, fresh quality pilot pending.**
  Before new RGB synthesis, freeze a small native CUDA capability gate on the
  already-scored H97 PUBLIC baseline/DWPose only. This is engineering feasibility,
  never a new quality experiment, retuning or evidence of 3D improvement. Exactly
  15 zero-latent full-body native heads; first graph supplies 2n×5 observation-only
  autograd Jacobian (at most20 rows), minimum/maximum singular ratio≥1e-5, analytic
  XY derivative check. No optimizer/candidate/proxy/new raster or private read.
  New contract: bounded XY=.30tanh and three native Euler increments
  .30/sqrt(3)*tanh; camera Z/identity/articulation/expression/K fixed byte-exact.
  Check actual native root-Euler limits[3:6] and complete V18439/KP308/J127/204
  controls/SO3 ABI, zero parity≤1e-5m. Do NOT reapply or waive the failed D96
  all249 dense-limit policy: fixed remainder is unchanged, not clamped or claimed
  physiologically valid. Head must be body-only, all weights frozen while input
  autograd remains enabled. No rigid-centroid shortcut: actual root pivot/axes/
  parameter transform/LBS affine proof is missing, so retain the full native head.
  Whole load/hash/forward/gradient budget120s H100/32GiB/4threads/network-none;
  exact Torch2.5.1+cu124, deterministic algorithms without warn-only, TF32 off,
  no kernel relaxation. Count attempts BEFORE head, returns and validations
  separately; retain failed partial receipt. Source/model/all15 public inputs
  rehashed before/after. New results/root5-native-capability-v1, no overwrite.
  New untouched synthetic recipe and generic explicit-cohort public observers
  are prepared separately; no H98 quality read or fit is authorized by API PASS.
  Capability source audit:164 focused tests PASS3.29s, full5901PASS/one
  optionaltrimeshSKIP83.64s, Bash syntax/diffcheck PASS. Statically complete
  25-file code-only closure97,152B encoded; no data/model/render transfer.
  Actual native capability **PASS**33.913146s, source
  1fc17325f2878405e24a3fc4dbef2871596fffd4, script
  `735da6c9691f0450bb7882d8053d39c9ea1c9802bc537b34d01ffa01f54d4a87`,
  receipt`2371fc2714a17c5a40d19f63110a48c4eb531b6309f9242d893ac7a40b0a2cdb`
  (45,208B/0444), actual25-file dispatch97,560B. All15 attempted/returned/
  validated heads and20 attempted/completed autograd rows. Observation-only
  rank5 ratio.023670807; analyticXY max error6.50e-6, within declared tolerance.
  Worst zero-state V/KP/J parity8.68e-7/6.24e-7/5.06e-7m, controls byte-exact,
  SO3 max entry error2.38e-7. GPU peak allocated4,178,142,208B/reserved
  4,192,206,848B; source/assets/all15 public inputs rehashed. Zero optimizer/
  candidates/private reads; terminal0/MainPID0, GPUidle75MiB. This establishes
  local native feasibility, NOT efficacy or full numerical optimization replay.

  New H98 preparation frozen: validation/root5_rgb_v1, own new3×5 full-body
  moving scenes/left-right-left interactions, changed native shape/scale/yaw/
  motion/bottle offsets/textures. Actual manufacturing249 control+shape limits
  checked for animated andneutral states; native truth face indices stay int32
  and typed SHA retained on every frame. Own bottle geometry receives fixed1.04
  manufacturing scale once; no prediction-dependent shrinking. TRUE focal1280
  equals generic publicprior1280 in both branches intentionally: fixed-camera
  isolation, NOT unknown-calibration validation. Public contract stdlib-only
  exact readonly15 RGB+manifest; no recipe/camera/GT fields. Private rig/truth/
  renderer receipt never mounted in observers. Two reference forwards and15
  rasters onlyAzure; strict120s/32GiB/network-none preparation. Generic observers
  take explicit immutable cohort, reuse existing native functions without their
  old mains/globals. Freeze automatic Grounding/SAM human+object masks,15Body/
  15MoGe+firstRGB identity native decode/reference,15nativeDWPose separately.
  Own44prep tests/108combined PASS, independent44tests plus360 randomized tiny
  scenes PASS;14-file prepare closure~48.9KB. This is new data manufacture,
  not independent accuracy evaluation. No fresh fit/candidate/private score yet.
  Generic observer audit:24 own/210 combined tests PASS1.12s, native fixed
  F32 pointmap/K and automatic-mask pixel-area guards retained, actual complete
  source_identity() succeeds without Torch import. Independent source+mount
  review PASS: masks/baseline/DWPose containers mount only required public
  inputs, pinned assets/notices/native sources; no private truth or alias mount.
  Original loaders/decoders/validators reused, stage scripts/producer bindings
  immutable; all outputs/inventories/sources/assets rehashed. Same frozen
  observer source generates all3 public stages, renderer revision may differ.
  32-file ordinary observer import closure116,464B encoded below128KB.
  Frozen preparation+observer full source suite5969PASS/oneoptionaltrimesh
  SKIP83.95s, Bash syntax/diffcheck PASS before immutable Azure dispatch.
  Fixed-depth fitting policy frozen before observing new labels:5 zero-initialized
  native XY/Euler latents withZ fixed, same10 training/7heldout landmarks, binary
  positive equal-confidence support≥6/32px each axis; radialHuber5 plus physical
  prior sigma.15.60 evaluated states/59Adam updates perframe, LR.01/.9/.999/
  eps1e-8, first minimum total objective. Full native head only; per-frame
  observation-only rank5≥1e-5, zero-state parity≤1e-5m, root bounds only; fixed
  remainder/Z/shape45/PCA28/K/object proxy byte-identical. Freeze15 common
  metric-aligned MoGe object proxies using shared baseline hard rasters BEFORE
  optimization.900 objective native heads+15final replay/candidate heads,
  885updates/≤300 initial Jacobian rows,15baseline+15final hard rasters;
  whole300sH100/32GiB/4threads/netnone, strict same native CUDA/TF32/determinism.
  Keep60 evaluated latents/projected training points/objectives and59gradients
  perframe, for schedule validation and separate final native replay BEFORE
  any private geometry access; no900-step duplicate optimization. Automatic
  human silhouette IoU safeguard requires every frame degradation≤.01;
  evaluate only after all15 candidates frozen, never select via heldout/IoU.
  Safeguard outcome joins unchanged primary camera-PVE median gain≥5%, no
  clip regression>5%, all per-hand/clip relative-object error regression≤5%.
  A complete fit/safeguard rejection may still receive its ONE planned private
  scoring for scientific diagnosis; rejection stops this recipe on this cohort,
  no retuning. Native feasibility/2D objective decrease alone never imply adoption.
  Actual fresh render **PASS**7.235268s, producing077a03910d3568634c3b0d0e0b8ae0978303ce28,
  script`608ce90af1ed555e59bcdd8b4c984c86dc1f68c51e0be53b74fcc6392c64a265`,
  private preparation receipt`3c3ac975eda3f0b7975a7f0ef210075d89b26b2c91f910d46e14945a93150a2a`
  (30,085B/0400). All15 different new RGBs,2attempted/returned/geometry-accepted
  MHR batches; manifest`16d909d206ea9a71f402c01a699e69d63603e5b1366e1e24d55ac208846f466b`
  2,196B/0444, truth typedI32 faceSHAf6748e29…5eacd6 unchanged.14-file dispatch49,196B.
  Actual automatic masks **PASS**16.095937s, observer8084688a4d84bbad9ba8c0580e4d1b2803745511,
  script`cb48d7d0a53cfd7b9f5c7ebe9e0904af91e844d187813fd1f973538b158cd4b5`,
  receipt`3f509385e3cf1e711580cd34e0f55de55d95a2cf75a5122141a6c94c202a0050`
  22,415B/0444;30DINO/30SAM/15encoders, all15public cases, zero private reads.
  Actual fresh baseline-v1 **FAIL**1.252773s BEFORE model/inference/private read,
  receipt`dfc9107594d1ea82ba9f1f3cb9c643d21fc6248e2185c43a3fb2ff83fa229f2c`
  (3,971B/0444), same808 observer.0Body/MoGe/native/reference calls. Exact traceback
  identifies semantic receipt `results/mhr-finger-semantics-v4.json` at
  protocol.identity(): original canonical file is0644, while public-only reader
  requires no host write bits. PublicRGB/manifest/mask reports are0444 as required.
  Preserve this packaging failure and original semantic bytes/mode. Explicitnew
  baseline_v2 will use original native regular/hash/semantic validation for this
  reference receipt ONLY under the existing read-only bind, never relaxing
  immutable public outputs or changing predictions. Reuse frozen808 masks with
  exact canonical historical observer-file SHA and unchanged helper hashes;
  new observer revision generates baseline_v2 and firstDW, no mask regeneration,
  old source patch, model/optimization/quality retuning or labels.32-file observer
  dispatch117,008B each; all units terminal/GPUidle63MiB. No fit/quality executed.
  Engineering-only complete code-control ceiling128→160KB for ordinary native
  fit/public/replay/quality import closure (~133KB fitter). Do not hide imports,
  duplicate validators or drop source checks to save4KB; no model/data/render
  bytes transit locally and no scientific budget/gate/numerical policy changes.
  Predeclared separate replay:60latents/projected-points/objectives/59gradients
  reconstruct manualF64Adam and losses within atol1e-6/rtol1e-5, selected
  candidate/physical delta byte-identical to first minimum;15full native best
  state replays withgeometry≤1e-5m/controls fixed, before any private access.
  Corrective observer-v2 implementation audit:26own/147combined tests PASS;
  independent exactGit808 AST confirms existing prediction algorithms unchanged,
  historicalsource/helperhash substitution is metadata only. Preserve exact
  v1FAIL3971B/dfc910… and source cb48…; no model/private observations existed.
  Read-only semantic reference regular/hash check restores original loader
  behavior only, keeps all public artifacts strictly444; v2exclusive namespace.
  Full corrective source suite5971PASS/oneoptionaltrimeshSKIP84.34s and
  Bash syntax/diffcheck PASS before new baseline_v2 dispatch.
  Corrective baseline-v2 actual **PASS**45.007966s, producing
  feae71ea16a1d942f08e95ccafc131b6467dffb9, observer
  `0e034078bf9026b338783867293aa5f95f8c1955b5b21760b7916408ec496018`,
  receipt`675fbad6dd6888a37d7ee7f36dc968d9276de0a79eb534aec90eceb35da8668e`
  (49,759B/0444).15Body/15MoGe/15raw-parity/15keypoint/15shared
  native heads+one official reference; all15 cases retained, fixed clip identity,
  zero private reads. Independent reference mean fidelity .000154–.000250mm
  establishes representation consistency, NOT geometric accuracy. Original
  baseline-v1 FAIL receipt/hash/mode unchanged; no previous failure rewritten.
  First fresh DWPose actual **PASS**2.735390s, samefeae/0e034…,
  receipt`73472e495492002e88ee74e7bf59f1f38914a987d59353581b0b89016037aeea`
  (35,962B/0444); one CPU session/15native133-keypoint calls, original float64
  list feed/keypoints, float32 scores, no cast/channel modification/neck134,
  private access or GPU. Final inputs/assets/source rehashed. Both units terminal.
  Mandatory six-field public pins configuration binds actual completed manifest,
  masks, baseline-v2 and DW receipts and stage-specific source revisions; no
  dynamic discovery/placeholder/fallback. Separate CPU public contract owns
  all candidate/proxy/continuous60-state59-update replay validation; native fit
  and replay use it without importing old stage mains or private readers.
  Implementation172focused tests PASS2.34s, independent115public/fit/observer
  tests and actual native signature/source/mount audit PASS. Ordinary fit code
  closure33files133,976B before tiny actual-pin config (<160KB). Top-level
  fit native_backwards/optimizer_updates count real885, replay counts zero.
  No fit, replay or private quality executed at this receipt freeze.
  Full frozen public+fit source suite6060PASS/oneoptionaltrimeshSKIP84.34s;
  actual-pin complete static closure34files134,268B encoded control, Bash
  syntax/diffcheck PASS before first immutable native fit dispatch.
  First public native fit-v1 actual **FAIL**.010366s in public_integrity,
  sourceb248f129e79c0bfcc5e5ef2a05d59999e6e91aee, fitdriver
  `df60eb9990fbdcb8931f7a508f808bf73121de5ea4bd5d97b48e98d1680067d9`,
  receipt`ceb6a65fbdc59d78872737901b08fc182accfa1a18abef4b5de41b04fb77d4b8`
  (2,811B/0444). Exact missing CODE/infra/run_dwpose_smoke.sh source-only
  provenance dependency: static import closure did not include .with_name()
  source-hash targets. ZERO model/proxy/native/Jacobian/Adam/candidate/private
  work, no scientific rejection or tuning observation. Preserve v1 bytes/source.
  Engineering-only v2 correction: explicit unique root_fit_v2 output, strict
  original failure SHA/bytes/revision/source/error/zero counters/lists required
  before public input/model access and rehashed after; old receipt RO bind.
  Runtime bundler now retains committed literal sibling source filenames,
  including finite source-hash lists and their ordinary transitive closure;
  absent explicit .with_name source dependencies fail before Azure dispatch.
  External/vendor names are not invented as own code.68bundle tests PASS,
  including actualH98 wrapper closure; independent109public/fit tests PASS,
  scientific helper ASTs match b248 exactly. No optimization/mask/camera/rank/
  prior/budget/tolerance/quality-gate changes; replay remains firstroot_replay_v1.
  Corrective36-file complete fit closure136,392B before finalGitmetadata.
  Corrective full-suite first audit6081PASS/2FAIL/oneSKIP80.05s: two historical
  bundle tests expected source-only fit references to be omitted. New complete
  closure correctly retains these hashes; update ONLY source-inventory assertions,
  preserve their128KB control caps (measured119,528/121,052B), all scientific
  metric/helper/oldproducer bytes unchanged.164focused legacy+bundle PASS9.18s;
  final177corrective bundle/public/fit PASS2.15s.
  Final frozen corrective fullsuite6083PASS/oneoptionaltrimeshSKIP78.19s,
  no scientific recipe/producer changes; readyuniquev2 nativefit dispatch.
  Corrective fit-v2 dispatched13720afb6f75bed59e449ff5e5f033d88e49a153,
  actual36-file code archive136,984B. First bounded observation at108s:
  7candidates/8traces,443nativeattempts,435Adam/backwards,15proxy rasters,
  no private reads; running report not immutable, no PASS/quality inferred.
  Separate quality source frozen before private execution: required actual
  fit+native-best-replay report revisions/SHA/bytes/source/config pins (no
  placeholder); all15traces reconstruct continuousAdam/objectives/rank, strict
 15native selected-state replay gate, source/checkpoint/region/clipfixedABI.
  QualityCPU120s/8GiB/netnone, all15frames scored ONCE evenIoU safeguard fails;
  no oraclefit/alignment/frameomission. Seven unchanged historical own metric
  functions extracted to reusable root_rgb_metrics (exactAST/constants from
  own5124edad…source); do not import obsolete stage drivers/transitive fit.
  Complete quality41files155,144B before futureactualpinconfigs, no codecapraise.
  Native model/reference host0644 stays under oldcanonicalRO validators;
  ownedcandidate/proxy/replay/private/config/source files remain strictly444/400.
  Entire public lineage is validated beforeGT and rehashed after; no blanket
  strict444 reader applied to alreadyauditedreference0644. Paired-baseline
  depth is not bitidentical to actual zero-native points (parity≤1e-5m), so
  paired analyticXY comparison is diagnostic, never an extra inconsistent gate;
  actualnative analyticXY/Jacobian gate remains source/receiptbound unchanged.
 43qualitytests/220combined PASS4.41s; full6126PASS/oneoptionaltrimeshSKIP81.48s,
  Bashsyntax/diffcheck PASS. No private H98 quality read yet.
  Independent read-only qualityaudit PASS43tests: Git077render+5helpers
  exactbytes, legacy7metrics exactAST, fit/replay4ed1…field compatibility,
  full249 manufacturing/private topology/source/mount provenance; no blocker
  found. Only actual fit/replay PASS and realqualitypins authorizeGT scoring.
  Actual native fit-v2 execution **PASS**252.099728s (300s budget), source
  13720afb6f75bed59e449ff5e5f033d88e49a153, driver
  `4ed1b4d60c89a47b083359815673f2e5a61d50dffdf25ca5bec2c15ec6efa03c`,
  frozenreceipt`3fd709d185e32412edb0d4a0325960e1ef63f535bfc992253178d04a55ac5aa1`
  (796,553B/0444).All15candidates/traces/300initialJrows,900objective+
 15finalnativeheads/885backwards+updates/30rasters returned/validated exactly;
  peakallocated4,178,142,208/reserved4,188,012,544B, sourceassets rehashed,
  zero private reads, original bootstrapfailure unchanged. Terminal unit.
  Automatic human silhouette safeguard **FAIL**: worstframeΔ−.024868986,
  clipmeans−.019316461/−.020968655/−.009918749; no candidates selected via
  guard or omitted/exportblocked. This recipe already cannot satisfy all
  preregistered quality gates. Still execute the planned independent selected
  native replay and ONE private paired score for diagnosis, not to override
  rejection or retune. Metric accuracy remains unmeasured at this point.
  Actual immutablefitpin written (completed SHA+producerrevision only);
  nofit/public/helper source edits after this observation. Next separate
 120s/15-native-head/zero-optimizer replay, then actualqualitypins ifPASS.
  Actual independent public native-best replay **PASS**45.580706s, producer
  bed76ef37596ef26588d449d8109e8be8b0ca254, unchangedfitdriver4ed1…,
  receipt`b76d5b4bb28833a44cb223d6adfb457f1e9fb347df7b5a5c8d4718f27138a084`
  (58,254B/0444).All15full60/59 continuous trace audits PASS;15selectednative
  heads returned/validated, zero optimizer/backward/additionaltraining/raster/
  private calls. WorstV/KP/J replay6.84805e-7/5.33589e-7/4.80548e-7m,
  controls0/SO3entry2.98023e-7; peakGPU allocated/reservedsamefit~4.18GB.
  Native replay verifies representation, not accuracy. Allsourceinputs/assets
  rehashed; terminalunit. Actualten-field qualitypins nowbind bothcomplete
  receipts/revisions/bytes/sources and publicconfigSHA. No placeholder/dynamic
  fallback/source edits. One plannedCPUquality maynow scoreall15frozenframes;
  silhouette FAIL stillforces recipe rejection regardless of private metrics.
  Postactualfitconfig220focused tests PASS2.54s, no numericalpolicychanges.
  Actual H98 paired private quality execution **PASS**22.739052s, source
 4c18295726b43f3d6728a87370faa797bfe9005e, driver
 `c75e68a8bdaa870154072402f885c504cebbd0fb7cf393828964d52893960c47`,
 receipt`7c48b7b73c827786799a4273c20c164fa2747b3abedfd2aa2e74a035a95201f1`
 (58,099B/0444).Complete15frames scored ONCE after all public60/59trace and
 independent15-native-best replays PASS; zero private optimizer/model calls,
 noGTalignment, fullprivate/public/source/asset rehash; no frame omission.
 **H98 scientific hypothesis REJECTED, not adopted.** Camera-PVE clipmean(cm)
 baseline20.542976/13.786135/9.077390 → refit20.737680/14.087849/9.775683;
 relative gains−.947786%/−2.188538%/−7.692662%, median−2.188538%.
 Median≥5% and no clipregression>5% gates FAIL; all6 perhand/clip relative
 object-vector regression≤5% PASS. Automatic silhouette no-loss>1pp FAIL as
 alreadyobserved. HeldoutRGB mean1.809558→3.004909px (worse), neverusedforfit.
 Native feasibility, clean replay and training objective decrease did NOT
 improve heldoutgeometry; no victory over CARI4D/fullHOI/real-domain claim.
 Stop root-only recipe onthisscoredcohort; no retuning orsecondqualityquery.
 Nextpriority is observation/model-bias diagnosis on untouched factorialRGB,
 distinguishing2D detectorbias frommorphology/articulation/globaldepth error
 BEFORE new optimizer freedom. Prompt-conditionedbodyarticulation conditional
 onusefulindependent2D evidence; notmoreZ freedom orGEM-X installation first.
 Final postactualreceiptconfigs fullsuite6126PASS/oneoptionaltrimeshSKIP81.61s,
111focusedquality/bundlePASS2.68s; qualityactual43-filearchive156,200B<160KB.
  Frozen-score scalar diagnosis (no extra private labels/model/fit/score):
  common objective mean .442975→.351758, while centeredPVEclipmeans(cm)
  3.733301/3.901715/4.055680→4.050873/4.319675/4.442132 worsened too.
  Baseline signedcentroid-Z errors(cm)20.411840/13.649304/8.159291 dominate
  cameraPVE; nativefixedZ root rotation cannot remove this metric depth bias.
  CandidateZ centroid20.532886/13.787849/8.333527 (externalZ was bytefixed,
  not every vertexZ underorientation). Avoid inferring detector vs articulation
  cause fromthese15cases alone; factorial diagnosis remains necessary.
  Challenge primaryCD uses first-human-only Sim3 shared withobject (audit),
  unlike this deliberately camera-space isolated gate. Future new-cohort
  quality must also preregister that actual global evaluation convention and
  object/contact/temporal quantities; do not retrospectively replace H98
  failedgates or score it again undera favorable alignment. CameraPVEalone
  is usefuldepth diagnosis, not evidence ofchallenge ranking superiority.
  All exact H98publicfitFAIL/fitPASS/replayPASS/qualityPASS receipts unchanged;
  lastGPUidle45MiB/0%,451GBused/574GBfree. Noheavyartifact downloaded locally.

## H99 — preregistered morphology/appearance/occlusion diagnosis

New untouched `validation/factorial_rgb_v1`: 24 own RGB cases,
2 morphologies ×2 diffuse appearances ×2 bottle depth orders ×3 identical
articulations. Morphology shape first2 (.31,.09)/(-.28,.18), free scale68
.04/-.03 (locked values0), legal native249 bounds; no clipping/repair.
One K1280/1024×768 and framing from both neutral identities; no per-case
camera adjustment. Same human V/J/KP/camera bytes across matched appearances
and occlusion. Own bottle manufacturing scale1.45, elbow-centred Z±.25;
12 front/back pairs must expose≥64 extra human pixels BEFORE inference or
abort manufacture (no recipe repair from model predictions). This controlled
render is not a photorealistic/real-domain or interaction reconstruction test.

Native truth uses reference352e… V18439/J127, exact frozen checkpoint
`head_pose.keypoint_mapping` F32[308,18566] from Bodyb5a2…; map V/J cm/100
BEFORE YZflip and common sceneR/T, no row normalization/semantic guessing.
Existing bundled Body MHR SHA352e…/696110248B is byte-identical to reference
(previous completed Body receipt independently inspected; metadata only).
Preparation additionally runs6 same249 parameters through each rig and
requires exact joint names/topology, V/J≤1e−5m; reference2 +bundled1 batches,
24 rasters,120s/32GB/4CPU/offline H100. Public manifest exposes24 original
RGB identities/dimensions only; all factors/geometry/rig/camera remain private.

After actual preparation PASS, commit exact manifest pins only; masks180s
48DINO/48SAM2/24encoders (`person.`/`bottle.`) without human prompts. After
actual mask PASS, commit exact producer/source/report pins. Framewise original
Body24 +24 native parity +24 full308 heads (300s/32GB), native DWPose133
24 unmodified F64-list-feed calls/one CPUORT session (180s/8GB). No MoGe,
proxy, optimizer, shared identity fitting or score-dependent selection here.
Every24 output immutable and reread; source/model/input bytes rehashed.
Invalid required inference fails execution, never drop/replace a case.

Before first private diagnostic, commit actual Body/DW/render receipts and
source revisions/hashes/sizes. One CPU120s/8GB read of all24 frozen cases.
Primary descriptive measures: raw camera native PVE, centroidXYZ, centred
PVE/RMS (RMS²=centroid²+centredRMS²), joints, derived COCO17landmarks,
8 limb proportions; Body2D vs DWPose2D on SAME native-positiveCOCO17indices
per frame, no score weights or GT visibility. Report Bodyall17 and all positive
counts. Equal landmark means→3pose means→8group macro, no missing-case drop.
Next automatic prompt pilot has supporting evidence ONLY if relative DWPose
error gain≥10% in≥6/8groups, every24frame has a positive pair/all8gains are
defined, and both appearance/occlusion levels occur among supportinggroups.
Otherwise deprioritize that prompt hypothesis; no threshold retuning.
Always report12 appearance and12 occlusion paired contrasts using common
positive-index intersections across compared levels; Bodyall17 separately.
Factor labels private quality only. These3poses/2morphologies are not independent
people; no significance or anatomical measurement independence claim (truth
landmarks share the pretrained mapper definition, not RGB predictions).

Separate gauge diagnostic: one proper positive full18439 first-human Sim3
pergroup, frozen across3poses, transforms+raw camera errors disclosed. Reuse
existing tested Umeyama helper (`joint_rgb_evaluate`, row@R.T convention).
NOT official alignment subset or fullHOI score; no object prediction/contact/
temporal metric in this diagnostic, no per-frame/object fitting/K correction.
H97/H98 scored cohorts/failed gates remain immutable; no extra score query.
Public contracts verified before any private read; zero private model/optimizer
calls. No adoption/submission/CARI4D-superiority claim regardless of outcome.

Independent prep/observer/quality source audits; 109 focused tiny tests PASS
1.68s before first Azure H99 dispatch. All heavy arrays/RGB/models remainAzure.
Full suite6235PASS/oneoptionaltrimeshSKIP84.48s. Source-only closures
prepare17files52,288B; observer33files115,488B; quality40files146,476B
(<160KB). Three independent read-only audits complete. Commit beforedispatch;
no actual H99 runtime/accuracy result has been observed at preregistration.
Before any H99 quality/public outcome: original Body loader receipt audit
requires1101 learned parameters and113 explicit immutable rig buffers, not
only matching reported source hashes. Quality also reports preregistered
equal8-group macro means.68 focused quality/observer tests PASS1.56s.
Actual H99 preparation PASS13.422197s, producingcommit
dbd888b7711729e3672e56ea127616c1b517c070, renderer
b2731a6bf4971ec2b112cdd489c51bff437b77b7ac77f0519603424bb4a948be,
private receipt75ca10272ac7d15a0d39081bbf424b51141e4aca072899d6df6bb0dcc77a3b36
(27,428B/0400), public manifest7b0ae37ebd6009d24887b999558f86cbd8e2b85d5eb164128ae6ac38326c4664
(3,511B/0444). All3native batches/24rasters returned; bundled/reference
V/J maxerrors0.0m, names/topology exact. All12front/backpairs useful,
minimum994 newlyvisiblehuman pixels. No inference/quality performed.
Terminalunit/H10039MiB0%. Actual manifestpins onlynow committed, nofactor
labels or geometry in publicmanifest; nextseparate48DINO/48SAM inference.
Private diagnostic strengthens recorded nativeDW feed/output trace: exact
24listF64[3,384,288] supplied, delegated unchanged, two nativeSimCCF32
[1,133,576]/[1,133,768] byte identities matchrecords. No inference/score
policy changed or private labels accessed.36tiny quality tests PASS.
Actual H99 automatic masks PASS22.212740s, sourcebebe9f5ab99d84346520acc559c8799b12d3c0fd,
observer9c69c9d8f7951ef03db2cad73d45766c95b605de70fdcaa907b13bf67142d667,
receipt1c901b3f681b064583d1646328a59a19a3f58d101924be2c312ba29f51422f6a
(31,507B/0444). All48DINO/48SAM/24encoders and24outputs complete,
private_truth_readFalse, terminalunit/H10033MiB0%. Actualmaskpins committed
onlyafterreceiptPASS; unchangedobserver/sourcehelpers/camera/cohort.
143 focused quality/observer/bundler testsPASS1.64s. NextBody/DW independent
serialstages, no private scene mounted and nofitselection.
Actual H99 framewise original Body PASS28.015011s, producingcommit
7133a59c0617dc48d3afeceb27f73466c8767396, unchangedobserver9c69…d667,
receiptcca29fa08c4ba442118a6ac8742b5d008e7e29e66f8f30c99ed6dc6f28827e82
(33,893B/0444). All24Body/24freshnativeparity/24full308heads attempted and
completed;24outputs immutable/reread, sources/assets rehashed, nofit/private.
Terminalunit/H10027MiB0%; fullsuite6242PASS/oneoptionaltrimeshSKIP82.36s.
NextnativeDWCPUstage sameobserver/source/image/actualmaskconfig, separate
unit/output, no private mount. No accuracy results seen at this point.
Actual native H99 DWPose CPU PASS3.861546s, producingcommit
15508652c0b92fbd3cac0b577ba06ad06f247e4c, unchangedobserver9c69…d667,
receiptde4804dc7140adbdda430cac1b23b2c5c06ebe3f48f37001b80f976f079c7fe1
(54,127B/0444).OnefreshCPUORTsession/all24originalnativecalls completed,
F64coordinates/native133 validity preserved, disposableprefixremoved;
no private labels/model-fitting. Terminalunit. All24BodyandDWfrozen before
actual qualitypins nowcommitted (exact3producerrev/source/SHA/bytes +public
configSHA). One120s private diagnostic next, all24cases/all8groups retained;
no model/optimizer/private labels passed to any producer. No scienceoutcome
observed beforetheseactualpins; preregisteredruleunchanged.
Actual H99 private diagnostic execution PASS4.731652s, source
0314cd0380eed79e02015ce772dae229e63c644d, driver
54f67adeabbe0611a4ba287edac55f5687396eec19c07bb42ed0c98ecd4de234,
receipt87d66f54aa31896f467e5faf749add68e1608ae9fbf36a7272b7a81af71630ad
(66,534B/0444).All24cases scored ONCE after complete public24Body/24DW
arrays+nativecallprovenance verified, predictions frozenbeforeprivate,
zero private model/optimizer calls; allinputs/assets/private/source rehashed.
Execution success is NOT scientific hypothesis support. **IndependentDW
prompt-evidence hypothesis REJECTED**, automaticpromptpilot NOT authorized.
All408/408COCO17nativeDWpositive, fullcoveragePASS; relativeDWerrorgains
by8groups = −158.4863%, −121.9856%, −91.1497%, −76.9623%, −178.0110%,
−136.1079%, −117.0072%, −99.6485%; supportinggroups0/8 (required≥6).
Bothappearance/occlusion supportgatesFAIL. No selectivelandmark/error/
confidence reweighting, no DWelbow-only rescue/threshold retune onthiscohort.

Equal8-group macro Bodyall17/paired2.336319px vsDW5.107629px; rawcamera
humanPVE14.354771cm, centeredPVE4.857469cm, signedcentroidZ13.911338cm.
Firsthuman-only full18439 Sim3(mean3poses/group)3.858476cm; oneproper
transform/group fixed3poses, scales .974524..998771. This removes gauge
error for diagnosis, NOT inference/fullHOI/officialsubsetscore/CARIranking.
Plain/stripe appearance means Body2D2.033976/2.638661px, centeredPVE
3.711756/6.003181cm, firsthumanSim3PVE2.647056/5.069896cm. Matched
humanV/J/KP/Kbytesverifiedidentical; descriptiveappearance dependence,
notpopulationstatisticalsignificance or proof2Dmapperanatomicalaccuracy.
Morphology0/1signedZmean9.512900/18.309775cm. Do not infer that adding
Zfreedom or a new detector cures this; H97/H98rootrecipesremainrejected.

Stop DWprompt path; preserve allH99frozenreceipts, nosecondqualityquery.
Next scientific priority: publicRGB appearance/temporal consistency with
clip-constant identity, tested against an unchanged SAMEidentity baseline
and a matched SHAM on an untouched cohort, then real-domain/fullHOI
validation. Research nativeSAM/GEM-X representation and licenses first;
no newstack assets just because H100available, no promisedCARIvictory.
All5H99units terminal; lastH10027MiB0%, noheavydata/model/RGBtransferred
locally.143focusedtestsPASS1.73s immediatelybeforeactualqualitydispatch;
actualqualityarchive43files148,268B<160KB.
PostactualH99quality/sourceconfig184focusedtestsPASS1.86s; cleanworking
source and historicalrecipes retained. Next concrete action is a NEW
public-only photometric/native-medoid mechanism test, before preregistering
newquality; not resurrectDWelbow-only prompts or addrootfreedom. Literature
audit supplies exactAPI/coordinate/SHAM/nativeidentity constraints; no extra
checkpoint required, allGPUdata remainsAzure. H99marksPROGRESS (actual
scientificrejection changesnextmethod), not a blocker or completedsubmission.

## H100 — public-only native photometric mechanism preregistration

Previous goal turn was PROGRESS: actual H9924-frame diagnostic rejected
DWprompt evidence and changes next method, not a wait/blocker. H100 is
mechanism validation only, not a substitute for requested fullHOIsubmission.
NewoneRGB `validation/photometric_native_v1`, no scored H99image reuse.
Ownmanufacture: shape first2(.23,−.16), free68scale.02/locked0, named
leftupperarm.21/elbow.36/rightupperarm−.015/fingers.16; actual249legal
animated+neutral recipe. Neutral wholeactorframing8px, K1280/1024×768,
stripedownappearance and bottle1.3. Reference352e…2batches/1raster.
No truthgeometry/rig/parameter/camera array stored; only originalRGB,
RGB-onlymanifest and reproducibility receipt. No inference receives the
manufacturer report/mesh/camera/labels.

Singlezeroargument frozenjob with THREE isolated offlinecontainers: manufacture
120s, automaticDINO/SAM2personmask120s, Bodymechanism180s; eachH100/32GB/
4CPU, imageb47…/grounding53b… pinned;3s outergrace/10skill. Source-only
archive≤160KB; no data/model/RGB downloaded toMac. Entire output route
exclusive, no restart/replacement of a failed namespace.

Threephotometric inputs fixedorderedgamma(1,.8,1.2), deterministicuint8
LUT floor(255*(x/255)^gamma+.5), SAMEoriginalautomaticmask/bbox/K.
ThreeSHAMgamma1copies (sixactualordinaryBodycalls total) samefixedseed0
withstrictTF32off/CUBLAS/determinism, no newweights/decoderprompts.
Gamma1bytes must exactlyoriginal; original/transformedRGBhashes logged.
Everyordinaryprediction gets independent freshnativeblockparity plus308KP
head; actual originalBody1101parameters/113rigbuffers strictloader unchanged.
Anchorshape45/PCA28/globalZYXEuler/Tcam/hands108/zeroexpr comefromfirst
originalRGB; proposeONLYnativebody133. Nativeusesfirst130 body slots,
actualhandcolumns68:122overwritten, controls0:6+136:204 bytefixed.
Fullreturned204 checks fixed0:6/68:122/136:204 byteexact, all V18439/KP308/
J127positivefinite and127properSO3. body133includes6skeletaltranslation
controls130:136; this is nativeBODY-block consistency, NOT purearmrotation
or allvertexZ/physicalshape fixed. No obsolete D96dense249baselineguard
is silentlywaived: this new mechanism has finite/proper/nativeparity+
bytefixedblock guards, no physiological/qualityclaim.

Sixfixednativeheads then existingprediction medoid perimage fromthree
fullcameraV sets: F64mean square vertex distance sumtoother2, earliest
exacttieoriginalgamma1; noEuler/geometricaverage, GT/IoUselection or scale
alignment. SHAM3rawpredictions must EXACTnativearraybytes matchoriginal,
SHAM3fixedheads EXACTfixedoriginal; SHAMscores0/index0. Baseline/SHAM/TTA
selectedexistingproposals independentlynative-replayed once (3heads),
maximum V/KP/J/control/R errors≤1e−5. Expected6Body/6parity/6KPheads/6fixed
heads/3selectedreplays,15immutableNPZs reread plusreceipt, source/assets/
originalRGB/mask/semantic rehashed. Anyfailure stops route; no epsilon
relaxation/gamma search/privatequery/omission.

PASS proves only that the mechanism runs and SHAM is sound. It DOES NOT
authorize adoption/submission or a superiority claim. IfPASS, preregister
a NEW24human-only quality pilot againstsameclipidentity baseline with
sharedbaseline-firsthumanSim3, no retuning H99; fullreal-domain/object/
contact/temporal verification and faithful frozenParquet remain necessary.
Independent nativeAPI/source/purepolicy/manufacture audits and156focused
unit/bundle tests PASS0.39s; completeordinaryclosure33files106,504B.
No actualH100modeloutput/accuracy observed beforethispreregistration.
Fullsuite6330PASS/oneoptionaltrimeshSKIP85.77s;156focusedtestsPASS after
final sourceguards. Independent runtimeaudit confirmsnoTorchimport before
GPUstage, exactnativecontrolcontract/SHAM/budgets. SAM2installedsource
nowrehashedafterautomaticmaskinference (notjustweights); 1101nativelearned
parameter/113rigbufferloader evidence+actualhandindices retained explicitly.
No actual H100output/privatequality read while adding theseintegrityguards.

### H100 actual unsupported-native-kernel failure; H100b preregistration

Actual source8c68fba8d4f9034c2e19ed2fa9eac394c9f64758, completeclosure33files
107,164encodedB. Render PASS4.254044s, receipt21e81b9dd3c6a7c1436382aaa8f4528d0caf56c71b0a81957cf7e7b0e3410b41
(1,873B/0444). Automaticmask PASS10.722988s, receiptd268cc0bd750cf2eb336da390534faadafd8c1271d442cb28e653cf0c8a349c9
(6,602B/0444). NativeBody FAIL18.691310s BEFOREfirstprediction: official
prompt_encoder.pyL241 `grid.cumsum(dim=0)` lacks PyTorch2.5.1 deterministic
CUDA implementation. Exactly1Bodyattempt/0completed/0decoderheads. Receipt
e136b3cd417fd72d54368885e32163250514b692acda9c808f6ea98207a0caaf
(17,361B/0444), driverd8ada243dd8bd234aaf640f2bd40f56f0e3abfa7d671829780a7898e54822e5f.
Unit terminalfailed/lastGPU137MiB0%. No quality/privatequery, no native
prediction seen; H100 remains FAIL, not a scientific rejection or PASS.

H100b is a separate180s public-only engineering contract, newuniqueunit and
`capability_replay_v1` output. Reuse original immutableRGB+automaticmask via
exacthistoricalproducer/source/reportSHA+bytes and currentpre/postidentity;
originalbodyFAIL scalarreceipt preserved and independently pinned. Bind original
source at its SAMEcanonicalpath; expose no failedNPZ/private/modelreference.
Unmodified official learned model/cumsum/source/checkpoint, seed0/TF32off/
CUBLAS:4096:8/4threads; explicitly `deterministic_algorithms(False,warn_only=False)`
for all6ordinary predictions andallnativeheads. Actualalgorithm/TF32state checked
before every branch, no scoped kernel patch, hidden warmup or inference retry.
Runtime inability is not excused by `warn_only`; this newcontract does not claim
PyTorchglobally guaranteed deterministic kernels. ORIGINAL exactRAW+FIXED SHAM
allarraybyteparity, zeroSHAMmedoid/index0, native≤1e−5replays, same15artifacts/
6Body+6parity+6KP+6fixed+3selectedcounts remain mandatory. If SHAM fails, stop
honestly; no tolerance relaxation or claiming seeds guarantee atomics.
PASS only demonstrates empirical repeated-input fidelity on this fixture and
process, not crossprocess/allframes/accuracy/adoption. Gamma/grid/mask/K/policy
unchanged.163focusedtinytests PASS0.38s before anyH100boutput/privatequality.

H101 code/planning may proceed independently, but no quality manufacture or
inference dispatch until H100bPASS and complete newquality preregistration.
This engineering failure is actionable progress, not external blockage.

### H101 renderer frozen pending mechanism PASS; not dispatched

Newhuman-only24RGB cohort `validation/human_photometric_v1` (8groups×3poses),
noH99/H100scoredimage reuse. Manufacture recipefixed: shape(.18,−.11)/(−.21,.14),
free68scales.025/−.02/locked0; namedleftupperarm.16+.05f/elbow.24+.06f/
wrist.008−.012f/rightupperarm−.018(f−1)/fourleftfingers.10+.03f; ownyaw
.10+.03(f−1),cameraX.01(f−1). Plaingarment(.34,.35,.37) vsfrequency20
stripes(.45,.40,.35)/(.22,.29,.38), samehumanV/J bytes; bottle nuisance1.35
front/back±.25Z around actualnamed l_elbow, NOTanobjectprediction/HOIproxy.
Commonwhole2neutralidentityframing8px/K1280/1024×768. Native249limitsstrict
(noadaptation), reference6animated+2neutral in2calls, actualbundled352e same
asset/source6animated1call maxV/J≤1e−5m/fulltypedtopology/source/acqrehash.
All24originalRGB+manifestonlypublic;9fieldhumanV/J/F/K/visibility/raster/group/
frame truth private400/dir700, fullrigprivate400. All12occlusioncontrasts
≥64newlyvisiblehumanpixels, bothentitysupport≥64, geometrybytesindependent
ofappearance/occlusion, otherwiseFAILnotreciperepair. Oneoffline120s/32GB/
4CPUrendercontainer; onlyreference+bundledMHRassets, no2GBcheckpoint/model
inference.42tinyprotocol/render/wrappertestsPASS; closure20files52,224B.
Code is prepared only: noH101manufacture/inference/privatequality dispatched.
Quality inference/decision guards must be fully preregistered separately first.

H100b independent frozen/currentAST-source audit READY: historicalpublicreader/
helperSHA unchanged, oldsource canonicalRO, complete34fileclosure108,016B,
139focusedcombinedtestsPASS0.70s/bash/diffPASS. RuntimeactualSHAMstillunknown.

Actual H100b execution FAIL20.478390s, sourcee83146a0a85f1530813228e2feb2a9b5a8f8a02b,
driver32cd5626e77b272b70f3bb4dafe91b6a706d2dbcc4e4f1bf205cfc28cfde8ec6,
complete34filearchive108,584B. Receipt110eebee952eb1fcd2c6db4b359a1756831ff1ad1d4549f8807f4f62c6a0e11c
(22,005B/0444). FourBody/4parity/4KP/4fixed heads completed; firstRAWsham0
failed allnativearraybyteequality despite boundednativeblockdecode parity;
0selectedreplays/accuracyqueries. Unit terminalfailed/lastGPU131MiB0%.
H100b remainsFAIL; no relaxedSHAMgate, H101 noexecution/privatequality.
Next boundedscalar-only failure diagnosis compares existingfrozenraw/fixed
original andsham array differences to localize learnedblocks vsnativegeometry.
No learnedmodelcall, image/geometryquality scoring or recipechange. Fullsuite
6379PASS/oneoptionaltrimeshSKIP83.96s beforethisresult; localchildtestRTKfix
42PASS2.50s. Actualsemanticjointnames contain no `elbow`, so H101manufacturer
currently fails its namedjoint preflight: correct only via actualsource/name
evidence, not numericaljoint guessing or rendering tochoosea betterfixture.

Scalar-only existing-artifact H100b diagnosis (no model/raster/quality calls):
learnedblocks themselves differ, not just decoder vertices. OriginalvsrawSHAM
maxabs globalEuler1.19209e−7, body1331.00210e−6, hands2.38419e−7,
shape7.15256e−7, PCA1.90735e−6, Tcam9.53674e−7; expressionexactzero.
RawV/KP1.43051e−6m, J9.53674e−7m, rotations2.20537e−6. Fixedcamera/
identity/hands remainbyteequal byconstruction, but body133 andgeometrydo not.
This localizes no unique cause: native intermediateMHRfeeds learneddecoder
(sourceaudit), whileotherlearnedops/JITcould also differ. Do not dismiss
smallactualdifferences or relaxstrictSHAM. No privateaccuracyqueried.
Actualsemantic127jointnames show leftlower-arm origin `l_lowarm` (index76),
not `l_elbow`; H101notdispatched source name corrected toactualjoint
`l_lowarm`, not guessednumericindex or postrenderrecipetuning. Allframing/
recipes/budgets/gates retained; originalunexecuted sourcecommit preserved.

### H100c scoped native operational replay preregistration

Same onepublicRGB+automaticmask, exactoriginalH100source/render/mask/failed
strictreceipt pins; H100bfailedreceipt110e…e11c/22,005B bound too. Newunique
`capability_scoped_v1`,180s/32GB/4CPUoffline, no private label/RGB download.
Joint NEWexecution policy, not a causal attribution: learnedestimateCUDA
ordinaryFalse/warnFalse (officialcumsum untouched); serialwrapper on EVERY
`head_pose.mhr_forward` (includingintermediatedecoderheads) saves enabled/warn
flags, enablesstrictTrue/warnFalse, delegates the exactoriginal boundmethod
withunchangedargs/outputs inside JIToptimized_execution(False), synchronizes
CUDA, validatesflags and restoresfinally. No concurrent/reentrant scope,
unsupportedopfallback/warmup/kernel/sourcepatch. Intermediatenativegeometry
feeds learneddecoder, so unchangedsource/weights do NOT imply numerical
equivalence to H100b. Strict-vsJIT cause is not isolated by this jointpolicy.
Allactualscopedhead attempts/returns/validated/restore phases retained; count
includesnativeintermediateheadsplus21existingparity/KP/fixed/selectedheads.
ExactoriginalRAWandFIXED SHAM3repeats/gamma1identitybytes, native≤1e−5full
replay, positivefiniteSO3/fixedcontrols/medoid15NPZ/noaveraging andall6Body
coverage unchanged. PASSwouldonlyempiricallyqualifythissixbranchfixture
mechanism, notcrossprocess/allframes/nativeaccuracy/adoption. AnySHAM
mismatch stops; no epsilon/gamma/privatequery rescue. H100/H100bFAILpreserved.
111focuseddatafreeexecutionpolicy/provenance/core/bundle testsPASS0.46s,
actualH100coutput notyetseen. H101staysunexecuted pending mechanism PASS.

PreparedH101observer is public-only code, notlaunched: masks24DINO/SAM/encoders
120s; Body600s6branches×24=144inferences/144parity/144KP/24frameanchors/
144fixed/72selectedreplays, 360NPZ+actualnativeLBSmetadata. Clipfirstshape/PCA,
frameoriginalroot/camera/hands fixed acrossmethods, exactSHAMandexistingVmedoid.
Its ordinaryCUDA policy is currently NOTqualified (H100bfailed); do not dispatch
it or claimthispreparationis a validatedqualityexperiment. Frozenhuman-only
paireddecision purepolicy16testsPASS (8groups median5%gain/no>5%group/hand
regression/every24frame automaticIoU−1pp); missingfinalqualityproducer pins,
evaluator and mechanismroute prevent any adoption. Noobject/contactclaim.

H100c independentaudit identifiedexceptioncleanupgap; fixedbeforefirstdispatch:
CUDA drain attempted withinstrictscope even iforiginalraises, syncfailure
recordedwithoutreplacingoriginalexception, nestedfinallyalwaysrestoresflags/
method/busy. No retryoforiginaloperation; completePASStrace requires actual
synchronizedTrue.112focusedtestsPASS0.50s incloriginalerror+syncfailure
cleanup;266combinedresearch/core/bundle testsPASS3.90s beforethisguardfix.
Completeordinarysourceclosure36files109,912B beforefinalcleanup addition.

H100c dispatched from18c605dcf0b045db1472d6de6e5d9b95dd17283f, exactcomplete
36filearchive110,640B, uniqueunitworld-reward-h100c-native-scoped-replay.
Finalread-onlyexecutionauditREADY; fullsuite6440PASS/oneoptionaltrimeshSKIP
87.29s (finalexceptioncleanupaddition separatelytested),62focusedcore/new
policytestsPASS5.39s. Dispatch/active status alone is not a mechanism PASS.
Only own local disposable Python/pytest caches removed; all Azure frozen
failures/nativeartifacts/historicalsource preserved, no heavy data here.

IndependentpreparedH101consumer audit85testsPASS: strictinventories reject
hiddenextras, clip/frameidentity/SHAM/medoid+allmetricgates coherent. Two
metadata-only preexecution gaps identified: topology dimensions/ranges lacked
typednativeSHA, frozenarrays did not yetvalidate6branch/3replay/bbox/K
metadata. Fixthese before anyqualityconsumer; neither gap authorizesH101
execution or weakens actualcall/SHAM requirements. No new quality observed.

Actual H100c mechanism PASS23.146740s, producer18c605dcf0b045db1472d6de6e5d9b95dd17283f,
driverf14b882e5074d5dd59d2b76d179ce0f96d05ae0c3acc0269fe84338156bad0bd,
receipt09434f40e41f772f8e718465a16c720d8eedfebc5d74f8a80334208083f4ba50
(51,610B/0444). Actual6Body/6parity/6KP/6fixed/3selected and57scopednative
heads complete/validated/synchronized/restored; methodrestored, fullsource/
asset/RGB/mask rehash and15frozenartifacts reread. All3RAW+FIXED SHAMexact
byteidentity; index0/scores0. OriginalRGB medoid index0/gamma1; pairwiseVmean
square distancesoriginal→gamma.8=8.263913e−6m²,→gamma1.2=1.932132e−5m².
Mechanism works but thissinglecase makes nochange andprovides NOaccuracy
evidence. JointstrictMHR+unoptimizedJIToperationalroute qualifiesfixtureonly,
notcausalattribution/crossprocess/allframes/quality/adoption. H100/H100bFAIL
remainimmutable; no rescoring/privatetruth. Unit inactiveexit0/lastGPU125MiB0%.
Next actualscience requires NEW24human-only pairedquality preregistration,
SAMEscopedexecution on every method/nativehead, all24/8groups retained and
existingSHAM/native/sourcegates; never use thisindex0 toadjustgammas.

## H101 — untouched human-only photometric quality preregistration

Mechanism H100cPASS required and nowobserved; noH101RGB/prediction/quality
exists yet. Frozennew24rendererrecipe above unchanged exceptactualnativejoint
name `l_lowarm`. Use nativepolicy exactlyH100c: ordinarylearnedCUDAFalse/
warnFalse/seed0/TF32off/CUBLAS:4096:8, allMHRheads strictTrue/warnFalse/JIT
unoptimized withsynchronize+restore includingintermediategeometry. No new
weights/gamma/camera/objectrecipe search. Reference-onlymanufacture120s,
automatic24personDINO/SAM masks120s, learnedBody144calls/native600s, final
private diagnostic120s; each32GB/4CPUofflineH100, separateexclusiveoutputs.
All heavyRGB/geometry/models remainAzure. ExistingH99andH100failures immutable.

All8groups2morph×2appearance×2occlusion/3realadjacentposes counted, noinvalid/
confidence/visibilitysubset. Publiconly24RGB+manifest; masksautomaticperson.
queries(originalRGBonly), fixedwholegrid/bbox/K1280. Sixbranches/frame ordered
original,gamma.8,gamma1.2,sham0,sham1,sham2 (latterRGBbyteexact). Clipfirst
originalshape45/PCA28 sharedall3methods/3poses; frameoriginalEulerroot/Tcam/
hands108/facezero sharedmethods, BODY133onlyproposal. Expanded68scales
clipbyteconstant; actualfixed0:6/68:122/136:204 checked. BODY133 includes
sixskeletaltranslationcontrols130:136, not purearmrotation or allvertexZfixed.
144Body/144freshparity/144KP/24frameanchor/144fixed/72selected replayheads,
1,392scopednativecalls (58/frame), allactualattempt/completedcounts required.
360NPZ(6raw/6fixed/3selected×24)+actualnativefaces/LBShandmetadataimmutable
andreread beforeprivate. RAWandFIXEDSHAM3allarraybytes mustbaselineexact,
SHAMzero/index0; independentselectedreplaysV/KP/J/controls/R≤1e−5. Selection
existingfull18439cameraVmedoid3gammas, F64meansquared distances/sum/firsttie,
no average, imageconfidence, IoU/GTselection, camera fit or label-driven prior.
NativefaceI32hashf674…acd6/source/semantic/modelpins andallsourceasset/RGB/
mask pre/postrehash. Anypublicmechanism/coverage failure stops BEFOREquality.
If everyframe selectsoriginal with no actual geometrychange, retain outcome
as no useful mechanism/no improvement, never tune gammas onthiscohort.

Freeze render/mask/body receipts/scripts/canonicalproducingcommits inone
explicitqualitypinsconfig AFTERproducerPASS andBEFOREprivateevaluation.
Quality firstvalidatesall24×15arrays/publiccallprovenance/native1392scopes,
not onlyJSONstatus. Onlythenreads ownprivatesynthesistruth9fields/rig.
No producer/inference receivesprivatecamera/pose/shape/visibility/geometry.
Oneproperpositivefull18439Sim3 from SAMEconstrainedbaselineframe0 pergroup,
sharedALLmethods/ALL3poses; no perframe orcandidate-specificalignment.
RawcameraPVE/centered/centroidXYZ, alignedfullhumanV/J andseparateLBShands
reported. Primary equal3posemean within8groups: pairedrelativealignedhuman
PVEgain median≥5%, everygroup regression≤5%, everygroup/EACHhand aligned
PVEregression≤5%. Zerobaseline relativegainundefined =>rejectrelativegate,
retaincase; nohandabsoluteincrease allowedfromzero. SHAMmetricsmustexact
baseline. Also every24frame TTArawcamera humanIoU against SAMEfrozen
automaticmask mustnotdrop>1pp vsbaseline;72complete scalarGPUrasters
baseline/SHAM/TTA, noSim3forsilhouettes/noGTvisiblemaskgating/occlusiondrop.
All24scoredONCE evenifqualitysafeguard fails; no rerun/rescue/thresholdsearch.
Jointsecond-difference across3adjacentposes is descriptivenofps², not a
fluiditygate oroptimization; zeroaccelerationdoesnotproveaccuracy.

ExecutionPASS andsciencehypothesissupport reportedseparately. Thishuman-only
syntheticpilot NEVERauthorizesfullHOIsubmission/adoption or CARI4Dranking:
noobjectprediction/contact/CD-O/PENmeasure, nonphotorealisticmanufacture and
knowncamera limitations explicit. If scientificrejection, do not retuneH101:
priority separatelypreregistered nativefullHOItemporal96+realframes with
validinferredmetricobjectmesh andsameidentity/camera, then realdomain checks.
Finalpacker/Parquet/all30clips/sharedHOIframe/licenses/registration/Gitcommit
accessibility stillrequired. Preregisteredbefore firstH101manufacture.

H101 preprocessing/metrics/scoped-policy73focusedtestsPASS0.71s while
finalnativeobserver metadata+samequalifiedscopedroute andprivateevaluator
are beingcompleted/audited. NoH101manufacture, inference or quality dispatched;
allgate thresholds andnewrecipe above alreadypreregistered.

Preexecutionnumericboundary fix: evaluate inclusive5%regression as
after≤before×1.05, inclusive1ppIoU asafter≥before−.01, avoiding cancellation
at exactlythethreshold. Samepreregisteredthresholds/noepsilon/newtolerance;
nextafter-overboundary rejected,17purepolicytestsPASS0.04s. NoH101qualityseen.

H101independentscience/rules auditREADY beforemanufacture:8groups aretwo
parametricmorphs crossedwithsamegeometry nuisance variants, NOTeight
independentpeople or statisticalSOTA/real-domain evidence. Gammaaffectswhole
RGB includingbackground, so evenPASS wouldnotisolategarmenttexturecausality.
KnowntrueK matchesgenericprior bydesign, nolearnedcalibrationproof. Full18439
baselinealignment differsfromofficialsubset; fixedhandcoefficients canstill
movehandpositions throughbody, hencehandsafeguards. Keepallrawcentroid/
metricgauge diagnostics even ifalignedgain passes. Existinglicenses/
trainingoverlap remain unresolved; norenderer/SHAMPASS clearsfinaleligibility.
127focusedmanufacture/policy/bundle testsPASS0.56s; observer/eval source still
beingcompleted, noH101job/GTscore. Budgets/gamma/coverage/gates unchanged.

H101 final pre-execution observer/evaluator audits completed: qualified scoped
route and all360 artifact/metadata checks, actual native topology/LBS lineage,
baseline-first shared Sim3 and complete24/72 raw silhouette diagnostics. The
independent audit caught a uint8 0/255 mask incorrectly treated as boolean in
the evaluator; corrected explicitly before any manufacture/GT, with a real
PNG/reader regression test. No threshold/recipe/selection change. Final159
targeted testsPASS18.98s; earlier204 observer/policy/bundle testsPASS12.35s.
One full-suite run started while the new fixture was being corrected failed
its moved-path identity check (6510PASS/1FAIL/1optionalSKIP), not a GPU/quality
outcome; final frozen full suite is being rerun. Azure preflight: freshH101
namespace, H100NVL95,830MiB/125MiBused/0%,574GBfree, previousH100c exit0.
Only code/scalars transferred; no challenge/private quality queried.

Final frozen local full suite6511PASS/1optional-trimeshSKIP96.93s, nofailure.
Committed full Git-metadata code closure: manufacture20files52,508B,
observer36files116,008B, evaluator40files134,188B, allbelow160KBcontrolbudget.
H101 ready for first manufacture/public inference; no private quality or scores
yet. Launch manufacture/masks/body from the same frozen source revision; record
their immutable scalar receipts only after body dispatch, preserving that pin.

H101 first manufacture **PASS10.372355s**, source
`05ef12488cc10bce68c233a92ea73a71005c44de`, renderer
`359e1fe495fa5ac7dc87d482a8ad90a6074455a9f68690099549eb2433da3eee`.
Receipt`62960a7f30d1ea7c58a4c1fba320b9f06ae01e8d889a0c836a0d73ddf91010bc`
(29,468B/0400):24RGB/24rasters,2reference+1bundled native forwards,
allfactorsgeometryindependent/sourceassetsrechecked, actualnamedl_lowarm76.
Public manifest`2e100730f0d32fcc29fca9c5f317de7afc28e2f63cbd8e1f7218d6319abbe0d4`
(3,515B). Unitinactiveexit0; no RGB/geometry copied locally/privatequality.

H101 automatic masks **PASS17.316456s**, SAMEsource05ef124…/observer
`2c68d9b12da166b36cc28c29ab93cfebca5960ed917339d19b12f524e0f553da`.
Receipt`4ac9cb33bf5478d62430d6346a925d776116895472c79511d9a819131a44276d`
(19,433B/0444):all24automaticpersonDINO/SAM2/encodercalls andmaskoutputs,
fullpublic/source/assetsrehash, no privateinput. Unitinactiveexit0/GPU113MiB0%.
Body observer nowdispatched from SAMEimmutable05ef124…bundle116,024B,
unit`world-reward-h101-human-photometric-body`;600sbudget/144learnedcalls/
1392strictnative scopes required. Quality pins NOTyetcreated; waitallpublic
mechanism/replay/provenancePASS, noaccuracyquery orproducerrestart.

H101 Body actual **PASS121.499012s**, SAME05ef124…observer/runtime source,
receipt`3ca209a3421583850d582b7e0309b1f943b8bb4c56ee72404b94b09020d0ba60`
(684,258B/0444), unitinactiveexit0/GPU107MiB0%. All144Body/144parity/144KP/
24anchor/144fixed/72selected and1392scopedattempt/return/validation complete;
all24RAW/FIXEDSHAMbyteexact, methodrestored, all360arrays/rigfrozen+reread,
fullsources/inputs/assetsrehash/no privateinput. Publicmedoid selected original
22/24frames; gamma1.2group01frame0, gamma.8group06frame2. This is NOTaccuracy;
two changes are retained and entire24quality cohort mustcount, no gamma tuning.
Explicit actualrender/mask/bodycommit/script/receiptSHA/bytepins nowrecorded
in configs/human_photometric_quality_pins.json BEFORE any privateevaluation.
Next ONE120s privatepaired diagnostic; all8groups/24frames/72rawrasters,
sharedbaselineSim3/hands/gauge diagnostics and predeclaredsafeguards unchanged.

H101 single private diagnostic actual **executionPASS/scienceREJECT26.073962s**,
source`b745d797886aa1a813928ede19d9027a343f9830`, evaluator
`9f9cf0e096735108988932607c47d3274c65b9a91b32dc3361f153377f9df57d`.
Receipt`628a434b76d23b2b1c0b7e288854bc54067c5fcd25f615e8245b11e97bb912c4`
(43,620B/0444), unitinactiveexit0/GPU101MiB0%. All24scored/72rawcamera
rasters after fullpublic360/provenance/1392scope validation, zero learned/
optimizercalls, allpublic/private/sourcehashrechecked. SHAMmetricsbaselineexact.
Median pairedgroup humanPVEgain **0%** fails≥5% primary; group01 regresses
0.0695632%, group06 regresses0.135115%, other6unchanged. Equalpose/group mean
alignedhumanPVEbaseline[2.901889,2.601242,2.621744,2.610424,2.062002,2.244002,
2.586449,2.669548]cm; TTAchangesonlygroup01→2.603051/group06→2.589943cm.
Group/eachhand5%safeguards andeveryframe1ppIoUsafeguard pass; worstIoUdrop
−0.00152953(−0.152953pp), maxhandabsoluteincrease0.0763622cm. Safeguards are
not efficacy evidence: two changedproposals gotworse/no supportinggroups.
Decision: abandon gamma-medoid as a useful accuracy method, no tuning/rescore
onH101/noadoption/noHOI/domain/CARIclaim. Keep exactruntime mechanism for
future nativeexecution, not method-levelimprovement. Next minimum public
96-realframe constrainednativeHOI pilot/generalproductionpath; independent
FULL-HOIclearance still separate. Existingreports/privatearrays retainedAzure.

## H102 — public96 shared-identity native HOI engineering pilot (preregistered)

Purpose: unblock a complete native reconstruction/export path, **not** evaluate
accuracy on challenge video. One untouched chronological prefix of episode15,
original frames0..95, no padding/drop/reencoding. Reuse exactly15 SHA/size-bound
automatic Track1 sources: original501 Body parameters, masks/RGB/depth, one
inferredK1920, predicted common metric scale, fixed aligned object mesh and
ICP/Viterbi rigid poses. No historical CoCoNet/refined outputs or private labels
are inputs. Azure-only/offline fresh `validation/cari96_public_v1`; old outputs
remain unchanged. Explicit source pins in `configs/cari96_input_pins.json`.

Before native geometry/neutral-height/materialization/caches, choose shape45 and
scalePCA28 from original frame0 and repeat byte-exactly across96; preserve all
moving root/body/hands and zero expression. No quality-based identity selection.
This is a **constrained-identity ablation**, not unchanged original CARI4D.
Preparation does no learned inference, fitting, image rendering or scoring.

Predeclared preparation gate:180s, four routes each six16-frame chunks: new native
V18439/J127/KP70; direct native204 parameter replay; independently decode the
saved shared initializer again; official reference FP32 model/FP64 residuals.
Native/direct/saved-replay maximum point error≤.01mm. Official reference mean
point error≤2mm **every original frame**, same earlier fidelity threshold;
reference maximum point error retained as a diagnostic, never substituted for
that gate. All geometry finite/positive cameraZ and exact native topology;
expanded68 scales clip-constant. Complete attempt/return counts, source/model/
helper/initializer and new artifact pre/post hashes, saved arrays/metadata/frame
indices exact. Historical raw-initializer projection/roundtrip checks are labeled
historical, not new shared-identity projection claims.

Three independent source/ABI/firewall reviews READY, no concrete blocker.
HDF5 tests initially exposed only a missing temporary parent in the tiny success
fixture; corrected that fixture, production exclusive-output behavior unchanged.
Actual tiny HDF5 read/write/payload-reread now executes locally after installing
small pinned h5py3.14.0/joblib1.5.2 dev dependencies (no model/data download).
Final90 focused testsPASS0.31s/noSKIP including all10HDF5 cases and historical-
metadata/attempt-ledger regressions. Frozen full suite6601PASS/1optional-trimesh
SKIP103.56s. Azurepreflight: namespace/unit/log absent; previousH101quality
inactiveexit0, H100101MiB/0%,574GBfree. NoH102 GPU job, quality query,
submission or efficacy claim yet; commit/code closure checked before dispatch.

If preparation passes: one native96-frame CoCoNet window with original config/
checkpoint and raw outputs unchanged; only identity deltas zeroed on an owned
composition copy. Then unchanged native refinement and direct full consumer
replay/export. Require fresh explicit preregistration/provenance for each stage,
no reuse of prior predictions as a new result. Independent FULL-HOI validation,
license eligibility, NVIDIA registration and accessible GitHub producing commit
remain separate from an engineering PASS and required before final submission.

H102 preparation actual **PASS19.676357s**, immutable source
`03ccfa1de49b5b13605a304f4dfb394c16f54ea5`, script
`ebd0378a92ba4474b65e4f7743499915a1dd7668a697c1bf24f64fb5ad8dad84`.
Git-runtime closure16files34,112B encoded, SHA-XZ
`e653c6708f391899edbdbe0c58f2375686f43019b6fe372ebe726f236f41c161`.
Unitinactiveexit0/GPU95MiB0%; allfourroutes6attempts/6returns: all96 new native
geometry, direct204, savedinitializer V/J/KP replay and official reference.
Sharedidentity/sourceassets/helper/newoutput pre/post rehashPASS. Official
mean/worstframe0.000419625/0.000843682mm; maxpoint0.002371742mm diagnostic.
Allstoredpayloads/attributes reread, fixedK/metricobjectgeometry unchanged.
Engineering ABI only; no accuracy or challenge quality claim. Next native
one96-window composition-constrained forward; original AMP/ordinaryCUDA policy,
no newly introduced class/factory hooks or claimed deterministic network replay.
Prepare immutable receipt
`26467461692e44ed5c4f595653b3e32a73d9e585ba2065a1565c77c36a0e9618`
(8,090B/0444), all13 actual outputs pinned in `configs/cari96_prepare_pins.json`.
The first readonly scalar-pin query had a Python dict syntax typo and returned
no pins; corrected query verified all13 output SHA/size/readonly and receipt
links, no producer rerun or geometry change. Source/public RGB never downloaded.

H102 next-stage gates preregistered before dispatch:
- Forward:360s, original commercial checkpoint/config/native96-window/stride96,
  render32/crop8/buffer2/no input cache, native AMP thenFP32. One original
  function-code clone overrides only composition lookup; shape45/PCA28 deltas
  zeroed on a fresh copy, raw original deltas and original global unchanged.
  All96 outputs, rigid object poses, original hands/face, clip-constant identity,
  complete automatic masks/contact, checkpoint/decoder/K/mesh/source/pins/raw
  pre/post hashes and saved bundle reread. OrdinaryCUDA/noWarn/TF32off;
  no deterministic forward replay claim. Exact1attempt/return/verified/hook.
- Refinement:1200s, unchanged native full96 public parity config300requested/
  301actual updates. Optimize original body rotation controls and objectT only;
  all rootT/rootR/hands/identity/face/internaltranslations/objectR/K/mesh fixed.
  No new strict backward policy, original collision4000/hand assets. Exactone
  optimizer attempt/return/verification; nativehistory0..300, savedreread,
  source/raw/observations/mesh/pins/decoder pre/post hash and fixedblocks exact.
- Direct export:180s, decode refined7blocks **without another identity change**,
  six16-frame native V/J/KP/direct204, freeze pose136/scales68/shape45/zeroexpr
  and same alignedlocalobject mesh/scale1/poses; six storednative and six
  official-reference replays. Native maxpoint≤.01mm/reference mean≤2mm each
  frame, complete attempt/return counters/source/output hashes and strict96
  schema/object camera geometry checks. No LM/inverse fit/alignment/model
  inference/quality query. Aligned local mesh frame is valid because human/
  posedobject remain in onecamera metric frame; no object shrinking/remeshing.

All outputs are public engineering96-prefix artifacts, **not** complete501
episode predictions, independent accuracy validation, final Parquet or verified
victory. Each subsequent dispatch must first bind actual preceding PASS receipts
in immutable config, inspect gates, run tests and commit a clean source closure.

Final H102 chain checks:240 focused testsPASS1.46s; fullsuite6750PASS/2SKIP
106.29s. The two skips are absent optional local trimesh and the absent tiny
official-kit fixture, independently confirmed by60PASS/2SKIP0.38s; neither
skips any H102 contract. Independent pinned-primary-source ABI/mount audit READY
(170,436B of source only, temporary cache removed), allthree wrappers bash-nPASS.
Prelaunch corrections: accept owned nonmasked NumPy memmaps, cast actual native
I64 faces to canonical I32 for topology hashing, replay allseven saved native
blocks independently, and require exact refinement additions while preserving
all observations/raw/contact/pr_initial/fixed blocks. These are algorithm-level
ABI guards, not per-episode labels or quality adjustments. Azure preflight found
forward namespace/unit/log absent, preparation inactiveexit0, H10095MiB/0%,
573GBfree. Next one immutable forward dispatch after clean Git closure check.

H102 first forward **FAIL before network0.151416s**, source
`cd7d3b9c7d17c62d16c5dd7596ccba038ffa718d`,21-file50,652B Git closure
(SHA-XZ6506d855c61804b0650a2175b274d3c1fe3cfc1c17e415bada5b347e0b9bed63).
Attempt/return/hook/Hub counters all0; unitfailedexit1/H10095MiB0%.
Cause: code-relative provenance literal `infra/run_cari96_prepare.sh` was omitted
by the archive selector, which previously followed sibling literals/imports only.
Correct selector rather than bypassing the generator hash: include exact infra/
own-package code-relative literals and their full transitive closure, fail closed
if absent. Added synthetic and allthree actual H102 closure regressions. First
failed namespace remains frozen; new `validation/cari96_forward_v2` is reserved
for a new source-bound attempt with identical data/weights/numerical gates.
Failure receipt d6271c5c0c68c785c375d91b1997e0c88dcc80f6c0588c599db0064721a7f324
(1,177B), driver e419724d701552d167fa0a8ad9d7c721109cf58ed695dae726d9b31904a762e4.
Corrected archive/gate checks311PASS4.22s; fullsuite6753PASS/2sameSKIP105.27s.
Additive original-timeline helpers separately82PASS0.11s: full501 has actual
windows0,96,192,288,384,405, first-occurrence terminal ownership480..500,
finite last decoder chunk5, actual unchanged caller indices validated readonly.
No H102 generator or numerical gate changed by those helpers; not yet executed
on full video. Newforward preflight confirms absent namespace/unit/log, old
failure terminal, H10095MiB/0%. Recheck actual committed closure before dispatch.

H102 forwardv2 **network returned / validatorFAIL66.929384s**, source
`2d0fa9ac9b68a16bfaf18c50eeab5efc5d7bfc64`, driver
99854a70cb6e705cf8028b454d65356904711ad7750d6bc589d084f7c339b6ca,
22-file50,808B Git closure SHA-XZf25133cb1daa216424b9705027bf819344933c69704ee2a659f63c776f27f98a.
One forwardattempt/return, twoHubloads, onehook/delegate/verified complete;
no final forwardPASS, refinement or export. Failedreceipt
1b9897e5043cb03acb5c3c36d2e0ec97fe82212296f6b9234f2e114d2dd2f0f7
(5,808B), nativebundle3241f42401904d7b9ae0c493eff54fdec84c523c520b26d28e3d33b9aa99d996
(73,829,436B) retained Azure only. Readonly scalar inspection: hands byteexact;
face differs in signed-zero bytes only, all6,912 values remain exactlyzero.
Independent source audit: frozen hand deltas explicitly zeros_like; face has no
head; pinned delta.py always performs FP32 init+zero, including−0→+0. Correct
validation to require EXACT native FP32 addition bytes plus numericalidentity
and exact frozen rawdelta zeros, with no epsilon/restoration/output changes.
Also preserve an unused Infinity training-config default in source/nativebundle
but record only finite active inference settings plus full untouched config
fingerprint. No change to upstream prediction or active inference settings.
Seven regression cases added; focused357PASS2.00s includes generic fullidentity
contracts (129PASS0.11s separately). Newforwardv3 namespace, no oldbundle reuse.
Readonly Azure operation check confirms exactnativeaddition forbothblocks,
rawhanddelta positivezero bytes exact, faceheadabsent,4,267initializernegative
expressionzeros become positivezero; no numericalchange. Fullsuite6980PASS/
2sameSKIP118.34s. Newv3 namespace/unit/log absent/H10035MiB0%; no failedproducer
rerun in place. Each fresh attempt uses unchanged public96 sources/gates.

H102 forwardv3 **FAIL before network1.936760s**: newly recorded supervision
source had an incorrect directory (`lib_mhr` instead of actual
`learning/training`). Source15a3d46f79ba113bd07fa2de29bb52910d296897, driver
6b14fc02ffd9e96c5a2ebc4bb578cea76ac7521824fd1524aca11c6a11b0f6be,
receipt ef422d5196adab781b96f90df7020f3b4b5cabb00f9fa376ce6d56f50ea5098a
(1,172B), zero forwardattempts. Exact primary import and remote file SHA now
verify `learning/training/mhr_supervision.py`,20,826B/e7f4c5f8991e312eec69a760204758d44edaf9f29f9456ec181d288241686635.
Correct path and add regression; no source/gate/model/numerical change.
Newv4 namespace/unit/log preflight absent; focused228PASS2.25s. Additive generic
fullclip source auditor independently155PASS6.33s; no new GPU/data read by it.

H102 forwardv4 actual **PASS72.173626s**, source
`d8513283a92d870db75184f64c39fd1262fde9d7`, driver
37ade3cc40cd06f7857934445beea9175286df33162e13cb7e58a70e96c2bcf7.
22-file51,640B Gitclosure SHA-XZ2e1180a925f2bdc03649de96b5464c9d96b67e7c20fd9b29e2c44475711f3823.
Oneforwardattempt/return/verified andonecompositionhook/delegate/verified,
twoofflineHubloads; all96 source/identity/K/masks/contact/raw/asset checks plus
savedbundle rereadPASS. Unitinactiveexit0/H100103MiB0%. Immutable receipt
8a0cdf1b5aec4c67e47a775b473dc183bfd20e18af610ea15fb5e5acc7514fc8
(50,377B), nativebundle6e9a8b00b781a411753f629e01b9abf7e2d93015d81d557b5e29037f9675829c
(73,829,436B); actualcompleteinventory pinned before refinement. No strict
network determinism, independent accuracy or full501 submission claim. Next
one unchanged full96 native300requested/301update refinement. Generic source
audit156PASS1.70s separately; newfull501 inputpins referonlyexistingpublic
15sourcefiles, neverold CARI predictions. No newfull501 GPU run yet.

## Full-video shared initializer — preregistration 2026-10-03

Additive full-N producer, first execution reserved for original episode15/501
frames. Reuse the same fifteen SHA-bound public input files without copying
RGB, masks or depth. Explicit episode and source pins; exclusive new Azure-only
`outputs/episode_000015/cari_shared_prepare_v1`. No former CARI predictions,
GT, inference, optimizer, rendering, inverse fit or quality-based selection.

Choose original frame0 shape45/PCA28 before model construction, geometry,
neutral-height or caches; preserve all other native blocks and original frame
indices. Re-decode joints/keypoints, retaining previous projection/roundtrip
claims as historical only. Budget600s; four actual routes over32chunks, the
last containing exactly5frames: native V18439/J127/KP70, direct204 controls,
frozen saved native replay, official FP32-model/FP64-residual replay. Same native
max-point≤.01mm and official mean-point≤2mm **every frame**, exact topology,
positive cameraZ, clip-constant expanded68 scales, source/helper/model/artifact
pre/post hashes and saved-byte reread. No score, efficacy or submission claim.

New payloads only: shared_initializer.pkl, direct_parameters.npz, target.npy,
plus concise source-bound receipt. All three payloads frozen before replay.
Original HDF5 and rigid object geometry/poses remain unchanged read-only inputs.
43 tiny callback testsPASS0.23s, Bash syntaxPASS; root removes draft-only absolute
test paths. Combined focused gates366PASS3.60s. An initial test command named
the wrong Azure test file and executed no tests; corrected without source/gate
changes. Additive archive regression requires all ten provenance helpers and
fullclip pins. Independent final audit and fullsuite precede actual dispatch;
H102 direct export remains separate and is executed first.

Independent full-video review **READY**,419focused testsPASS1.87s; native
20,514B primary decoder source SHA/constructor/direct-head/flip/translation ABI
confirmed. No workspace/model/data writes or duplicate GPU jobs by auditor.
Root fullsuite7180PASS/2same optional SKIP106.26s; no shared-prepare contract
skipped. Actual source archive/clean commit is checked before dispatch.

H102 native refinement actual **PASS48.828237s**, source
`2feff1a8e2170da3312bfc26d961e37b977687ad`, driver
`ece3ff2c42303c2f6e9237ab78cb53b61fba40af261ac744e511b170e8f11ed6`.
26-file70,688B Gitclosure SHA-XZ
`219aa4b4175cee603f60c8d38c889e9587d8ebd5b328230bc24b52b3720696d6`.
One optimizer attempt/return/validated,300requested/301actualupdates, all96
original frames. Every fixed parameter/raw/input/observation/contact/pr_initial
and aligned object geometry preserved, saved bundle reread and source pre/post
checksPASS. Unitinactiveexit0/H10097MiB0%. No empty automatic observations in
this prefix. Receipt44d4a3dd5bc20a7f9be629ef1cad7a46d199a7447335af07fecee2e160e6cd2d
(5,834B), refinedbundle07c614481fcdb9d13a930d39e0a4b6f34952f235686f10cb7b8bdb4128df2730
(73,840,928B), all0444. Actual pins frozen before export. A readonly inventory
query used unavailable Python3.8 hashlib.file_digest; rerun with streaming SHA256,
no artifacts/producer changed. Export and full501prepare namespaces verified
absent. Engineering execution, not independent quality/CARI4D improvement.

H102 direct export actual **PASS16.485373s**, source
`78b9005471496448acc38fa7308940b0093306df`, driver
`7c46111d623ee4c32154da2e151ff0465edee812c1d03c5dca1fd14eb5519372`.
Actual Gitclosure31files80,172B, SHA-XZ
`7eaa2f3fe4780f2052b9e69e29453bbe8d42cc315cfc1ca9c66d4c4e35cbf38b`.
All four routes6attempts/returns/validated; full96 official replay
mean/worstframe0.000413737/0.000770197mm, maxpoint0.002632832mm diagnostic.
Original refined7blocks/sharedidentity/raw/contact/masks/source/assets/fixed
aligned2044-vertex object/scale1 unchanged; object camera roundtrip exact0m,
savedparameters reread and strictTrack1Episode96PASS. Receipt
95b432ba4b91c2cb10625da2488bcf5460b4aaf30283ce0e08bc4b00e5e481df(7,526B/0444).
Frozen trajectory6d5c998c30722bf2908d5cedd77170edce316c3ec016af5c403a18d716816718
(97,381B); nativearchive507ec525fb535656fde0857f5a6cc089ffd950f1512443b2abcc4e659c061678
(473,077B),targetf1331b114fd48f2fab755fe2606c2152a4882aac7d0eea14b56d54bb462e6b38
(21,241,856B),unchangedGLBdb97398bf5c1a45eca41ca42d6b521408a92a89d9618563b39b7c3ac889d7b32
(74,596B). Unitinactiveexit0/H10091MiB0%. No LM/newfitting/identitychoice,
complete501 reconstruction, finalParquet, accuracy result or CARI4D victory.

Full501 shared prepare dispatched from same clean78b9005 source: actualclosure
25files50,288B, SHA-XZdb8cbc83a22dbebecc3e58c2744c80b09ee8d31482c3988fdbb161b9927f3be1.
No old full501 CARI outputs reused; complete public sources remain Azure only.
Future clean-episode input producer now records the dataset revision already
verified by its input audit. Do not edit historical receipts or fabricate a
missing historical field; legacy acceptance remains exact-hash-specific.

Full501 shared preparation actual **PASS19.863002s**, producer78b9005,
driverba5d431ace4a55f1bfcc3f5bb926685a5270fa206b8a7bef831aeeeaa7c1d0b6.
All four routes32attempts/returns/validated, last chunk5, full original501
coverage and source/model/helper/identity/savedpayload checksPASS. Native
direct/replaymax0.002870950/0.000953791mm; officialmean/worstframe
0.000425200/0.000843682mm, maxpoint0.002709790mm diagnostic. Receipt
cd17771616be2f2aff97f840b750e6da4beee78b17e5dc614a2ca60ba495a7de(23,871B/0444).
Unitinactiveexit0/H10085MiB0%; actual complete outputs pinned before forward.
Preparation fidelity is not reconstruction accuracy or a submission result.

## Full-video HOI production chain — preregistration 2026-10-03

First execution: full original501 episode15, same automatic public sources,
fresh exclusive outputs `cari_shared_forward_v1`, `cari_shared_refined_v1`,
`cari_shared_export_v1` under that episode. No older predicted bundle is an input.
Stage-specific complete pins bind each actual passing predecessor before the
next dispatch; keep existing c96 helpers and prepared full-video generator intact.

- Forward900s/outer903: original native96 windows/stride96/terminal405 and
  first-occurrence ownership96/96/96/96/96/21; one overall native forward,
  six composition/delegate/verifications, two offline DINO loads. Identity
  fixed before construction. Original native initialization decoder batch8
  (full501tail5), rendering32/crop8/buffer2/AMP and config/checkpoint unchanged,
  no materialized input cache. Read actual pinned caller indices at sole
  composition hook, corroborate all7 initialized blocks/object input poses and
  decoder identity, capture untouched raw/composed/contact for each window,
  compare exact owned assembly and saved reread. Only identity deltas on owned
  composition copy become zero; native original global/network unchanged.
  Ordinary CUDA, not a claimed exact deterministic network replay.
- Refinement7200s/outer7203: one unchanged full501 native optimizer,
  batch0/fullclip300requested301actual updates, report_every100. Preserve
  original native floating history0/100/200/300, full batch at every record;
  only original bodyrotationcontrols[:254] and objectT optimized. All remaining
  params/root/hands/shape/PCA/face/objectR/internaltranslations/K/mesh/raw/
  input/pr_initial/contact/observations fixed; original proxy4000/hand assets.
  Saved full-bundle reread and source/assets/helper/predecessor hashes intact.
- Export600s/outer603: no new identity choice/inference/optimizer/LM/alignment.
  Full501 finite16frame chunks, last5, four32call routes: native V/J/KP,
  direct204, stored native7blocks/V/J/KP/topology replay, official FP32-model/
  FP64-residual replay. Native≤.01mm maximum point and official≤2mm mean
  **every original frame**, same fixed gates. Freeze full pose136/scales68/
  shape45/zeroexpression72 and actual aligned local object geometry/scale1/
  nativeposes/inferredK, strictTrack1EpisodefullN and object camera roundtrip.

All three drivers are generic explicit episode0..29/N>=96, not changed constants
or stale prefix arrays. Public inputs/masks/depth/model weights remain onAzure;
only new output writable/offline, no old CARI or other tracks/GT access. Separate
source-bound stage consumers rehash original15, actual predecessors and code/
native/reference/optimizer/model assets. Engineering PASS does not establish
external accuracy, license clearance, finalParquet or a CARI4D victory.

Independent disjoint implementation/source audits followed by root review;
combined generic forward/refine/export262testsPASS1.75s/noSKIP, actual peer
module imports included. All three wrappers BashPASS. Correct actual native
history iteration dtype(float), original body asset location and fixed aligned
object metadata were verified against primary source before dispatch, not
repaired in predictions. Complete generic closures stay below160KB code-only
control ceiling; archive regressions require previous producer provenance,
shared identity/timeline and actual full501 preparation pins. Fullsuite and
actual Git-metadata closure are checked before any full-forward GPU execution.

Root fullsuite **7443PASS/2same optional SKIP120.47s**, no generic stage skipped;
three added archive regressions plus generic gates337PASS2.65s. All ten actual
full501 preparation helper bytes remain unchanged from its78b9005 generator.
Forward fresh namespace/unit verified absent with actual frozen preparation
inventory. Launch requires the clean producing commit, no uncertain retry.

First full-forward dispatch fromfb727b7 failed **before output creation or GPU
execution**: host bootstrap imported `cari_full_forward`, which imports NumPy;
the VM's control-only Python3.8 has no NumPy. Unitexit1, no forward receipt,
H10085MiB0%. Exact traceback established the cause; no predictions or prepared
inputs were changed. Remove only the numerical producer import from the host
path enumerator: original fifteen input pins remain checked there, and complete
prepared-payload validation remains inside the pinned container before Torch.
Do not install numerical/model dependencies on the control host or relax gates.
Four actual stage bootstraps now execute with `python -S` and numerical/model
imports forbidden; valid paths and invalid episode/hash gates covered12tests.
Combined bootstrap/fullstage/archive **349PASS5.26s**. All prepared generator
helpers and full forward Python driver are unchanged. Retry uses a new unit/
log and clean producing commit, retaining the still-absent exclusive output.

Full501 forward retry actual **PASS118.209749s**, producer
8c338c0d3feff3f8a1e6ff5bd6a767fedc94908d, unchanged Python driver
ac07d96a6bb8bad8aa8977b74f7b266600bbcc26fc92bf07ba6419be0142fd2f.
Actual32file76,016B encodedGitclosure. One overall forward/return/validated,
six captured/compose/delegate/verified windows0/96/192/288/384/405, owned
96/96/96/96/96/21; original501 complete, actual caller/initializer/object-pose/
decoder identities and raw/contact assembly preserved. Two original offline
DINO loads; source/predecessor/assets/helpers pre/post and saved-bundle reread
PASS. Receipt42999af278aeb9d0c9bf477e60df320a6b3d04470e477dec75bd3cdd88017c18
(68,584B), bundlec20c4269c3605b181cd566d2af0164ff9ee72afe2017614a257a47daa849fb68
(383,360,892B), both0444. Unitinactiveexit0/H10025MiB0%. Actual complete
pins committed before native refinement. No GT, old prediction/cache, quality
result, finalParquet or CARI4D superiority claim.

Additive stdlib-only stage inventory emits tiny source-bound next-stage pins:
independently supplied actual producer revision/script SHA, complete readonly
inventory hashed before strict JSON and after, no model/geometry loads or file
mutation. Python3.8 dict-union runtime incompatibility caught during root review
and removed;137testsPASS plus all four real bootstrap testsPASS149 total0.66s.
It inventories prior numerical execution, not independent geometry or quality.

Additive full-native episode consumer, not old LM conversion: exact five
externally pinned export files rehashed before JSON/trajectory-only NPZ load,
actual generic predecessor/source/helper chain and unchanged aligned GLB
verified. Full eleven-key owned FP32/I64/F64 trajectory retains501original
frames, shared identity/scale1/zeroexpression/inferredK and strictTrack1Episode
schema/positive-camera object roundtrip. Other native archives/target/models/
videos remain hash-only. Independent48test audit followed by root combined
inventory/bootstrap/fullstage/archive/consumer **534PASS3.32s**. No actual full
export or independent quality demonstrated by these tests. All numerical
producer helpers remain unchanged; actual clean Git closure precedes refine.

All30 integration audit: existing clean route still ends in old unconstrained
forward/LM conversion; a separate frontend-only route and actual source-pin
inventory are needed, never implicit reuse/overwrite. First audit all30 original
N/HW/fps/SHA and readiness with no media transfer or label reads. Hard occlusion
limits persist in body crops, fixed scale anchors, object seed/depth support and
input-mask prep; downstream zero-mask acceptance is not a frontend cure. Keep
automatic masks unchanged, full timeline and confidence explicit. Do not add
filled masks, static/interpolated-only final poses, test-specific thresholds or
frame deletion. General bidirectional/temporal recovery needs non-challenge
validation before adoption.

Historical501 frontend object-pose3929.56s/CPUprepare3582.24s dominate; all16563
linear projection74.1h is a planning estimate, not ETA. Serial oneH100; CPU-only
prep may overlap another episode's GPU work with isolated outputs onAzure.
Conservative250GiB working envelope vs latest actual573GiB free. Source15,
dependency reports and actual producer artifacts remain frozen; cleanup only
after verifying no current/future source manifest consumes an intermediate.
No extra VM or heavy local transfer was needed. Separate code closures must
remain below160KB control ceiling as all30 actual pins accumulate.

Full501 native refinement actual **PASS268.838208s**, source
76c67928579a7046485129645245f328d520f015, unchanged driver
b874a91cd0754e88c758daac2863ca6a621eedcc320e627dba8aadaddc5060b6.
Actual40file110,588B encodedGitclosure. One attempt/return/validated, original
300requested301actual full501updates/history0/100/200/300, all fixed blocks/
raw/observations/source/assets/helper/predicted mesh and saved rereadPASS.
Native postopt human mask is empty on59frames, object0; original224masks both0.
Those59automatic crop observations remained zero, no filling/deletion, and
unchanged full native optimization completed. This is neither a full frontend
occlusion recovery proof nor quality evidence. Receipt
b9947c21ec2b809c993c52617766c558f0590444a773eab4271f753495295a84(14,316B),
bundlefb2460279117512ba91d6a826fa9965f053c6ee562942829043f193faf9424f2
(383,399,392B), both0444. Unitinactiveexit0/H100147MiB0%. Actual pins frozen
before direct export; no thresholds, poses or immutable producer helpers changed.

Root fullsuite **7829PASS/2same optionalSKIP124.26s**, followed additive
readiness tests separately (created after collection); combined tiny source/
inventory/frontend/consumer/archive gates553PASS13.28s. Frontend route has
conservative deadlines600/5400/7200/7200s and cooperating scoped GPU lock,
ending at original input prep, not old learned/conversion route. Existing child
image-tag/broad-mount behavior is unchanged, not newly claimed hardened.
Source-pin inventory uses independently supplied original structure and actual
producer hashes, exact15pre/posthash and report-only audit, not GT/heavydecode.
Readiness audits only selected30RGB+2metadata hashes/container metadata,
never sample values/other tracks/labels/model files. All unverified occupied
outputs remain opaque; no automatic reuse or resume. Actual Azure readiness
execution and full export are still separate gates.

Full501 direct native export actual **PASS34.580988s**, producer
bf54ceb75d3b4240f24639184de47a7ac209f02e, unchanged driver
967b45eb50959aa3854f01d1baae89fb946679f68bbaa8aff8a2d59287b27ee0.
Actual43file116,432B Gitclosure SHA-XZ
3138aebd081352da221858c7218861d15a791c174449f7a3c9eed733448dadb4.
All four32attempt/return/validated routes cover original501/tail5; native
direct/replaymax0.002870962/0.000953936mm, unchanged .01mm point gate.
Every original official-reference mean passes unchanged2mm gate; complete
mean/worstframe statistics still to inventory, not inferred from native values.
Original refined7blocks/raw/contact/masks/source/model/helpers and saved
payload rereadPASS; aligned2044vertex object scale1/camera roundtrip0m retained.
Receipt947a97d9abfa8ee712ea5adf9747e95aa74f6a9006c34a95801434e0e19768c6
(23,205B/0444); actual complete four payload pins committed for the consumer.
Unitinactiveexit0/H100141MiB0%. Full prepare/forward/refine/export engineering
chain now passes, not yet all30/finalParquet/independent accuracy/CARI4D win.

Additive stage-inventory export support retains actual missing-field semantics,
all previous gates and exact five-file hashes; new export fields mirror recorded
full-N fidelity/source/object-roundtrip proofs. Root combined387testsPASS2.06s,
no numerical producer helper changed. Clean-launch guard rejected a readiness
dispatch while a disjoint agent had uncommitted additive work; no Azure command
or unit was created. Commit all verified work before actual readonly preflight.

Full-native episode CPU consumer gate added separately: pinned existing image,
networknone/noGPU4GiB2CPU,300s/outer303, exact public/predecessor/source assets
readonly, exclusive one444JSON receipt. No new trajectories, model execution,
fitting, rendering, repacking or Parquet. Real peer imports and callback lifecycle
25testsPASS followed root source/inventory/consumer/readiness/archive checks.
All original prepare/forward/refinement helper bytes verified against their
actual producing commits10/16/9files. A first local source_helpers call correctly
rejected writable checkout files; rerun direct Git-byte comparisons, not altered
file modes or remote evidence. Actual consumer runtime remains to execute.

Actual full501 official reference export: mean0.000430992957mm,
worst original frame0.000833857652mm, diagnostic max point0.002441872758mm;
501 per-frame means, unchanged2mm gate. No geometry threshold changed.
The CPU episode consumer wrote immutable receipt
b0645f81745314fc684f923a2e827820095d0b6a9c97a6cb40a95e7566b57ba0
(3,826B/0444); complete actual source chain/trajectory details are inventoried
by the real consumer. Actual **PASS11.832886s**, producer1c635b5,
driverbcf588e20de2f47b6a20f29570726122cb9b11a0c728e5fb2b3b7f54e0a0b28f,
62 original source bindings,501 full owned trajectory/schema/object roundtrip,
export pins/source helpers rehashed. Unitinactiveexit0, no GPU/model/optimizer/
render/Parquet calls. It does not invoke the official packer or produce Parquet.

First actual all30 readonly readiness stopped after5.014187s with
`Exactly one selected RGB video stream metadata required`; producer1c635b5,
script9821ff7601bd887ad73b627fb9033bf20976df8cb06dc1995007d990dbb00e56.
No decoder/model/GT fallback, inference or frontend output creation occurred.
Inspect the actual ffprobe metadata envelope before any correction or retry;
retain the failed receipt/log and original source. This is not data readiness PASS.
Actual original episode0 metadata-only ffprobe confirms exact1536x1152/30Hz/
790frames plus an empty `programs: []` envelope. Accept only this specific empty
optional envelope in addition to `streams`; unknown fields/nonempty programs
still fail. Stream dimensionality/frame/rate/hash gates, no-count/decode/fallback
policy and120s deadline remain unchanged. Retry uses a new source/unit/log.

Latest fullsuite initially exposed three synthetic CPU consumer callback tests
inheriting Joblib from an earlier test module. Scope optional-import isolation
to the callback fixture, restore worker state and add a polluted-worker regression;
the production fresh-process guard and numerical helpers remain unchanged.
Focused order regression182PASS2.44s; fullsuite
**8025PASS/2same optionalSKIP124.12s**. Independent callback tests26PASS0.53s.
Exact ffprobe-envelope regressions plus consumer/inventory/actual Gitclosure
**464PASS2.66s**; fixed dimensionality/rate/frame/no-fallback rules retained.
Read-only frontend audit confirms the generic route is one fail-fast engineering
pilot, not an all30 occlusion guarantee: automatic masks may be empty, but Body,
scale anchors, object frame0, visible-depth ICP and input prep have explicit
support requirements. Leave them strict; preserve failures rather than patch
individual episodes. Existing three image tags must resolve to verified actual
images before launch; root scheduler excludes all GPU jobs, since the cooperating
flock does not cover older units. Never claim this resolves license eligibility.

All30 readiness retry actual **PASS2.210132s**, producer
ce0519fb99125ee0b847cd89c9f37ea6a684738a, driver
d985426860041ec1f6bf8c2398f15ce7fd30f5de0718e9eee3a2e9de05235e3a.
Thirty original1536x1152/30Hz clips,16,563frames; selected32 hashes verified,
35 other legitimate manifest files never read,30 metadata-only probes, zero
decoded frames/model/private/GT reads. Manifest
3df960ce0f594b8f51675b21bb070925de7aa87a583332674eb89b0e90fc6263
(13,081B), readiness log33e8b0ef4c7613596ab27885e33a45b3926391166db0e58448006ecb6222ca19
(27,378B) stays onAzure; no full log transferred. Exact original lengths0..29:
790/668/866/592/747/668/816/777/634/415/577/877/405/425/442/501/
360/419/535/443/549/563/533/552/420/365/399/440/366/419.
Only episode0 and15 frontend targets occupied, still opaque/unvalidated by
readiness; deterministic first clean episode1/N668, not a visually selected case.
Actual613,831,860,224free bytes,106,636,103free inodes and~306GiB RAM available.
All three expected images present at their verified IDs; upstream7c0d clean,
required assets/camera gate present, UID1000 and no compute jobs. MoGe weight
snapshot is an existing audited HF symlink, not a new regular-file assertion;
existing model producer checks remain unchanged. License clearance is separate.

Dispatch the exclusive episode1 frontend-only engineering pilot fromce0519f:
53file103,212B closure SHA-XZ
bf0ef94a5041138b3269748e76067a21acb78cb40695e872f918d3a108aac751.
Fixed masks600/7initializers5400/objectpose7200/CPUprepare7200s,
no historical predictions/CoCoNet/refinement/LM or overwrites. Actual outcome
pending; serial H100, potential CPU overlap only after GPU lock release.
Actual dispatch returned active/MainPID711539; no second GPU job started.
Complete latest suite **8033PASS/2same optionalSKIP125.77s**, including exact
Azure ffprobe compatibility and callback test-order regressions.

New own-data feasibility prerequisite only: complete closed/outward/manifold
mesh, original face/vertex coverage, conservative self-intersection and full
human/object triangle intersection plus solid containment audits. Adjacent faces
are exempt only after proving intersection is their original shared simplex;
coplanar overlap/tolerance ambiguity/exact touch fail. All possible pairs use
complete sphere/AABB broadphase, never sampled collision points; positive
<=2mm gaps are proximity, not true contact/force closure/biomechanics. No face
deletion, repair or scaling. Float64/tolerance1e-8m and bounded30s/2M candidate
pairs explicitly limit the numerical certificate, not an exact-arithmetic proof.
Own analytic54testsPASS0.92s; root combined120PASS1.00s. Native18439vertex/
36874face human closure/embedding/runtime remains unverified onAzure.

Separate private one-state manufacture capability being implemented, not run:
fixed full194vertex/384face own asymmetric bottle/scale1, native shared zero45/
zero68/zero72 identity with legal204 controls, automatic actual distal thumb/
index surface patches, actual opposed normals and .5mm positive target gaps.
No human AABB proxy, old synthetic/challenge poses, rendered RGB or inference
quality. Fresh neutral embedding and native autograd/finite-difference checks
must pass before bounded legal thumb/index plus object rigid-pose fitting.
Hard300s/100 total native forwards including diagnosis/final replay; failure
stops before rendering or cohort manufacture. Only private output and exact
reference model/code mounts; no dataset, labels, vendor, old predictions or
historical truth mounts. One-state feasibility cannot establish full96 temporal
HOI validation or method adoption. H100 execution waits for frontend GPU release.

Own-source/publication audit:638 tracked text files/~6MB,331 reachable producer
commits; no tracked/history assets, private labels, credential files or concrete
token/private-key literals found (invalid credential-shaped tests excluded).
No upstream implementation bundled; external runtime/licenses remain separate.
Public repositorymrprokl/world-reward-v2d created, existing main/history pushed
without rewritten producer commits; exactce0519f page unauthenticatedHTTPS200
and remote main hash match. Future producing commits must be pushed too.
This removes the missing remote/access blocker, not registration/runtime-license/
one-command final all30 reproduction/Parquet/quality checks. No Kaggle upload,
organizer message, gated assets, private synthesis truth or credentials published.
Complete surface-audit/source suite **8087PASS/2same optionalSKIP124.84s**;
all numerical production helpers unchanged. Public main32c62a2 now includes
that own-surface source and actual all30 readiness documentation.

Episode1 automatic masks actualPASS63.422972s, receipt
a26684ab3524d2d7af85fbd8883268eddf6370a6482a5e873d1ce0712e0d1604(9,034B).
SparseBody/depth/shared gauge16.275605/8.571827/3.616264s PASS.
Grounded object generation32.633463s PASS,31,334vertices/62,616faces before
the separate topology-preserving pack/pose gate; receipt
8d87466bba415ee5ae5c86da3d719cbf4e0a6e9ea6e37a23f88072414a929981(15,645B).
Full Body668original frames actualPASS217.117887s, predictions
db11c5eb14fdf43f400b01f64b1efc2367b5875c65876c4e68e821688b78b981;
native geometry/input dataset5f683... verified. Fulldepth/adapter/objectpose/
input-preparation still pending; no early numerical/fullchain or quality claim.

Episode1 full depth actualPASS344.130609s for all668original frames, receipt
487f2353dd7226a152125dfdeaca5ca2852fb9aa924af1e8a648845f61c17570(1,011,785B).
Body-to-native adapter actualPASS8.617577s, receipt
da9fc9ad12457915279888bc5de25bf0333f8c2b3dc9a13d0468663143425d14(2,540B).
All seven initializer stages returned; full object pose now running, observed
CUDA PID720064/1,886MiB. No second GPU job, completed input-preparation,
new frozen input pins or all30 reconstruction claim.

Private own-grasp capability implementation frozen for later execution, not run:
77 tiny tests pass, including actual generic dispatcher namespace and host
marker semantics. One fresh neutral native MHR state, full194V/384F own bottle,
legal249 limits, fixed zero45/68/72 identity/scales/expression; only native named
left thumb/index free controls and object rigid6DoF are manufactured. Full native
neutral closure/embedding gate before anatomical patch selection or optimization;
actual CUDA autograd checked against four legal five-point forwards. Bounded
Gauss-Newton with uniform feasible updates, never control clipping; at most86
objective states,93 planned batch1 native calls including diagnosis/replays,
100-call/300s hard cap and reserved30s full-surface certificates. Both frozen
output reread and native replay must reproduce original bytes. Actual native
closure, reachability, differentiation and runtime remain unverified. Positive
.5mm opposed surface gaps certify proximity only, not touching, force closure,
full96 motion, RGB frontend capability, quality or adoption.

Only readonly committed own-code/reference-MHR file and private output mounts;
no public input, historical pose/results, render, vendor or private truth input.
Canonical dispatch revision/archive markers and entire readonly source closure
checked; original model/helpers/metadata/native-input bytes verified before/after.
Failure leaves concise private0400 receipt only; own transient NPZ files removed.
Numerical upstream/shared501 producers and conservative surface helper unchanged.

Complete own-grasp/source regression suite **8164PASS/2same optionalSKIP124.83s**.
No native/GPU execution or quality query during local tests. The corrected actual
immutable source namespace is jobs/<revision>/run_own_grasp_capability/code;
generic Azure dispatcher unchanged.

Targeted throughput audit retains original object-pose execution unchanged.
Episode1 actual200/668frames at1,224.703523s, not an allframe outcome; existing
H100 remains occupied. Earlier canonical-tree full-candidate test already failed
useful-speedup gate1.2794<1.3, so do not repeat or lower its threshold. New tiny
own CPU diagnostic (NumPy2.4.6/SciPy1.17.1, <0.4s) finds nontied8192/2048
queries1.99x and three14-iteration ICPs1.79x, exactR/T/status/inliers/iterations,
residualdelta<6e-18m; not H100/end-to-end evidence. Rotated equidistant/near-trim
cases instead diverge by40mm translation or6vs3 iterations at nearly unchanged
residual. SO3 floating query equivalence is not universal discrete parity. Keep
cache/batch proposals unadopted; any future distinct ambiguity-aware candidate
needs original fallback or failclosed, exact branches/NN/trim/candidate path and
original camera/mesh parity plus measured>=1.3x whole-frame throughput. No
hypothesis reduction, resizing, relaxed acceptance or running-source mutation.

Official packer actual source/import audit completed onAzure, six small files
pinned in the new one-episode CPU gate. Actualpack_track1(args)->None supports
an exact episode-only row layout; no missing29 predictions needed. Original
read_sample reads all columns before selectingrow_id, so pass only a new CSV
made by our existing Arrow columns=['row_id'] projection of the hash-bound
originalsample. Originalsample3,495,851B/SHA1db56c5ec7267cdb7a68d42928f85c092c6b8e9bfbc0dc98a2b020d7a48bc621,
740,780rows/9,877scoredframes/30episodes. Never read its XYZ as predictions.

New official gate retains complete source-bound501frame episode, writes exactly
six full trajectory arrays and byte-identical aligned GLB into disposableAzure
scratch, invokes unchanged officialpacker once, checks originalscoredindices/
exactcontrol/clipidentity and full cyclic-oriented triangle multiset after
official welding/padding. No second simplification/geometrytolerance. Scratch
including temporaryParquet deleted finally; sole444 JSONreceipt. NoTorch/Joblib/
model/scorer/render/upload or finalall30 artifact. Focused48PASS1.58s, combined
122PASS1.89s, complete **8212PASS/2same optionalSKIP124.04s**; sourceclosure
47files126,632encodedB within160KB. Actualexecutionpending.

Dependency preflight fails before packing: existing pinned CARIimage has
NumPy1.26.3/SciPy1.16.3/pandas3.0.6/trimesh5.1.0/fast-simplification0.2.0,
but PyArrow absent. Do not mutate it or call a schema-only pass packing success.
Build a separate CPU child image adding only Apache PyArrow19.0.1 cp311
manylinux2.28x86_64 pinned wheel42,084,055B/SHA
49a3aecb62c1be1d822f8bf629226d4a96418228a42f5b40835c1f10d42e4db6
from public PyPI metadata; wheel fetch/build staysAzure, no dependency upgrade.
Actualnewimage/receipt binding must precede dispatch. GPUfrontend continues;
this CPUruntime work can overlap without GPU duplication.

Separate CPU image builder ready:41 mocked testsPASS0.13s, actual primary
PyPI metadata revalidated against the literal pinned42084055B wheel before any
download. Source-bound exactAzurehost entrypoint, noGPU/records/modelreads;
120s actual acquisition alarm including blocking reads,300s whole driver.
Offline/no-pull/no-deps build from verified unchangedCARI parent, own unique
childtag, original5packageversions plusArrow19.0.1 CPU import required. Original
parent reverified. Canonicalreadonly code/archive markers, deterministic recipe
owner label, exclusive444 receipt, temporary download/context cleaned. Timeout
cleanup targets own labeled classic-builder containers only; hard daemon
termination is explicitly not proven, so inspect failed build workers before
any retry. No global prune/tag replacement/sourceimage mutation. Nativepacking
stays blocked until actualnewimage/receipt pinned; no GPUjob duplicate.

Complete CPU image builder/source suite **8253PASS/2same optionalSKIP127.34s**;
unchanged directexport/sharedconsumer/crosssurface/own-grasp sources verified.
Freeze builder before its first AzureCPUdispatch; one exclusive image tag/report,
no implicit repair/rebuild.

Actual original CPU imports NumPy/SciPy/pandas/trimesh/fast-simplification pass
in sourceimageb47e with noTorch/Joblib loaded; pandas tinyframe(1,2). First
separateArrowCPUbuilder dispatched from764c3bce2f8c8a6192d5f68b455a45274e59efd8,
21files16,672encodedB, unitworld-reward-official-pack-cpu-image-build-v1.
Actualbuild outcome pending. The existing frontend retains its GPU; no model or
dataset transferred locally, no extra H100 or replacementCARI image.

Arrow CPUchild actualbuildPASS68.365245s, unitinactive/dead/exit0. Producer
764c3bce2f8c8a6192d5f68b455a45274e59efd8/driver8b380376230c7439ba058a27e6acc5a0911d55600ba71f737264c28bb113a717.
Actualimage sha256:1a04b1930f713ef9ffb411489e80ddebbce59a5ce26e713add4095cd9b5303f0,
receiptce9a8ba45b14b3faefbdeb43a6f18e410f1bd1a95f0dd19e17727871a71812a2(3,563B/444).
Python3.11.10; Arrow19.0.1actualCPUimport, fiveoriginalpackages preserved,
parentb47e unchanged; onlyArrowadded and scratchremoved verified. NoGPU/
Torch/Joblib/records/models read. Root runtimepins now record actualobservations;
officialpacking checks exactreceipt/source/image before first invocation.
Episode1 stillactive300/668objectframesat1,935.131503s, nofulloutcomeclaimed.

Full actual build receipt inspected; committed runtimepins bind its exact bytes,
builder revision/helpers, Arrow wheel and immutable childimage. Pure runtime
validator accepts observed fields; launch will hash-check the original receipt.
Parent/native export pins remain b47e, no fabricated relabeling. Runtime-bound
official gate tests77PASS; complete **8282PASS/2same optionalSKIP124.02s**.
No actual official packing yet; freeze/push the actual producing code before
one CPU-only episode15 smoke. No benchmark accuracy or all30 submission claim.
