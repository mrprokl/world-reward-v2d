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
