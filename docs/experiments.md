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
