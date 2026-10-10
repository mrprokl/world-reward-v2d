# Saved gain audit — 2026-10-10

## Scope and controls

Frozen cohort `[9, 1, 14, 7]`; original baseline `de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba`,
SAM frontend `b658881079871508c6b3ec14d001dc1299956996`, full4D producer
`052ba1554e9a573d566713a99a61d89a5f27681c` unchanged. Independent audits ran on
Azure CPU, reusing saved predictions: zero fits, model calls, GPU requests or
heavy local transfers. Parallel independent code/scientific audits; Azure control
commands serialized to avoid competing Run Command invocations. No oracle/GT,
manual labels, camera/shape/scale changes or production adoption.

This is measurement of existing candidate gains and limitations, not a new 4D
reconstruction or held-out performance evaluation. Reprojection uses fitting
tracks; forward/reverse agreement has no independent segmentation truth.

## Mask recovery gain / new integrity and consistency measurements

Coverage: **329/372** originally empty observations now have native reverse masks
(88.44%); episode 1: 244/244, episode 7: 85/128. Remaining 43 empty observations
are episode 7 frames **375–417 inclusive**. Full timeline retained. Every original
person mask and originally nonempty object PNG remains byte-identical. Native
negative presence and unevaluated/unknown states are not overridden.

Agreement on frames where both original forward and reverse masks are nonempty:

| Episode | Common frames | IoU median | IoU p10 |
|---|---:|---:|---:|
| 1 | 257 | 0.99127 | 0.94902 |
| 7 | 593 | 0.99327 | 0.96650 |

These common-frame numbers **do not validate the missing-frame predictions**.
Recovered area / original visible-area median: 0.653 (episode 1), 0.384 (episode
7); recovered person-overlap median 9.36% / 4.45%, episode 7 p95 66.38%. These
descriptive values can reflect genuine occlusion/contact or error; no area,
component or human-overlap threshold is used to delete predictions.

Two deterministic gap-boundary/uniform QA boards were produced on Azure only:
170,806 and 178,222 bytes. They are automatic unverified predictions, not labels.
Visual approval and external occlusion evaluation remain pending.

## Four saved poses / episode 9, same 33 tracks and 415 frames

| Candidate | RGB mean px ↓ | Centroid acceleration median / p95 m/s² | Active anatomical distance mean m ↓ | Decision |
|---|---:|---:|---:|---|
| Original | 8.7317 | 7.3158 / 21.3894 | 0.007087 | Preserved control |
| RGB | 3.4310 | 0.4160 / 6.2851 | 0.342615 | Reject: lost interaction |
| RGBD | 4.6531 | 12.6669 / 44.8435 | 0.111373 | Reject: more translation jitter |
| RGB + contact | 3.8116 | 0.6181 / 35.6794 | 0.075995 | Reject: tail jitter/contact; not converged |

Acceleration and anatomical distance are prediction-only conservative QA, not
truth. No candidate passes all predeclared gates. Episode 14 lacks adequate KLT
texture, so pose evidence does not cover all four videos. **No verified 4D gain**.

## Bottlenecks, evidence rather than presumed causes

1. **Correlated inferred depth.** Residual = inferred axial depth minus original
   pose prediction, not sensor error. Frame common residual absolute median
   7.82 cm, within-frame absolute deviation about median 5.94 cm; lag-one
   correlation 0.944. Correlation between residual second difference and object
   centroid Z acceleration is 0.862 in RGBD, versus 0.403 original / −0.026 RGB /
   −0.084 contact. Strong evidence that RGBD follows this changing evidence, not
   proof of which trajectory is correct. Ideal IID-noise DEV was insufficient.
2. **Anatomical witness continuity / joint optimization.** 247/268 adjacent active
   hand pairs change selected mesh vertex (92.16%). Contact-candidate top ten
   acceleration frames account for 52.96% of total acceleration magnitude;
   median near a witness switch 5.03 m/s² versus 0.320 otherwise. Motion itself
   can cause both changes; correlation does not justify freezing a sliding grasp.
   The capped optimizer was not converged. Persistent contact patches and
   reliability need separate evaluation, not stronger arbitrary smoothing.
3. **Weak independent evaluation and textureless coverage.** Only episode 9 has
   this four-way pose comparison; episode 14 supplies no KLT tracks. Current
   fitting residuals and common-frame mask consistency cannot establish general
   improvement. Remaining occlusions require unknown-observation handling, not
   forced amodal masks or disappearing objects.
4. **Runtime bottleneck is pose initialization, not this audit.** Original
   per-stage wall times: object pose 2,441.64 s (episode 9) / 1,895.99 s (episode
   14), versus refinement 293.24 / 215.08 s. Input preparation also takes
   762.28 / 819.54 s. These stages dominate actual reconstruction; recorded
   walls are not an additive parallel total. Profile repeated ICP/surface work
   and input preparation next; reuse clip-constant geometry and batch independent
   work without losing observations. Adding GPUs without this profile is not
   itself an efficiency fix.

**Deprioritized hypothesis:** mask-centre hand contamination is only **3/10,489**
RGB-supported queries (0.029%); 10,473 are object-only. This does not validate a
feature's whole patch or its initial surface attachment, but does not support
claiming widespread tracked-hand contamination. Canonical points are rank 3
(SVD 1.2963, 0.9138, 0.4448); all 415 local projection Jacobians have six singular
values, median normalized condition 27.02. These are local geometry diagnostics,
not global depth/shape correctness or confidence.

## Next bounded experiment

Before another challenge fit, stress the fixed fitter with independent
manufactured common-mode/temporally correlated depth and sliding-contact cases.
Then use two new contiguous external RGB-D clips, selected by metadata before
labels; audit source/license/overlap and sensor schema first. Infer from RGB
only, seal predictions, expose sensor depth/segmentation only to the evaluator.
Measure error Z, temporal error, occlusion false observations and coverage. No
sweeps or parameter selection on these held-out clips or challenge outputs.

One comparison of the existing factor with a reliability-aware alternative may
follow, only after its settings are frozen on separate DEV. Reuse one observation
bank and parallelize disjoint clips; do not rerun body/object models or render
while a numerical gate has already rejected the candidate.

## Reproducibility / runtime

Pose audit producer `1b4bb80f2cb0987f3c7e92c3a67755c4d8067281`: 11.28 s CPU.
Recovery audit producer `72d8982b303fe44013d17ebbc3be5a86a3497e73`: 196.87 s CPU,
including full native PNG/provenance rechecks and bounded RGB QA decode. Native
recovery itself previously cost 483.42 s; it was **not repeated**. Immutable source
transport/control latency is additional, not included in these CPU runtimes.
Actual artifacts remain under `/srv/scenesmith/world-reward/results/` on Azure.

Recovery report: 12,850 bytes, SHA256
`0fddef671534d002fc4bbcc6a6ac9bb8c2011c8c640a8b638ef16ef19ece0662`.
Pose report: 272,362 bytes, SHA256
`e72a2e0d061c7a86d17533c80930df21829c98a3838c3664ee44b41c4612ef1c`;
its detailed 860-source ledger stays on Azure, not copied locally.
Both actual audits completed; owned containers were removed. Runtime image pinned
to `sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7`;
readonly sources, offline CPU containers with independent ownership locks.

**363 targeted tests pass**. A wider suite was stopped after 3,529 passes; two
historical tests independently still fail (clip inventory fresh-process import,
prepare AST historical assertion). Their code/tests are untouched by these
changes. An additional wide-suite interrupted test is not a regression diagnosis;
the whole repository suite is **not claimed green**. No score or leaderboard
improvement is claimed.
