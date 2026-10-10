# Real temporal/contact follow-through — 2026-10-10

Baseline remains unchanged. These are algorithmic ablations, not manual episode
labels or a demonstrated leaderboard gain. Original full timelines and geometry
are preserved. All media/models/compute remain on Azure.

## Frozen hypotheses actually executed

1. **Contact association**: episode 9, 415 frames, same 33 automatic RGB tracks,
   271 activated hand/frame entries. J1 versus hard anatomical pool versus soft
   pool; identical 300-evaluation cap, coefficients frozen outside challenge.
   This tests the complete saved-pose fitting stage, not a rerun of every model.
   Gates compare the same J1 contact witness, RGB fidelity, linear/angular
   acceleration tails and RGB motion increments against original and fair J1.
   Real contact truth is unavailable; these remain conservative output QA.
2. **Coherent video depth**: native metric Video Depth Anything Small versus
   original MoGe, three TUM RGB-D sequences, contiguous original RGB ranks
   200–295. Same scenes as the previous sparse pilot, disjoint frames: not new
   scene generalization. RGB-only predictions must be sealed before private
   registered sensor values are mounted for scoring. No calibration, scale/shift
   fitting, challenge GT or sequence-matched FORM-HOI.

Metric gates were frozen before prediction: median AbsRel improvement >=5%,
median original-pixel temporal depth-change error improvement >=10%, no sequence
regression >5%, paired support >=95% overall and per-frame. Temporal metric is
Eulerian and uses RGB timestamps with nearest registered sensor offsets <=20ms;
it is not material-flow tracking or exact sensor-time velocity. Adjacent repeated
sensor images and unavailable sensor observations are not temporal evidence.

## Actual execution failures and corrective action

- VDA asset v1 failed before inference: authoritative HF checkpoint redirected
  to `us.aws.cdn.hf.co`, missing from allowlist. The verified exact publisher
  host was added; no model/settings change. V2 acquired 21 pinned files in
  **16.13 s**. Receipt SHA256
  `2e566c445983bdd79a4aaee71997861acb785578f43892f132d7a9a190da51fa`.
- TUM acquisition passed in **66.32 s**, 288 full-resolution RGB frames plus
  public manifest; two sensor observations missing, 274 unique sensor frames.
  Original publisher archives were removed after selected byte validation.
  Receipt SHA256
  `514eab2a0b333bb855d1db3aaad4a051167d4cfbdbebe0b828bacf98debc7581`.
- First GPU inference stopped in **4.05 s**: native temporal module requires
  `easydict`, absent from the image. No prediction or private truth read. Owned
  container absence verified. Failure report SHA256
  `6bccfd6303e0dbbe362bafa3723781c9379c48fd60d5dbed8699e3945cc14bf3`.
  Corrective execution v2 uses the official pinned 6,804-byte pure-Python
  `easydict==1.13` wheel in an isolated readonly overlay with LGPL-3.0 notices;
  no global pip, Torch update or modification of VDA. Native dependencies were
  audited again, and the remaining runtime modules probed on Azure.
- Contact hard-pool stage previously used single-point distance evaluation.
  V2 enables identical-face-set batching for J1/hard as already used by soft;
  tiny complete fits are bit-identical in rotations/translations/diagnostics.
  Each worker atomically seals its trajectory and receipt before parent QA,
  so a later worker timeout cannot discard completed fits. No concurrent
  duplicate contact job is allowed; v1 has to terminate first.

Producer for runtime corrections:
`9bd3c8c3b92168ff58c8c87399248f9f3e8549ab`.
**257 targeted tests passed in 6.08 s**; no claim the historical whole repo suite
is green. Local files are code/tests and concise receipts, not heavy outputs.

## Explicit limitations / next transition

### Completed contact and external sensor results

Contact producer `4ef9a1b5290d656f6c20f9d4aa6a781b4bfba019` completed
in **1,146.46 s**, all three fits converged (J1 65, hard 72, soft 67 evaluations).
J1 / hard / soft fit times: 143.36 / 1,096.83 / 996.51 s; no model calls.
The larger cap removes the prior non-convergence excuse; it is not the remaining
contact failure. Hard/soft pass every frozen gate except anatomical contact
distance. Their similar outcomes weaken the hypothesis that hard switching alone
explains the remaining separation. All full trajectories are saved.

| Route | RGB mean px | Linear acceleration p95 m/s² | Same original anatomical gap mean m |
|---|---:|---:|---:|
| Original | 8.7317 | 21.3894 | 0.007087 |
| Fair J1 | 3.8118 | 34.4416 | 0.075971 |
| Hard pool | 3.7947 | 11.3330 | 0.086674 |
| Soft pool | 3.7931 | 11.7292 | ~0.0866 |

Original bounded full-pool gap mean 0.00960 m becomes 0.04085/0.04141 m
(hard/soft), so this is not only a change of the selected finger witness.
Whole-anatomy coverage and factor-cost audit now runs on these sealed results,
without another fit. This will distinguish missing pool coverage from genuinely
misplaced predicted human/object geometry. Next structural alternatives are
contact memory or joint relative ray-depth correction, not arbitrary weights.
Report pin: 26,482 B,
`af0874f8debf7979cc2c873c74901a6b6013b8c3e74e5016964df891df37ba52`.

VDA/MoGe blind predictions completed **59.37 s**; sensor evaluation **6.32 s**.
All frozen sensor gates **pass**, same paired coverage 100%.

| TUM sequence | MoGe / VDA AbsRel | AbsRel gain | Eulerian temporal-error gain |
|---|---:|---:|---:|
| Freiburg1 desk | 0.48625 / 0.16291 | 66.50% | 17.96% |
| Freiburg2 xyz | 0.62105 / 0.11389 | 81.66% | 7.33% |
| Freiburg3 office | 0.26383 / 0.14635 | 44.53% | 40.98% |

VDA forward times 3.48/3.00/2.97 s versus MoGe 13.70/13.19/13.04 s.
These are measured H100 forwards, not control latency or complete pipeline time.
Median AbsRel gain **66.50%**, temporal gain **17.96%**, no sequence regression.
Sealed prediction report SHA256
`117dcd936408158ec3e4b9d3d823535c4d98be294d2b798ac73f6140d8629da1`;
evaluation report 4,504 B SHA256
`e7c16df2d488115539f4f3dd2455dcbf9c21fb9e23dca1a2dedb0fa122f2e3af`.
Owned containers absent, exit 0. Retain VDA as a depth candidate and advance to
the full-T pose/contact stage; no automatic production adoption.

### Anatomy diagnostic: replace an expensive failed question, not its result

Full-hand exhaustive minimum audit under
`8fbbbca70c330adff4dd8fb6f3ebdb28dbd2e7e9` exceeded its **600 s** limit;
no anatomical result is claimed. Failure receipt 359 B SHA256
`7682e7f17a47f85456dabbc7ba8a9da409a730667a53cc39a3ee0a77f93501dd`.
Owned CPU container removed, exit 1. Do not rerun with a larger arbitrary cap.

Alternative `79f8a2032088b1219602c64abaa7f3ee47db885c` completed **11.86 s**:
every active hand's 2,318 vertices participates in conservative lower/upper
bounds, and one exact continuous-surface witness tightens the upper bound.
**628,178 anatomical point bounds per variant**, original faces retained.
Intervals overlap the original in all 271 hand/frame rows; no certified worsening
or improvement of the *whole-hand minimum* can be claimed.

However, the upper bound certifies a closer anatomical point outside the chosen
eight-point pool on **232/271 hard-pool** and **233/271 soft-pool** rows
(original 146/271, J1 215/271). This is positive evidence of limited fixed-pool
coverage, not proof of physical contact or correct hand placement. Whole-hand
upper mean is 0.00709 m original versus 0.02348/0.02344 m hard/soft; these are
upper bounds, never labeled exact gaps. Prioritise changing-patch contact memory
and independently test relative placement, rather than simply strengthening
the same incomplete association.

Receipt 16,390 B SHA256
`06fbb21965ef264bd6a61a3d75c2fbe4000b50df1f4e366a8b88d24b798daa73`.
Owned container absent, exit 0; no fit, model or heavy transfer.

### Actual full-T depth-to-pose follow-through

Producer `426a9cb74923af24389e2b7c118a4790908e088f` now runs the actual
415-frame RGB sequence through VDA, the same predicted human's single positive
clip scale, object-track depth, and two fixed-geometry R/T fits (without/with
soft contact). It reuses upstream masks/human/object rather than rerunning all
models. This is the full temporal fitting chain, not a complete new frontend
or human-articulation optimization.

VDA without contact has already converged: **23 evaluations / 3.37 s** fitting,
10,349 depth observations across all 415 frames. RGB 8.7317 -> 5.4914 px, but
linear acceleration p95 21.3894 -> 48.6669 m/s² and original anatomical gap mean
0.00709 -> 0.33283 m. It cannot be adopted on its own. The shared human scale
1.733574 achieves 4.01% median human-depth consistency, which does not establish
object-depth accuracy or relative interaction geometry.

The combined soft fit subsequently completed: **25 evaluations / 599.06 s**,
converged, versus 23 / 3.37 s without contact. Complete chain **630.71 s**,
including native VDA **12.16 s** on all 415 frames. There are 10,382 total
supported depth observations (10,349 post-anchor observations in the fitter).
Soft-contact candidate RGB 5.4927 px, acceleration p95 **50.2173 m/s²**, same
original anatomical gap mean **0.31480 m**, versus 48.6669 / 0.33283 without
contact. Both fail the predeclared contact and linear-acceleration gates; all
other gates pass. They are finished experiments, not pending or non-converged.
Neither replaces the baseline. Owned GPU container absent, exit 0.

**Next structural hypothesis:** the verified incomplete fixed contact pool needs
changing-patch memory, and the object-only fitter needs actual joint human/object
relative placement under retained image evidence. Existing native MHR parity
optimizer already has articulation/contact/silhouette/penetration/temporal terms,
but freezes human translation, hands and object rotation and has no independent
2D body loss. An explicit extension can reuse this solver and differentiate
human translation, rather than add another ad-hoc SE(3) solver. It requires
automatic DWPose observations on the original timeline, a zero-extension parity
control and external validation before a real comparison. Do not silently cap
the existing 33 tracks to the legacy point adapter's 32-query contract, invent
barycentric attachments or reuse the baseline validator that forbids new DOFs.

Latest integration checks: **188 targeted tests passed in 6.30 s** using an
isolated temporary directory. A previous concurrent local pytest attempt had
seven setup errors because its default shared temporary directory disappeared;
the isolated rerun passed without modifying algorithm tests or thresholds.

The frozen protocol's `moge_fov` prose says "native inferred" incorrectly.
Actual implementation retains the earlier RGB-diagonal focal prior (800px at
640x480), fixed across the clip. This is disclosed in prediction reports; no
sensor calibration or post-evaluation parameter change. Do not interpret a
gain as removing this camera-prior confound or full 4D superiority.

Passing sensor depth gates permits a new saved-full-T 4D evidence ablation; it
does not automatically replace the production initializer. Passing contact QA
permits Azure-served visual comparison. Either failure must identify a concrete
next hypothesis, not trigger a coefficient sweep or be mislabeled success.
Model training overlap is not independently attested; source/license checks do
not establish leakage freedom. No new submission has been made.
