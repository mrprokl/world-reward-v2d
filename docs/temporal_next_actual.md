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
