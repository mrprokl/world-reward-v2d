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
