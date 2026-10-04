# Specialist hand observations: proposed full-timeline CPU pilot

**Status: preregistration draft, 2026-10-04. Not implemented or adopted.**
No new RGB, annotations, model, or runtime has been acquired for this pilot.
It tests automatic 2D observation availability, not identity, tracking, contact,
shared metric geometry, full HOI, or superiority over CARI4D.

## Question and closed evidence

Can a palm/hand specialist produce useful original-image 21-point observations
over complete manipulation clips, without requiring a hand at frame zero?
The closed DexYCB subject-01/02 bank failed its required-class gate at frame zero.
No private hand visibility was measured: that is **not a demonstrated visible-hand
false negative**, and neither that cohort nor its thresholds will be retried.
MediaPipe's audited model card explicitly excludes object-holding occlusions.
Success here is uncertain; cheap falsification should precede any Boots or 3D job.

## Freeze before acquisition or annotation values

- Fresh subject `20200820-subject-03`, original 100-sequence lexical inventory,
  indices **[4, 39, 74]**, camera **836212060125**, every original RGB frame.
  All three are in the publisher's `s0_test` selection; no substitution or
  visibility-based trimming. Exact filenames/counts freeze from archive headers
  before inference. The complete original archive is **12,197,037,343 bytes**;
  acquiring three clips still requires Azure-side gzip download/scan.
- Original RGB 640x480, unmirrored and uncropped. Private retention is only the
  corresponding original label NPZ bytes, separated from predictor mounts.
  No depth, camera calibration, mesh, MANO, other camera, or challenge recording.
- MediaPipe **0.10.21**, source `cad7f3ab99ebf175947e40c5252c642612aae927`;
  official float16/version-1 task bundle, generation **1682480004222387**.
  Its known bytes/MD5 and independently published wheel SHA are in the source
  audit. Future measured task SHA is not a fabricated publisher SHA.
- Synchronous **IMAGE**, CPU delegate, `num_hands=4`; all native detection,
  presence, and tracking thresholds **0.5**, with no custom NMS, crop expansion,
  handedness filtering, fallback, smoothing, or temporal interpolation.
  Four is a deliberately generous fixed capacity, not a learned scene fact.
- One separate pinned Linux x86_64/Python 3.11 CPU environment; do not mutate
  qualified SAM2/Boots images. Native observation budget **600 s inclusive** of
  load, original-input hashes, all frames, and post-checks; cleanup grace 30 s,
  4 CPU/8 GiB, no GPU. Acquisition/runtime construction are separate bounded
  prerequisites, not unreported inference time. Budget failure preserves FAIL;
  no partial-timeline output is accepted or reused as a complete result.

## Raw observation contract

One native call per original frame, including empty frames, in original order.
Retain ragged frame offsets, every native returned hand, and native slot order.
Slots are **not persistent physical identities**. Preserve normalized xyz and
21 pixel xy values (`x*640, y*480`), without half-pixel adjustment or clipping.
Record an empty returned set as missing observations, not a synthetic hand.
Record capacity saturation whenever four slots return; it is a censoring risk,
not proof that exactly four hands exist. Model errors remain errors, not empties.

MediaPipe order: wrist; thumb CMC/MCP/IP/tip; index, middle, ring, little each
MCP/PIP/DIP/tip. A primary-source audit now verifies **17 joints only**:
MediaPipe indices `[0, 5, ..., 20]` map identically to DexYCB `[0, 5, ..., 20]`
(wrist; index, middle, ring, little/pinky MCP/PIP/DIP/tip). The pinned
[DexYCB README L160–161](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/README.md#L160-L161)
states that `joint_2d` uses the `joint_3d` order, and its
[joint names L59–81](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/dex_ycb_toolkit/dex_ycb.py#L59-L81)
match the corresponding MediaPipe native enum. DexYCB names thumb indices
1–3 MCP/PIP/DIP whereas MediaPipe names them CMC/MCP/IP; exact anatomical
equivalence is **unverified**. Exclude **all four thumb joints** from joint-error
evaluation; retain all 21 raw predictions. No nearest-joint fit,
error-minimizing remap, mirror, or source calibration. Block thumb EPE and
any all-21 accuracy claim. Source identities: README 40,820 B, SHA256
`e19797f352bb5615b43b5a3ed4a6a193cee4eee2cd1cd1f5859615165081bb7c`;
`dex_ycb.py` 8,713 B, SHA256
`f73074505bb822b01178dc7aae9778f5107efea479d43423f4fe2dc37224d8ad`;
MediaPipe `hands.py` 6,132 B, SHA256
`4eea13e0f63eae5cf2dc21448ea78df38f4c90a2a36f43f1db8d8797f788b51a`.

Retain native handedness category/score as classification metadata, not detection
confidence, per-joint visibility, actor selection, or calibrated probability.
There is no public Tasks palm/presence score or per-joint confidence to invent.
Returned landmarks are inferred coordinates; none certify measured visibility.
World coordinates are hand-centred, not shared human/object camera coordinates.

## Freeze first; private evaluator second

All three complete outputs and source/model/runtime identities freeze before
opening any private segmentation or joint values. No labels in predictor mounts.
Private evaluation reads only `seg` and `joint_2d`, never `joint_3d` or poses.
Hand pixels are `seg==255`. All-`-1` joints can mean no visible hand **or missing
annotation**: do not convert this sentinel into a certified negative frame.

Report separately for each clip and pooled, keeping every original frame:

1. Raw returned-hand count, nonempty rate, empty spans, saturation, and runtime.
   These are observation availability, **not accuracy or visible-hand recall**.
2. Annotated-positive coverage: for frames with nonempty `seg==255`, construct
   the GT hand pixel bounding box. Construct each predicted box from its 21
   original xy, without padding. Positive box intersection is a deliberately
   weak association diagnostic, not a calibrated IoU threshold. Exactly one
   overlapping prediction gives a unique match; none is missed, multiple are
   ambiguous. Preserve/report all other predictions; never select minimum error.
3. Matched-hand Euclidean xy error in pixels over the **verified 17-joint subset**
   and finite, in-grid, non-sentinel GT joints; per-frame mean and joint-weighted
   pooled mean. Report the valid-joint and uniquely matched denominators alongside
   errors: conditional low error must not hide missing/ambiguous hands.
4. Unmatched/ambiguous annotated-positive frames, extra/bystander-like slots,
   outside-grid predictions, and annotation-unscorable frames separately.
   Extra predictions are not proven false positives where bystanders are
   unannotated. `seg` empty plus joints `-1` does not certify hand absence;
   unsupported negatives, handedness correctness, and bystander identity remain
   **unscorable** without independent annotation semantics.

Diagnostic falsification: any clip with annotated-positive frames but **zero
unique overlap matches** rejects the useful-observation hypothesis. A clip with
no positive annotation cannot satisfy this gate; mark inconclusive, not PASS.
Passing this minimal gate only warrants examining coverage/error distributions;
it does not license adoption. No hand-error/PCK, coverage percentage, or contact
acceptance threshold has been calibrated by this pilot. Freeze any later
acceptance rule on a separate cohort before new validation, not on these labels.

## Optional comparison and stopping

If affordable and frozen **before** private evaluation, run the existing
GroundingDINO fixed `hand.` detector on the same complete RGB timelines, original
`.3/.25` detector/text thresholds and `.7` NMS, all boxes retained, no required
frame-zero hand, no SAM2/Boots. Budget 300 s total; report box-coverage diagnostic
and runtime only. It has no 21-joint output, so no invented 2D hand-error baseline.
If omitted or incomplete, make no relative-improvement claim. No prompt retry.

Stop on source/rights/runtime/cohort mismatch, incomplete timeline, or the stated
falsification gate. Keep results/decision and immutable receipts; discard owned
temporary extraction/build noise. Do not proceed automatically to SAM2, motion,
contact, hand-conditioned object poses, or challenge inference.

## Sources and unresolved rights/overlap

Primary links and exact audited pins are in
[MediaPipe audit](mediapipe_hands_audit.md) and
[fresh hand validation audit](hand_validation_sources_audit.md).
Those cite the publisher, toolkit split/format, native Tasks API/graph, original
task metadata, wheel metadata, and model card; none proves current inference.
MediaPipe source/card declare Apache-2.0; exact bundle/dependency licenses still
need acquisition audit. DexYCB is CC-BY-NC-4.0: research permission is not an
unconditional commercial/prize/submission waiver. Subject-03 appears in other
DexYCB training splits, and exact MediaPipe training membership is unpublished.
Neither dataset/model challenge overlap nor leakage-free validation is established.
