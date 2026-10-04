# Automatic hand availability: IMAGE versus native video memory

**Preregistered; native API and acquisition qualified, masks not yet evaluated or adopted.** Dex03 and Dex04 remain
closed. The rejected 17-point ablation is not extended or retuned.

Hypothesis: temporal memory can maintain an automatically seeded hand hypothesis
when the per-frame IMAGE detector has no proposal. It cannot detect a hand never
proposed, certify physical identity, contact, visibility or 3D, or recover every
late-born hand. Forward/reverse SAM2 already exists in the official baseline;
this experiment tests its contribution relative to IMAGE, not a novel tracker.

## Frozen experiment

The first published not-previously-evaluated DexYCB subject, excluding01–04, is
subject05. Camera836212060125 and lexical clips4/39/74 are fixed before acquisition
or annotation values. Retain every original frame. CC-BY-NC4 and unknown model/
challenge overlap keep this diagnostic-only; a new subject is not a new domain.
[Publisher](https://dex-ycb.github.io/),
[official archive](https://drive.google.com/file/d/1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS).
Announced archive size12,815,420,651B; its checksum must be measured on Azure.

A uses unchanged MediaPipe IMAGE4/.5 and all usable21-XY boxes, native SAM2 IMAGE
box-only. B seeds all usable boxes at the first automatically supported frame in
two independent native video states. Original slot numbers are hypothesis IDs,
not recognized identities. Reverse owns earlier frames; forward owns anchor and
later frames. Native logits>0 only; no prompts/points/reseeding/fusion/GT selection.
No anchor means explicit full-T abstention. Keep empty masks and all seeded slots.
Use the qualified installed upstream SAM2, not a new package or unverified
low-memory wrapper. Numeric JPEG staging copies original bytes, without reencoding.
Use explicit CPU video storage, GPU state and synchronous loading, matching the
qualified native API smoke. Check the native state retains the full original
frame count and 480×640 grid before seeding; no resampling or shortened clip.

## Gates and scope

Before fresh RGB, a bounded data-free native CUDA smoke must verify two-slot
seeding at a middle frame, independent forward/reverse states, exact yielded
positions/IDs, finite `[N,1,H,W]` logits and unchanged source/assets. Shape/API
success is not segmentation quality. Reuse the existing isolated runtime/lock.

Freeze all full-T masks before the separate CPU evaluator opens segmentation
values. Score only annotated-positive frames, including empty predictions as
zero; unlabelled frames are unscorable, not negatives. Require positive pooled
Dice delta, no clip regression, fewer positive empty unions and no increase in
object-label pixel count on the same positive frames. No-positive clips are
inconclusive and stop, never discarded. Object contamination is not semantic
false-positive truth. Record background pixels and IoU separately.

Budgets and exact policies are in the committed JSON protocol. Failure closes
the recipe: no anchor/threshold/seed/clip tuning, no Dex04 rescue. Even passing
this diagnostic cannot establish CARI4D superiority or competitive eligibility.
