# Gemini / SAM3.1 → unchanged full4D: causal regression diagnostic

## What this experiment answers

Do better automatic initial pair boxes and their propagated masks improve the
same downstream reconstruction? It **does not** claim improved held-out 4D
accuracy, calibrated scale, resolved jitter, or a win over CARI4D.

The original random sample is retained: population 30, seed `20261008`, four
uniform draws without replacement: `[9, 1, 14, 7]`. These clips have since been
inspected and therefore are regression QA, not held-out validation. Failed clips
stay in the denominator; no manually labeled prompts or replacement draws.

## Frozen chain

1. Read existing automatic Gemini boxes at original frames 0/14/29. No API replay.
2. Native SAM3.1 fixed identity prompts → original-resolution person/object PNG
   masks over the complete original timeline. Empty observations are retained.
3. Fresh SAM3D Body inference from the new person masks, sparse and full-video.
4. Reuse MoGe2 RGB-only depth **only** if the original immutable numerical source,
   actual weight SHA, official source-video SHA, original timeline and every NPZ
   hash match. Original receipts are copied byte-identically; a separate ledger
   identifies the actual old producer. No cache is presented as new inference.
   The known two-link HF/Xet graph and independently pinned readonly model are
   verified without modifying the shared cache. Original acquisition metadata
   may be writable; its sole canonical path, bounded bytes and unchanged stat/
   hash are checked explicitly, without claiming permission immutability.
5. Re-estimate one human-anchored depth scalar/shared inferred camera. Run
   SAM3D Objects once from original frame-zero RGB and the new object mask with
   that aligned pointmap. One canonical mesh and one clip-constant scale.
6. Preserve the qualified whole-surface mesh budgeting route; do not shrink,
   delete, repair or close object components to optimize the metrics.
7. Run the unchanged full-frame rigid initializer: finite octahedral orientation
   candidates, image-gated partial-depth ICP, whole-clip Viterbi candidate path.
8. Native shared human/object CARI initialization → forward render/compare →
   native contact/silhouette/temporal refinement (300 requested steps) → direct
   full-timeline export, with every upstream oracle mode explicitly disabled.
9. Render exact exported human and object next to original RGB with a fixed
   assumed virtual floor. No per-frame alignment, camera, recentering or scale.
   Grid is 0.5 inferred metres, **not** proof of physical calibration.

All data, checkpoints, inference and rendering stay on Azure. Original baseline
`de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba` is untouched. Separate source entry
`run_gemini_full4d` is explicitly allowlisted in the same strictly revision-bound
full4D namespace; all source/inode/hash/oracle/numeric gates remain enabled.

## Efficiency and fail-fast gates

The first frozen clip is a complete wiring scout. At most two remaining episode
workers overlap CPU copy/mesh/serialization with one learned GPU worker under
one H100 lease. No duplicate GPU inference, failed-run quality retries, cached
body predictions from a different actor, or new model installation. The original
native budgets are upper limits, not runtime predictions. Object pose estimation
still performs many full-resolution renders per original frame; it is currently
the expensive stage and is deliberately unchanged in this causal experiment.

Coverage, correct source/grid, finite SE(3), one shared frame, one fixed geometry
and complete masks/exports are hard execution gates. Fewer than 40 visible
inferred object points currently causes an explicit pose-stage failure; no static
pose or made-up interpolation is substituted. Scientific failure is retained,
not a successful reconstruction. Visual QA is not a quantitative accuracy score.

## Research insights and next controlled change

- **CARI4D (CVPR 2026), §3.2:** combine RGB-only and RGB-D pose hypotheses,
  occlusion-aware comparison and forward/backward recovery before the learned
  render-and-compare/contact stages. Native docs recommend SAM3, not recurrent
  semantic VLM calls. They explicitly warn about long occlusion and permit a
  better-observed reconstruction frame. Our first-frame chain is preserved in
  this ablation; automatic best-view selection is a separate future experiment.
  [Paper](https://arxiv.org/html/2512.11988v3#S3.SS2),
  [native custom-video docs](https://github.com/NVlabs/CARI4D/blob/main/docs/custom_video.md).
- **FoundationPose (CVPR 2024):** initial registration and subsequent tracking are
  separate operations; track from the previous pose rather than re-registering
  independently at every frame. It consumes RGB-D and a model/reference; using
  our inferred monocular depth introduces uncertainty, not ground-truth depth.
  [Paper](https://arxiv.org/abs/2312.08344),
  [official demo](https://github.com/NVlabs/FoundationPose/blob/main/run_demo.py).
- **BundleSDF (CVPR 2023):** persistent keyframe memory and pose-graph refinement
  are the general mechanism for reducing drift while registering an unknown
  object. The paper's RGB-D result cannot be transferred as a performance claim
  to Track 1 monocular depth. [Project](https://bundlesdf.github.io/).
- **GRC-Pose (September 30, 2026), §3.2–3.3:** canonical-geometry-to-scene
  correspondences, inlier memory, robust branches and a temporal posterior are
  relevant ideas. This is a paper-only lead; release/checkpoint/provenance and
  applicability to shared clip-constant scale are not verified. Do not install
  an unverified model or adopt benchmark calibration/per-frame Sim(3).
  [Paper](https://arxiv.org/html/2609.39116v1#S3.SS2).

The old causal diagnostic exposes two different errors: early segmentation of
the wrong object, and depth/pose jitter even when RGB appears stationary. New
boxes/masks address the former. Fixing the latter requires a separate measured
sequence-level SE(3) objective using persistent RGB/canonical correspondences,
confidence and contact evidence—not forcing objects static, smoothing each
frame blindly, regenerating a mesh every frame, or tuning episode-specific
weights. Keep metric accuracy and input-relative proxy evidence separate.

## Local checks

`tests/test_gemini_full4d.py` uses tiny manufactured metadata/payloads to verify
cache provenance rejection (GT, source calibration, wrong model, changed code,
gaps, wrong hashes), fixed cohort, honest allowlisted source entry, and preserved
immutable receipts. It does not test native GPU correctness. Native outputs must
pass their existing numerical producer gates before any visual is published.

## Native low-cost follow-up identified (not yet adopted)

The pinned native `MHRParityPostOptConfig` already exposes
`freeze_object_rotation=False`; the current wrapper explicitly freezes it.
Rotation correction is therefore unavailable in this causal run. An independent
controlled follow-up is A=current frozen rotation, B=same native optimization
with rotation unlocked, C=B plus automatic persistent RGB point reprojection.
B−A and C−B separate mechanisms. No new weights are required for B; C requires
qualified attachments/tracks and positive-weight gradient/runtime qualification
of `src/world_reward/joint_point_objective.py`. Its zero-weight delegation gate
is not evidence that positive-weight joint fitting works.

Native optimizer source: NVIDIA `video_to_data` revision
`7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`,
`reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py`,
SHA256 `84e0e818a3bc0935bb30b75fcd82fd7c5e3730ed812864594cd759697ddb406b`.
Existing declared CoCoNet/BootsTAPIR assets must be rehashed on Azure before reuse.
Choose/calibrate any new objective on qualified non-challenge data; never force
static objects or erase real motion to reduce jitter/acceleration proxies.

## Fail-fast observation gate for subsequent runs

The first full SAM3.1 run reports 244 empty object-mask frames in episode 1
and 128 in episode 7. The unchanged per-frame pose initializer requires at
least 40 inferred visible object points at every frame: a zero mask necessarily
fails regardless of depth. A later orchestrator revision authenticates the
original source, full mask inventory, tracking/seed RGB checksums and native
area/gap summaries before any learned 4D calls. It reports these as unsupported
native absence, not true physical absence, and retains the four-clip denominator.
No mask filling or fabricated trajectory is substituted. The source-bound area
metadata is not claimed to be an independent PNG decode or accuracy score.
This fail-fast code is not retroactively injected into running immutable jobs.

## Progressive full-video delivery

Legacy publication still requires three/four complete clips. A separate explicit
partial protocol publishes **all** currently completed one/two clips, derived
from a snapshot of the same original four-clip numerical report; no caller
selection and no unfinished clip gets a video. Numeric producer, publisher and
snapshot hashes are retained separately. Each actual full-T export/render must
already pass its existing source/geometry/timeline gates. Statuses are frozen at
capture, not falsely live. Remaining/failed clips stay visible in the denominator.
Private MP4 previews remain bounded to 2 MB each and stream only on demand,
without local video files or tokens in browser URLs. Tiny partial metadata has
an explicit 64 KiB ceiling; the legacy 16 KiB protocol is unchanged. Cleanup is
restricted to exact original publisher-owned names and conditional ETags.
