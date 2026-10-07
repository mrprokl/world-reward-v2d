# Qwen masks → geometry → full 4D

## Frozen first geometry gate — 2026-10-07

User approved the limited Qwen3-VL/SAM2 visual screen. Preserve its full original
mask streams, model, task prompt, seed and all receipts, including the original
preview-only failure. No human-supplied test labels or per-clip correction.

Before full-video CARI reconstruction, run the unchanged historical native
SAM3D Body, MoGe2, human-anchored scale and SAM3D Objects functions on exactly
episodes 8/9/26 and frames `[0,T//2,T-1]`. Object generation uses frame zero,
seed zero and the same inferred camera/pointmap; no oracle/camera calibration.
The object has **no trajectory yet** and must not appear static at later frames.
Body proposals remain frame-independent, **not** final clip-constant identity.

Configuration: `configs/qwen4d_initializers_v1.json`. Four stages per clip,
900 seconds each, total host budget 4,200 seconds, exclusive existing H100 lease.
No model acquisition, automatic retries or geometry-repair route. Independently
failed clips retain their failure and the same protocol continues on other clips.
Preview cap 180 seconds/180,000 bytes per clip, fixed JPEG quality 60/320px cells.
All heavy outputs remain on Azure. Only those tiny QA images may cross locally.

Store candidates under `experiments/qwen4d-v1-<source-commit>/outputs`, never
the baseline `outputs`. The common output helper changes storage paths only;
default legacy layout and numerical estimators are unchanged. Native workers
cannot write baseline outputs, model weights, source, data or global caches.
Mask import verifies original SHA-pinned reports/prompts plus measured full-T
PNG inventories; the absence of historical independent PNG pins is disclosed.

Read-only visual comparison: original RGB, Qwen/SAM2 masks, historical frozen
full-4D mesh export, new three-frame initializers. All files consumed are bound
before/after display. No fitting, camera realignment, component removal or labels.
This historical-vs-initializer view is **not** a controlled final-4D A/B score.

**Stop gate:** missing/off-target body or object, visibly wrong scale/geometry,
camera/provenance disagreement, native numerical failure, or resource deadline.
Then investigate the general failure, not manually adjust that test sequence.
If useful, user reviews tiny visuals before spending compute on full-T body,
depth, rigid trajectory and native CARI shared-identity refinement/export.
Only external non-overlapping validation can establish generalization; this
challenge development pilot cannot establish victory over CARI4D.

## Cleanup

Removed 166 local regenerable cache/obsolete-preview files, 9,121,802 bytes.
Baseline comparison images, current Qwen images, immutable source, pins, results,
credentials and environments remain untouched. Historical baseline artefacts
stay read-only until the comparison has consumed them; dependency-bearing
sources are not deleted merely because the old predictions were wrong.
