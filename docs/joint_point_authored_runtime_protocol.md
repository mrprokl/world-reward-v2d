# Fresh authored real-MHR point-objective runtime pair

**Prospective only, 2026-10-05.** The companion
`configs/joint_point_authored_runtime_protocol_v1.json` freezes one new three-frame
control. Its bounded caller has executed and **failed exact A/B parity**;
runtime qualification and accuracy improvement remain unproven.
EP21's original insufficient-query failure
stays closed; neither its inputs nor any earlier authored control is reused.
This is a small new caller of existing operators, not another inference pipeline.

Implementation: `infra/joint_point_authored_qualify.py` calls the original
decoder/render/crop and existing paired sequencing, retaining `pr_initial`
before execution to preserve the unchanged native result ABI. Actual Python
kernel entrypoints are observed by a scoped profiler, not monkeypatched.
Root279 tests pass with10 Torch-dependent skips; these are caller/control tests,
not real GPU evidence. Complete installed Body/native source and independently
pinned model bytes remain runtime prerequisites. One fresh Azure dispatch is
executed at `1b1ac133ea59aeef732a6552b9a86dbb173494c1`; its failure is frozen in
`configs/joint_point_authored_failure_pins.json`.

Actual manufacture14.838s/fullT3 succeeded;32/32 mask-quantile attaches were
distinct, supported and unambiguous. Two constructors and all four probes
returned with actual contact/render and step181 Kaolin sign/distance calls.
A completed301 updates; exact initial/loss/gradient comparison then stopped B
**before its optimization**. Native29.908s/host33.990s, source posthash and owned
container cleanup verified. Twelve complete native artifacts remain on Azure.
The602-update pair and full-result parity were **not achieved**.

A CPU-only audit of retained states/probes/result, without decode/loss/optimizer
replay, localized the initial difference to146/2304 `reference_foot_vertices`
F32 entries (max4.768e-7m); other initial values/optimizer/scheduler were exact.
Forward diagnostics/contact also differ; step181 total and penetration differ
by2.384e-7. Gradient maxima differ up to1.655e-4. This is not proof that the
point subclass caused the differences, nor that backward atomics explain all
of them. Seed/TF32 settings do not establish baseline bit reproducibility.
The exact gate stays unchanged; no tolerance, repetition or rescue of this
scene. A separate fresh **original-vs-original** control can test native
self-reproducibility before attributing this failure to the extension.

## Exact source and model prerequisites

The primary code is NVIDIA V2D revision
[`7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`](https://github.com/nvidia-isaac/video_to_data/tree/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80).
The JSON records exact URLs/bytes/SHA256 verified from primary text in RAM:
optimizer `84e0e818…406b`, MHR layer `a753ab8e…8c23`, native body/schema/rotation
converters, postopt crop, hand-surface specification code, native renderer and
Body head. These pins are not proof of all transitive installed dependencies.
Before any decode, authenticate the actual b47e4450…380a7 image, complete live
native/Body source and existing model/acquisition receipts. Verify the exact
4000-vertex collision proxy and hand specification against decoded faces.
Supply the actual checkpoint and TorchScript model; explicitly select an absent
compact-buffer path. No guessed model/checkpoint hash or substituted decoder.

PyTorch3D transforms/distance kernels, Kaolin collision, nvdiffrast CUDA renderer,
Transformers scheduler, native MHR/Body head and their installed sources must be
real and authenticated. Their runtime availability remains to be tested here.
Native/model license eligibility and training/challenge overlap are **unverified**;
user grants are not a license waiver. This runtime test authorizes no submission.

## One deterministic scene, before seeing its outputs

Use original indices/names `0,1,2` / `000000,000001,000002`, 640×480 and the fixed
JSON OpenCV K. Construct F32 MHR parameters through the primary converters:
zeros133 → valid body260, zeros27 → hand54 for each side. Do **not** use zero260
as a neutral rotation basis. Root six-dimensional identity is interleaved
`[1,0,0,1,0,0]`. Shape45, scale28 and expression72 stay zero and clip-constant.
Scale28 is a PCA coefficient vector: head line226 computes
`loaded_scale_mean + scale_params @ loaded_scale_comps`; zero is not factor0.
The real head outputs cm/100, then MHRLayer applies its single `[1,-1,-1]` flip
and camera translation. No extra axis/metric conversion is allowed.

Decode actual full vertices, joints, keypoints and faces. Human translation is
`[0.01*t + 0.005*t*t, 0, 4]` metres; this supplies nonzero acceleration at T=3.
The new asymmetric tetra's four F32 vertices and four I64 faces are literal in
JSON, with outward faces `[021,013,032,123]`. The ideal decimal determinant is
positive `0.000814791 m³` (six-times-volume); no numerical tolerance/repair is
selected from outputs. Both arms use the identical actual F32 loaded geometry.

Place it once at the mean frame0 decoded vertices selected by the authenticated
specification's **named left_hand** row, offset `[0,0,-0.045]` metres. Left/right
order is primary `MHR_HAND_ORDER`, not an inferred physical identity or guessed
wrist ordinal. Add the same authored translation increments, keep R=I and scale1.
This placement may be occluded or insufficiently near the actual hand: that is a
legitimate whole-control FAIL, not permission to reposition or rerender.

## Actual render and native bundle ABI

Use native `Utils.nvdiff_color_depth_render` / `RasterizeCudaContext`; render
human/object separately and compose nearest positive camera Z with fixed colors.
Exact equal positive depths fail instead of inventing a visibility tie-break.
Full masks are bool, background depth0 invalid. Use the **original**
`build_postopt_crop`/`stack_postopt_crops`: K256 is their result, not a guessed
resize. The declared contract is
`cari4d.smplh_postopt_full_resolution_crop.v1`; cropped masks are binary F32
`[3,256,256]`, cropped intrinsics `[3,3,3]`.

Bundle schema `cari4d.mhr_wild_inference.v1`, `gt={}`, full seven F32 parameter
blocks, proper F32 `pose_abs[3,4,4]` and authored contact logits `[1,-1]` each
frame. Metadata records manufactured-not-inferred, GT false, actual object path,
native aligned/storage frame revisions, two identity transforms and cache=None.
The existing `cari_full_refine.validate_source_bundle(...,3)` must pass without
fabricating a CoCoNet/Track1 producer receipt. Effective left contact must be
positive in **all three frames**; authored logits are not network predictions.

## Same native optimizer, zero-weight extension

Create A with the actual `MHRParityPostOptimizer`. Attach quantile queries only
after reading A's actual frame0 `_object_state` and native-loaded F32 mesh:
32 mask-raster quantiles, mask-only selection before rays/depth, existing min8,
distinctness and ambiguity gates. No refill, camera snap or analytic pose stand-in.
Evidence has full3 finite projected XY256 and all future support false: a
numerical attachment control, **not** a tracker. Explicit scale1/weight0 and the
runtime reference are not calibrated parameters for positive-weight fitting.

Reset Python/NumPy/Torch/CUDA seed0 before each constructor; TF32 off, original
deterministic-algorithms policy unchanged. Preserve native 300 requested steps,
full batch0, fixed object rotation/internal translations, real contact/render/
penetration/priors/Adam/scheduler; only report interval100 and explicit asset paths
are supplied. Before updates, compare initial state, optimizer/scheduler and
all loss/metric/gradient bytes at steps0 and181. Confirm real kernels executed,
including the penetration path at181. Run301 updates each, 602 total; compare
full results/history exactly after removing **only** B's point-objective metadata.
No tolerance, positive weight, loss stub or frozen/static optimization substitute.

## Evidence, budget and stop rule

Freeze full authored raw arrays and actual decoded geometry on Azure **before**
the pair; keep both full native outputs, four complete probe records and hashes
afterward. No sample/challenge records, prior cohort, GT, inherited prediction or
human observations enter. Local holds only this protocol and scalar receipts.
Manufacture180s + pair1200s =1380s inclusive of loads, hashes and sealing; outer
1440s allows bounded30s owned cleanup, not extra compute. Missing source proof,
decode/crop/contact failure, too few rays, ambiguity, nonfinite values, parity
mismatch or deadline ⇒ STOP without resampling/rerender/refill/reposition/rescue.
Even PASS establishes only this real runtime pair, never CARI4D accuracy,
physical contact validation, positive-weight calibration or adoption.
