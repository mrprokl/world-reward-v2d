# Prospective native A/A numerical repeat control

**2026-10-05 — not executed.** This is one distinct authored control of the
original native MHR optimizer, not a replay, learned inference or HOI accuracy
experiment. Both constructors are original `MHRParityPostOptimizer`; no point
subclass or point-optimizer factory runs.

The first A/B control `1b1ac13…` remains **CLOSED FAIL**. Its two initial states
differed only in `reference_foot_vertices` (146/2304 values, maximum
4.76837158e−7 m). Probe gradients and some forward metrics differed; at step181
the penetration/total loss differed by 2.38418579e−7. These observations do not
identify a cause or establish that the point extension caused the differences.
The retained CPU audit reads saved arrays only, not new native execution.

## Frozen scene and unchanged dependencies

`configs/joint_point_native_repeat_protocol_v1.json` independently binds the
original authored protocol by its complete SHA256 and inherits its exact source,
model/acquisition, installed Body, renderer/crop, contact/Kaolin and runtime
contracts. Primary NVIDIA optimizer revision
[`7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`](https://github.com/nvidia-isaac/video_to_data/tree/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80)
has source SHA256 `84e0e818…406b`. Existing checkpoint/model pins and full source
checks are mandatory before decode. Native source and image are not modified.

Construct a NEW T=3 scene, original indices0,1,2, 640×480, same fixed OpenCV K
and real compact-zero MHR body/hand conversion. Zero scale28 remains PCA
coefficients, not zero physical scale. Human x(t)=0.0125t+0.0075t², y=0, z=4 m.
The new asymmetric tetra uses exact dyadic coordinates with denominator64:
`[-3,-2,-2], [4,-1,-1], [-1,5,-1], [1,1,5]`, and outward faces
`[021,013,032,123]`. Both arms use the same native-loaded F32 geometry, R=I and
scale1. Placement remains the source-bound named left-hand vertex mean at
frame0 plus `[0,0,-0.045]`, then the new human translation increments.
No earlier raw control arrays, QA-selected placement, rerender or repair.

## Actual sequential A/A test

Use the existing caller with explicit `--control native_repeat` and a distinct
`joint-point-native-repeat-<revision>` result namespace. Default authored A/B
behavior remains separate. Semantic arm names are `A_native_first` and
`A_native_second`; neither is a zero-weight point extension.

Retain the SAME sequential lifecycle and shared real MHR layer: reset native
Python/NumPy/Torch/CUDA seed0; first original constructor, pre-update probes0
and181,301 native updates; second original constructor, pre-update probes0
and181,301 native updates. TF32 remains off; deterministic-algorithms policy,
Adam/scheduler, all native losses, full batch0, fixed object rotation/internal
translations and reporting100 are unchanged. No warmup, cache reset, alternate
decoder or deterministic-kernel override is added to make this pass.

Once after the first constructor, retain the unchanged 32 mask-raster quantile
query diagnostic and its min8/distinct/ambiguity gates. This is availability
only: do not bind point evidence, invoke a tracker or attach any point loss.

Compare **exactly** every retained initial state value, optimizer/scheduler,
probe loss, metric, gradient and native kernel count at0/181. Confirm real
contact/render and penetration/Kaolin execution. Probes181 still use the initial
parameters; they activate the penetration schedule, not181 completed updates.
Any exact mismatch stops before the second run.602 is a target, never a claim
when the second arm did not complete. If both runs complete, compare complete
native results/history without removing metadata. No tolerance or excluded
cache fields; signed zero and every original array remain part of comparison.

## Evidence, budget and interpretation

Keep full manufactured numeric inputs/RGB/Z/decoded geometry, source bundle,
quantile diagnostic, two initial snapshots, four complete probes and both full
result bundles on Azure. Seal source/models/inputs/results and owned cleanup.
Only concise counts/hashes/decision reach the local workspace.

Budget: manufacture180s, pair1200s, total1380s including sealing; outer1440s and
bounded owned cleanup30s. No extra compute, retry, rerender or adaptive gate.

FAIL means this native baseline repeat is not bitwise qualified under the frozen
lifecycle. It cannot attribute causality to the extension. PASS qualifies only
this new A/A control: not universal numerical determinism, old A/B recovery,
calibration, tracking/accuracy, submission eligibility or adoption. Any later
extension comparison needs separately frozen new support. Shared-layer warm
state, CUDA reduction order and backend nondeterminism remain hypotheses until
independently discriminated; no PyTorch3D result is evidence about nvdiffrast.
Model license eligibility and training/challenge overlap remain unverified.
