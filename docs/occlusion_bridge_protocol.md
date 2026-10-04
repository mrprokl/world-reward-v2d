# New CPU occlusion-bridge initializer gate

This experiment is an **initializer operator**, not full RGB validation, contact
proof, final reconstruction or a verified improvement over CARI4D. No challenge
records, model inference or new data acquisition participate.

Frozen before the first run: seed **310427**; 12 scenarios, each 48 frames:
accelerating carry, curved carry, free motion, slip, regrasp, noisy measured hand,
each with a centred 3- or 12-frame object-observation gap. Fixed asymmetric
surface witnesses evaluate pose displacement in the original camera gauge;
there is no alignment, shape/scale fitting, missing-frame deletion or static
final trajectory. Synthetic poses manufacture observations; hidden poses are
used only for evaluation, never passed to the solver. Missing inputs are NaN.

Method: three consecutive observed object/hand poses on each gap side. RMS
hand-relative translation dispersion ≤0.03 m and rotation dispersion ≤0.15 rad
permit hand-conditioned initialization. Mean relative rotation uses a proper
polar projection. Smooth endpoint residual correction retains both observed
anchors. Missing hand evidence, slip/regrasp instability or insufficient anchors
use explicitly inferred endpoint SE(3) interpolation. Leading/trailing gaps
without both observed endpoints abstain; no last-pose extrapolation is supplied.
Relative stability proposes a branch, **not proof of contact**.

Comparators: endpoint SE(3) interpolation and last observed pose. Gates:
100% coverage of the 12 bilateral fixtures and bit-identical observed poses;
≥10% surface-displacement improvement for every carried gap; no >5% error
regression in any noncarry stratum. Last-pose results are diagnostic, not the
sole comparison. Tests additionally require explicit abstention for leading/
trailing gaps, proper SO(3), strict source validity and no mutation.

Run once with `python tests/test_occlusion_bridge.py`; aggregate result is emitted
without pose arrays. Any scientific gate failure rejects adoption: retain the
decision, do not retune thresholds or reuse the cohort to manufacture a pass.
Even a pass would require a separate new validation using actual automatically
inferred camera-frame hand/object observations before challenge use.

## Frozen first-run decision

CPU unit contracts: **16 passed**. The unchanged scientific cohort run returned
**PASS**, cohort SHA256
`05453436f09687c8fddb2202afbe62f911ed38bdbd7c5c6f44249ee7dcfecf63`.
Minimum carried-gap gain was 0.9999999999998699; no frozen gate failed. This large
gain is unsurprising because those fixtures construct the object consistently
from the measured hand. It establishes operator mechanics, not realistic model
accuracy or permission to adopt it in challenge reconstruction.

Unseen slip/regrasp occurring entirely inside a fully hidden interval can leave
both visible anchor groups apparently stable. The method cannot identify that
event without additional evidence; its hand branch remains a prior. Likewise
stationary fallback motion is possible when endpoints coincide, but is not
claimed to reflect actual hidden motion or rewarded as a quality improvement.
No thresholds or fixture parameters were retuned following this result.


## Next independent RGB availability-ablation gate (not yet run)

Freeze a NEW authored 4×48 cohort: accelerating carry, free motion, slip and
regrasp. Hide only automatic object-observation rows 9–11 and 20–31 after
freezing all 192 model-inferred observations. Both branches receive identical
RGB, camera, shape, scale, predicted object mesh and named camera-frame joints.
This tests missing-observation initialization, **not natural occlusion or physical
contact**, and cannot authorize challenge adoption by itself. Existing rejected
closed-human/raster references and already exposed cohorts remain rejected.

Fail fast in this order: independent nearest-triangle FP64 ray/camera/depth
microgate; four fixed RGB anchors with automatic masks/Body/MoGe; all observations
and single clip gauge; source-bound named127-joint frame and independent RGB
hand-support checks; freeze both complete output trajectories; private authored
pose evaluation only afterward. Finite hand joints alone are not validity.
Without enough RGB evidence, the unchanged operator must take its explicit
endpoint branch or abstain; no ideal-hand or per-episode rescue.

Use full asymmetric-surface camera displacement without alignment. Require
carry improvement ≥10% versus endpoint interpolation and no free/slip/regrasp
regression >5%, complete original indices, byte-identical observed poses and no
leading/trailing extrapolation. Report absolute pose/motion errors and all
abstentions; do not retune or reroll after the first result. A model-inferred
synthetic result is still not external natural-video generalization.

## Four-anchor predicted-geometry QA — frozen before first evaluation

The four new RGBs and all automatic masks/full Body/MoGe predictions have
completed and are independently byte-pinned. Before manufacturing 192 frames,
run CPU-only QA on the *predicted* geometry, with no private authored mesh,
pose, camera or labels available. Keep all predicted triangles and the original
640×480 K800 gauge; no alignment, shape/scale fitting or per-record adjustment.
Require predicted human-only silhouette IoU ≥0.70 against the automatic person
mask for every anchor. Human-only rendering includes self-occlusion but cannot
remove human regions hidden by the object; report this limitation rather than
deleting geometry to improve IoU. Derive a left-hand proper camera-frame basis
from the actual named 127-joint wrist/index1/middle1 positions. All three must
have positive Z, lie in the original image and be within 5 pixels of automatic
person-mask support. Right-hand results are diagnostics, not a selection rule.

These fixed gates establish only a silhouette/support **proxy**, not independent
RGB finger-keypoint accuracy, contact, metric accuracy or bridge adoption.
Degenerate/unsupported geometry stops the pilot; never supply ideal hand poses,
retune the thresholds, alter the bottle/seed or fit this cohort after its result.
Store scalar decisions and original byte identities only; all heavy arrays and
masks remain on Azure. Even PASS still requires clip-constant identity/scale,
RGB-inferred object geometry and independently evidenced hand observations
before the unchanged availability-ablation can be run.
