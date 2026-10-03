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
