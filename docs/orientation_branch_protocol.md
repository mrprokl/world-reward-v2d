# Orientation hypotheses retained through joint fitting — prospective

**Design only, 2026-10-05; not implemented or executed.** This is
a new conditional synthetic component experiment, not a replay or amendment of
the translation-only point study, RGB-only end-to-end validation, or a CARI4D win.
No new model/data are needed. Freeze a complete recipe, code/runtime identities,
arm ordering and budgets in a new namespace before manufacture or prediction.

## Question and actual novelty

[CARI4D v3, §8](https://arxiv.org/html/2512.11988v3) explicitly reports large
FoundationPose 180° flips beyond refinement's capture range. Its
[released CoCoNet](https://github.com/NVlabs/CARI4D/blob/71fa7cbe46081467edadd11ab534b0c14aa9d913/learning/models/refine_tempnet.py)
already mixes human/object spatial-temporal information; the original
[postoptimizer](https://github.com/NVlabs/CARI4D/blob/71fa7cbe46081467edadd11ab534b0c14aa9d913/learning/training/opt_refineout.py)
already exposes object-rotation optimization. Thus neither adding rotation nor
another seed search is new. Our existing native24+1 pool and Viterbi already
select globally **before** continuous fitting. The proposed difference is to
keep several whole-sequence orientation hypotheses until the **same coupled
objective** has fitted each, then select one full trajectory without truth.
[4DAnyone RCP/TCR](https://arxiv.org/abs/2608.20335) motivates sharing bounded
context, not invented views, metric observations or a replacement HOI generator.

## Minimum causal comparison

All arms receive one frozen native candidate pool, automatic masks and Boots
tracks, original frame/query IDs, identical t0 material attachments, native F32
mesh, K, scale, human initialization, contact logits and complete joint loss.
Shape/identity, human root/hands and t0 object pose are fixed identically; the
same body-control policy, priors/reference tensors, gradient settings and301
updates apply. Contacts are not relabelled by the selected branch. No pose
averaging, GT seeds, new tracks/refill or branch-dependent observation filtering.

| Arm | Initial trajectory | Object rotation during fit |
|---|---|---|
| A | Original pool/Viterbi top1 | Fixed, as in the current point study |
| B | The exact same top1 | Unlocked for t>0, proper SO(3) parameterization |
| C | Top1 plus up to three diverse whole-T paths from the **same** pool | Same unlocked domain as B; fit each, select one globally afterward |

**C versus B tests delayed branch selection; B versus A tests rotation freedom.**
A versus C alone is confounded. B is C's first branch and its frozen result is
reused, not rerun: at most five fits per scene, not six. Independent constructor
state/gradient qualification must precede comparisons; do not infer exact native
repeatability from seeds or reuse the failed parity cohort.

Paths preserve one common t0 and every source frame. Candidate diversity uses
only their predicted rotations over automatically supported frames; the path
generator, angular separation, bounded search cap and deterministic tie order
must be fixed on development data. If the pool supplies fewer than two genuinely
distinct admissible paths, record insufficient hypotheses; do not inject a
truth flip. C selects the lowest **common, unregularized-observation plus native
regularization objective**, evaluated identically over full T after every fit;
retain individual components. Branch-specific initializer priors or different
support denominators invalidate selection. Ties use original branch order and
are reported as unresolved, not evidence of correct material orientation.
The common t0 anchor makes this a test of **later orientation loss/recovery**;
it does not prove correction of CARI4D's initially wrong material association.
That would need separately qualified branch-dependent attachments, not relabelling
the fixed canonical witnesses after seeing their tracked trajectories.

## New cohort and universal calibration

Use two new DEV and four untouched reserved T24 scenes: fixed authored textured
objects, nonzero translation/rotation acceleration, interacting articulated
human and one contiguous middle occlusion with visible endpoints. Cover an
asymmetric shape and a geometrically half-turn-symmetric shape with asymmetric
texture in both DEV and reserved strata; distinct trajectories/material seeds.
The exact dimensions, pose formulas, occlusion intervals and native feasibility
checks remain to freeze, **not** choose after inspecting their output. Manufacture
mesh/K/masks/t0/contact controls are explicitly synthetic oracles, identical in
all arms. Future poses/tracks/visibility are never supplied to fitting; Boots
observations are computed once from RGB and sealed before private scoring.

Perfect geometric symmetry cannot reveal material orientation from shape or
geometric contact. Asymmetric texture can help only if the common t0 material
association and subsequent automatic tracking actually carry that information.
Support, rank and an alternative-orientation likelihood diagnostic are required;
an indistinguishable observation is an honest ambiguity, not a successful fit.

Calibrate outside the challenge, once on these two DEV scenes: point noise scale
and weight, rotation learning-rate/parameterization, rotation-prior strength,
path diversity/search rule (maximum4 branches), and a common branch-score
normalization. Also freeze query count/support, contact/temporal weights, step
count, numerical tolerances and tie/ambiguity rules globally. No per-scene tuning
or reserved-label inspection. Do not transfer the current translation study's
weight silently: its translational gradient balance does not calibrate rotation.
Prefer its already-qualified operators/configuration when valid; any change is
explicit. The generator must not read future/private truth to create candidates.

## Scoring, cost and stop decisions

Freeze **every branch** and selected full-T outputs before private evaluation.
Use camera-frame object surface/material-point displacement and SO(3) error,
object-to-wrist relative error, human PVE/MPJPE, actual native proxy penetration
(not an exact solid certificate), and translational/angular second-difference
error against known dynamics. Report visible and occluded intervals separately;
no initial/per-frame/per-object alignment, static substitution or frame deletion.
Report geometric symmetry-equivalent and material-orientation errors separately,
never use GT symmetry or orientation to select an output branch.

Proposed component go-gate, frozen before reserved prediction: C versus B median
object/material and object-to-wrist error gain≥10%; neither has a reserved-scene
regression>5%; human/proxy-penetration/dynamic errors do not regress>5%; full24
coverage and proper rotations. A zero baseline error requires zero candidate
error and supplies no relative gain, not a division or artificial gain. These four pairs are
descriptive effects, not statistical noninferiority or real-video proof. Report
B versus A even if C's gate fails. No winner is declared from RGB loss alone.

At most30 fits across six scenes: native301 updates, full-batch T24, **120 s per
fit / 3,600 s fitting total**, plus manufacture600 s, one Boots load/six calls900 s,
native pool construction plus path generation1,200 s and scoring/source
sealing300 s: **6,600 s inclusive compute**. Oracle manufactured masks/depth are
common controlled inputs; do not silently charge an unbudgeted model frontend
or present this as a production RGB-only pipeline.
Verify on new DEV only that this fixed ceiling is feasible; do not enlarge it
after reserved execution. B reuse and unchanged observations expose C's extra
compute explicitly. Sparse/ambiguous attachments, missing hypothesis/support,
invalid gradient/geometry/shared references, any failed branch, nonfinite pose,
deadline or inability to retain all frames ⇒ close the experiment without retry,
scene deletion, lower thresholds or partial PASS. Geometry/runtime feasibility
must pass before rendering/Boots; calibrated DEV evidence must pass before the
reserved fits. Production adoption requires later independent lawful RGB-only
validation and license/overlap clearance, not this conditional component test.
