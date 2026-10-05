# Frozen joint pair scorer — before new FIT observations

This recipe supplements, and does not change, the frozen64-image protocol.
24FIT images alone fit B;8DECISION are diagnostic, not a hyperparameter/threshold
grid;32TEST are opened once. All creator-profile groups are disjoint. Neither
author disjointness nor this implementation proves pretraining/challenge overlap.
No labels or candidate reference boxes enter feature construction or scoring.

## Common automatic bank and baseline A

Use the complete ordinary person/object proposals, native DWPose133 crops and
optional complete HOI-DETR routes. Exact original **pixel-coordinate tuples**
pool automatic person/object aliases before normalized evaluator transport;
this is not a physical identity claim. All coordinates/groups remain through
raw→unique maps. No GT aliases, threshold, top-one or missing-positive repair.

A ranks ALL unique P×O pairs by the lexicographic negative distance of the object
center outside the closed person box, then person/object center distance, both
divided by that original person box's diagonal. Exact tuple ordinal ranks permit
the existing scalar retrieval evaluator without weighted sums or a tie breaker.

## B:30 values +30 numerical-availability indicators, one linear score

Reuse `InteractionCandidateEvidence`'s ten columns. Across both sides and exact
box aliases, five distances each pool min/max/mean of **distinct supported
values**; person/object IoU pools max; four raw scores pool mean. Duplicating an
exact value does not increase its mass. Missing joints do not become anatomy:
unavailable descriptor transport is0 plus availability=false. Two baseline
geometry distances add two values. This leaves20 base+2 geometry descriptors.

Eight HOI descriptors pool all person aliases/sides/native-pair/generic-object
routes: minimum wrist→hand-box distance; minimum hand-root→hand-box distance;
maximum direct→generic IoU; minimum direct→generic center distance; maximum raw
interaction margin; geometric-weighted means of margin, hand raw score and
direct-object raw score. Distinct complete route evidence is pooled once.
The fixed weight is `IoU/(1+wrist_distance+root_distance)` using supported
original-image-diagonal distances. Zero-IoU routes remain in the raw max/min
descriptors but contribute zero to weighted means. No owner/contact probability
is inferred. Empty/no-supported HOI produces zeros with false availability,
never a dummy native pair. Sides have no named left/right model coefficients.

## Private FIT labels and numerical recipe

After automatic features freeze, evaluator-only matching supplies a boolean
mask of observed-positive **joint** P/O pairs. The image-balanced loss is
`LSE(all scores)-LSE(observed positive scores)+0.5*||coefficients||²`.
Unknown alternatives are implicitly suppressed by its denominator: this is NOT
binary negative GT, unbiased PU/contact classification, OFF training or a
calibrated probability. Multiple positives permit any-positive retrieval, not
recovery of every positive, and the linear objective need not be convex.

Exactly24 FIT slots/unique authors are supplied; missing-positive or no-ranking-
alternative records are counted and omitted from the undefined/uninformative
loss, never repaired/replaced. At least12 usable authors are required or FIT
fails. They remain misses/support diagnostics in their evaluation denominators.
Available usable FIT values alone determine mean/std, equal image then equal
pair mass, conditional on feature availability. Constant/absent scale=1.
Unavailable standardized values=0; append30 availability indicators. Intercept
is fixed0 because adding an image-wide constant cancels from the listwise loss.

One zero initialization, no RNG/restarts/grid. NumPy FP64 full-gradient Armijo
descent: initial step1,50 possible halvings, decrease constant1e-4,max1000 steps,
gradient infinity norm≤1e-7,120-second total FIT deadline. Every objective,
gradient, scale and returned score must be finite. Nonconvergence/deadline fails;
no alternate optimizer or partial model. A stationary solution is not a global
optimum or empirical success. Source/runtime NumPy and inputs must be pinned by
the calling producer; unit fixtures do not qualify real runtime or model data.

## Evaluation boundary

Freeze B coefficients/scaler and complete A/B scores before TEST references.
Use the existing positives-only joint retrieval metric on all32 frozen records:
missing access/bank, abstention or unknown top pair means an annotated positive
was not retrieved, not a demonstrated false contact/owner. Keep localization,
unresolved-positive and exact-alias counts. Never reopen old closed pilots,
retune B on DECISION/TEST, or claim task selection/hand ownership/temporal3D or
a CARI4D victory from this static component. Implementation PASS is not quality
PASS; config is `configs/openimages_joint_pair_scorer_v1.json`.
