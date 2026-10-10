# Next contact hypothesis: feasible native pose SQP

## Actual diagnosis, not another weight sweep

Root-reported full-real producer `c25f0dfd866b8ac3df56ea03c33a68d19b456341`
completed in **96.371 s**. Both original supported records returned the complete
moving-A fallback, **no gain**; all nine global-alpha proposals were rejected.
Episode 9: placement failures around frames 99/108. Episode 14: placement
failures around 231/233 or exact-contact rejection. Unsupported original records
1/7 remain in the denominator. These are execution summaries supplied by root;
the detailed sealed remote report was not independently downloaded here.

The preceding baseline-reference failure was numerical, not evidence of wrong
coordinates: root measured native-vs-saved-A maximum vertex errors of
`9.83e-7 m` / `5.96e-7 m`, larger than the unchanged `1e-7 m` witness slack.
Recomputing the SAME automatic IDs' bounds on actual native A before any proposal
fixed that reference mismatch. Both reference and verification use `(p-t) @ R`;
the exact inverse in the placement primitive transports corrections only.

Three structural limitations remain:

1. A scalar alpha explores just **one clip-wide direction** in a high-dimensional
   articulated space. If one active constraint has positive directional
   derivative at A, every positive small step in that direction can be invalid.
   One difficult frame blocks unrelated valid improvements elsewhere. More
   dyadic trials do not create a new feasible direction.
2. A shared relative translation cannot change the separation of two anatomical
   hand witnesses. B changes the native body pose despite frozen hand controls;
   the two wrist/hand placements can therefore require incompatible translated
   object-surface tubes. Searching four surface branches does not repair the
   articulation itself.
3. Near alpha zero, SO3/SO2 retraction and native FP32 decode are quantized. Taking
   smaller steps cannot guarantee monotonically vanishing geometric error.
   This is separate from the first two structural failures.

## One selected replacement

**Feasible-start sequential convex / SQP optimization of native articulation and
the two actor translations**, initialized at actual decoded A. Keep the original
body/root/object/hand identity restrictions, full timeline, image observations,
and published native objective coefficients. Replace a weighted contact tradeoff
with hard SAME-witness inequalities:

`g[t,s](q) = exact_all_triangle_distance(native_vertex(q), object(q)) - native_A_bound[t,s] <= 0`.

At each outer iteration, get native pose/translation Jacobians and closest
surface points on the complete object. Solve one sparse trust-region QP for a
descent direction of the existing training-image/native objective subject to
linearized contact inequalities. Use the objective's Gauss–Newton metric and
numerical regularization, not new manually weighted contact penalties. Apply
SO3/SO2 tangent updates in the audited native ABI; do not linearly mix meshes,
6D rotations, or the 136 direct-export controls.

Restore second-order contact violations with **pose plus translation variables**,
then freshly decode and check exact full-surface distances. Rebuild closest-face
associations at accepted iterates; an old barycentric anchor is a local proposal,
never an immutable claim of where a sliding hand must touch. Unsigned proximity
is not penetration/physical feasibility: retain native penetration/silhouette
terms and explicit final limitations. Fixed QA witnesses remain a conservative
comparison contract, not a whole-hand contact oracle.

This differs from the rejected method: the optimizer can rotate the two native
kinematic chains in different feasible directions while preserving the video
motion; it is not restricted to A-to-B interpolation or translation-only repair.
It is still a local hypothesis, **not a promised solution/global optimum**.

## Primary literature and what is actually borrowed

- [Open-CHOIR, 2026; v4 Oct 7](https://arxiv.org/html/2605.20992v4), sections
  4.2–4.3: rectify relative placement before association, then periodically
  rebuild barycentric contact proposals while preserving image constraints.
  It supports the architecture; its manual seeding, generative weights and
  soft confidence thresholds are **not** imported into our prediction contract.
- [CRISP, 2025](https://arxiv.org/abs/2502.01055): sequential convex QPs with
  adaptive trust regions for difficult contact-constrained problems. Borrow the
  sparse local step/restoration strategy, **not** its complementarity/dynamics
  assumptions or a claim that its convergence theorem covers our surface union.
- [Multi-contact whole-body retargeting, 2022](https://arxiv.org/abs/2206.00542):
  optimize full kinematics with SQP near feasibility boundaries rather than
  repairing end-effectors only by global translation. Robot force-equilibrium
  assumptions are not evidence about monocular human reconstruction.

## Bounded DEV → actual experiment, before any adoption

1. **Independent authored DEV**, separate fixed seeds for verification: moving
   two-chain bimanual contact, curved/sliding surface, release, occlusion of image
   observations, and a case where translation/one-direction continuation fails
   but articulation succeeds. Check complete chronology, hard-contact residual,
   full path/increment retention, and native SO3/SO2 Jacobians against finite
   differences. A small native-decoder Jacobian check is a runtime contract, not
   external HOI accuracy. Freeze trust-region/restoration numeric policy here.
2. **Actual saved full9/14**, same A/B inputs and native layer, no model/pose-init
   rerun. Predeclare **20 SQP steps / 900 s**, preserve every frame and all fixed
   contact activations. Batch16 native decode; at most two witness constraints per
   active frame; block-diagonal kinematics plus banded temporal objective. Cache
   fixed topology/image inputs and use an exact conservative surface broadphase;
   no all-hand-by-all-face quadratic matrix or 301-step optimization replay.
3. Seal complete native parameters, directly decoded136/68 controls and full
   geometry before final same original A/B QA. Training factors remain training;
   reserved human-image observations are **not optimization targets**. Require
   original RGB/contact/motion nonregression gates without tolerance retuning,
   then lightweight RGB/A/C visuals. External FORM metrics remain a separate
   after-seal evaluation, not a selector for these records.

Report actual objective decrease, each exact gate, QP/restoration status, decode
count, runtime and geometry checks. If there is no feasible local descent under
these constraints, preserve dynamic A and expose the local constraint/dual
diagnostic; do not claim global impossibility or a verified CARI4D win.

## Implemented core / authored DEV only

`native_pose_sqp.py` uses a **positive diagonal proximal metric**, not a full
Gauss–Newton Hessian: the full-frame objective gradient may contain temporal
terms, but each QP has only 133 local tangent/translation variables and <=4
linearized witness rows (two-sided at an unsigned zero-gap cusp). Its <=4 dual
variables are solved with box clipping and an active-system polish, followed by
explicit primal/duality checks. No dense clip-state matrix or all-triangle
pair matrix. Closest-face queries still examine the complete surface through
the existing exact conservative broadphase.

Native callback ABI is `[T,2,3,127]` anatomical camera-space Jacobians for
23 right-SO3 blocks plus 58 SO2 angles; human/object translation contributes
6 further variables. Full native V/J/KP/136/68 geometry remains callback-only.
Callbacks are checked before and after budget consumption; returned over-budget
proposals are discarded. A caller still needs an external wall-clock kill to
interrupt a callback that never returns. Frozen shape/hands/global rotation,
internal translations, IDs, activations and chronology are rechecked.

The QP aims **inside**, never beyond, the unchanged exact contact bound using a
half-ULP FP32 witness-coordinate reserve capped by the existing numerical slack.
This prevents restoration from endlessly rounding onto the wrong side of a
boundary. It tightens a proposal target; it does not increase the accepted gap
or learn a threshold from the videos. Exact full native decode remains decisive.

Independent authored seeds `60261010/60261011`, 96 moving frames with explicit
inactive/release frames: the uncorrected target's hand span exceeds object width
plus both baseline gap radii, proving translation-only infeasibility for that
state. Pose SQP reduced its known angular objective **2.45926→0.11767 (95.22%)**
and **2.46826→0.13095 (94.69%)**, respectively; six accepted steps / eleven
synthetic full-geometry callbacks each, about 1.3 s on the local tiny assay.
Every original active witness satisfies the original exact bound, root-Y motion
increments remain intact, and frozen fields are preserved.

The first DEV attempted a cap on total human-centroid path; this rejected useful
added articulation, illustrating why a motion magnitude cap is not a truth
metric. The authored test instead checks its independently known root-Y motion
increments. **No real/challenge gates were changed.** The actual full RGB/contact/
acceleration gates remain separately required during future native integration.
Two seeds share one authored mechanism, not a held-out real HOI benchmark.
`14` tiny tests pass, including native tangent finite differences, an incompatible
local QP, unsigned cusp, invalid callbacks, QA fallback and budget exhaustion.
There has been **no actual MHR/SQP GPU run, external accuracy validation, adoption
or submission**. The synthetic direct controls are schema fixtures, not a claimed
native MHR export. Curved/sliding real contacts remain to be tested.
