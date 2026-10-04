# Mask-conditioned frame0 query quantiles — proposed operator control

**Frozen design: 2026-10-05. Not implemented, executed or adopted.** This is a
new sampling hypothesis, not a replay of the closed EP21 query-availability
failure. No challenge records, future tracks, private labels, fitted camera,
model calls or permission-dependent external dataset enter this control.

## Existing primary implementation and distinct hypothesis

The authoritative current source is
[`select_query_indices`](../src/world_reward/point_pose_cost.py) and
[`canonical_surface_queries`](../src/world_reward/point_surface_queries.py).
It tests 16×12 full-image centre-grid pixels, keeps the first32 qualified hits
in raster order and requires8 distinct canonical witnesses. Rays and tracker
queries use original `(x+.5,y+.5)` pixels. Depth support, first-surface hits,
original-face barycentrics, proper supplied SE3/K and boundary/tangent/coplanar/
depth-tie rejection are independent requirements. The current operator and
its 64·eps64 numerical margin remain unchanged.

**Hypothesis:** a small automatic mask can contain many usable pixels but fewer
than8 of the192 full-image grid centres. Allocating a fixed query set to the
mask itself can remove this sampling limitation without improving a wrong
shape, pose, mask, camera or tracker. A mask-bounding-box lattice is not selected:
empty concavities and occluder gaps still consume its lattice points.

Proposed B, for each object's original frame0 automatic bool mask `M`:

1. Enumerate all true pixels `(y,x)` in original row-major order; let `N=|M|`.
2. Set `Q=min(32,N)`. If `N=0`, retain an explicit empty candidate set.
3. For `i=0..Q-1`, select rank `floor((2*i+1)*N/(2*Q))`, using integer arithmetic.
   Ranks are distinct because `N>=Q`. Freeze these original pixel indices now,
   **before consulting depth validity, mesh/raycast results or any future RGB**.
4. Form queries `(0,y+.5,x+.5)`. Apply the same supplied mesh/SE3/K, depth-support,
   first-hit, barycentric, ambiguity and distinct-witness requirements as A.
   Retain all candidate-slot support diagnostics; only qualified witnesses
   enter a successful result. Fewer than8 witnesses means FAIL/abstention.

No erosion, border snapping, crop/resize, pixel displacement, rank shift,
score ranking, repeated points, component selection, extra candidates, refill,
future-track selection, pose fitting or per-clip parameter exists. An ambiguous
supported ray fails the call rather than being replaced by another pixel.
Raster quantiles spread **mask mass**, not texture or surface information;
they can miss a tiny disconnected component and are not a confidence measure.
No mesh face/component is removed by sampling.

## Scalar observations before every availability gate

For both A192 and B≤32, record: mask area; candidate count; candidates in mask;
in-mask candidates with valid inferred depth; first positive surface hits;
ambiguous-ray count; distinct qualified canonical witnesses; status/reason.
Report every case, including failures, without retaining dense outputs locally.

- `A candidates_in_mask <8` proves insufficient sampling of the supplied mask,
  not that the automatic mask or initial pose is correct.
- B has mask candidates but loses depth support: observed-depth availability
  conflict, not automatically a pose error.
- B has depth support but too few surface hits: shape/pose/K/mask disagreement;
  this scalar cannot uniquely identify its cause.
- Enough hits with ties/near-boundaries still fails. Diversity/availability
  alone does not establish correct correspondence, identity or6D observability.

## Fresh fixed cohort: four positives plus four negative controls

These are **new analytic operator fixtures**, not rendered/native segmentation
quality data. Their supplied masks, geometry and pose are declared inputs,
never presented as automatically inferred observations. No previous R35/18-case,
D105, bridge, YCB48–50, Dex01–05 or challenge cohort is reused.

All cases have original `H=48,W=64`, FP64
`K=[[64,0,32],[0,48,24],[0,0,1]]`, `R0=I`, `t0=(0,0,0)`, and one original
triangle `V=[[-5,-5,2],[11,-5,2],[-5,11,2]]`, `F=[[0,1,2]]` int64 unless
explicitly amended below. All three vertices are in front. Pixel-centre rays
through the complete image lie strictly inside this triangle: its projected
bounds are `(-128,-96),(384,-96),(-128,288)`, far from any tested pixel.
Default depth validity is true everywhere. Interval notation is half-open.

| Case | Supplied frame0 mask / one explicit amendment | Expected B |
|---|---|---|
| P1 small off-centre | `7<=y<11` and `9<=x<17` (32 pixels) | 32 witnesses |
| P2 one-pixel thin band | `y=19` and `8<=x<56` (48 pixels) | 32 witnesses |
| P3 concave visible mask | `8<=x<52` and (`4<=y<12` or `32<=y<40`), union `8<=x<16,12<=y<32` | 32 witnesses |
| P4 occluder gap | `4<=y<44,4<=x<60`, minus `16<=y<32,16<=x<48` | 32 witnesses |
| N1 absent mask | Empty mask; other inputs unchanged | No-anchor FAIL |
| N2 absent depth | P3 mask; depth validity false everywhere | Support FAIL |
| N3 inconsistent anchor | P3 mask; supplied `t0=(40,0,0)` | No-hit FAIL |
| N4 unresolved geometry | P3 mask; append an identical original triangle face, without welding/removal | Depth-tie FAIL |

P3/P4 describe concave/occluded **masks**, not a demonstrated concave3D shape or
human occlusion detector. Fixtures deliberately isolate allocation from model
error. P1's A-grid has2 in-mask pixels; P2's band contains no A-grid row. P3/P4
contain more than8 A-grid pixels. These counts follow from the fixed grid
formula, not from an experiment whose failed cases are replaced afterward.

Before any operator measurement, one future controller must write/freeze the
case definitions, exact full-array F64/I64/bool hashes, implementation/source
identities and all constants. Generate each fixture once by the formulas above;
no random seeds, resampling, result-selected rotations or fixture replacement.
Construction/hash or expected-support inconsistency itself stops the cohort.

## Independent reference, budget and immutable decision

Use a separate scalar plane/triangle reference, not the vectorized production
intersection expressions: at query pixel `(x+.5,y+.5)`, a positive plane hit has
`p=((x+.5-32)/32,(y+.5-24)/24,2)`. Solve original triangle barycentrics as
`((6-px-py)/16,(px+5)/16,(py+5)/16)`; nearest depth is2. Apply N3 translation to
the reference triangle; N4 has two indistinguishable first-depth faces. No GT
pose perturbation, source geometry transformation or post-hoc alignment occurs.

**Single inclusive CPU budget:30s**, from fixture construction through A/B
queries, repetitions, reference checks, input hashes/postchecks and scalar
report. No GPU/model/render/network or native tracking. Exactly8 paired cases,
two deterministic calls per operator/case; no parameter sweep or retries.

PASS requires all of the following, fixed before measurement:

- B yields32 distinct, supported, unambiguous witnesses in all4 positives;
  A's P3/P4 successful availability must not regress.
- B resolves P1/P2 sampling scarcity where A remains below8; no refilling A.
- N1/N2/N3 abstain and N4 rejects its tie; none produces accepted invented points.
- Successful ray depths, canonical points and barycentrics agree with the
  independent reference to `atol=1e-12,rtol=0`; query pixels/ranks agree exactly.
- Output/support/order are repeatably byte-identical, original inputs unchanged,
  no hidden candidate refill or removal of mesh faces, and elapsed<=30s.

Any failed condition closes this cohort/operator version with all scalars kept;
no loosened tolerance, shifted pixel or replacement fixture. Passing only
qualifies this deterministic availability operator on these synthetic cases.
It does **not** reopen EP21, justify adoption or show native-mask accuracy,
real-object pose/gauge correctness, Boots track quality,6D identifiability,
training-overlap clearance, full interaction quality or a CARI4D gain.
