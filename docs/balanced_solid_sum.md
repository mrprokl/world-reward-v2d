# Balanced exact component-volume sums — unqualified implementation

## Change and scope

The standalone EPECK query now accumulates each component's **unchanged original
face determinant** with binary carries. An occupied level holds an exact sum of
`2^level` consecutive terms. Adding a term merges equal-sized blocks; finalizing
occupied levels from low to high gives the same rational sum with logarithmic
addition depth. Components remain independent; no original face is omitted.

`Kernel::FT`, parsed binary64 coordinates, per-component origin, determinants,
zero/sign predicates, `/6` diagnostic conversion, all manifold/intersection/
containment predicates, resource bounds and output schema are unchanged. No
eager exact conversion, coordinate scaling, perturbation, sorting or epsilon is
introduced. Consumed slots release redundant handles, but the retained lazy DAG
may still occupy **O(F)** memory. `CGAL::to_double` can use approximate intervals:
reassociation does **not** guarantee bit-identical diagnostic doubles. It does
preserve the exact rational sum and hence its mathematical sign.

The separate 131072-term ASan experiment supports a generic recursive-depth
mechanism; it does **not** establish the cause of the original EP25 termination.
Original binaries, source snapshots and FAIL receipts remain preserved.

## Required qualification, before any production use

Freeze a new source SHA, independently pinned release headers/libraries, image,
compiler flags and binary SHA. The CGAL/API/build provenance and existing gates
must remain intact. Local rational/source tests are not native qualification.

Run exactly **15 fresh procedural controls**, frozen before native evaluation,
using the existing input/forest contract with newly specified coordinates:

| Cases | Required result |
|---|---|
| Two cavities, disconnected hollow roots, island in a void plus another root; each at identity and one predeclared proper similarity (6) | Native certificate and exact expected forest/signs; every source face and vertex retained |
| Touching components, crossing components, self-crossing surface, open surface, duplicate face, exact degenerate face, trailing token, out-of-bounds index (8) | Original categorical rejection; no substituted certificate or geometry repair |
| Nested positive-oriented shells (1) | Native geometry certification followed by the unchanged material-forest rejection |

The control owner must freeze exact F64/I64 arrays or exact malformed input bytes
and all expected categorical outcomes before execution. These names/categories
are not permission to replay an earlier failed geometry or resample a control.
No tolerance is added to force diagnostic parity; all existing downstream gates
remain applicable, and an unexpected change is a STOP.

Then qualify **four fresh geometric controls**, also frozen before native calls:
two new curved families with cavities/disconnected material roots, each under
zero-origin and a fixed translated/scaled chart domain. Apply the full six-stage
whole-solid compiler: original source, QEM candidate, F32 GLB, default-eight weld,
unmodified official packing and original fixed-scale metric bake. Require every
original topology/embedding/forest/birth-fidelity/volume/CD gate; no source loss,
repair, budget relaxation or per-case retry. Record every actual query and QEM
call. The cached QEM algorithm does not change in this experiment.

**No production replay** until these native qualifications and independent source,
binary, image, full receipt and cleanup checks all pass. Any later EP25 replay
would be an explicitly authorized technical experiment in a fresh namespace,
not a retroactive PASS or evidence of better reconstruction/competition quality.
Linked CGAL rights remain GPL-3.0-or-later or a separate commercial licence;
Apache-2.0 glue does not establish competition eligibility.
