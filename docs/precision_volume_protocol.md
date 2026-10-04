# Precision/backend controls v1

**Prepared, not adopted.** The original measured binary SHA is frozen below.
No challenge episode, masks, RGB, predictions or private
ground truth are mounted. Historical failures and helper hashes stay immutable.

The only replaced validity rule is whole-mesh area threshold → exact stored
triangle non-collinearity. Static explicit wrappers replace no runtime globals
or bytecode; AST comparison proves all other original topology, volume,
birth-map, OBJ, pack and containment statements remain unchanged. Private
float32-to-float64 promotion for exact predicates preserves represented values;
original arrays and subsequent original-dtype computations are unchanged.

## Fixed controls and rejection gates

1. **Graded tetrahedron identity**, 7 vertices/10 faces: exact planar subdivision
   of one corner, scale `2^-30`, positive whole volume `1/6`. Its small legitimate
   faces fail the old global-area guard without challenging the unchanged
   global-volume rule. Identity candidate, GLB export, official pack and one
   metric-scale bake must preserve all geometry.
2. **Graded sphere with inward ellipsoidal cavity**, outer icosphere level 4,
   corner scale `2^-30`, inner level 2 radii `[.22,.176,.198]`, center offset
   `[.1,0,0]`. More than 4096 original faces invoke the **unchanged** original
   source-volume-constrained QEM once. No other geometry algorithm or fallback.
3. A truly coincident/collinear triangle rejects the whole geometry. Native
   cross-product squared norms must remain positive/finite; exact predicates do
   not repair underflow or serialized quantization loss.

At source, candidate, serialized export, official pack and baked metric stages:
full oriented closed vertex-link manifold validation; no self-intersections;
all inward vertices contained in the outward surface; original birth-face/shell
bijection and vertex births; **≤1% source-diagonal sampled Chamfer and ≤5% each
original shell and net volume**; unchanged geometry ancestry. Source arrays are
hashed before/after. GLB quantization may not merge vertices or alter triangles
beyond stored float32 values. Packing must preserve exact cyclic oriented
triangles; only official zero padding is excluded from its meaningful view.
Metric scale `.375` is baked once, never fitted or retuned.

One remote network-none CPU run, **4 CPUs / 16 GB / 3600 seconds total**, at most
**900 seconds native QEM**. Fail any gate → retain tiny failure report, delete
only owned temporary scratch, stop without retuning/retry/adoption. Report only
concise scalar measurements/hashes, never arrays or media. No GPU or local data.
The read-only container drops all capabilities, forbids new privileges, and
uses a 512 MB temporary filesystem. Only the official audited helper file is
mounted, not the submission-kit directory. Runtime source/marker/helper/build/
binary hashes and file identities are sealed before/after, including on failure.
The 3603-second outer deadline is followed only by bounded, exact-owner CID
TERM/KILL cleanup; no other container is stopped and no prediction is retained.
Success requires both the controller report and the host cleanup receipt.
Both final receipts are sealed mode `0444`. Post-computation provenance sealing
is included in measured elapsed time; the separate outer deadline bounds it.
The 2031-byte official helper was directly audited: only future/numpy plus lazy
fast_simplification/trimesh imports, no sibling or dynamic imports. Its single
file mount is therefore the complete submission-kit source closure.

Original image: `sha256:c8fb1632a6908a82aeeeb73c36a00f17a26b81f95f4d2d498c53f75894e21137`.
Original build receipt: 3429 bytes,
`8c5441162603260fb768dad688ed7d47ddf29a79de6de53fde603c84994c28ec`.
Original sources and binary build-info must match the frozen helper manifest.
Original measured binary:
`eb606febe824f8f0a308e1d78c7677b2ecddc6ffdd0cdfc8f9a3a04fcd8055e7`.

This is an engineering fix test, not new accuracy validation or evidence of a
leaderboard win. Exact triangle predicates do not prove exact native embedding.
External YCB comparative research remains the scientific priority. Production
integration needs a separate uniformly applied frozen protocol after this PASS.
