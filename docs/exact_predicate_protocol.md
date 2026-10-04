# Exact triangle predicates — engineering protocol

**Status:** new precision-only gate; no production adoption or leaderboard claim.
The historical global-area failures and their original helpers remain frozen.

## Hypothesis and method

A face-area cutoff based on the *whole mesh* extent can reject non-collinear
triangles merely because another component is large. Separate exact validity
of represented coordinates from numerical conditioning and reconstruction
fidelity. Do not lower a tuned cutoff, remove a face, or recover hypothetical
pre-export positions.

For each original finite float64 triangle, compute its three projected 2D
orientations. Outward `nextafter` intervals propagate through coordinate
subtractions, four-corner interval products and determinant subtraction. An
interval excluding zero certifies a nonzero orientation. Otherwise resolve all
ambiguous predicates with exact dyadic `Fraction.from_float` arithmetic. All
three exact determinants zero rejects the **complete geometry**. There are no
tunable length/area tolerances. Return scalar diagnostics, not selected faces.
The filter fails closed unless binary64 gradual-underflow checks pass.
Positive but poorly conditioned triangles are not mislabeled exactly zero;
downstream readiness can still reject them. Quantization loss remains loss.

## Predeclared fail-fast controls

- All data-free tests must pass: every projected plane, exact collinearity,
  repeated/coincident positions, cancellation, product/subtraction overflow,
  norm/product underflow, representable scaling/translation, float32 loss,
  tiny disconnected components and untouched inward winding.
- Fixed synthetic seeds **7601/7602/7603**, including 10,000 faces, must agree with
  independent all-exact rational predicates. No source arrays may change.
  The filter's 6,144 adversarial projected intervals must themselves enclose
  the exact dyadic determinants or fall back as numerically ambiguous.
- One 10,000-face mixed-scale CPU benchmark must complete within **60 seconds**;
  failure rejects this implementation. Benchmark is engineering, not accuracy.
- Do not modify legacy helpers or relabel old failures. Any later integration
  is a new frozen general backend protocol, applied uniformly, not per episode.
- Before any candidate is usable, retain separate full topology/orientation/
  vertex-link checks, whole-source intersections, component/cavity preservation,
  unchanged **1% diagonal Chamfer / 5% per-shell and net volume** fidelity,
  native birth-map integrity, original canonical geometry and official packing
  fidelity. No normalization, repair, component loss or shape shrinkage.
  Existing global-volume readiness remains unchanged unless a separately frozen
  precision protocol proves its replacement; this gate does not replace it.
- Exact predicate success cannot certify native QEM arithmetic, float32 export,
  universal embedding, ground-truth reconstruction or superiority to CARI4D.

## Primary sources (all pre-September 2026)

- [Shewchuk, 1997: Adaptive Precision Floating-Point Arithmetic and Fast Robust
  Geometric Predicates](https://www.cs.cmu.edu/~quake/robust.html).
  [Original public-domain implementation](https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c):
  adaptive orientation uses a fast filter and exact fallback. We implement
  independent outward intervals plus Python rationals, not copied C code.
- [CGAL 5.6.2 EPICK](https://doc.cgal.org/5.6.2/Kernel_23/classCGAL_1_1Exact__predicates__inexact__constructions__kernel.html):
  exact predicates do not imply exact constructions. No CGAL dependency added.

Challenge diagnostics may reveal an engineering false positive, but cannot
choose acceptance tolerances or validate reconstruction quality. Record new
engineering results separately from the immutable historical failures.
