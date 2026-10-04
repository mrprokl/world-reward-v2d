# Surface LOD: official contract and prospective backend

**Prepared 2026-10-05; not implemented, qualified or adopted.** This corrects an
interpretation of the official scorer, not any historical experiment gate.
EP25 and all other closed failures remain FAIL; no episode reroll, changed
threshold, mesh repair or production query is authorized by this document.

## 1. Audited primary sources

Official revision: `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`.
The [official submission-kit archive](https://raw.githubusercontent.com/nvidia-isaac/video_to_data/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/docs/v2d_challenge/assets/v2d_submission_kit.zip)
was inspected as public code in RAM, without samples, assets or ground truth.
Only the following Track1/shared sources support this correction. SHA256 pins:

| Archive-relative file | Bytes | SHA256 |
|---|---:|---|
| `v2d_submission_kit.zip` | 695124 | `b1f8f703c2e772684a57d048cf28f40c7b9b54cf7a61aff6917b46d826915e1f` |
| `v2d_submission_kit/v2dlb/mesh_budget.py` | 2031 | `42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0` |
| `v2d_submission_kit/v2dlb/mesh_common.py` | 2175 | `aaeac13286c839a7d7e271888613875c934e46de094b2ac928ff2ff01c5a1f06` |
| `v2d_submission_kit/v2dlb/mhr_submission.py` | 24344 | `06fbd58d07bae1c583a92598f879771a4805ccea3e8346605975bcf5007b1fa3` |
| `v2d_submission_kit/v2dlb/mhr_metrics.py` | 25647 | `73077b65b5c3e0204307feef784317aab8c733d7462b6b8e6f4f397f7b91d2e0` |

`mesh_budget.py:27–40` loads with `process=False`, constructs a `process=True`
welded mesh, simplifies only when over the face budget, checks the resulting
face/vertex budgets, then pads. **It does not assert closure, manifoldness,
positive volume, cavity nesting or embedded-solid validity.** This is not a
guarantee that its weld or simplifier preserves a participant's surface.
`mesh_common.py:15–43` validates integer face indices and samples surface area;
it requires finite positive total area, not a closed solid.

`mhr_submission.py:246–262` explicitly defines generalized winding on a
**triangle soup**, saying it "degrades gracefully across holes, which a
participant's mesh may have". For a query point, the oriented solid angles give
`w = sum(atan2(numerator, denominator)) / (2*pi)`. At lines401–429, penetration
uses `abs(w) > 0.5` and unsigned distance to the nearest triangle. Exactly
zero-area padding contributes no surface; there is no solid-forest assertion.
Lines450–466 evaluate the participant's own hands against its own object,
with the shared alignment and object scale. `mhr_metrics.py:281–293` defines
CD-O's unsigned symmetric surface distance as
`mean_p min_q ||p-q|| + mean_q min_p ||q-p||`, not a volume discrepancy.

Native CARI sources at the same revision are also relevant, not new dependencies:

| Source under `reconstruction/modules/v2d_cari4d/lib/cari4d/` | Bytes | SHA256 |
|---|---:|---|
| [`lib_mhr/contact.py`](https://raw.githubusercontent.com/nvidia-isaac/video_to_data/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/contact.py) | 9646 | `d4e8a92845d75a7bae962f312dee4d747587c39157a978908293a5645beb6d5c` |
| [`lib_mhr/sdf.py`](https://raw.githubusercontent.com/nvidia-isaac/video_to_data/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/sdf.py) | 5723 | `5858709fd16a3b8b0932a928f92be011f7249ecb33ecf9f9e7249e9aa8a42423` |

`contact.py:58` loads/concatenates the object's scene with `process=False`;
surface sampling and closest points do not require an object solid.
`sdf.py:24–78::object_inside_human_penetration_loss` requires the **human** mesh
to be watertight for Kaolin inside tests. That requirement must remain unchanged;
it is not an object-closure requirement.

## 2. Three different contracts

| Contract | Required interpretation |
|---|---|
| Official representation/scorer | Fixed4096-row budgets, valid indices/coordinates, positive sampled surface, proper poses, full timeline and shared gauge. No object-solid assertion in the audited paths. Full official submission validation still applies. |
| Reconstruction fidelity | No object/component deletion, shrink, hole filling, sign repair, static motion, source calibration or manual episode fitting. Preserve the represented interaction and quantify LOD error. A physical thin sheet may be open; holes in an estimated solid can instead be reconstruction error. |
| Numerical backend qualification | A backend declares its supported domain and verifies serialization/births. Exact embedded-solid forests remain appropriate for the **solid** backend, not a universal prerequisite for every surface representation. |

Format acceptance is not physical validity: a self-intersecting soup can pass
schema/scorer code without representing a plausible object. Mixed orientations
can change winding even though a global orientation reversal cannot change
`abs(w)`. Never reorient or remove surface to improve PEN. A surface backend
does not claim a reliable signed volume, material interior or contact certificate.

Current overconstraint is concrete: `src/world_reward/mesh_budget.py::_inspect`
is called before even its under-budget identity return, and rejects open or
zero-volume components. The default `infra/object_pose_smoke.py` calls this
operator. Its eight attempts retain components, but all require closed meshes.
Do not alter that historical operator, its evidence or the qualified solid route.

## 3. Phase1 priority: faithful under-budget open surface

First implement one separate identity operator, **without QEM or new models**.
Domain: finite, positive-area, consistently oriented vertex-manifold surfaces
with optional boundaries, including a disconnected one-triangle component.
Non-manifold or ambiguous inputs abstain as unsupported; this is a backend
domain restriction, not a statement that every such official record is illegal.

For sources already within both4096 budgets, preserve native arrays/face order
and all meaningful faces/components; do not invoke `process=True` as a repair,
compact away surface, fill holes or require nonzero volume. Audit native raw
loading, GLB POSITION/node transforms, F32 representation and the unmodified
official packer. Check exact represented oriented-triangle multiset across
packing, component correspondence, boundary loops and all nondegenerate faces.
Only official padding is excluded from the meaningful view. Record the actual
runtime/helper versions; serialization loss or unintended weld merges fail,
not trigger retries. F64-to-F32 coordinate rounding is reported separately
from additional packing changes, not silently called exact source identity.
Retain clip-constant geometry, original positive scale applied once and the
same full-T pose/human algorithms. Identity LOD is not improved3D estimation.

New own-authored tiny controls precede any external or production use: a curved
open sheet, open tube, and disconnected sheet plus isolated triangle; malformed
index/zero-area/non-manifold cases must fail without mutation. They must exercise
the real loader/official packer, not only fabricated metadata. No prior failed
validation fixture or episode is reused to select policy. This is the cheapest
useful scope: establish whether lawful open under-budget sources can traverse
the existing reconstruction pipeline without accidental geometry changes.

## 4. Phase2 proposal: one boundary-aware QSlim LOD

Only after Phase1, add a thin explicit native adapter, not a copied solver.
[libigl revision40e7900ccbd767f1f360e0eb10f0f1a6432e0993](https://github.com/libigl/libigl/tree/40e7900ccbd767f1f360e0eb10f0f1a6432e0993)
is already a source-bound dependency; MPL-2.0 notices remain required.
`include/igl/qslim.h` (1957B,
`188941d1d1d608dd59dcbd1f5c0c69f9faaecdbbd1ff6c48658af3d4110b9d31`)
explicitly supports open edge-manifold meshes. `qslim.cpp` (3452B,
`96b5c7c009c9b9539b200d77158cd90093d8b7d0803dcd4b2f357b77bb5bfa40`)
adds fictitious boundary-to-infinity faces, boundary quadrics and optional
intersection blocking; outputs `U,G,J,I`, then removes only fictitious faces.
The old closed OBJ parser and volume veto are not suitable for this new domain.

Use **one global target4096**, no alternate targets, allocation rerolls or
fallback. Check both budgets at completion. Every source component participates
in the objective and survives, even a single triangle. Decimation legitimately
reduces tessellation; "preserve all surface" is not a promise to keep more than
4096 original faces. Real source-face births and a contraction ledger must
verify component bijection, boundary quotient/homeomorphism, no cross-component
or boundary-loop merge, no local face reversal and no newly collapsed F32 face.
Transactional callbacks veto component disappearance and boundary destruction.
Do not treat vertex birth `I` alone as the full source-to-output vertex map.
Preserve upstream intersection blocking, but disclose its floating-point scope;
it is not exact embedding certification. Queue exhaustion or unsupported input
fails the whole proposal without healing. Recheck Phase1 serialization gates.

The seemingly cheaper `fast_simplification0.1.13` is MIT, commit
`4a193efa8c3270c8fef6c9a71c7b8b9432f83343`, but its border/flip heuristics do not
prove these invariants. `replay.py` (7217B,
`31154bca4f0f052758485b77837f11d42f5df19a1931c63d05a3d29f89c5f3ab`)
lines195–213 cleans triangles, reassigns isolated points and removes outliers;
it is not a faithful face-birth witness by itself. No such cleanup is adopted.

## 5. Fidelity evidence, cost and adoption prerequisites

Mandatory low-cost gates: complete provenance/pre-post hashes, budgets, all
components/boundaries, births, orientation and represented packing fidelity.
For approximated surfaces, independently measure bidirectional surface error
**per component and boundary**, not only pooled CD or a percentile. Signed/net
volume is not used for open surfaces. RGB silhouettes/depth/occlusion agreement
can diagnose this video-only hypothesis, but cannot prove held-out3D accuracy.

A certified Hausdorff upper bound is a **separate optional qualification**,
not an official rule or a pretext to block Phase1. Adaptive subdivision of all
triangles can bound directed distance using a verified target-surface witness
and cell radius (distance is1-Lipschitz); account for numerical rounding and
check both directions. Large/near-contact soups can be expensive. A bounded
run may report INCONCLUSIVE; sampled `trimesh.proximity` or CD must not be
relabeled a certificate. No sampled deletion or percentile replaces a claimed
maximum. Exact embedding certification is likewise optional/domain-specific.

**Still missing:** Phase1 implementation/runtime controls; Phase2 transactional
boundary/component ledger and serialization tests; a frozen new external3D
cohort with clear rights/overlap, error thresholds, CPU/memory/deadlines and A/B
acceptance rules before values; actual full-source fidelity and HOI/coverage
measurements. New procedural3D controls qualify representation/LOD only, not
CARI4D quality. External inference is RGB-only with private metric truth isolated.
No challengeGT, per-frame alignment or tuning on closed cohorts is permitted.

Priority order is Phase1 identity correctness → Phase2 single-call LOD/fidelity →
external reconstruction comparison; expensive exact certificates only where
their extra physical claim is needed. No QEM is developed or launched here,
no PASS is asserted, and no production default or adoption decision changes.
