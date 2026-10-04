# Surface LOD: official contract and prospective backend

**Updated 2026-10-05; Phase1 identity implemented/native runtime independently
verified; Phase2 design and implementation remain unqualified. Not adopted.** This corrects an
interpretation of the official scorer, not any historical experiment gate.
EP25 and all other closed failures remain FAIL; no episode reroll, changed
threshold, mesh repair or production query is authorized by this document.

The first standalone Phase2 build (`4ccefd3`) failed before compilation/QEM:
the new source-inventory adapter imposed a 2MiB per-file cap on the inherited
Boost `typeof/vector200.hpp` (2,328,744 bytes). An offline call of that same
immutable header function reproduced the exact failure; the original 1,418-file
libigl/Eigen inventory remained hash-identical. The correction uses the existing
hash helper's 64MiB capacity for Boost only, preserving canonical regular
single-link files, complete hashes, publisher pins and the libigl/Eigen cap.
The original sealed FAIL is retained; a fresh build is required before the
already-frozen two-positive/one-negative QEM cohort. No geometry gate changes.

That corrected build (`e5f82ea`) now passes independently: same CPP, existing
compiler/image, all 1,418 libigl/Eigen and 14,322 Boost files verified twice;
the retained ELF and host/native receipts are frozen in
`configs/surface_qslim_build_pins.json`. Build-only wall time22.814s, zero mesh/QEM
calls. The numerical cohort, physical fidelity and real HOI accuracy remain
unqualified; a successful compiler is not a successful reconstruction method.

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


## 6. Phase1 executable qualification (2026-10-05)

`infra/surface_identity_qualify.py` now defines a data-free offline CPU test
using the existing pinned official packaging image1a04b193…303f0. Only code,
six original Track1/shared kit source leaves, two native mesh-loader sources and
the independently frozen CPU build receipt are mounted. No sample predictions,
video, body/model assets, renderer, dataset, GPU or new package installation.
Three new exactF32 inputs are fixed globally before measurement: curved4V/2F
sheet; open square8V/8F tube; sheet plus disconnected triangle7V/3F. Three raw
negative domains (index, collinear, nonmanifold) must abstain without mutation.
All six complete arrays and manifest are frozen before any predicate/call.

Each positive gets one owned GLB export, actual native CPU loader and full-scene
source-authenticated replay, then one **unmodified** official `budget_mesh` call.
Exact oriented surface+F32 representation, all component/boundary geometry and
zero-only padding must agree. Three budget calls, six native loader calls
including authority replays, zero QEM; no retry, source healing or tuned tolerance.
Owned GLBs remain on Azure scratch and are removed, leaving only tiny receipts.

Budget is **60s inclusive native** from initialization through fixtures, imports,
checks, cleanup, posthash and receipt write; Docker outer63s. Host integrity,
Docker control, independent exactCIDabsence and final sealing are separately
bounded. Daemon failure is not container absence. Failure retains phase/counts
and fixture/source manifests; only receipts are sealed0444 under0555. Root238
manufactured tests PASS0.61s; these mocks do not establish the real runtime PASS.
Actual source/image/helper/artifact rechecks and terminal execution must precede
any production hypothesis. Full Parquet packing, scorer acceptance, physical
embedding and reconstruction accuracy are separate, still unproved here.

## 7. Phase2 preimplementation contract: fixed-boundary native surface QSlim

**Preimplementation contract retained below; current build-only evidence above.**
The standalone compiler is now authenticated; QEM execution/qualification are not. Section6
and the frozen Phase1 sources keep their original meaning. This section specifies
the first prospective Phase2 variant, not a relaxation of any closed experiment.
Implementation review and Phase1's actual terminal/runtime proof precede a new
source freeze. EP25, earlier fixtures and all challenge failures are excluded
from method selection and this qualification cohort.

### Native construction and committed-event ledger

Reuse the pinned libigl setup from Section4: `connect_boundary_to_infinity`,
`edge_flaps`, native point-to-plane quadrics, native QSlim cost/placement callbacks
and a custom `decimate` call with explicit pre/post/stopping callbacks. The closed
proxy is **algorithmic only**: original open surfaces remain open. Neither the
closed-source OBJ parser nor the original volume/solid/cavity veto is inherited.
The original face count `F_source` separates real from synthetic birth IDs.
After decimation, strip **only** synthetic `J >= F_source` faces and their unused
synthetic vertex; keep every surviving real face in native order. Do not classify
real faces as padding from the presence of vertex index0. Verify final `J` range,
real face lineage and all original components before publishing any candidate.

`I` identifies original surviving vertex births, **not** a coordinate equality
or a map of every original vertex to its replacement. Track the real vertex
quotient through actual contractions separately. Record/update the committed
ledger **only when `collapsed == true`**. In failed post-collapse events, `f1/f2`
can be undefined: never inspect them, update face counts or retain a speculative
transaction. Real-face/component counts and live/null face state must reconcile
against final `G,J,I`; synthetic events cannot consume the last real triangle
of a component. Single-triangle components are protected, not filtered.

### First variant: freeze every original boundary vertex

Any edge incident to an original boundary vertex has cost `+infinity` and cannot
collapse. Original boundary coordinates/edges are fixed, not moved toward a
penalty target. Introduce no boundary weight or tuned aggressiveness. A source
with **at least4096 referenced boundary vertices is INAPPLICABLE** to this
variant; do not switch policy or discard a loop. Too many fixed/interior vertices,
protected tiny components or queue exhaustion can also prevent budget completion.
Under-budget sources retain Phase1 identity rather than invoking QEM.

Preflight establishes consistently oriented vertex-manifold topology with
optional boundaries. Illegal link changes, cross-component mergers, removal of
a component, local face reversal and collapsed representedF32 triangles veto
the proposed transaction. Boundary loops and component correspondence remain
mandatory. Keep upstream floating-point intersection blocking with its declared
limitations; do **not** add a closed-volume, material-forest or exact-embedding
acceptance requirement to this surface backend. It cannot certify physical
interiors or resolve pre-existing self-intersection/non-manifold sources.

One target4096, one invocation per positive fixture, no target retry, reordered
seed or fallback. Stopping requires both real face and real vertex counts
`<=4096`, plus the declared serialization checks; final official arrays have
exactly4096 vertex and face rows through unchanged official padding. Do not
mistake the proxy face count for real output budget. All real components,
oriented vertex links and fixed boundary geometry must survive exportF32,
decoded native/raw readers and the **unmodified** official weld/packer. Compare
complete meaningful oriented surfaces, component births and boundary loops;
zero-only official padding is excluded separately. No sign flip, healing,
hole filling, scale fit, meaningful-face deletion to repair or component removal.

### Fresh procedural controls, defined before any QEM values

Use a curved dyadic grid patch: for integer `i,j` from0 through `N`,
`x=(i-N/2)/32`, `y=(j-N/2)/32`, `z=(x*x+y*y)/16`. A retained cell `(i,j)` has
faces `(v00,v10,v11)` and `(v00,v11,v01)`, consistently oriented in that order.
A square hole removes cells whose two indices both lie in `[a,b)`.
Construct only vertices incident to retained cells, in lexicographic `(i,j)`
order, before creating the source arrays; this is the authored open shape,
not removal or repair of an existing source. Store exactF32 coordinates/I64
faces and freeze every complete source array/hash before preflight or calls.

1. **Fresh curved holed patch:** `N=64`, `[a,b)=[24,40)`. Exactly4000 vertices,
   7680 faces and two boundary loops/320 boundary vertices; one component.
2. **Fresh disjoint holed patches:** two patches with `N=48`,
   `[a,b)=[20,28)`, second translated by `(4,0,0)`. Each has2352 vertices,
   4480 faces and two loops/224 boundary vertices. Combined4704 vertices,
   8960 faces, two components/four loops; every component participates.
3. **Inapplicable boundary excess:** open square-sided tube with exactly2048
   samples on each rim. In each z-plane0/1, traverse four sides with512 samples
   each, `s=k/256`, `k=0..511`: `(-1+s,-1)`, `(1,-1+s)`, `(1-s,1)`,
   `(-1,1-s)`. Connect successive corresponding rim points with two oriented
   triangles. Exactly4096 vertices,4096 triangles, all vertices on the two
   boundaries. Expected INAPPLICABLE **before QEM**, despite fitting the row
   budgets; test the explicit Phase2 domain decision, never erase a rim.

Assert these combinatorial counts, exact positive triangles, referenced-only
source arrays and fixed-boundary eligibility before either positive QEM call.
All sources are fresh data-free controls, not external accuracy evidence.
The negative is a policy test, not a reason to change Phase1's identity domain.
Freeze procedural definitions and source/helper hashes; any source, native or
packing failure stops the cohort without adjusting these definitions.

### Cost, utility and unresolved accuracy gate

Use the existing source-bound libigl/Eigen CPU build environment, no downloads,
installs, model/GPU runtime or new dependencies. Prospective limits are
**4 CPUs/16GB; compile<=600s; whole control cohort<=300s**, including source
preflight, both native calls, actual loaders/official packing, posthash and owned
scratch cleanup. Each native call is clamped to the cohort's remaining budget;
the second is not guaranteed300s of its own. Total compile+qualification<=900s;
hard outer deadlines and bounded ownership-checked cleanup must be frozen with
the future controller. Host source publication/daemon preflight have a separately
reported bounded scope, not a false claim that the whole lifecycle takes300s.

These controls target real boundary-preserving simplification/representation,
not3D estimation. Report full-source per-component/boundary surface distances,
births, serialization changes, native attempts/returns and actual collapse/veto
counts. Exact source-to-LOD surface equality is not required for simplification;
exact **LOD-to-serialized/packed** identity is. A new accuracy threshold/cohort
must still be frozen before external or production adoption; this section does
not invent a numerical tolerance. Optional Hausdorff/embedding certification
from Section5 remains separate and must not become an implicit solid-only gate.
No budget success, silhouette improvement or procedural PASS can establish a
gain over CARI4D or authorize production defaults/submission changes.


### Phase1 actual evidence and separate host failure

Frozen producer e57113213164188434735903a49efe84b012ec13 completed native PASS
0.684506492s: three exact controls, six loader calls including authentic replays,
three unchanged official budget calls, zero QEM. Native13067B
SHA89e5ee946c9d88a2c168ba314bbd16798f9f8301bd0190316c8a21926fc511e8;
full six-source manifest bfb8bb8dfce620c8f083f5840cd0c7ecb84f9d8be3f4c234a3658cfd2df90166.
The **host unit remains FAIL**, with no host report: final cleanup expected a
capitalized Docker error, but the actual pinned CLI emits lowercase exact
`error: no such object: <sameCID>` with return1/empty stdout. Original1324B log
SHAaba29014312cdbe2606ee0e79dd814a6ab89b78b7da96c1ab29ccd7be7c60f46 is retained.

An independent data-free audit verified all226 immutable source files/archive
and readonly ledger, current official/native/runtime hashes, complete six-input
fingerprints recomputed with stdlib struct, original native counters/results,
posthash and **actual exactCID absence**. No control/loader/model was rerun.
`configs/surface_identity_qualification_pins.json` binds this narrowly qualified
native evidence and explicitly records the historical host FAIL. It is not a
backdated host report, whole Parquet qualification, arbitrary-mesh proof or3D gain.

Future driver/wrapper accept that exact lowercase sameCID/return1 variant;
daemon failure, extra text, foreignCID, wrongreturn and timeout stay rejected.
Snapshot mount inventory is restricted to code and its two original markers.
Root318 tiny tests PASS0.71s; tests do not replace the actual independent audit.
Phase2 still needs real build/transactions and full serialized/packed fidelity.


### Phase2 implementation freeze before native build

A separate328-line `infra/surface_qslim.cpp` now assembles native libigl QSlim
quadrics, fixed-boundary +infinity costs and its actual floating-point AABB
intersection callback. Local pre-collapse transactions protect real components,
vertex links and represented F64/F32 normal activity/orientation. Only committed
collapses update the ledger; final J/I births and a complete original vertex
quotient replay reconcile component/Euler/fixed-boundary/unused-vertex bytes.
No inherited solid/cavity/volume gate, source repair or original solver rewrite.
Native output remains F64; one required GLB F32 cast must be reported separately
from exact F32-LOD-to-loaded/packed identity. None is runtime-qualified yet.

`infra/surface_qslim_build.py`/wrapper build this standalone source in the existing
pinned CPU imagec8fb1632…21137, with original authenticated libigl40e790/Eigen314739
headers, compiler and Boost inventory. No CGAL child, package installation,
models, meshes, QEM calls or data; only compile and source-bound `--build-info`.
Native600s inclusive/host900s inclusive/outer960s include posthash and receipt
sealing, with bounded owned cleanup. ExactCID absence requires actual return1,
exact sameCID message and empty or empty-array stdout, never daemon errors.

Root194 manufactured tests PASS0.63s. This is source/API/fixture-contract review,
not a real C++ compile, collapse qualification, geometry accuracy or adoption.
Fresh Phase2 control cohort from Section7 remains frozen before native values.
The implementation status at freeze above is historical; the October5 build-only
PASS and pins at the beginning of this document supersede only compilation status.
