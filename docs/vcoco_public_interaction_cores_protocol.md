# Public pose callback and explicit16/48 interaction core — source only

These are reusable pure/callback seams, not new executed producers or a quality
qualification. No Azure, model, actual RGB/reference, FIT/CAL or inference study
was run while implementing them. Historical helpers and default16 behavior are
unchanged; no source/global profile mutation or test-record retuning is used.

## Public pose loop

`infra/vcoco_public_pose_core.py` exports
`observe(images,banks,load_bank,decode,infer,save,check,records=None)` and the
original11 `FIELDS`. Its function AST is **exactly** the original
`vcoco_person_pose_observations.observe`, SHA256 of attribute-free `ast.dump`:
`525495e177be8303d5e890fa12f9470d58967bc95526b2d2f82404b27531158b`. Only imports differ: `hashlib` and the stdlib-only
`mediapipe_cpu_runtime_verify` require helper. No replica/completion, historical
receipt, checkpoint, grant, role/reference or model package is imported.

The original loop checks every endpoint bank before the first inference, calls
`infer(rgb,ids,boxes,scores)` unchanged and preserves all person IDs/count/order,
frame0/grid and original array fingerprints. It returns `(records,total_people)`.
The loop is cardinality-generic; the surrounding public48 reference and adapter
must enforce all48. P0 passes an empty original census to the supplied qualified
adapter, which must retain its no-full-image-fallback rule. The loop itself
neither loads a session nor validates/serializes133 joints: those unchanged
`PersonPoseObservations` and11-array save contracts stay in the public adapter.
Caller owns callback authentication, deadline, cleanup, sealing and failure
postchecks; partial callbacks/records never imply completed inference.

## Explicit complete48 join

`infra/vcoco_full_interaction_core.py` exports the same six-argument
`reconstruct_interaction` plus keyword-only `population=16`. Only exact integer
16 or48 is supported (bool rejected). Default16 retains historical slot holes
and acquired ordinal semantics. Explicit48 accepts slots/ordinals0..47 and
requires slot==ordinal; it never compresses, renumbers or synthesizes records.
It joins **one** row, so full48 completeness/source/receipt validation remains
the caller's responsibility before invoking any row.

The original frozen helper
`infra/vcoco_interaction_observations.py` remains10476 bytes,
SHA256`e8333e3180fe009f08bb14201b20919a6bec823164d19c7f21d2124eae56a2a1`.
The new core copies its exact numerical body and snapshot/identity contracts.
Only a population prelude, `_grid` bound parameter and grid call arguments
change. Tests compare the unchanged native numerical body AST and default16
raw bytes, feature/support arrays, route metadata and frozen copies. This is
not a wrapper importing the historical producer/private modules.

All17 endpoint,11 pose and19 HOI arrays are byte-pinned and joined in a shared
original image/grid/frame. Original133 joints, all900 raw GDI queries and3600
OWL patches,1500 HOI queries/tokens, duplicate class routes and every Cartesian
H→O/O→T pair survive. Genuine P0/K0 and missing/off-grid keypoint support remain
explicit; no missing pose file is manufactured. AllP×2×3600 and K×3600 evidence
is built by unchanged `world_reward` numerical contracts, without selector,
proximity winner, threshold, alignment, score fitting, contact or ownership.

Native-source closure: the pose loop needs only itself and
`infra/mediapipe_cpu_runtime_verify.py`. The join core needs itself plus
`src/world_reward/{__init__,person_pose_observations,hoi_detr_observations,
owlv2_object_observations,owlv2_candidate_bridge,interaction_candidate_evidence,
interaction_tuple_evidence}.py` and installed NumPy. Callers mount only their
fresh public projections and numerical files; no original private receipt or
RGB/reference role metadata is needed by the join. Sources/runtime/arrays must
be authenticated before and after by the future caller.

Qualification:42 dedicated authored controls plus40 frozen original join
controls, **82 PASS0.76s**. Cases include default16 parity, full48 slot47,
P0/K0, original callback AST, dtype/ID/flags/grid/bounds/tampering, mutation,
partial errors and denied eager/private/model imports. Initial controls had
five test-fixture/API typos (wrong support field and mutation of a future rather
than active bank), corrected without changing the copied numerical algorithm.
Owned test fixtures are removed. No actual native full48 pose/HOI/join or
accuracy/performance result follows from these CPU source controls.

Root independently ran the same82 controls: PASS0.77s; exact fixture roots
removed. An initial root collection command omitted the established infra/src
test import path (zero tests executed); corrected harness only, no source change.
