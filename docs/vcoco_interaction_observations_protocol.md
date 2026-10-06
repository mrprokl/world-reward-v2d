# Blind saved interaction join — source-only seam

`reconstruct_interaction(endpoint_row, endpoint17, pose_row, pose11, hoi_row,
hoi19)` joins one genuine original still image. No filesystem, RGB decode,
inference, optimization, references, scoring or new transport is involved.
The host import is stdlib-only; NumPy/numerical contracts load on invocation.

The root caller first authenticates whole sources, runtime/checkpoints, sealed
complete producer receipts and file SHA pins. This helper only checks numerical
array metadata and cross-bank consistency; fingerprints are **not provenance**.
It accepts exact original17 endpoint,11 DWPose and19 HOI fields, including full
900 GDI/1500 HOI native arrays and3600 OWL patches. DWPose is not synthesized if
its output is unavailable. Same opaque32-hex image ID, original slot, acquired
ordinal, grid and frame0 must agree. No identity is inferred from ordinal alone.

Pose person IDs, original boxes and detector scores must byte-match the supplied
endpoint census, in unchanged order/dtype. Native133 validity is recomputed as
raw score>0; in-grid is separately recomputed. The old episode/six-array saver
and its decoded-RGB field are not fabricated or reused. Raw missed/off-grid
coordinates, duplicate HOI query IDs across classes, negative pair logits,
original NMS indices, object-target pairs, zero-area/out-of-image OWL proposals
and native masked `-inf` GDI text logits are retained.

Existing `PersonPoseObservations`, `HOIDetrObservations` and
`Owlv2ObjectObservations` validate their native contracts. The latter checks the
FP32 square-pad inverse and row-major3600 IDs. The unchanged lossless OWL bridge
and `build_interaction_candidate_evidence` create every P×2×3600 base row,
P×2×K local tuple and K×3600 route bridge. No P×2×K×3600 feature tensor, topK,
threshold, nearest association, union, selected route, alias fusion or scorer
is introduced. Genuine P0/K0 keeps empty local routes and the complete OWL bank;
absence is not predicted OFF/no-contact. Missing supported coordinates remain
NaN with explicit numerical-support flags, not filled anatomical observations.

The return object includes the three reconstructed contracts, existing evidence
and all three immutable byte-backed raw array mappings. Supplied array metadata
is checked before and after construction; no input values are modified. This
is an enabling source/ABI seam, **not an executed DWPose join**, physical-person
identity, owner/contact validation, temporal association, calibrated likelihood,
3D reconstruction or benchmark adoption. Future dataset/phase guards and fixed
image populations remain caller obligations.

Source-only qualification: author209 related tests PASS0.54s, independent40
dedicated tests PASS0.26s, root263 related tests PASS0.68s. Actual endpoint17,
DWPose11 and HOI19 producer fields/save metadata match; no actual pose inference
was run. All owned fixtures removed. Helper10476/`e8333e3180fe009f08bb14201b20919a6bec823164d19c7f21d2124eae56a2a1`.
