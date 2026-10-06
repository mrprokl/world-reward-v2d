# Host-only actual48 acquisition input seam

Prospective source/tests qualification only. No new acquisition, model, FIT/CAL,
role values, pixel decoding, selection, resampling or performance claim here.

## API and immutable originals

`authenticate_actual_acquisition(code, revision, entry, helpers, checkpoint)` in
`infra/vcoco_fit_cal_acquisition_inputs.py` returns `(records, public_proof)`.
The supplied namespace belongs to the NEW caller. All imported helper origins
must match that code; no mutable ENTRY profile or old-source import is used.

Original acquisition producer: `d938e7c40c4fd90353fbe92ebd2970303a639b9b`,
316 files/321 source entries, closure
`f8c01005b0c2ba5d055784753d9cb8d903d1b1a6f02a5a3c483665f0ddee3b6b`;
original XZ6 archive SHA256
`567424d0eef72e5e6edfab82e7054aaddf01864d942d9d1f8446a4a859113d29`.

| Immutable artifact | Bytes | SHA256 |
|---|---:|---|
| acquisition/report.json | 133045 | 900596ae02a7b6fc60a79353f03517689fc8f540c1842b9ae3421df1114bf2c0 |
| inputs/manifest.json | 9754 | e98657edc8e081ed5b101edf22cdbb490b77bf36384e8952eff49f82d0c7b93c |

The exact qualified acquisition/context helper bytes are frozen independently.
Existing explicit-context authentication checks the complete old freeze/census/
prepare/three-upstream source lineage and 41 inputs/assets, including old pilot
reference HASHES and JPEG byte hashes. No original run/census/reproduce is called.

## Saved-proof reconstruction

Authenticate the new caller with `context.authenticate(...)`, then authenticate
the original d938 namespace with `f.original(...)` and its original helper list.
Deep-copy the new context proof; replace only its current_source and the three
current-parent/marker states with their authenticated ORIGINAL d938 counterparts.
Nested original-freeze proof, all upstream state, metadata/cohort identities and
private parent remain identical. Compare the JSON-normalized copy to the saved
original report. Neither the caller proof nor module globals are mutated.

This is a saved-integrity bridge, not a claim to rerun historical computation.
Byte/mode/marker checks and source state are actual; historical execution facts
remain those of the immutable original receipt and its independent saved audit.

## Full48 byte and privacy contract

Require original PASS/complete, 48/48 acquired, FIT32/CAL16, zero missing or byte
aliases, fixed recipe, original source/posthash/seal flags and budget. Root stays
700, acquisition and inputs exactly500, owned single-link regular leaves400.
Check the acquisition directory has only its report, and inputs has exactly49
leaves: the original manifest and 48 original flat JPEG files.

Every frozen slot is joined to its private original publisher metadata, license
ID4/CC-BY2, grid and original S3 URL. Original SHA256/MD5, no historical/new alias,
JPEG header (not decoded content) and opaque mapping must agree. Keep all48;
there is no smaller-bank rescue. The public six-key manifest is byte-pinned.

Returned records are ONLY `image_id,file,bytes,sha256,width,height,path` with
opaque IDs and canonical public paths. Returned proof contains source/report/
manifest identities, aggregate48 count, hashed private-context state and public
artifact byte/inode state. It does not contain private native IDs, split, photo,
publisher rows, roles, boxes or reference values. Negative scope flags are explicit.

The host calls this API before and after its operation and requires complete
equality. **Never mount this helper, private report, metadata cohort, context or
role helpers in the native prediction container.** Native inputs are the original
public manifest and JPEGs plus a tightly whitelisted source/model runtime; strip
the host-only path from projected six-key records if a JSON projection is needed.

Publisher individual-image grant metadata is not independent creator identity,
author-disjointness or final submission clearance. Pretraining/challenge overlap
and role-versus-anatomical-ownership/contact limitations remain unverified.

Source-only qualification: author93 related controls PASS0.77s (22 dedicated),
root132 related controls PASS0.90s. Fixtures removed; no actual new caller or
model inference is implied by those checks.
