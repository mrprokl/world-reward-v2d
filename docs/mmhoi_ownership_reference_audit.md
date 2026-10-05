# MMHOI: a concrete static ownership reference, not yet an executable study

2026-10-05. Independent primary-source audit; no Azure call, archive, image,
mesh, calibration, model or annotation bulk downloaded. Tiny published schema
examples were read, therefore exclude the **whole audited scenario** from any
prospective FIT/CAL/RESERVED population. No challenge reference was consulted.

## Actual release and rights

[MERL's official download page](https://merl.com/research/downloads/MMHOI)
points to concept DOI10.5281/zenodo.17711785. The actual immutable release is
[version17711786](https://zenodo.org/records/17711786), published2025-11-25:
`MMHOI_release.zip`,93,682,766,723B, publisher MD5
`3f461a6e4962e772f8acd0b718d0da8c`. This MD5 is not a computed whole-archive
SHA256. Release metadata6723B/SHA256
`0101d7acb7dedf176ff5486a474818d34a2072b4e937635657e4b7879527378d`.

The publisher explicitly grants **all data CC-BY-SA4**, not merely website or
metadata rights. Preserve attribution and applicable ShareAlike obligations;
our code cannot relicense the dataset Apache. Separate SMPL-X/model/source
rights and exact pretrained overlap remain unverified. Real recorded RGB and
published action/body-part references are useful here without using SMPL-X
parameters, multiview imagery, calibration or reconstruction truth as inference
inputs. External reference generation using multiple cameras is not permission
to supply those cameras to a Track1 predictor.

The archive's actual
[README](https://zenodo.org/api/records/17711786/files/MMHOI_release.zip/container/MMHOI/README.md)
is6876B/SHA256
`30db4e21175441b0a0cdd76d5391b7ff8004ede5851aee7caf5b38bbaee278a5`.
[Author repo6032da46](https://github.com/kaenkogashi/MMHOI/tree/6032da465e2346ce47278b7352db6f6565dac980),
2026-05-25, contains only README2040B/SHA256
`c64d34fb9d3ef385104acdc0c9660a9725c9e917c84031c8b8522498350b9acf`.
No executable method/checkpoint/software licence is established. Later v2
object-part affordance labels require email access and are not presumed granted.

## Direct relations actually observed, not nearest-box pseudo-truth

Schema-only inspected prefix:
`MMHOI/sequences/20240412_personA_personB/20240412_C_5K3__30skip/00271/`.
Do not use this scenario to fit, calibrate or evaluate the prospective learner.

| Member | Bytes | SHA256 |
|---|---:|---|
|`PARAM/action.csv`|375|`ae9fe4ca556553f4ada5d71a1451b5259e22354a214a41a719dc5b303050e276`|
|`final/C_5.csv`|127|`27adec6d45b747293a81121fd62f7ed073b907f390ad9cff78343ff2b08280b1`|
|`PARAM/bbox_0.csv`|366|`71cbe61790cdbc9bcc6c9c1d2f7a80aef2e3d77b9429b83a5d0094c0b9d07139`|

`action.csv` has no header: eight coordinates, personID, object name, verb.
The inspected record has all six2-person×3-object rows, including hold, sit and
no-interaction. This single schema example does not prove an exhaustive label
policy for every frame. `bbox_0.csv` maps each entity name to four coordinates
plus undocumented geometric fields; coordinates agree with action rows.
`final/C_5.csv` is object-row/person-and-object-column relational matrix with
body-index sets and distinct `none`, `-`, `YES` cells. A hold verb coexists with
none in this matrix: **actions and contact cannot be treated interchangeably**.

Missing verified semantics: the14 body-index dictionary; all sentinel meanings;
XYXY/XYWH order and camera conventions; scope/lifetime of entity IDs; exact RGB
to reference frame map (one camera3 image has a different index). Do not infer
them by fitting projections, inspecting the most favourable model outputs or
treating no-interaction as verified hand OFF.

## Next fail-fast gate

**GO for bounded metadata feasibility; NO-GO for anatomical fitting/evaluation.**
Zenodo exposes individual archive members through its container API; the original
camera0 RGB `0_00271.jpg` is individually exposed, HEAD200/397657B, content unread.
The inspected root listing truncates at955 entries and covers only part of one
scenario. It is not a complete census, and named folders alone do not establish
synchronization, subject disjointness or uninterrupted video availability.

On Azure only: establish complete inventory using an authenticated documented
interface or bounded central-directory reader; acquire only original
schema/split/CSV members needed for a metadata census. No93GB archive download
before showing that this solves a useful research gap. Freeze scenario/session
disjoint FIT/CAL/RESERVED before RGB, excluding all earlier schema-read scenes.
Prove mappings/sentinel meanings from primary sources, otherwise stop that scope
and ask the author a precise question. No arbitrary numerical body mapping.

Once decoded: RGB-only complete automatic bank, all persons/objects preserved,
endpoint support and direct person–target retrieval measured separately from
side/bodypart/contact. Raw predictor has no labels/IDs/camera/source geometry.
Freeze A/B outputs before references; retain misses and multiple positives.
No static result certifies temporal identity, metric3D quality or a CARI4D win.

## Actual API boundary and frozen inventory gate

Independent backend-source audit catches a dangerous false shortcut:
[`invenio-records-resources/extractors/zip.py`3afc2588](https://github.com/inveniosoftware/invenio-records-resources/blob/3afc25887baf7e2e48067326425e652b8661511b/invenio_records_resources/services/files/extractors/zip.py),
14565B/SHA256
`e5c8721771e22735e872a7f97779ca9a48218d9b690c3a856800c9fc235657bf`,
caps files+directories at1000. The observed955+45 explains truncation;
`total` counts returned files, not the archive. **A directory container link
streams a ZIP of that directory, not a paginated JSON inventory. Never GET it
for a census.** Deployed backend version is unknown; these observed semantics
are consistent with the audited source, not exact deployed-version proof.

The original publisher file metadata763B/SHA256
`1e7204a4196f11461755d6f30b8e6b97e00742a4d6e2f135794a090c6414031c`
binds version UUID`f74a5b6d-72a3-4fc6-b7d5-f0b35126696b`, file
`d359ea34-9233-4302-8308-f94aff1748ee`, bucket
`fcb5c16d-d1d3-4900-ad0d-854bef1453a2` and the original content URL.
HEAD is200/93,682,766,723B/Last-Modified2025-12-01; no ETag or Accept-Ranges
was announced, so operational Range support is **unverified**.

`infra/mmhoi_inventory.py` freezes one Azure-only attempt: pin metadata before
ranges, no redirects/retries,180s inclusive, ≤128MiB central-directory bytes,
≤1million members,8MiB request chunks. Read only the final22B/ZIP64 trailer and
exact central-directory region. Require206, exact Content-Range/length/encoding/
Last-Modified **before any response body**; a200 archive response fails without
reading it. No comment scanning into members, CRC/data extraction or whole
archive authenticity claim. Complete safe names/types/lengths and counts alone
may pass; CSV meanings, frame map, cohort, RGB and ownership remain unqualified.
20tiny authored tests PASS0.35s including ZIP64 sparse offsets, unsafe names,
budget rejection and zero body reads on invalid HTTP headers. Independent source
audit confirms the exact763B metadata ABI/firewall; no actual Azure result yet.

Original7fa90fd attempt **CLOSED technicalFAIL0.424977s**, receipt2440B/SHA
`b56bf71490d2ec3521467e0bd7499a3a6157cd3ad7f61dc63c39744651e85d08`:
the server returned206 and correct22B range but **omitted Last-Modified**.
The frozen strict firewall rejected before reading its body:0archive bytes,
0member/CSV/RGB reads,0outputs, unit exit1. This is not a failed scientific
ownership hypothesis and is not silently retried.

Separate header-only Azure diagnostic0.265103s confirms exact206/Content-Range/
Content-Length, Accept-Ranges bytes, no ETag/date. No body read; original failure
byte pin unchanged. Fresh explicit transport-v2 keeps every census/scope/budget
unchanged, accepts an **absent** date (never a conflicting date), and pins the
original publisher version/file metadata **before and after all ranges**. It
does not invent CAS or whole-archive authenticity. Separate output/config/source,
original failed receipt remains immutable and verified before/after.24tiny
transport controls PASS0.06s; actual v2 inventory/audit still required.

Actual94c6f96 transport-v2 **CLOSED metadata-budgetFAIL0.866485s**:98trailer
bytes read,0central-directory/member/CSV/RGB bytes. Separate exact trailer-only
diagnostic reproduces all three original range SHAs and ZIP64 fields:
782,626members,153,100,982central bytes at93,529,665,643. This exceeds the frozen
128MiB gate, so original failure remains; no same-namespace rerun or partial
inventory is called PASS. Original publisher98trailer bytes and file metadata
are sufficient to dimension a **new metadata-only** census, not to sample a
favourable subset or change any scientific ownership population.

Prospective explicit **inventory-v3** uses these exact782,626members and
153,100,982central bytes, total153,101,080Range bytes including the98B trailer,
600s inclusive deadline/20s per request/no retry. Original v1/v2 receipt pins are
required unchanged before/after; a separate namespace prevents overwriting the
failed gates. Asset identity, exclusions and no-label/no-RGB scope are unchanged.
No directory budget is enlarged in response to a quality result. This is a new
dimensioned metadata gate only; actual completion/audit remains required.

Primary [paper v1 §3.3/4.4](https://arxiv.org/html/2510.07828v1) confirms trained
annotators label framewise actions and involved body parts, all person×object
pairs; neither is defined as binary physical contact. README lines98–111 name
camera0–3 and bbox/action members but do not specify coordinate order, anatomical
indices or sentinels. Published temporal chunks share scenarios and must not be
mistaken for subject/session-disjoint validation. A positive person–object action
study is distinct from hand-side/contact evaluation. Supplement PDF still unread;
no numerical mapping or OFF label is inferred from an ambiguous sentinel.
