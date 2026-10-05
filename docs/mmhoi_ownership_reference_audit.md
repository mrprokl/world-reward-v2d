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
container interface or bounded central-directory reader; acquire only original
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
