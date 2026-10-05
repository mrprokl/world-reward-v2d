# Fresh real-RGB proposal validation — October 5, 2026

## Decision

Use a **new32-slot Open Images cohort**,16DEV/16RESERVED, for SAM2 proposal
localization only. This is immediately implementable with cached Azure metadata
and the already-qualified SAM2.1 AMG runtime; availability of32 suitable new
authors has **not** been measured. Run the new metadata-only census first. Never
reuse the closed144-ID pilots, closed64-slot GDI study (including14missing slots),
their original photos/MD5s/URLs or their publisher-author profiles. The rejected
HO-Cap pilot also stays closed and is not an input. No acquisition/inference is
authorized by this audit or census implementation alone.

The old authenticated census has270eligible records/232raw author-profile URLs,
but subtracting64records is insufficient: all historical authors must be
excluded and URL aliases canonicalized. Remaining availability is unknown.
The existing acquisition selector does not exclude *historical authors* and
must not be reused unchanged. The new driver reuses only unchanged geometry and
identity functions; original studies/configs remain untouched.

## Legal and overlap boundaries

[Google's primary license statement](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses)
grants annotations under CC-BY4; images are listed CC-BY2, with an explicit
instruction to verify every individual image. [CC-BY2](https://creativecommons.org/licenses/by/2.0/legalcode.en)
allows reproduction/derivatives, including commercial use, subject to its
conditions; retain creator/title/original URL/license attribution and mark any
published modifications. Do not confuse this with a code license or an NC ban.
The existing strict Flickr JSON-LD check verifies a matching creator-name,
photo landing and license declaration, not independent creator-account/title
ownership, consent, privacy or all third-party rights. Keep RGB private on Azure.

SAM2 code/checkpoints are explicitly Apache2 at
[source2b90b9f5](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md).
Its SA-1B/SA-V/internal pretraining is not proven record-disjoint from OI or the
challenge. This experiment is **study/photo/metadata-author-disjoint**, not a
certified unseen-pretraining benchmark. Publisher-author grouping does not
prove different depicted people, events or near-duplicate images. OWLv2/SAM3
explicitly involve Open Images and are not interchangeable validation arms here.

## Frozen metadata census, no HTTP

New files: `infra/proposal_external_census.py`, matching wrapper/config/tests.
Seven original cached inputs are byte-pinned: boxes, relationships, triplets,
image metadata, old16, old128 and the frozen old64 publisher-metadata cohort.
Old64 cohort31043B SHA256
`441bdc57e101cb4e8291ca6bbebc554ca397619e2a892469d9e410a16359a456`.
No RGB, model output, previous metric, HTTP, GPU or current challenge input.

Reuse original record gates: at least two nonintersecting countable person boxes,
two nonhuman/non-body-part boxes, one uniquely bound positive `holds` pair and
Rotation exactly`0.0`. Flags group/depiction/inside must be exactly zero;
occlusion/truncation remain. These are conservative metadata selection gates,
not physical identity guarantees or inference assignments. Missing historical
author identity fails instead of asserting disjointness. Historical Flickr
route variants are exclusion identities only, never expanded rights requests.

Exclude all208IDs plus canonical author profiles, original-photo IDs, URLs and
MD5s where available; no cross-author photo duplication. Deterministically sort
by SHA256(`world_reward.oi_proposal_external_v1/`+ImageID), then ImageID, retaining
one author/photo/MD5/URL. Require32slots before any HTTP. First16DEV, next16
RESERVED. Too few slots => INCONCLUSIVE_CLOSED_NO_RGB, no replacement or another
sample seed. Seal a metadata-only cohort with no reference boxes/relations.
Inclusive180s CPU census budget, original-source/input posthash, separate output
namespace. Actual result and downstream authorization remain pending.

## Smallest subsequent scientific measurement

After a successful census and separately frozen acquisition/inference recipe,
verify each creator grant and original MD5/JPEG/orientation. All32selected slots
remain; inaccessible records are misses, never replaced. Native SAM2.1 Hiera-L
AMG author defaults are32x32/.8/.95/NMS.7/no crops. Preserve every returned mask,
bbox and quality value; no largest/nearest-person/hand pruning or per-image query.
Freeze full native banks before private references are exposed. DWPose/HOI and
learned scoring are unnecessary for this proposal-capacity gate.

Primary: image-balanced **bbox proposal recall at IoU>=0.5**, against all positive
annotated countable nonhuman/non-body-part bbox rows, with exact duplicate
annotation rows deduplicated and classes preserved. Every fixed slot contributes;
inaccessible/empty bank contributes0. Report object-row recall, raw/duplicate
reference counts, proposals/image, bytes and runtime. A fixed one-to-one matching
diagnostic may expose composite proposals covering multiple reference boxes;
do not silently change matching after outputs. Set numerical support/rappel
go-no-go thresholds in the *next* recipe before first inference, not this audit.

This is actual object-localization support from mask-derived boxes, **not mask
IoU, exhaustive precision, semantic objectness, ownership, contact or task
selection**. OI boxes cover annotated positive classes, not every unknown region.
Body/class hierarchy duplicates must not be renamed into physical identities.
Native SAM mask quality is not calibrated objectness. Publish DEV outcomes first;
freeze the next algorithm before opening RESERVED once. A closed failure is not
rescued by thresholds, extra prompts, resampling or additional model arms.

True mask-IoU needs separately acquired publisher segmentation PNGs; they are
not among the cached seven inputs. [Original format](https://storage.googleapis.com/openimages/web/download_v6.html)
defines `MaskPath`/`BoxID`; its CSV box is the annotation starting box, **not the
mask's bounding box**. Instance masks omit small/ambiguous instances, so do not
invent missing mask ground truth. Avoid that acquisition until bbox coverage
justifies it. No ownership/adoption/CARI4D/leaderboard win follows from this gate.

## Actual census and bounded acquisition implementation

Actual producer`d1de716d9d6e94165d14224d17dee36fe2ef2fe5` completes census
PASS3.274094s:208excluded IDs/197canonical excluded authors,
175eligible remaining records/164distinct new authors. Frozen32cohort18076B SHA
`4db28db7b8c9beb0ce58e574e8f012f05c86ef7447ff056a269d4b04cd74ed33`;
report3847B SHA`5c9362852da341159ceefdfe37e4b866a06840ff4d69fbf281424d38b28e3849`.
Parent's independent census audit and actual acquisition result are separate.

New acquisition driver reuses unchanged original creator-rights/HTTPS/MD5/JPEG
mechanics:32fixed slots,6workers,300s inclusive,16MiB/JPEG,15s socket timeout,
zero retry/replacement. Full current/historical census source and all seven
metadata inputs rehash before/after. Private root-only rights/ledger400 and
directories500 retain every slot, including failed downloads. JPEG originals
move by the same inode to`inputs/image_<original-slot:06d>.jpg`, unchanged hash,
no duplicate or re-encoding. Public`rgb_proposal_inputs.v1` contains acquired RGB
only and exactly six fields per image; no original ID/author/split/relations.
Missing slots remain solely in the private manifest, not fake images. No GT,
model, GPU, reference geometry or scientific success is implied by acquisition.

## Next prospective real source after closed OI32 zero-RGB acquisition

The OI32 acquisition is now CLOSED with zero qualified RGB. Do not recover its
photos via another URL, change rights/query thresholds, replace records or
reopen OI240/HO-Cap. The next practical real-RGB source is **COCO2017 validation,
restricted to publisher-reported CC-BY2 images**, for proposal recall only.
This is a new dataset/cohort, not an unseen-pretraining or ownership benchmark.
No COCO image, complete annotation archive or model was acquired in this audit.
Cached COCO annotations on Azure have not been confirmed; root must inventory.

### Actual primary availability and licensing

[COCO Consortium terms at a historical July2020 source pin](https://github.com/cocodataset/cocodataset.github.io/blob/aaa6a5a0cc24bf1350247169cc512edd7ddf28b9/dataset/termsofuse.htm)
are2370B SHA`bd019f88ee44c29b2f19c5b99888cf5bc2e7c16f57b6e52af8ed3a60462e8bdd`.
Annotations have an explicit CC-BY4 grant. Images remain their Flickr creators'
copyrights under individual licenses; the Consortium is not their licensor.
The [native format](https://github.com/cocodataset/cocodataset.github.io/blob/aaa6a5a0cc24bf1350247169cc512edd7ddf28b9/dataset/format-data.htm)
(15380B/SHA`a15ea326fbb7a7d5adc349aa8bfd615e442886b0c5cccb819cc9d5bce3185897`)
contains each image's `license` identifier, `flickr_url`, dimensions and official
`coco_url`, and original instance `bbox`/polygons/`iscrowd`. It contains **no
creator name/profile or original MD5**. Those absences are disclosure limits,
not grounds to invent an author-disjointness guarantee or an endless identity
verification infrastructure.

The [official download source](https://github.com/cocodataset/cocodataset.github.io/blob/aaa6a5a0cc24bf1350247169cc512edd7ddf28b9/dataset/download.htm)
names bucket`images.cocodataset.org`. Its original annotation archive is publicly
accessible via authenticated HTTPS:
`https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip`.
Actual HEAD200:252907541B, Last-Modified2018-07-10,
ETag`f4bbac642086de4f52a3fdda2de5fa2c` (**not a publisher SHA256**), version`null`.
Local `https://images.cocodataset.org/...` had a certificate verification error;
the canonical S3 endpoint passed standard TLS, with no verification bypass.
Only a65536B Range206 prefix was read in RAM, extracting the publisher license
table, not RGB or per-record annotations. Fullarchive SHA must be computed on
Azure if it is not already cached; no whole-file SHA is asserted here.

That table explicitly maps **id4** to`http://creativecommons.org/licenses/by/2.0/`.
Do not assume all COCO photos have that license:1/2/3are NC variants,5isBY-SA,
6BY-ND,7"no known copyright restrictions" (not CC0),8government-work metadata.
The prospective narrow cohort uses id4 with an exact corresponding license URL.
[Flickr's primary CC explanation](https://www.flickr.com/creativecommons/)
recognizes the creator grants; it does not waive third-party/privacy rights.
The license assertion is the **publisher's recorded individual-image grant**,
not independently authenticated creator-account ownership. Keep raw RGB and
labels private; retain license plus original Flickr photo URL for provenance
and attribution, supplement creator/title when supplied. Do not claim creator
verification if a legacy landing page is unavailable, and do not publish a
raw-image bundle lacking appropriate attribution.

### Concrete32-slot census before any new predictions

1. Authenticate an existing official`instances_val2017.json`, or acquire the
   original252.9MB annotation ZIP only on Azure. Record whole-file hashes and
   original archive member; do not fetch1GBvalRGB or18GBtrainRGB. Metadata only
   may select records; no reference boxes/polygons/captions enter inference.
2. Candidate records: exactlicenseid4; at least two distinct positive-area
   **noncrowd person instances** and two distinct positive-area nonperson
   instances, with at least one nonintersecting pair of each. `iscrowd=1`
   describes a collection, never two physical people. Resolve original category
   named`person` from the native catalog, without turning categories into model
   queries. Preserve occlusion/multipart polygons and original grids.
3. Extract Flickr photo IDs from native static URLs; exclude all240closed OI
   photo IDs and any known URL/content identities before selection. Exclude
   historical author groups **only if actual creator metadata is supplied**;
   otherwise state author-disjointness unverified. COCO image IDs are a different
   namespace, not evidence of photo disjointness. Full old-original MD5 checks
   cannot be inferred from COCO metadata; reject detected exact-byte duplication
   after original download, without replacement.
4. Sort once by SHA256(`world_reward.coco_proposal_v1/`+zero-padded COCO imageID),
   take32slots16DEV/16RESERVED, photo IDs unique. Too few => closed noRGB. No
   selection using pixels, captions, model predictions or download success.
   Freeze cohort and recipe before each image request. Download only official
   filenames via the same canonical bucket
   `https://s3.amazonaws.com/images.cocodataset.org/val2017/<file_name>.jpg`;
   actual32availability remains untested. Seal SHA/size/header/grid; retain
   unavailable/right-uncertain slots as explicit misses, zero replacement.
5. Run the already-qualified unchanged SAM2AMG once per acquired RGB. Freeze
   complete proposal banks before private annotation evaluation. Primary remains
   image-balanced bbox recall@IoU.5; report noncrowd mask-IoU recall separately
   using original polygons only after a tested native decoder, not invented
   boxes-as-masks. No precision over unannotated categories, ownership, task
   selection or temporal/3D claim. Same source/defaults for DEV and RESERVED;
   thresholds/gates fixed before DEV, no RESERVED reuse after failure.

COCO-trained DWPose/HOI and many vision backbones overlap this dataset by design;
SAM2/web-pretraining exact overlap remains unknown. This is useful **real-domain
proposal-capacity validation**, not leakage-free held-out superiority. Author-
disjoint new own captures would be stronger but require human participation;
the already-frozen authored synthetic stress can proceed independently now.

### 4DAnyone is not the proposal unblocker

[4DAnyone source9cc2aa23, September23](https://github.com/ant-research/4DAnyone/tree/9cc2aa230fe5d364da2c5634ec3dabadadcb0d60)
is available, but its README recommends one person; its native GVHMR route calls
`Tracker.get_one_track` and uses a full-image person box after an empty tracker.
It does not expose a high-recall general-object or person-to-object association
module. Motion source13196B SHA
`13ef1744bff6b5046ca11d38e686e271b7a9311e18783520ddec07d6d7cdbd3a`.
Its generated reference views remain priors, not observed validation truth.
Reuse the structured-context idea later; do not acquire another multi-GB stack
to solve this immediate missing-object-proposal gate. No new code/model run or
independent real-validation PASS is established by this section.

## Actual metadata and label-free closure

New census producer d1de716 qualifies175candidate records/164new author profiles
after all208ID and197historical profile exclusions; freezes32slots. Actual
3.274094s. Independent complete source/Git and7input before/after hashes plus
exact metadata selection replay pass. New acquisition924e206 completes4.102033s
with0RGB/32missing (26HTTPError,6ValueError;12creator-rights and20original-RGB failures); no model or reference boxes
opened. This fails the predeclared>=12acquiredDEV gate and closes the study
INCONCLUSIVE before further inference/labels. No URL substitution, retry,
resampling, new query or cohort rescue. Transport PASS is not dataset usability
or a quality result. Full acquisition source/input audit remains pending.

## Prospective real COCO execution protocol

The minimal COCO producer now has separate `--census` and `--acquire` phases.
The first authenticates the full official annotation archive and reads only its
original validation member. It freezes32metadata-selected slots and32separate
private reference JSONs. Acquisition requires the independently recorded census
and cohort SHA/byte pins; it hashes, but never decodes, the private references.
All32slots remain, including unavailable or byte-duplicate records. Both members
of a detected duplicate group are censored, without favor or replacement.

The existing unchanged native SAM2AMG accepts a fresh COCO public-RGB namespace;
no model/default, mask filtering or prompt changes are introduced. The new
CPU-only box evaluator uses every native XYWH exactly as saved, without inclusive
pixel correction, clipping or deduplication. FixedIoU.5 primary/.25/.75 diagnostic
and image-balanced all-slot recall gate.6 are frozen before first RGB/model.
At least12acquired DEV images are required before any reference evaluation.
Missing images score zero; no misleading successful-acquisitions-only gate or
all-slot instance-micro claim. Reserved references open once, lazily, only after
the DEV gate. Polygon IoU is explicitly deferred until an original decoder is
qualified. Capacity is not precision, semantic objectness, ownership or3D.

Local generated fixtures and private-loader spies pass170combined tests in.41s,
including57COCO preparation tests; no real RGB/model or measured real recall yet.
COCO publisher-reported individual-image CC-BY2 and annotation CC-BY4 are kept;
creator/author identity and pretraining overlap remain unverified. The whole
252.9MB annotation acquisition, selected originals and inference stay on Azure.
