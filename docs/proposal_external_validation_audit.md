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
