# Next ownership reference: a publisher-hosted relation dataset

2026-10-05. Source/metadata feasibility only; no Azure call, annotation archive,
image, checkpoint or new benchmark execution. The latest ownership acquisition
is **closed**: 68/96 acquired, FIT23/CAL10/RES35 below24/12/36. Do not retry its
HTTP failures, reduce denominators, reseed OpenImages or select an available
subset. Exclude **all432 historical slots** (336 earlier +96 latest), including
unavailable photos, using publisher photo IDs/URLs and known byte identities.

## 1. Visual Genome — best next metadata gate

The [original author's UW mirror](https://homes.cs.washington.edu/~ranjay/visualgenome/about.html)
explicitly licenses **Visual Genome under CC-BY4**, not merely its Python code.
The [primary paper §5.1](https://doi.org/10.1007/s11263-016-0981-7) identifies
Creative-Commons Flickr photographs from the COCO/YFCC100M intersection. This
supports a genuine publisher data grant; it does not silently replace each
creator's original image license. Join preserved COCO/YFCC creator/license
metadata and retain attribution. A valid historical grant does not inherently
require a fresh live Flickr request. Different CC variants, missing rights
lineage and creator identities remain separate checks; publisher CC-BY4 is not
a blanket proof that every underlying photograph is commercially cleared.
[Multimedia Commons' primary terms](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/)
explicitly retain creator-selected image licenses. No author was contacted.

The [release/schema](https://homes.cs.washington.edu/~ranjay/visualgenome/api_readme.html)
provides image IDs with `coco_id`/`flickr_id`, instance `object_id` and XYWH boxes,
and directed subject/object relationships with predicate and relationship ID.
Scene graphs link `subject_id`/`object_id`: this is genuine **instance-pair
reference**, unlike deriving ownership from nearest boxes. Person aliases,
duplicate nodes and v1.2/v1.4 field differences require explicit source-backed
handling. Dense annotations are **not exhaustive negatives**; no anatomical
side, observed OFF, temporal identity, physical contact or unique task target
is established. The paper describes region-derived relations and human-checked
co-reference merging (§§4.4–4.7); preserve published IDs, do not relabel them.

**Access is materially better than per-photo Flickr acquisition.** HEAD-only
checks of [official download links](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)
returned200 and `Accept-Ranges: bytes`:

| Publisher asset | Actual Content-Length |
|---|---:|
| `data/dataset/image_data.json.zip` (UW) |1,780,854|
| `data/dataset/objects.json.zip` (UW, v1.4) |55,323,929|
| `data/dataset/relationships.json.zip` (UW, v1.4) |77,904,473|
| `https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip` |9,731,705,982|
| Same directory, `images2.zip` |5,471,658,058|

These are **header observations, not SHA-authenticated downloads or successful
206 member reads**. The advertised349/709MB are not the compressed HEAD sizes.
Use the author mirror, not an assumed still-valid API at the old domain.

**Smallest next action:** separately freeze an Azure-only metadata census
(~135MB compressed objects/relations/image metadata, zero RGB). Authenticate
publisher bytes/version and archive safety; exclude432 photo identities before
examining their annotation rows. Join per-image licenses from an authenticated
existing COCO metadata cache where covered; unknown rights are rejected, not
rewritten. Report creator-known/unknown rather than claim author disjointness.
Count crowded images (≥2 person instances, ≥2 object instances) and published
positive manipulation edges under a fixed, documented predicate mapping.
Predeclare capacity and FIT/CAL/RESERVED population before RGB. Only then seal
a new deterministic cohort and qualify bounded ZIP-member206 access. Preserve
failed slots; no availability-based replacement. No whole15GB image download
is presently justified. This is a **new relation-label source**, not OI rescue.

## 2. HICO-DET — suitable pair labels, rights blocker

The [official publisher](https://umich-ywchao-hico.github.io/) supplies human/object
box pairs and interaction classes, exhaustively within its defined object/HOI
scope; this is not universal OFF supervision. Version20160224 is a7.5GB bundle.
[Original code](https://github.com/ywchao/ho-rcnn) uses
`hico_20160224_det.tar.gz`; individual random-access member delivery is unproved.
No explicit image-creator or separate annotation grant was found on the audited
publisher page/README. Third-party CC0 declarations do not resolve that gap.
Defer RGB acquisition pending genuine data terms, not another downloader.
The deployed HOI-DETR declares COCO/refinedHands23, **not HICO training**;
GEN-VLKT's proposed HICO/VCOCO checkpoints are a different model. Exact image
overlap remains unknown in either case.

## 3. ADE20K — reject for this question

[Primary terms](https://ade20k.csail.mit.edu/terms/) explicitly separate
noncommercial-research/education images (MIT does not own their copyright)
from BSD-three-clause-style software/annotation permission. Segmentation/parts
do not supply published person→held-target IDs. It may test endpoint masks,
not correct-owner retrieval. Do not spend acquisition/GPU budget on it here.

## Comparability and source pins

VG/COCO exposure in detector/backbone training is possible; DWPose and HOI-DETR
have declared COCO-family training. Exact checkpoint/VG/challenge overlap is
**unverified**, never leakage-free. A photo-disjoint new study can measure
conditional positive pair retrieval, not unseen-pretraining generalization or
a CARI4D/Track1 victory. Keep endpoint recall separate from relation retrieval;
all automatic candidate pairs and misses survive. FIT-only learning, CAL-only
predeclared calibration, RESERVED opened once; unannotated pairs stay unknown.

Tiny HTML byte identities fetched2026-10-05, total76,952B; no linked assets read:

| Primary text | Bytes | SHA256 |
|---|---:|---|
| UW VG `about.html` |8984|`29ec54c0554a4e12c4f695afd7f6bbe7a77db411090ee3031268950ea38a3a7b`|
| UW VG `api.html` |8197|`43a80892dbc5f5ca6215ad75f7bf3d57c510ddd7444d2d9025165f52e80f7774`|
| UW VG `api_readme.html` |42785|`127a62312d2b5b93a103e85d6deadd5804e6c2ce5ace6ae879b1826d4eccc306`|
| HICO official root |9523|`d1d9fa2c8bcba0a5544c871a8e1a13fdd5fa7cc0f1a0b449fbe2622d0678c2b9`|
| ADE20K `terms/` |7463|`d5984affc025b8240ce7333772049e38ac0eb3fe73687c5d5a3cd99c8aaaa8aa`|

**Decision:** one bounded VG metadata/rights-lineage census next; HICO deferred,
ADE20K excluded. No benchmark is claimed fully acquired, eligible or validated.
