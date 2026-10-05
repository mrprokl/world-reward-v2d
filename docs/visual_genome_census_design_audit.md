# VG crowded held-pair census: predeclared design

2026-10-05. Prospective metadata-only: no archive/annotation rows/RGB/models/Azure,
cohort selection or FIT/CAL/RESERVED opening here.

## Inputs and rights

Authenticate actual acquisition report, full producer Git/XZ/markers and five
files (3ZIP+2HTML) before rows. First-read SHA≠publisher attestations. Include
independent release/alias text pins below in the new closure.

Existing COCO2017 **val5000 image/license metadata** only:
ZIP252907541B/SHA113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268;
report3562B/SHA c8dcee4e4f78b56a5ba8b9f644f94507715a9c46f6c05d0a305f8ce977f58786,
producer4657c8b45f733a1043c5a8af2f5b9d6ac027196a. Authenticate its original
source/member; stream `images`/`licenses`, skip `annotations` values. No old
individual references, train images, predictions or media.

COCO license4 must resolve to its recorded `http://creativecommons.org/licenses/by/2.0/`.
VG `coco_id` must identify one val image; `flickr_id` must match its original
`flickr_url` photo ID. Conflicting IDs/grid/links reject. This preserved
**publisher-recorded image grant** supports the same narrow research-use basis
as prior COCO—not independent creator authentication. Creator UNKNOWN; no
author-disjoint/consent guarantee. Retain source/license/attribution, no raw
redistribution. A valid historical grant need not require live Flickr. Outside
this authenticated subset: rights-unknown/ineligible.

## Exclusions before semantic annotation access

Authenticate complete historical432 slots: OI240 +COCO32 +endpoint64 +closed
ownership96, **including unavailable slots**. Read only their publisher identity
metadata; hash any embedded old reference files without opening values. Union
Flickr-photo IDs/URLs, known MD5/authors. Resolve VG image crosslinks first;
exclude old/ambiguous records before semantic object/relation parsing. New VG
MD5/author absence stays UNKNOWN, not exclusion-complete. Future RGB must check
MD5 before models and preserve unavailable slots.

## Source-backed schema and fixed semantics

[Release v1.4](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/readme_v1_4.txt)
leaves images/image_metadata unchanged from1.2; updates objects/endpoint boxes.
Join unique `image_id`, not order/URL basename. Relations carry
`relationship_id`, predicate, subject/object `object_id`/XYWH/synsets/names.
Positive finite in-grid XYWH only; no clipping/nearest-IoU/reconstruction.

The [data readme](https://homes.cs.washington.edu/~ranjay/visualgenome/api_readme.html)
uses singular `name`; preregister **exclusive** `name:str` or `names:list[str]`
variants retaining raw strings. Other/conflicting forms, duplicate native IDs,
missing endpoints or endpoint/global-object XYWH/synset/name-set conflicts:
**ineligible**, never merged/repaired. Structural ambiguity closes census.
Equal boxes/different IDs remain ambiguous raw nodes, not certified people.

Person whitelist: exact `person,persons,man,mans,woman,women,womans,womens,
boy,boys,girl,girls,human,humans` from publisher groups. No substring/stemming
or future expansion; other decorated aliases deliberately uncounted.
Paper §§2.2/4.8 confirms ambiguity; unknown labels≠nonperson/OFF. Conservative
targets: exact79 nonperson COCO category names from authenticated catalog;
other nodes UNKNOWN/not eligible targets. This avoids counting unknown people/
parts as targets. Predictor retains ALL generic proposals, not this vocab filter.

Freeze **only exact case-sensitive predicates `holding` and `holds`**; the
publisher joins them. No `wearing`/`with`, reversal or later synonyms. Require
direct P→target edge, ≥2P IDs/≥2target IDs and positive boxes. Positive retrieval
only; no exhaustive negatives/anatomy/contact/OFF/task/temporal truth.

## Bounded census and output

Incremental stdlib top-level JSON array reader, each raw row≤16MiB, duplicate-key/
nonfinite rejection, UTF-8/bracket/string-safe skipping of excluded semantics.
Eligible image index+one row at a time; no1GiB `json.load` or expanded-file dump.
Complete stream/CRC/source/input byte+mode postchecks,1200s/16GiB. Predeclare
≥96 distinct eligible photo IDs: **capacity lower bound**, not author guarantee
or selected cohort. Insufficient closes before RGB. All rejection denominators;
sealed counts+private metadata-only ID/rights ledger. No public references,
selection, splits or FIT values written.

Primary text pins (no annotation/image values): readme746B/SHA
788334c8c396b15869cea8cf3097e837a8f0752fefececb9c10ea3d28089be0e;
[relation aliases](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationship_alias.txt)
122102B/15f7f64802c95c5bf5b5690457566b1b19ee64eac7a94d8b8cbb3a4378f2ec7c;
[object aliases](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/object_alias.txt)
60166B/0c8e059fc31eeebfd98231f5789892da8ae33bfa00434c70aee969dc6eaa853b.
Photo-disjoint≠pretrained-disjoint; exact overlap unverified. No adoption/CARI4D claim.

## Source qualification, not actual census

Root169 tiny manufactured tests PASS0.62s across streaming/census, acquisition,
cache/cost and launcher contracts. Independent parser review found no blocker
in actual default callers; generic nonpositive/noninteger limits now reject
before reading the source. Excluded semantic values, ID field order, native
conflicts, source-receipt ABI and late-publication demotion are tested.

No real annotation row has been consulted. Native authored parser cost control
now closes its600s gate:697.606746929s projected including120s reserve. Actual
37.772462458s covers three64MiB streams; the projection is not a bound. Keep
that control closed, never rerun it.

Prospective census **v2**, before the first semantic row: unchanged parser,
scientific eligibility,96 threshold, exclusions432 and source/rights contracts;
1200s inclusive/1215s outer, fresh output `visual_genome_ownership_census_v2`.
This supersedes the unexecuted600s v1 design, not a failed real census or
historical cohort. The measured697.61s projection fits the new budget without
altering data or algorithm; actual timeout remains fail-closed. Capacity/rights/
quality remain unverified until their respective actual stages.
