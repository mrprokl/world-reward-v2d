# V-COCO — renewed rights evidence and prospective route

2026-10-05. Primary text/tree audit only, cutoff September2026. No annotation
values, photos, archives, model, Azure job or cohort selected. This is a delta to
`vcoco_role_validation_audit.md`, not a reopening of any closed study.

## New material evidence, not a license badge

Original [README](https://github.com/s-gupta/v-coco/blob/489cc4db74f2f10ab4b134f67da3874afbf245ab/README.md)
line3 explicitly says this repository **hosts the dataset and associated code**.
The complete [root tree](https://api.github.com/repos/s-gupta/v-coco/git/trees/517cee7d005cf57fa4e6e5f5bc1eaeb3b3118ac7?recursive=1)
contains the role JSON files alongside a root
[MIT license](https://github.com/s-gupta/v-coco/blob/489cc4db74f2f10ab4b134f67da3874afbf245ab/LICENSE),
copyright2016 UC Regents, allowing unrestricted use and sale with notices.
There is no separate data license/exclusion/NC file in that tree. This gives a
**reasonable repo-wide MIT interpretation for the committed role data**, stronger
than a code badge alone; it does not prove an explicit annotation-specific or
competition-specific grant. The standard wording names software/documentation;
helper headers also inconsistently name Simplified BSD. Therefore commercial
use is **not shown forbidden**, but the scope ambiguity remains disclosed.
The complete12-item public issue index yielded no licensing clarification.
Do not infer that CC-BY-NC is universally contest-forbidden in other projects.

Commit489cc4db74f2f10ab4b134f67da3874afbf245ab is2017-06-16 (`data v3 & script`),
root517cee7d005cf57fa4e6e5f5bc1eaeb3b3118ac7. Exact original public file metadata
below was read from Git tree only; **no JSON annotations were downloaded**.

| Role file | Bytes | Publisher Git blob SHA1 (not SHA256) |
|---|---:|---|
| `data/vcoco/vcoco_train.json` |3135892|`fa808965b680c2baee3bb5f9041335515ef0af5e`|
| `data/vcoco/vcoco_val.json` |3590505|`fbd0af34173838b506a5054e88d559e3dac217d0`|
| `data/vcoco/vcoco_test.json` |6193877|`c633fc3aeca04ed7b14c8b5bf7ff5b68bb9552f8`|
| `data/vcoco/vcoco_trainval.json` |6722166|`017b349e9097ad3a194a3591e549380942f94876`|

Re-read README4031/SHA256`08894a6d5304a49d0586c9ffeef31a87021d264d2a40d7dbe663c585e72ef5a0`;
LICENSE1133/`f9eef0d2092d17798a6e11738be62c2ecd210bc52c9854d0d9ad5a38a2815c14`;
utils5985/`5b8ae544d79dde56cdd8ee55c47271eed01ba1f0161c665977d5e7a8c32955ac`;
mapping2823/`adf1d6506a187a55ce08496e8d245dd351e784597c8aa2815a54ba2e22f31850`;
eval17878/`eb6e765503bcc27fb73628476b641328e6f8bb26e4839f3f4645dc13afe72567`.
Root-tree JSON6041/`4fb5c800dbe126e3f98c587606c271184bdf6a2f3baeab937bb13d05d6e0700f`.

## What this reference measures

`vsrl_utils.load_vcoco` reshapes role-major `role_object_id` into N×K, preserving
role order; column0 is person **COCO annotation ID**, not proposal/image ID.
`script_pick_annotations.py` merges original COCO train2014+val2014 instances
and filters official split IDs. Resolve every nonzero role ID to the same
original image; do not substitute2017 annotation IDs. `label==1` denotes action
positive. Role ID0 is missing/unannotated object, **not OFF**. Unannotated agents
are ignored by official evaluation; unlisted pairs are not certified negatives.
Roles include object/instrument/support semantics, not necessarily manipulation.
Static person→role-object positives discriminate joint pair retrieval, not
left/right hands, anatomical ownership, contact, task selection or temporal3D.
The [2015 primary report](https://arxiv.org/abs/1505.04474) describes16K people in
10K images; no present multiperson/multiobject capacity count was measured here.

[Original COCO terms](https://github.com/cocodataset/cocodataset.github.io/blob/aaa6a5a0cc24bf1350247169cc512edd7ddf28b9/dataset/termsofuse.htm)
grant COCO annotations CC-BY4, **not image copyrights**. Original individual
photo CC-BY2 metadata/attribution remains separate from the role-data MIT reading;
public mirrors do not clear people/third-party rights or prove creator identity.
Existing DWPose COCO-WholeBody and detector COCO/web supervision can overlap.
Exact HOI checkpoint V-COCO/COCO role overlap must be declared known/unknown:
a V-COCO-trained arm cannot demonstrate unseen-label generalization.

## Small next qualifying experiment, only after a rights decision

**October6 decision:** accept the documented repo-wide MIT interpretation for
private research and metadata qualification, retaining the Regents license and
the inconsistent helper notices. This is a reasoned interpretation, not an
annotation-specific legal guarantee. It authorizes only a separately frozen
metadata acquisition; not photos, FIT, deployment or submission eligibility.
Individual photo rights and checkpoint overlap remain separate gates. No need
to keep treating the absence of an annotation-specific sentence as an evidenced
commercial prohibition. No publisher message or agreement has been obtained.

1. Explicitly record either acceptance of the documented repo-wide MIT reading
   for private research/evaluation with retained Regents/helper notices, or seek
   annotation-specific written clarification. No claim of unambiguous clearance.
2. Azure-only acquire authenticated original role files/splits and COCO2014
   metadata; verify native IDs/schema and per-photo grant. Census before pixels:
   ≥2 noncrowd people, ≥2 nonhuman objects, ≥1 localized positive role. Count
   positives, missing-role IDs and invalid endpoints; no model-based selection.
3. Exclude **all432 historical slots**, including unavailable ones: OI240,
   COCO96 and closed ownership96. Use original photoID/known MD5 unions, not
   release-year or filenames. Creator UNKNOWN means photo-disjoint only, never
   author/person/event-disjoint. These are new images, not a VG/OI/COCO rescue.
4. Prospectively freeze a tiny8DEV+8RESERVED acquisition/capacity pilot, disjoint
   official splits/photos, no FIT yet; fixed missing slots/no replacements.
   Predictor gets only opaque RGB six-key manifest, allP/all3600O/allnativeK;
   roles/GT boxes/splits remain private. After banks seal, measure joint endpoint
   coverage on DEV. Failed gate closes before RESERVED/learning.
5. Only if qualified, freeze a fresh learner experiment and positive-pair metric.
   Our continuous-XYWH retrieval is **not official V-COCO AP** (official evaluator
   uses clipped inclusive+1 IoU and two missing-role scenarios). Do not relabel
   role retrieval as ownership, contact accuracy, Track1 victory or eligibility.
