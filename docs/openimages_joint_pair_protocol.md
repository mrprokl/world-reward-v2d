# New joint-pair feasibility — October 5, before RGB

The two old Open Images studies are CLOSED INCONCLUSIVE. Read their authenticated
144 IDs only to exclude them; no old predictions or metrics enter this census.
This is a new **metadata feasibility census**, not a rerun, expanded pilot or
validation victory. It selects no images for inference yet.

Use cached original publisher bbox/VRD/triplet/metadata CSVs, pinned before run
in `configs/openimages_joint_pair_census_v1.json`. Preserve original classes,
coordinates and flags. Exact positive relation duplicates alone deduplicate.
Binding requires exactly one same-class IoU>=.5 box, including all duplicate,
group and unknown boxes in the ambiguity test. Count only group/depiction/inside
flags exactly zero; occlusion and truncation are not exclusions.

Prospectively require two countable human boxes with zero intersection and two
countable nonhuman/non-body-part boxes with zero intersection, at least one
uniquely bound positive, and publisher Rotation exactly `0.0`. This conservative
geometric lower-bound reduces hierarchy duplication; it does not prove physical
identity. It selects **records**, never model candidates. All 13 classes named
`Human ...` in the exact boxable catalog are excluded as object-count evidence,
not just the ten initially suggested body parts. Catalog12011B SHA
2fa47fb1b87e71c90b9fbee2dc1184eb8c59601bb811ea4424b200653cedff8e.

Feasibility requires >=16 eligible images and >=8 nonempty source author groups,
within180s CPU host budget. Below either gate: INCONCLUSIVE_CLOSED_NO_RGB. Above:
freeze a separate new experiment/split before original-image acquisition or
predictions. Individual creator grants/attribution, MD5, size, orientation and
decoded-header gates remain mandatory, with unavailable slots retained and no
substitution. Author groups/MD5 help split hygiene; neither establishes unseen
pretraining or absence of challenge overlap.

The subsequent narrow metric can compare **automatic P and O endpoints jointly**
against positive published `holds` pairs after predictions freeze. No reference
box may crop, gate a hand, select a person/object, or set an inference parameter.
Unannotated competitors remain unknown, never invented negatives. An unknown
winner, missing proposal or abstention means an annotated positive was not
retrieved, not a proven false positive. This cannot calibrate OFF/no-contact,
test left/right hand ownership, or establish the unique task target/full-HOI3D.
Those need different explicit external references. No CARI4D win claim follows.

All CSV/image/model bytes stay Azure. Record only pins, counts and decisions
locally. No new backbone or transport system is needed for this census.

## Actual metadata result and next frozen sample

Producerbf47f065c6e076d9a73e5084fe02d9b91cb45d9c: PASS2.844554s,
897 untouched relation images after excluding144oldIDs;296geometry-qualified,
270also publisher-orientation-qualified,232nonempty source-author profiles.
Receipt842858B SHA106fbe042343f5ce3462c89245c2e02eaf4dfaabad145782cc4b06caf4df30d7.
This clears only the metadata feasibility gate; no new RGB/predictions were read.

Freeze64slots from the eligible population sorted by SHA256
(`world_reward.oi_joint_pair_v1/`+ImageID), retaining at most one author profile
and one original MD5/photo identity. Exclude exact original-photo identities of
old144 records as well. First32are DEV, next32 TEST; DEV first24FIT and next8
DECISION. No failed-download replacement, no choice based on image content or
predictions. Creator and original-file gates are unchanged; original32TEST slots
remain the evaluation denominator even if rights/access/header/proposal support
fails. Source-author disjoint is not depicted-person/event/pretraining disjoint.

Learned-score comparison is a subsequent frozen protocol, not this acquisition:
positive-set listwise loss may rank observed positives above unknown competitors;
its gradient implicitly suppresses unannotated alternatives. Never call this
unbiased contact classification or manufacture absent-pair ground-truth negatives.
Multiple-positive linear loss is not guaranteed convex; numerical termination and
fixed fit recipe must be checked. Without any matched positive, a FIT record has
no defined positive-set loss and cannot be repaired with GT proposals. Preserve
such records as explicit support failures in evaluation. No OFF training here.

## Frozen scorer comparison — before new inference

Use exactly the same automatic retained P/object proposals in every arm. Queries
`person.`/`object.` are fixed across records, native GroundingDINO .3/.25 and
classwise NMS .3/.7 unchanged. No SAM masks are necessary for this component.
DWPose133 processes ALL retained P crops, no full-image fallback when P=0;
HOI-DETR supplies all original H→O routes separately. None receives reference
boxes, creator metadata, split, positive counts or selected task labels. Source,
preprocessing, checkpoint and precision are pinned before native execution.

Primary is image-balanced joint annotated-holds-pair top-choice retrieval on all
32TEST slots, matching BOTH automatic endpoints only after scores freeze.
Exact duplicate P/O coordinate tuples count once; their upstream pooled score
must be identical. Exact top ties receive fraction of unique top tuples matching
a known positive. Unknown winner, empty proposal bank, abstention, inaccessible
RGB or unscorable reference contributes0, not an asserted false relation. Report
proposal/localization support and resolved/unresolved positives separately.
Class-agnostic endpoint matching considers all original reference boxes, including
hierarchy/group overlap; ambiguous matches are explicit misses, not relabeled.

A: geometric automatic P→O ranking, lexicographic outside-object-center distance
and then center-center distance, normalized by original person-box diagonal.
This mirrors the old automatic actor affinity but scores ALL P×O competitors,
without first restricting to the top-scored object. It is not an anatomical
owner classifier. B: a small grouped linear learned scorer over immutable base
and optional HOI/anatomy features with numerical-availability indicators; its
exact pooling/features/fit recipe must be committed before FIT inputs open.
Do not change the baseline after seeing its outcomes. No TEST tuning, per-image
thresholds, reference crops, missing-positive GT proposal injection or human
challenge labels. Fulltuple/LEFT/RIGHT/OFF remain outside this experiment.

After B is frozen on24FIT and8DECISION, open32TEST once. Require >=6 acquired
TEST images with at least one jointly localized annotated positive, strict
positive mean paired B−A across ALL32slots, and a one-sided exact paired sign
test p<=.05 on nonzero image deltas. Ties stay in the mean/denominator but carry
no directional vote. Insufficient support=>INCONCLUSIVE_CLOSED; non-superiority
or mean regression=>REJECT. No widening/resampling/test switching on a closed
cohort. A positive outcome is promising only for observed static relation
retrieval and still needs explicit ownership/task/temporal3D non-regression
before any Track1 adoption.

## Actual frozen acquisition (no predictions)

Producer e2b6019f502e18694a221aa6b1818804ef3d6d83 completed in122.853752s.
Exactly64slots remain:50original files acquired (27DEV/23TEST),14missing,
100776311RGB bytes retained only on Azure. Cohort31043B SHA
441bdc57e101cb4e8291ca6bbebc554ca397619e2a892469d9e410a16359a456;
manifest76232B SHA57135ee4853afaff9d5820f11128fc2c8da9c42b2cb14b40597909c9f0f0eeb4.
Independent saved-only audit rederived the same metadata-only selection, all
original MD5/JPEG headers, seals and full source/input closure. It verifies
creator-name/license landing declarations, not independent account identity,
training/challenge overlap, decoded quality or prediction accuracy.

Two earlier preflight failures read no RGB and acquired no sample: Flickr
metadata identity parsing omitted set/direct route variants. Fixes broadened
only exclusion identity parsing, not the stricter new creator-page requests;
no failed selected slot was substituted and no predictions informed selection.
