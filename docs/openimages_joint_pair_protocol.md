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
