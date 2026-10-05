# V-COCO: fresh role-retrieval validation audit

October 5, 2026. **Source/protocol audit only:** no annotation archive, RGB,
checkpoint or Azure job was fetched/run. V-COCO is a 2015 static-image benchmark,
not a new September 2026 method. Proposed acquisition remains conditional on
rights and metadata feasibility; no source/data/model is declared overlap-free.

## Primary source identities and native v3 conventions

[Pinned repository](https://github.com/s-gupta/v-coco/tree/489cc4db74f2f10ab4b134f67da3874afbf245ab):
commit `489cc4db74f2f10ab4b134f67da3874afbf245ab`, June 16, 2017,
message `data v3 & script`. Files actually read and independently SHA256-hashed:

| Primary file | Bytes | SHA256 |
|---|---:|---|
| [README.md](https://github.com/s-gupta/v-coco/blob/489cc4db74f2f10ab4b134f67da3874afbf245ab/README.md) | 4031 | `08894a6d5304a49d0586c9ffeef31a87021d264d2a40d7dbe663c585e72ef5a0` |
| [LICENSE](https://github.com/s-gupta/v-coco/blob/489cc4db74f2f10ab4b134f67da3874afbf245ab/LICENSE) | 1133 | `f9eef0d2092d17798a6e11738be62c2ecd210bc52c9854d0d9ad5a38a2815c14` |
| [vsrl_utils.py](https://github.com/s-gupta/v-coco/blob/489cc4db74f2f10ab4b134f67da3874afbf245ab/vsrl_utils.py) | 5985 | `5b8ae544d79dde56cdd8ee55c47271eed01ba1f0161c665977d5e7a8c32955ac` |
| [vsrl_eval.py](https://github.com/s-gupta/v-coco/blob/489cc4db74f2f10ab4b134f67da3874afbf245ab/vsrl_eval.py) | 17878 | `eb6e765503bcc27fb73628476b641328e6f8bb26e4839f3f4645dc13afe72567` |
| [script_pick_annotations.py](https://github.com/s-gupta/v-coco/blob/489cc4db74f2f10ab4b134f67da3874afbf245ab/script_pick_annotations.py) | 2823 | `adf1d6506a187a55ce08496e8d245dd351e784597c8aa2815a54ba2e22f31850` |

The four initial sources total 29,027 bytes; the additional mapping script and
commit/root-tree metadata bring this audit's public-text reads to 36,253 bytes.

- `vsrl_utils.load_vcoco` (38–49) and `vsrl_eval._load_vcoco` (426–435)
  reshape serialized role-major IDs to **N×K**, and `ann_id`, `label`, `image_id`
  to N×1. The agent is role column zero, equal to the person's **COCO annotation
  ID**, not an image ID or row index. Obtain each action's ordered `role_name`
  from authenticated data; do not guess role/class counts from the stale README.
- `VCOCOeval._get_vsrl_data` (142–168) recognizes positives only at `label==1`;
  unannotated persons retain action `-1` and are ignored in evaluation
  (214–217). Annotated persons initialize action labels to zero. These action
  negatives do **not** make every unlisted person–object pair a negative relation.
  A non-agent role ID `0` means **no annotated role object**, and becomes internal
  `-1`; it is not an OFF/contact-negative label. Nonzero IDs must resolve to the
  same image's valid COCO instances. Preserve distinct positive action/role rows
  even when their physical endpoints coincide.
- `script_pick_annotations.py` (16–33,43–72) merges **COCO train2014 and val2014**
  instances, then filters by `vcoco_all.ids`, producing
  `instances_vcoco_all_2014.json`. Retain original image and annotation IDs.
  Do not substitute COCO2017 annotation IDs, infer a V-COCO split from a COCO
  filename, or assert photo disjointness merely because the release year differs.
  The evaluator checks split image IDs against its loaded V-COCO records (29–40).
- **Box convention matters:** native v3 evaluator (93–115,438–461) converts
  XYWH to `x+w−1,y+h−1`, clips to the image, rejects invalid reference boxes and
  computes inclusive `+1` IoU. The older utility (119–122) uses `x+w,y+h`.
  An official-comparable result must reproduce the evaluator, not silently use
  our previous continuous-XYWH proposal metric. Preserve raw references and
  separately report invalid-reference exclusions; never repair predictions.
- Native role AP first matches the **person** with maximum IoU, then its role
  object, at `>=0.5`; duplicate detections count as false positives (250–294).
  Missing-role **scenario 1** accepts only zero/all-NaN role predictions;
  **scenario 2** ignores the role location (266–276). Both normalize by positive
  action instances, including missing roles (219–220,314). README 79–81 gives
  occlusion/out-of-category motivations, not a per-record cause annotation.
  Localized-positive retrieval below has a different denominator and is **not
  official role AP**. No-role actions have no role AP (225–227).
  Roles describe action semantics, not necessarily grasped/manipulated objects
  (a support surface or instrument can be a role). Keep that scope explicit.

## Permissions and checkpoint comparability

The root LICENSE grants **MIT software and associated documentation** rights,
including sale, with notices; it does not expressly name the added role
annotation data. The Python headers instead say “Simplified BSD … provided in
LICENSE” (both files,1–8): retain the original notices and disclose this mismatch.
There is **no evidenced NC condition here**, nor an evidenced separate grant
clearing the added V-COCO annotations. Request/verify the publisher's applicable
data permission before acquiring them; do not extend COCO's grant by assumption.

[COCO terms](https://github.com/cocodataset/cocodataset.github.io/blob/aaa6a5a0cc24bf1350247169cc512edd7ddf28b9/dataset/termsofuse.htm)
explicitly grant COCO annotations CC-BY4. Flickr images remain subject to their
individual creators' grants and third-party rights. A verified CC-BY2 image is
not research-only by default; retain attribution/license/source and mark
published modifications. Cached COCO metadata alone does not supply verified
creator attribution or a new V-COCO data grant. No broad licensing waiver is
implied by public GitHub hosting or this research audit.

Audit the **exact deployed checkpoints**, not method names: DWPose's declared
COCO-WholeBody supervision, OWLv2's web/Open Images-related training, and a
HOI detector potentially trained/fine-tuned on COCO/V-COCO can overlap these
photos or labels. Record primary cards/training lists, verified overlap and
unknowns before inference. A V-COCO-trained HOI arm is an **in-domain supervised
reference**, not held-out proof that our selector generalizes. Fresh study/photo
splits do not certify unseen pretraining. Action-conditioned official AP is not
comparable to action-agnostic manipulation retrieval or Track1 task selection.

## Prospective metadata-first feasibility and split freeze

1. On Azure only, authenticate the eventual permitted v3 roles, original
   COCO2014 instances/image metadata and official V-COCO split files. Inventory
   eligibility and rights fields without RGB/model access. No annotation bytes
   or availability have been verified by this audit.
2. **Exclude every historical 272-slot photo**: 208 old OI slots, 32 closed OI
   proposal slots and 32 closed COCO slots, including missing/inaccessible slots.
   Authenticate their original cohort/metadata receipts, not predictions.
   Match canonical Flickr photo identity across HTTP/HTTPS/route variants, original
   image IDs where comparable, URLs and known MD5s. Report actual set cardinality,
   duplicates and unavailable identities; do not claim 272 unique photo IDs
   without the union check. Fail closed where freshness cannot be established.
   The closed COCO DEV failure remains closed; its reserved references stay
   unopened. No alternate URL, replacement or reanalysis rescues these photos.
3. Proposed minimal pilot: **32 FIT /16 CAL /16 RESERVED** images drawn from
   official V-COCO train/val/test respectively. Eligibility is fixed before
   predictions: ≥2 valid non-crowd individual people, ≥2 valid nonhuman countable
   objects, and ≥1 localized positive non-agent role bound to a nonhuman instance.
   “Multiperson” does not mean COCO `iscrowd` group annotations. Keep occlusion
   and all other people/objects; never select by model success. Census reports
   eligible images, localized positives and missing-role counts per action/role.
4. Deterministically order by SHA256(`world_reward.vcoco_role_v1/`+COCO image ID),
   with ID tie-break, then select the fixed counts, one photo per cohort. Use
   creator grouping across splits if authenticated metadata supplies it; otherwise
   disclose **photo-disjoint only**, not author/event/person-disjoint. If counts,
   rights or exclusion evidence are insufficient, close before RGB. Freeze a new
   protocol/cohort once; failed acquisition slots remain missing, without resample.

## What the next automatic experiment can establish

The predictor sees original RGB and opaque file/hash/HW identities only—not
GT boxes/roles/person counts, split, author, or selected target. Fixed global
prompts/class vocabularies must not become annotation-derived per-image prompts.
One all-person detector bank → unchanged all-person DWPose → generic-object bank;
retain **every person×left/right×object**, duplicates, native scores, off-grid and
missing-keypoint flags. HOI suggestions are optional features, never proposal
filters; zero HOI pairs must not erase generic objects. OFF/UNKNOWN remain
explicit abstention/availability states, not invented supervised negatives.

Freeze predictions before private references. Predeclare separate measurements:

| Measurement | Denominator / interpretation |
|---|---|
| Endpoint support | Person recall, role-object recall, and **joint endpoint coverage** at native-v3 IoU≥0.5 over localized positive roles; fixed slots/missing banks count as misses. This is the proposal ceiling, not ownership. |
| Positive relation retrieval | Image-balanced and micro pair Recall@1/5/10, with **both endpoints belonging to the same annotated positive pair**. Preserve all positive targets. Report end-to-end recall plus conditional recall where both endpoints were present; conditional gains must not hide detection misses. Matching/duplicate handling and ranking scope are frozen before inference. |
| Official role AP, optional | Original action/role vocabulary, score format and both missing-role scenarios, explicitly separated from our positive-only retrieval. Do not relabel Recall@K as AP. |
| Anatomical tuple | **Not measurable here:** roles do not label left/right hand identity, fine joints, contact or hand ownership. Wrist/root features remain automatic diagnostics, not truth. |
| Temporal identity / Track1 task target | **Not measurable:** static V-COCO lacks trajectories and the challenge's designated task target. Pair retrieval cannot certify actor tracking, full-T reconstruction or initial task accuracy. |

For positive pair retrieval, collapse left/right alternatives for the same native
person/object proposal slots by a predeclared score aggregation; V-COCO cannot
choose the correct side. Keep raw duplicate proposals and separate one-to-one
matching diagnostics. Do not let two side hypotheses count as two recovered
physical positives. Recommended primary: **global action-agnostic pair Recall@5**
over unique localized positive person/object annotation-ID pairs; preserve and
report the full action/role rows separately. Query-conditioned (action/role)
retrieval must have a separate name and denominator, never provide GT target
roles to a supposedly action-agnostic predictor.

FIT can train one predeclared low-dimensional relation ranker on permitted
annotations, using a documented multi-positive/positive-unlabeled objective;
unlisted pairs must not silently become verified negatives. CAL chooses only the
predeclared regularization/feature choices and an abstention **positive-recall**
operating point. Positive-only supervision does not justify calibrated ownership
probability, OFF specificity, exhaustive precision or unique task-target labels.
Before any RGB, freeze that small search space, endpoint go/no-go floor, primary
Recall@K criterion and uncertainty reporting in the next implementation protocol.
Do not invent measured thresholds or sufficient sample size in this audit.

Open RESERVED once, only after model/source/checkpoint/prediction pins and all
choices are frozen; publish misses, coverage, denominators and fixed-slot
confidence intervals. A failed endpoint gate stops relation fitting; a failed
reserved gate closes the study without prompts, retuning, substitutions or new
arms on those records. Success supports only this **positive-role retrieval
pilot**, not unseen-pretraining generalization, exhaustive ownership, contact,
submission eligibility, or a victory over CARI4D.
