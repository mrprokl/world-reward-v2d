# Fresh person + objectness endpoint validation — proposed V2

Source audit, 2026-10-05; prospective numerical plan approved by maintainer;
**not an executed endpoint experiment or adoption**.
The closed 32-photo COCO AMG study remains a DEV recall failure with its 16
RESERVED references unopened. The closed GDI `object.` study remains a proposal
capacity failure. Neither cohort may be rescored, retuned or reused here.

## 1. What the existing person bank actually does

- `infra/automatic_masks.py` uses local GroundingDINO weights, HF revision
  `12bdfa3120f3e7ec7b434d90674b3396eccf88eb`, and the fixed query `person.`.
  Its defaults are box confidence **0.3**, text threshold **0.25**, and NMS IoU
  **0.7**. The native Transformers processor receives
  `target_sizes=[(original_height, original_width)]`: its returned XYXY boxes
  are already original-image pixels, not resized pixels requiring another
  inverse transform. Original RGB/frame identity must accompany each bank.
- `prompt_selection.non_maximum_suppression` drops invalid/out-of-grid boxes,
  low scores and exact duplicate boxes; then greedy NMS suppresses IoU **>0.7**.
  Ties sort by box coordinates. It preserves every remaining person, not only
  the highest-scoring person. “Complete” therefore means complete **post-NMS**
  census, not every unthresholded model query. Preserve rejected/raw rows too.
- The automatic-mask driver later selects a single actor for mask initialization.
  Do **not** invoke that selector, SAM prompts, or its Track1-specific `main`
  for this external bank. Detailed saved observations there cover only the first
  three seed observations, not every original video frame or all 16 seed banks.
- The reusable fresh-photo native recipe is in
  `infra/openimages_joint_pair_gdi.py`: official processor/model inference,
  `retained_bank`, and `validate_native_text_logits`. It saves all 900 original
  boxes, 900×256 logits, token IDs/attention masks, all native postprocessor rows,
  retained raw-slot links and IDs. Native masked `-inf` logits must remain intact.
  Reuse only its person recipe, **not** its closed-cohort `main` or `object.` query.
- `person_pose_bank_probe.py` is a saved-bank CPU/DWPose diagnostic, not a fresh
  person detector. If downstream poses are later needed, reuse
  `infer_person_pose_frame` on **all** retained persons. N=0 produces genuine
  empty arrays and never calls upstream DWPose's artificial full-image fallback.
  For N>0 the original callable makes N sequential batch-one crop calls and
  performs its own inverse affine keypoint transform; no added offset/resize.
  Crop IDs and DWPose hand slots are not verified anatomical/person ownership.
  The existing saved-bank CPU route uses immutable B47 image
  `sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7`
  and original DWPose source `3dca5db79d9f9ffdd378753ddf6ec66535aace88`.
  This is optional downstream reuse, not a new GPU runtime or fresh-photo
  qualification. Existing interaction-tuple evidence retains all person/side/
  candidate rows; it must not be replaced by top-person or nearest-hand ownership.

## 2. Runtime, source and rights: established versus pending

The selected frontend image is
`sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252`.
`bridge_frontend_bindings.py` authenticates original build
`d1fcb8ad158cf4ad4b1f7fe2bba7ea533ae565cc`, parent layers, source, model assets,
and kernel gate `168a809369a294c1ab65154a585b83014c98d976`. That kernel gate
qualifies SAM2 operations, **not** GDI recall. Separately, the closed external
GDI run genuinely executed 50 RGBs/100 forwards; its successful execution is
reusable runtime evidence, not validation of its inadequate object proposals.

OWL assets V2 genuinely acquired and independently byte-audited the four public
assets. Parent reports actual native qualifier PASS 7.5842s and host PASS 10.8145s:
one load, three image embeddings, three objectness and three box calls, all 3600
patches per procedural image. Independent saved-only CPU audit subsequently PASS
2.6248s: all four original Git/XZ closures, exact saved FP32 banks/square inverse,
and installed source/wheel RECORD identities rechecked before/after, no new model
or GPU call. See `results/audits/owlv2_native_qualification_v1_actual.json`. This
qualifies execution, not endpoint accuracy or ownership; bind the actual receipts
and proof before photos, never substitute mock tests or metadata readiness.

Use original [Google OWLv2 checkpoint/card at
57beb61adb5abda3de4a9796bc35ae60bc4b9802](https://huggingface.co/google/owlv2-base-patch16-ensemble/tree/57beb61adb5abda3de4a9796bc35ae60bc4b9802).
Card 4838B SHA256 `7c7426bc5ec939a42d1f96fb093031b6263400cceac4129ebb941a0c8c11b9b9`
declares Apache-2.0, June 2023, and cites
[Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683).
Its data section names web/YFCC100M and COCO/OpenImages detection training,
explicitly **“to be updated for v2”**. Exact checkpoint training-photo/challenge
overlap is unknown. A new COCO split is photo-disjoint from **our research**, not
proved independent of model pretraining. Preserve notices; publisher declarations
are not blanket France/EU/competition clearance. No SAM source/license is needed
for this native person+OWL execution merely because the image contains SAM.

`owlv2_object_observations.py` uses original Transformers 4.53.3 image embedding,
objectness and box heads only: no text/class query, NMS or fallback. Preserve all
60×60 row-major patch IDs and native FP32 objectness **logits**, not calibrated
class probabilities. Original square bottom/right padding means box corners are
scaled by **max(original H,W)**, not separately by W/H. Outside-image and
zero-area boxes remain unchanged. `bridge_owlv2_candidates` is lossless transport.

## 3. Prospective photo-disjoint question and selection

**Question:** can the unchanged complete person detector and native OWL
objectness ranking expose distinct annotated endpoints at a usable object-bank
budget, without target labels/prompts? This is not actor selection or HOI quality.

Propose **64 fixed slots, 32 DEV + 32 RESERVED**, selected once from original
COCO val2017 metadata. Require each selected photo to have ≥2 non-crowd,
positive-area person annotations and ≥2 non-person annotations. Keep the old
metadata-only separated-pair condition and license ID 4 (CC-BY-2.0 photos).
Select by SHA256 of `world_reward.coco_endpoint_v2/` plus
zero-padded native image ID, then image ID; unique native Flickr photo identity.
Do not select on predictions, image appearance, difficulty or known recall.

Before selection, authenticate and exclude **all 240 prior OpenImages slots and
all 32 closed COCO slots**, including unavailable slots and unopened RESERVED.
Use their public photo/ID records only, never their reference values. Exclude
matching Flickr photo IDs and known exact-byte/MD5 aliases; do not claim arbitrary
crop/reupload/creator disjointness. If 64 eligible unique photos do not exist,
stop INCONCLUSIVE; no replacement or relaxed selection. Original JPEG acquisition
is one attempt per slot. Require at least **24 of the fixed 32 DEV photos acquired
before any model call**; fewer closes INCONCLUSIVE without reference scoring.
Missing slots stay explicit and score zero, never vanish.

Reuse the authenticated COCO archive cache and metadata parsing, not old results:
`annotations/instances_val2017.json` from publisher ZIP 252907541B/MD5
`f4bbac642086de4f52a3fdda2de5fa2c`. Freeze all IDs, splits, raw file/decoded-grid
pins, licenses and reference pins before inference. Preserve publisher/Flickr
attribution records where supplied; **author identity remains UNKNOWN**, not
verified attribution to a particular creator. Annotation terms are CC-BY-4.0.
Creator reconfirmation and exact model-training overlap remain unverified. Keep references
outside GPU mounts. Old `/srv/world-reward-data/coco_proposal_v1/metadata/cohort.json`
is an exclusion input only; do not open its `reference_*.json`.

## 4. Fixed metrics and proposed independent gate

Freeze the maintainer-approved numerical policy in a new config **before any new
prediction/ref score**; this prose authorizes no run.

- Person P: all unchanged post-NMS `person.` boxes. Object O: stable descending
  native OWL objectness logit, ties by original patch ID. Diagnostic budgets
  **32, 128, 3600**; primary **O@128**. These are evaluation-only views: production
  storage/bridge retains all 3600 rows; no top-K deletion or new objectness cutoff.
- Original-pixel continuous XYXY IoU **≥0.5**, no clipping or alignment. Persons
  match person instances; object proposals match every annotated non-person
  instance without category prompts. Crowd/zero-area annotations are outside the
  declared target population, not opportunistically removed after predictions.
- Primary per-image recall uses maximum-cardinality one-proposal↔one-reference
  IoU matching: duplicate proposals/one enclosing box cannot receive several
  instance votes. Also report conventional independent best-IoU coverage as a
  diagnostic, so matching conventions are explicit. Every selected image and
  every eligible instance remains in the respective denominator.
- Equal-image macro recalls, separate P and O endpoints. **Proposed capacity gate:
  macro P recall ≥0.70 AND macro O@128 recall ≥0.70**, first on fixed DEV, then
  independently on RESERVED only after DEV PASS. This is a prospective engineering
  coverage target approved before new outputs, not a fitted threshold, powered
  hypothesis test or CARI claim.
  No pooled P/O compensation or choice of 32/3600 after seeing 128 failure.
- Both complete predictor manifests must be sealed and all acquired native banks
  validated before reading DEV references. Any technical partial/truncated bank
  is INCONCLUSIVE, not scientific PASS. DEV failure closes the study; RESERVED
  references remain unopened. No hyperparameter update, retries or new sample.

Report per-image instance/hit counts, acquired/missing slots, P raw/retained
census, O full/inside/zero-area census, all three budgets, duplicate matching,
matched-proposal fraction and P/O paired coverage. An unmatched COCO proposal is
not automatically true background: its catalog is not exhaustive of all objects.
Full-bank recall can improve simply by dense geometric coverage; disclose its
budget/unmatched burden, never market O@3600 alone as an intelligent detector.
Box recall proves neither mask completeness nor person↔object ownership,
manipulation/contact, temporal identity, 3D geometry or better reconstruction.

## 5. Minimum implementation seam, not another framework

One fresh endpoint driver/wrapper/config can reuse existing authenticated source
and acquisition contracts. Host authenticates all exclusions/rights/64-slot
manifest; sends GPU **only** image ID, path, byte SHA and original H/W. Native
loads GDI once and OWL once, runs exactly one person query and one image-only
OWL bank per acquired photo, preserves empty P and every O patch, and seals raw
NPZ+complete ordered manifests. No DWPose/SAM/HOI/tracker calls are needed to test
endpoint capacity. CPU evaluator later opens only the new reference projection.

Reuse the native recipes explicitly, without importing closed-cohort `main` or
mutating frozen helpers/configs. Authenticated original runtime/model receipts
remain separate from the new caller's complete source binding. Use existing VM02
offline GPU-lock, owned-container, source/input/asset pre/posthash and sealed
publication policy. Proposed one-attempt native budget is 600s including sealing;
timeout closes INCONCLUSIVE, not an excuse to raise budget after values. Source
closure must include both native adapters, NMS and the original processor files.
First actual endpoint study, if approved, is this one—not another metadata loop.

Audited repository source at `5b03f2ffc78eeb3a8439770b67cd4923c32c4e58`:

| File | Bytes | SHA256 |
|---|---:|---|
| `infra/openimages_joint_pair_gdi.py` | 26539 | `6d5f5976de54b45ca179aed0c8d2ac3d9da2379b56334ec585326884d9887352` |
| `src/world_reward/prompt_selection.py` | 9971 | `04c9ed2ce9d004920da436d2e15ce0d464275598f122411eac87b47bb8c95ae6` |
| `src/world_reward/person_pose_observations.py` | 7614 | `c93fa24f4ce4bd6b5f4c53f4e48943b2c61d0c20a2211b36ef6d0ccb6b3ce375` |
| `src/world_reward/owlv2_object_observations.py` | 8748 | `8587489a860b44a74bc0149c67960128b6b12aa1e4d51bc07e0e52103da5b888` |

These hashes record this code audit, not independently qualified new prediction
receipts. No Azure/data/model operation or research-reference read occurred here.


## Actual outcome (frozen protocol, October5)

The64 fresh slots all acquired once on Azure. Native GDI+OWL33.8373s /host39.2775s
completed two model loads and64 calls per recipe, retaining all raw900+3600 rows.
Independent saved-bank/source/runtime audit PASS4.5162s precedes reference reads.

| Split (32 slots each) | P one-to-one recall | O@128 one-to-one recall | Gate |
|---|---:|---:|---|
| DEV |0.7874599358974359|0.8999335529666411|PASS|
| RESERVED |0.8119521103896103|0.8678985329186941|PASS|

Both use IoU≥.5, original continuous boxes and fixed equal-image denominators.
Independent scalar IoU and separate maximum-cardinality matching reproduce all
metrics and32 rows per split (DEV4.3555s; RESERVED4.5506s). RESERVED was mounted
only in a separate phase after independently audited DEV PASS; its audit did
not reread DEV references. One stdlib-host eager NumPy import failure preceded
all numerical/reference execution; its lazy-import repair changed no metric or
prediction and is recorded separately. No native GPU rerun, tuning or resampling.

**Decision:** this unchanged bank meets the preregistered endpoint-capacity gate.
Do not reopen these references for selector tuning, call this benchmark unseen
pretraining, or claim correct hands/ownership/contact,3D quality or a CARI4D win.
Next is an independently split interaction/ownership validation using the existing
complete anatomical/relation/temporal evidence, not a largest/nearest heuristic.
Exact receipts and qualifications are in `results/audits/coco_endpoint_*_v2_actual.json`.
