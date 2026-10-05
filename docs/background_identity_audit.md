# Background distractors: choose the interaction before tracking it

Primary-source audit, October5,2026; methods available by September30. This is
an architectural decision, not a corrected challenge prediction or quality gain.

## What CARI4D actually does

- [Original CARI4D paper, v3](https://arxiv.org/html/2512.11988v3), sections3,
  3.1,3.2 and7: reconstructs an already targeted person/object interaction;
  the object mask identifies the target. Its dynamic hypothesis selection
  chooses **poses of that object**, not a partner among competing instances.
  The mostly-visible initial object is an assumption, not an automatic selector.
- [Original preprocessing at71fa7cbe](https://github.com/NVlabs/CARI4D/blob/71fa7cbe46081467edadd11ab534b0c14aa9d913/prep/run_sam3_masks.py#L69-L151):
  `merge_masks_from_output` takes the union of all SAM3 instances for each text
  concept (69–80). Chunked segmentation starts a new session every300frames
  (95–151). This can merge matching bystanders; it is not persistent joint
  person/object identity. Independently read10349B/SHA256
  `7d74990996e13df4ac8f6d8e6c53b37d0d9a37db88bb26d0b515ecb93057d8d9`.
- [NLF initialization](https://github.com/NVlabs/CARI4D/blob/71fa7cbe46081467edadd11ab534b0c14aa9d913/prep/run_nlf_sepK.py#L86-L102)
  uses the supplied mask bbox and suppresses competing detections; missing masks
  can reuse an earlier mask. This cannot establish the actor. Read9405B/SHA256
  `72994fa0b769c830053a01130de7f58f030ec76c8d96ac622200daefbb98c092`.
- [V2D CARI4D README at our executed7c0d pin](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_cari4d/README.md#L58-L85)
  offers interactive target annotation. [Its SAM2 adapter](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_sam2/lib/video_to_masks.py#L86-L115)
  propagates supplied IDs0/1, not a learned interaction assignment. We cannot
  manually provide challenge targets. The independent audit finds these two
  files and FoundationPose byte-identical at the October5 V2D HEAD; no switch
  of executed producer or post-cutoff adoption follows.

Our automatic alternative has a separate vulnerability: `actor_selection`
first uses the top confidently separated **object detection**, then spatial
affinity chooses a person. A highly scored background object can therefore
produce a stable but semantically wrong pair. Mask/mesh agreement and low
identity-switch counts cannot certify this pair. EP8/9 quality remains rejected;
no hand reassignment, private prompt, parameter tuning or historical overwrite.

## Methods address different failure classes

| Component | Useful mechanism | What it does not establish |
|---|---|---|
| [DAM4SAM, CVPR2025](https://arxiv.org/html/2411.17576v2), [cutoff code](https://github.com/jovanavidenovic/dam4sam/blob/aad389b85c224bd408da8923e8fcb96410cf7e30/dam4sam_tracker.py#L121-L168) | Distractor-aware recent/anchor memory after an initial bbox/mask. | Which initial person/object is task-relevant. SAM2.1 is a separate release, not DAM4SAM. |
| [SAM3.1, March27,2026](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/RELEASE_SAM3p1.md) | Separate concept instances and multiplexed multiobject memory; lower bank execution cost. | Manipulation ownership. Never union its instances as the CARI helper does. Author speed/benchmark claims are not our measurements. |
| [MASA, CVPR2024](https://arxiv.org/html/2406.04221v1) | Appearance correspondence across a complete automatic proposal bank. | Person–hand ownership or physical interaction. Our pre-checkpoint CUDA failure remains closed pending an isolated original-source build. |
| [HOI-DETR, June2026](https://arxiv.org/html/2606.17384v1), section3.3 | Explicit learned hand→direct-object→tool-target links. | Hand-to-body owner, left/right slots, temporal identity or calibrated contact. Our128-image proxy study is closed INCONCLUSIVE, not adopted. |
| [SingleQuery-BHOI, September10,2026](https://arxiv.org/html/2609.12155v1), sections3,5,6.3 | One person query attaches body pose and both hands; a hand-to-entity relation matrix has off/self/other targets. | Published executable/checkpoint availability, robust crowded-video reconstruction, or correct Track1 predictions. |

The September paper is unusually relevant to our actual failure. It reports
strict tuple accuracy54.00/33.09/22.02% for1/2–4/≥5people, **conditional on matched
persons**. Crowding remains difficult; this is not an all-person retrieval score.
Training-label wrist-nearest/IoU/VLM verification is annotation construction,
not a permitted way to correct individual challenge labels. Our design must
retain unknown ownership, not inherit those rules as ground-truth guarantees.

## Availability and rights: fail fast before another model stack

The [SingleQuery project page](https://lgecto-ail-vil.github.io/SingleQuery-BHOI/)
marks code coming soon. Actual primary page22146B/SHA256
`666d7a1608b35a371ff15073bb4b7e7ebf281fe01938b38b58bb8be0fbf1ce93`.
Its separately announced code repository returns404 on this audit: no verified
published model to install. This is evidence of unavailable public access, not
proof that the authors have no implementation.

[Dataset release atb23fec9f](https://github.com/LGECTO-AIL-VIL/SingleQuery-BHOI-Dataset/tree/b23fec9f3aecfc80435ac329ad73320565a80926)
contains actual person/left-hand/right-hand/target IDs and no-contact/self-contact/
person-contact/object-contact validity flags. Source tree metadata shows
trainZIP36442716B/Git blob7f80165303a8a5dbea060f6413c3fbcb2ed8c339 and
valZIP2880378B/e320060f3c4fdd210515818b6e47cc0695d65ad0. These are Git blob
identities, not publisher SHA256 or an assertion about decoded archive contents.
Neither archive, examples, labels nor image bytes were read in this audit.

Only tiny primary texts were read:

| File | Bytes | SHA256 |
|---|---:|---|
| README.md |10362|389f6d0d37b1a156f3b17b00c0231e69fcf75be0028f44c0145caa1656b4d3f2|
| LICENSE-DATA.md |784|b5771123b84cd9babecd2ae4ce819d5723efa9e49c8bb92488f677c64be8b128|
| annotation_schema.md |18884|94605645e0a1122d39fbaf3ec9f47ef19177e75b84985c12a9618b20f05a1402|
| annotations/README.md |8527|2f3cb8f3508d6089a6d51904b6d4092b4d5d508c846427136b5d745342a2a70a|

The explicit annotation grant is CC-BY-NC4.0 for research/education. Competition
use eligibility is not assumed. Original COCO image rights remain separate;
the released annotations explicitly remove image license/source URLs, so those
must be joined from original COCO metadata. This is a promising **static
multiperson ownership reference**, not temporal/metric3D validation. Defer heavy
acquisition and training until those rights, overlap and a disjoint protocol
are cleared. Code-license badges never override data/model terms.

## Decision and next actual experiment

Prioritize **joint target identity and visible anatomical ownership**, then
memory robustness. Keep all automatic people/objects and competing pair routes;
body pose supplies person-attached wrists, HOI links supply candidate interaction
evidence, appearance supplies persistence. Null/no-contact/occluded ownership is
explicit. Proximity, depth, co-motion or one pair logit alone cannot select truth;
do not hardcode foreground, largest person, nearest hand, or moving object.

Reuse the existing observation/scene ledger and native HOI producer, not another
transport framework. First qualify person-attached wrist observations for the
whole bank and preserve all ambiguous hand-owner hypotheses. Then freeze a
new external comparison before labels: identical detector bank and downstream
CARI; spatial-only versus anatomy+relation selection. Measure proposal recall,
initial correct person–object pair and complete tuple precision/recall on **all**
records, counting missed people/objects and abstentions. Separately measure
identity drift on new videos; an externally assisted initialization is only a
tracker control. Final adoption additionally needs full-T body/object/relative
3D non-regression, no per-frame alignment and unchanged legitimate input scope.

No re-evaluation of the closed OpenImages pilots, no manual episode fix and no
claim that a literature result, procedural test or prettier overlay solves the
leaderboard objective. Useful memory operators may run in parallel, but they
must not be mistaken for the missing initial interaction selector.

## Executable person-attached observations

October5: independent source audit selects the already deployed native DWPose133
CPU model before a more expensive full-body pass. Original inference_pose accepts
all automatic person bboxes and returns one133-point row per input crop, in order,
via N sequential batch-one ORT calls. One whole-bank adapter call is not one
batched ONNX forward. Body wrists9/10 and hand roots91/112 are different predicted
landmarks; source metadata30735B/SHA256
`8c73971296517eadef6b7df6de92eb8943119cb467b3f4d488a6e062470cf179`.

The new person_pose_observations adapter retains IDs, duplicate competing boxes,
native scores/coordinates and misses, skips native N=0 full-image fallback, and
never selects a target. Parent109 combined tiny tests PASS0.42s; additional
automatic-census diagnostic tests bring125PASS0.25s. These are manufactured API
controls, not actual multiperson anatomical correctness.

Next CPU probe uses all saved automatic person proposals from EP9/26's first
three recorded seed banks, unchanged native model and separate fresh output.
These are actual original linspace frame indices, not frames0/1/2; remaining
seed positions contain counts only. No selected masks/prompts/Body/CARI enter.
It is challenge-video diagnostic inference, not an independent validation
cohort, hyperparameter tuning, identity correction or adoption. Actual execution
and independent output/source/input cleanup audit are still required.

A source audit also catches an upstream multi-person SAM3DBody mask pitfall:
process_one_image blindly reshapes provided masks to[N,H,W,1]. Input[H,W,N]
would mix people rather than transpose. Our historical N=1 calls are unchanged;
any future all-person Body call must use explicit person-first mask layout.

Actual fresh producer5e4255d CPU diagnostic PASS2.958846s: EP9 frames0/27/55
retain2/3/3 persons, EP26 frames0/26/53 retain2/2/2. One session executes14
original batch-one forwards; all six bank arrays stay on Azure. Independent
whole-source, frozen input and prediction hashes, readonly outputs and owned
CID/name absence pass; receipt24928B/
0d8227caf7d35df87547c3a7cd92cb89686d08372f22adc8ee3fd8b940aadac6.
The earlier9d8703 preflight path failure remains closed. This qualifies the
native complete recorded-bank transport only, not full-T observation coverage,
person identity, anatomical correctness, manipulation, adoption or accuracy.

### External validation is a separate remaining dependency

A bounded independent audit did not identify a fully qualified public
person/left-hand/right-hand/object ownership cohort. This is not proof that
none exists. DexYCB/OakInk2 are useful single-actor controls, not crowd tests;
SA-V supplies masks, not ownership; scene-graph role annotations do not supply
the complete tuple or establish their own data grant. COCO-WholeBody's primary
README license wording and linked CC-BY-NC terms differ: do not silently treat
it as an unrestricted annotation source. Previously closed OpenImages pilots
remain closed. None of these metadata checks establishes model-training overlap.

Two routes remain: obtain explicit permission for the relevant existing
annotations/images, or acquire a genuinely new independently annotated real
RGB cohort with author rights and participant consent, directly on Azure.
A proposed small capture is12 clips, separate2 FIT/2 calibration/8 reserved,
covering stationary targets, bystander manipulation, crossings/occlusion and
similar objects. It is **not frozen, available, captured or evaluated**. External
human labels are permitted; challenge labels/prompts are not. New capture after
checkpoint freeze would provide a temporal independence check, not universal
generalization. Full-tuple errors must count missed proposals and abstentions;
sparse annotations cannot certify unannotated temporal identity or3D contact.
No email has been sent, data acquired or comparison gain claimed.

### Prospective complete tuple evidence bridge

The pure interaction_tuple_evidence bridge now enumerates every person ×
left/right side × native HOI hand/object pair, including duplicate proposals.
It retains source slots/IDs, raw joint coordinates/scores, both relation logits
and separate native-valid/in-grid diagnostics. Fifteen named geometric/raw
features are evidence, not a learned selector or calibrated probabilities.
Missing native joints give unsupported/NaN distance features; finite off-grid
coordinates remain explicit diagnostics. All inputs stay unchanged. Parent148
combined tiny observation/bridge/display tests PASS0.21s; no real joint-bank
execution, anatomical correctness or external accuracy comparison follows.
A separate bounded Azure CPU preview will display all six already frozen banks,
not choose a target or infer labels from human QA. Colors are frame-local slots,
not stable temporal IDs.
