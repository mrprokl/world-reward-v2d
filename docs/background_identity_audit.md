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
| [MASA, CVPR2024](https://arxiv.org/html/2406.04221v1) | Appearance correspondence across a complete automatic proposal bank. | Person–hand ownership or physical interaction. Original SM90 operators now pass; secure full-checkpoint decoding fails, so appearance adoption is deferred. |
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

### Fail-fast boundary for appearance tooling

The full original SM90 extension now passes actual embedded-architecture and
DCNv2/RoIAlign controls; the separately frozen model run fails secure checkpoint
decoding. Static inspection finds training/NumPy/MMEngine objects in its pickle,
not a verified weights-only model. Do not weaken loading security or treat this
as ownership progress. A state-only publisher artifact is preferable. Any
original-weight extraction would require a new explicit, restricted symbolic
protocol, byte-identical storage/alias/stride and benign module-version metadata
preservation plus malicious fixtures. A general pickle VM is not a priority
framework component. If this cannot stay bounded and maintainable, defer MASA
and use already-qualified automatic instance/memory observations rather than
spend further research on unrelated infrastructure. Ownership remains the
primary hypothesis and needs an independently labeled legal external cohort.


## 2026-10-05 — Legal external multiperson validation route

A bounded primary-source review found a credible **Ego-Exo4D** route. The
[38-page model agreement](https://ego4d.github.io/pdfs/Ego-Exo4D-Model-License.pdf)
allows ML development/evaluation for academic, commercial and noncommercial
purposes and creation/distribution of annotations. It is explicitly a draft,
**not our signed grant**; an executed agreement and participant identity remain
required. Prior descriptions of this dataset as research/NC-only were too broad.
[Official access instructions](https://docs.ego-exo4d-data.org/getting-started/)
state free access, approximately48h approval and14-day AWS credentials. No form
submitted or media acquired. Thomas must supply the personal signature if needed.

[Atomic descriptions](https://docs.ego-exo4d-data.org/annotations/atomic_descriptions/)
distinguish camera wearer C from other people O, including a published two-person
example take `3c489f86-8896-4c86-8a5a-929999799d36`. This does **not** qualify
eight two-person/two-object clips or exhaustive hand ownership. Proposed route:
12 licensed short single-exocamera clips,2FIT/2calibration/8sealed, capture/site/
participant-disjoint; creator-authorized new external labels blind to outputs,
including misses/ambiguity/left-right/actor/tool-target. No challenge labels.
Only explicitly selected [448-resolution takes](https://docs.ego-exo4d-data.org/data/downscaled_takes/)
on Azure; default14TiB downloader is prohibited by this experiment's narrow scope.
Do not share raw RGB with third-party annotators without their own applicable grant.

IKEA ASM, Assembly101, EPIC-KITCHENS, RICH, BEHAVE, InterCap and Panoptic did not
provide an immediately qualified alternative. Their NC conditions are not a
universal competition ban: the [Creative Commons FAQ](https://creativecommons.org/faq/#does-my-use-violate-the-noncommercial-clause-of-the-licenses)
depends on purpose/use. Rights and multiperson/ownership coverage remain unresolved.
EgoExo is not among the declared DWPose COCO-WholeBody/UBody or HOI-DETR
COCO/Hands23 corpora, but unverified backbone/teacher overlap remains unknown.
**Gates:** signed grant; actual eligible sealed cohort; overlap audit. No validation PASS.


### Actual complete native joint-bank integrity, October5

Fresh explicit Track1 diagnostic7b41e93 loads the same full1796-state HOI model
and performs six original forwards (native42.945401s/host49.109799s). Pairs
0/2/5/6/4/6 join the14original person observations into0/12/30/24/16/24=106
unselected person/side/hand/object hypotheses. First empty bank preserved.
Independent CPU-only replay checks258arrays byte-exact and full source/inputs/
runtime/output cleanup before-after. This qualifies evidence transport, not
correct anatomy, ownership, temporal coverage, calibration, adoption or accuracy.
Never turn these challenge diagnostics into manually chosen targets/fit labels.
The next scientific gate remains an independently labeled legal external cohort.

## Complete generic bank, distinct validation question

Actual six-frame native diagnostic has14persons and106person/side/HOI tuples;
its first frame has zero HOI pairs. This is genuine missing relation evidence,
not evidence that no person/object interacts. The new pure
`interaction_candidate_evidence` seam therefore keeps every supplied automatic
person × side × generic-object proposal **independently of HOI**. All native
HOI-pair × generic-object routes remain a separate, unfiltered bridge. Duplicate
proposals, raw dtypes/order, missing/off-grid joints stay explicit. OFF/no-contact
and UNKNOWN/abstention are distinct future hypotheses, neither fabricated from
an empty detector bank. This is tested numerical plumbing, not deployed scoring.

For initial validation, separate three targets: static annotated relation
retrieval; anatomical hand ownership/full tuple; unique task-relevant clip pair.
SingleQuery predicts many interactions per person, not the last of these.
Open Images positive `holds` endpoints can test the first with automatic **both
P and O** endpoints; they cannot calibrate OFF or provide reliable absent-pair
negatives. Our old evaluator's reference-person containment is not automatic
owner validation. Old pilots remain closed; a new,144-ID-disjoint metadata
census qualifies crowded records before any new RGB/predictions.

Preferred small learned component after lawful external annotations are frozen:
grouped target scorer with standardized FIT-only numerical observations and
availability indicators; balance image/person/side, not number of duplicate
routes. Reuse solver mathematics, not the private YCB label/weighting contract.
Explicit no-contact labels are necessary to train OFF; positive-only observations
must never be relabeled as exhaustive negatives. No component is adopted until
actual external retrieval/ownership and full-T reconstruction checks support it.

## Proposal generation priority after the closed joint census

The new frozen64-slot joint study closed before labels: fixed GDI `object.`
has only1FITimage with both automatic endpoints, below12usableFIT. Do not
rescue that cohort. Segmentation, objectness, ownership and task identity are
different contracts; a high-quality background mask is still a distractor.

1. Reuse **SAM2.1 native AutomaticMaskGenerator**, already executable on the
qualified HieraL runtime. Original source18461B SHA
66df266dbe14412305ae3398f0ec1bb21b303a93216b102d767e6c4ee5d4c3d7,
https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/automatic_mask_generator.py .
Keep author defaults32×32/.8/.95/NMS.7/no crops, every original returned mask
and bbox. These are regions/parts/background, not certified physical objects.
Apache2 covers code/checkpoints; SA1B/SAV/internal overlap remains unknown.
Reuse mechanics only, not the closed rejected HO-Cap validation.
2. **OWLv2 native image-only objectness**, not a generic text prompt: original
image_embedder→objectness_predictor+box_predictor. Primary paper
https://arxiv.org/abs/2306.09683 ; Transformers4.53.3 source78926B SHA
98e94770ce96e7b6b09be14596fad0600aac7a60e47b5fda846464547ae4d05e.
Official google/owlv2-base-patch16-ensemble@57beb61adb5abda3de4a9796bc35ae60bc4b9802
safeweights619918824B SHAe1e130b9e404cf91a75ad45644c1da9d7fa5284085eecc864266a6923efb99e7,
Apache2. New acquisition/runtime not performed. Preserve complete patches;
predeclare recall@K views, no proximity-to-hand pruning. Official training
includes OpenImages/COCO/web, so a freshOIcohort is not pretrained-disjoint.
3. **SAM3 PCS separate instances + global external vocabulary** is later: code
https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40 ,
customSAMlicense permits use/research subject to conditions, not Apache/OSI.
Training includes OpenImages. WordNet vocabulary extraction can be globally
frozen independent of all dataset labels, but broad queries are expensive
and incomplete. SAM3D qualification does not qualify SAM3 PCS. Never union
concept instances or manually supply episode-specific target prompts.

Prospective gates before any next RGB: authenticate source/defaults/runtime,
freeze a genuinely new procedural multi-person/multi-object distractor DEV and
disjoint reserved stress recipe, all candidates/no truncation, bounded cost,
then object/joint-proposal recall at IoU.5 after banks freeze. Procedural engine
labels are external, not challengeGT. Synthetic success would only qualify
proposal mechanics; real owned/consented external captures or an executed
EgoExo grant are still needed for ownership/temporal/real-transfer claims.
No next manufacture, model execution, adoption or leaderboard gain claimed.

### Actual real transfer fail-fast, October5

The new synthetic native-AMG proposal-capacity result is not enough. Fresh
CC-BY2 COCO cohort32,215metadata-eligible after all240closed photo exclusions,
is independently frozen/acquired32/32. Unchanged HieraL AMG executes32calls/
1024complete native regions in23.819923s; all banks precede references.
FixedDEV16 bbox-IoU.5 macro recall is **.328394all/.185550human/.437592object**,
below the prospective.6gate. Original244instances88persons/156objects yield
79/15/64best-box recoveries. RESERVED references remain unopened; the study
is **CLOSED**, not a tuning bank. Independent saved-only raw-bank/DEV scalar-IoU
audit PASS4.294886s confirms this result; RESERVED references hash-only/unmounted.

Decision: reject native default AMG as the sole high-recall actor/object
proposer. Synthetic geometry is easier than real clutter/partial humans/small
objects, and regions remain parts/background. Keep the reusable region module
but do not adopt it as a physical-object or ownership solution. A substantively
new observation system should combine an explicit complete multi-person bank,
an objectness-capable complete object bank and automatic relational evidence;
not merely increase AMG thresholds/crops until this closed cohort passes.
Validate each endpoint first on genuinely fresh licensed external records,
then annotated joint pairs and full-T reconstruction. Neither successful RGB
acquisition nor1024masks establishes foreground target selection.

The next candidate has a source-audited **image-only OWLv2** seam, not another
generic `object.` prompt. Original Transformers4.53.3
[model](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/owlv2/modeling_owlv2.py)
78926B SHA98e94770…05e executes image_embedder→all3600patches→raw objectness
and normalizedcxcywh box heads, without text/class head, NMS/topK/threshold.
[Processor](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/owlv2/image_processing_owlv2.py)
28040B SHAb20c2be9…253f uses rescale1/255, bottom/right gray square padding,
original SciPy resize and CLIP normalization. Its exact native inverse multiplies
all corner coordinates bymax(H,W), retaining padding/out-of-image proposals.
FP32 source/config/whole-bank checks are numerical contracts, not detections.

Official prospective [google checkpoint](https://huggingface.co/google/owlv2-base-patch16-ensemble/tree/57beb61adb5abda3de4a9796bc35ae60bc4b9802)
619918824B SHAe1e130b9…99e7 is **acquired and byte-verified on Azure, not yet
native-qualified**. Processor425B
SHAcf3e3966…0064/config414B SHAba9df8c2…acb2 and Apache2 notice are separate
source pins; COCO/OpenImages/web training overlap remains explicit/unknown in
detail. No claim of unseen validation or general physical-object truth.
Root157combined numerical seam/person/candidate/scorer testsPASS.54s, including
28new OWLv2 fixtures. Next native runtime/strict checkpoint qualification must
precede a new photo-disjoint cohort; the closed COCO32 is not reopened.

Actual acquisition V1 closes on a transport-only failure before checkpoint
streaming: original publisher302→`us.aws.cdn.hf.co` was not in our outer exact
allowlist. Original three tiny assets/report stay immutable. Fresh V2 adds only
that observed CDN; the original MASA streamer/inner endpoint and all model pins
are unchanged. V2 PASS9.401160s; independent fullGit273files/XZmarker/4assets
posthash PASS1.425606s. No checkpoint decode/GPU/model/data or quality claim.
81 native qualification fixture testsPASS.21s precede its data-free H100 run.
The lossless objectness→generic-candidate bridge retains all3600IDs/FP32 logits,
zero/padded/duplicate boxes and source fingerprints without ranking or filtering;
111 related fixturesPASS.69s. It does not certify object or owner identity.
