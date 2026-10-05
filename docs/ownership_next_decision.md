# Next research decision: interacting person–hand–target

2026-10-05. Decision after the **closed** endpoint V2 study, not a new executed
selector experiment. Both fixed endpoint splits pass independently: DEV
P78.746% / O@12889.993%; RESERVED P81.195% / O@12886.790%. This establishes
candidate capacity only. No correct-owner, mask, contact, temporal or 3D claim.
Do not retune or reuse the64 photos for the next selector study.

## Scientific priority

Learn a joint person–side–target ranking on the **complete automatic bank**,
then validate persistent identity. More tracker memory cannot repair a wrong
initial interaction. No largest/closest/moving-object rule or per-episode prompt.

Reuse current native GDI-person, person-attached DWPose133, OWL3600 and HOI
observations, `interaction_candidate_evidence`, `interaction_tuple_evidence`
and existing temporal identity solver. Missing HOI must not delete OWL objects.
Raw scores/logits, support and outside-image flags stay distinct; no probability
or anatomical owner is claimed by an adapter. SELF, other-person, OFF and unknown
remain separate states; occlusion is not an observed OFF label.

## External reference choices

| Priority | Reference | Discriminates | Gate before acquisition |
|---|---|---|---|
|1|[SingleQuery-BHOI release b23fec9f](https://github.com/LGECTO-AIL-VIL/SingleQuery-BHOI-Dataset/tree/b23fec9f3aecfc80435ac329ad73320565a80926)|Person ID, anatomical side and target segment ID, valid OFF/SELF/PERSON/OBJECT|CC-BY-NC4 research/education annotation grant; competition/commercial eligibility and photo rights must be clarified. No available qualified executable model claimed.|
|2|[V-COCO 489cc4d](https://github.com/s-gupta/v-coco/tree/489cc4db74f2f10ab4b134f67da3874afbf245ab)|Positive person–object role retrieval|Explicit annotation grant unresolved; no anatomical side/contact/temporal truth. Zero role IDs are missing, not OFF.|
|3|[DexYCB](https://dex-ycb.github.io/)|Manipulated-object identity and object/hand trajectory with distractors|NC data grant and other asset rights; predominantly one actor, not a bystander benchmark. No24GB download before a useful subset path/eligibility.|

Primary sources, hashes, rights and precise schemas were independently audited
in `background_identity_audit.md`, `vcoco_role_validation_audit.md` and
`identity_association_protocol.md`. A public code license never clears separate
data/model rights. Primary annotation IDs, not nearest wrist/box matching chosen
after seeing results, identify the reference target. Published automatically
constructed annotations are not independently certified physical truth.

## Single discriminating experiment to freeze next

After an adequate reference grant and a metadata census:

1. Authenticate/exclude **every historical photo**, including all64 endpoint
   slots and unavailable old slots. Freeze a fresh photo-disjoint FIT/CAL/RESERVED
   cohort, population and counts before images/predictions. Unknown pretraining
   overlap stays explicit; do not claim creator/subject independence without IDs.
2. Generate one shared automatic bank for both arms. All persons, both anatomical
   sides, all candidate targets and explicit null states survive. Predictor sees
   RGB only, no reference masks/crops/counts/target IDs or source calibration.
3. A: groupwise learned ranker on automatic geometry/support. B: exactly the same
   ranker/bank, adding original HOI link/logit evidence. Balance images/persons/
   sides; aliases of the same endpoint do not multiply label weight.
4. Learn on FIT only, fix the single CAL procedure/model/acceptance rule before
   RESERVED. Freeze both predictions and parameters before held-out references.
   No rescue grid search, resampling or new threshold after failure.
5. Primary is correct **person+side+target** retrieval over the complete declared
   population, including missing endpoints/abstentions. OFF specificity is
   separate; invalid/UNKNOWN references never become convenient negatives.
   Report conditional retrieval, wrong-person/side rates and paired uncertainty.
6. Census lacks usable positives/null/confounders, source/rights fail, or native
   banks truncate: close INCONCLUSIVE before expensive fitting. Predeclare an
   adequately dimensioned paired quality/non-regression gate before actual data;
   this note invents no statistically justified numerical victory threshold.

Only a validated static selector may seed competing temporal hypotheses. New
video validation must include bystanders, stationary targets, occlusion and
release, followed by full-T shared-frame body/object/relative3D non-regression.
DexYCB covers only an object–hand subproblem. No selector adoption or leaderboard
win follows from endpoint recall, an authored fixture or a clean overlay.

## Minimal learner seam, not another transport framework

Preserve the historical closed positive-only `joint_pair_scorer` unchanged: it
pools sides and independently pools best route features, so cannot stand in for
coherent anatomical tuples or OFF supervision. Reuse the complete observation
contracts, not its feature pooling/24-slot research recipe.

The prospective side-aware core takes every person×side×target, where targets
are all original OBJECT proposals, SELF, every other PERSON proposal, and OFF.
This is `2P×(O+P+1)` rows. UNKNOWN is unscorable reference validity or abstention,
not a fabricated target bbox. Separate state scores from conditional instance
ranking: adding duplicate objects must not increase OBJECT evidence by mere
cardinality. Preserve native slots and ambiguous aliases, not invented physical
IDs. Exact geometry alone cannot establish two proposals as one physical object.

Each coherent native HOI route combines its own hand/person geometry, direct
target geometry and logit **before** reducing across competing routes. A/B share
the same geometric routes/reduction; only B adds relational logits. Do not form
an impossible route from three different minima/maxima. Process OWL3600 without
topK deletion; blockwise route reduction bounds memory without changing the
candidate population. FIT-only feature standardization and availability flags
remain separate from anatomical visibility or calibrated contact probability.

Fresh manufactured falsifications in `test_background_selection_controls.py`
demonstrate that the old gate accepts a stable high-confidence background pair
and changes owner when only object confidence changes. Complete evidence keeps
both persons, both sides and both objects; missing anatomy never becomes OFF.
214 relevant tiny tests PASS0.79s. These are algorithm/API controls, not dataset
validation, proof of correct ownership, or a change to any challenge prediction.
