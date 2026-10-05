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

New independent MMHOI audit finds a commercially unrestricted **CC-BY-SA4 data
grant** and individually accessible archive members with explicit person→object
action/bodypart matrices. This advances MMHOI to the next **metadata feasibility
gate**, not a new training/adoption priority by licence alone. The14-part mapping,
sentinels, coordinates, RGB alignment and full inventory remain unproved. See
`mmhoi_ownership_reference_audit.md`; whole inspected scenario is excluded before
any prospective split. No93GB archive or new model stack is justified yet.

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

## Executable coherent route core (not fitted or adopted)

`coherent_route_scorer` now accepts caller-supplied finite linear weights and
returns both arms on every original person×side×object slot. It computes
`base + max_k(tuple_k + bridge_k,o [+ B margin_k])`: the same native route's
geometry and logit meet **before** reduction. No incompatible independent
feature extrema, topK object deletion, side pooling or physical-ID fusion.
Object blocks bound workspace without altering the bank or arithmetic. Native
IDs remain aligned with explicit unavailability when the upstream Cartesian
bank has no person; unavailable IDs are not invented detections.

Active-feature numerical support is shared by both arms; unsupported base is
NaN, absent/unusable HOI leaves the base with route support false, never OFF.
This narrow raw-linear core does not learn parameters, standardize features,
model availability, assign a winning owner or calibrate probabilities. A future
FIT-only learner must explicitly define these before any held-out reference.
199 parent tiny tests PASS1.02s include50 dedicated controls, two independently
found zero-bank ABI corrections,3600-object block equivalence, permutation and
duplicate invariance, and falsification of an impossible mixed route. No real
data, trained selector, temporal improvement or leaderboard gain follows.

## Reference fallback after MMHOI schema closure

The complete MMHOI inventory and primary supplement still do not establish
numerical body mapping or bbox/camera alignment. Do not spend inference on
invented hand/OFF references. A fresh **OpenImages annotated-holds pair** study
can test the narrower correct-person+object retrieval on the now-qualified
person+OWL bank, without those missing anatomical semantics. This is a distinct
protocol, not a rescue of any closed GDI/objectness/endpoint cohort.

Before choosing records, exclude all240 historical OI slots (16+128+64+32),
all32closed COCO and64endpoint slots, including unavailable RGB. Join Flickr
photo/URL/MD5 identities across datasets; author disjointness only where actual
publisher profiles exist. COCO authors remain UNKNOWN. Extend the explicit
208-slot helper rather than accidentally omit the later32 OI or64COCO slots.
Freeze a new metadata-feasibility gate and FIT/CAL/RESERVED counts before HTTP;
then original creator/photo grants, fixed acquisition and no replacements.

Unlabelled sides remain latent, scored as **complete** routes before reduction.
Positive-set listwise fitting is incomplete supervision: it implicitly competes
against unknown alternatives, not true negatives/OFF. Exact automatic aliases
must not multiply loss mass; no geometry establishes physical identity.
Standardization and geometry parameters are FIT-only; an isolated B relational
coefficient is learned with the same fixed geometry/bank/support. Declare the
optimizer, stopping rule and CAL use before references, retain all3600objects,
misses/unresolved positives and abstentions. No fulltuple, task, temporal3D or
challenge improvement follows from such a positive-only static study.

### Frozen metadata-only feasibility gate

`ownership-pair-census-v1` reads only the four original SHA-bound cached CSVs
and six complete historical metadata ledgers (336 slots). It reuses the original
automatic relation-to-box census without consulting old predictions, reference
values or RGB. All known historical author/photo/MD5/source-URL identities are
excluded; COCO creator and original-byte identities remain explicitly UNKNOWN.
The deterministic one-author/photo/MD5/URL greedy capacity is a lower bound,
not maximum matching and not a selected cohort.

Before the sole Azure CPU attempt:180s inclusive budget, no network/model/GPU,
no retry, at least96 independent slots. Complete execution is a technical PASS;
`capacity_gate_passed` is a separate decision. Insufficient capacity closes this
gate INCONCLUSIVE without lowering its threshold or choosing replacement rows.
Sufficient capacity permits only a separately frozen, rights-checked new study,
never selector adoption or a claim of held-out accuracy. Outputs and all ten
inputs/source bytes and modes are sealed or checked before/after respectively.

Actual producer4dda08752ae95687b7b2840aab9b06032e2dddd4 closes **technical
FAIL1.099374s**, exit1, report5360B/SHA
6ba7dfd843e47bae9043927c4d5f65059b99df2a09941dc2ee49c708d9d3198b.
Historical336 slots/source/ten-input hashes and modes pass; no geometry, RGB,
network, model or cohort selection follows. Independent saved-only diagnosis
PASS1.1067s verifies289Gitfiles/294entries, all input pins and unchanged failure.
801relation IDs have metadata without duplicates/missing IDs;5of698 examined
new-record identity candidates have unsupported Flickr landing formats. The
first is HTTPS on the bare `flickr.com` host rejected by the exact `www` rule.
Diagnostic2417B/SHA79b179aa7ec43154d9ac1a18268dd18c60df614d96af7d05d0594167bd8e7bad.

A distinct explicit metadata-only v2 may **count and exclude** unsupported new
record identities, never guess/canonicalize them to recover photos. Historical
unknown identities remain a hard failure; original failed source/receipt stay
byte/mode-pinned before/after. Same180s/96-slot/no-RGB/class/geometry/capacity
contracts and no previous scientific cohort reopened. No usable capacity has
yet been measured, and no A/B study is authorized by this technical diagnosis.

Explicit v2 implementation controls:255 parent tiny tests PASS1.19s, including
56 census controls, strict old-mode behavior, rejection without URL changes,
unchanged96 threshold, original-failure posthash changes closing FAIL and no
per-record flags. This is an implementation check, not the actual census result.

Actual v2 producer8d7b438d0ec2d27868756e20721fb70f15e5d147 **PASS3.369368s**,
exit0/inactive-dead, report8286B/SHA
dbcdd0e99b93805746317d6f626b110f500d2f2fc9162a7aa20ba870e87049ad.
After all336 historical slots/229 known authors and unsupported identities are
excluded:139 geometry-qualified photos,132 independent author/photo/MD5/URL
slots, exceeding the unchanged96 capacity gate. All5 invalid new identities
are rejected without repair;45 positive pairs are unscorable. These are
metadata counts, not accuracy or96 individually licensed/acquired records.
No RGB/network/model/GPU or selection was performed. Independent saved-only
rederivation **PASS3.472462s**, one attempt/no repair, authenticates291Gitfiles,
XZ6/closure, original289-file failed producer, all ten inputs/modes before/after.
Its independent scalar IoU/binding/greedy census reproduces every count and
inventory22230B/SHA62e9e0879b7b06b70136bd90e9880f761ae0b7923ad33e8a15beabec8bc1092f.
Audit2464B/SHAe8c0ed21df1eae91acc89993dc5a32f63d649c84547f4caa25e3caabbefce35d;
later collected unit has unknown exit at audit time, while the root's earlier
terminal observer saw exit0. No new launch follows from unit collection. Freeze
a separate new study, including remaining COCO byte exclusions, before acquisition.

A bounded primary-only fallback audit identifies original OpenImages **TRAIN**
relations356,502,404B, boxes2,258,447,590B and boxable metadata638,407,721B
([publisher V6](https://storage.googleapis.com/openimages/web/download_v6.html)),
HEAD200 only. The current cached source is **TEST**, not validation. TRAIN
would be a distinct source partition, not permission to reopen a failed TEST
cohort; admissible capacity/individual rights remain unknown. Its3.253GB is
**not downloaded or needed now** because v2 clears metadata capacity. OWLv2's
OpenImages pretraining remains known at dataset level/unknown by exact photo;
neither a fresh source partition nor author disjointness makes this an unseen
pretraining benchmark or a Track1 validation score.

Before any future selection, also authenticate original **acquisition ledgers**
for COCO32+64, not just their cohort metadata. They contain original-byte MD5:
`/srv/world-reward-data/coco_proposal_v1/eval_private/manifest.json`,37509B/SHA
6f3346be5a0ccb065084226ad6cdaa2423dd0d460427d43823b9635f57141ae6,
producer4657c8b45f733a1043c5a8af2f5b9d6ac027196a; and
`/srv/world-reward-data/coco_endpoint_v2/eval_private/manifest.json`,72806B/SHA
53de338a0c23409e12afa2a24b39538ff9873d40fc18c2d142b512790bf5ae10,
producer7b557290140dc97c839590c31155fbaf50e442a8. Read acquisition metadata
only, never their RGB/predictions/reference values. Verify every32+64 slot,
schemas/acquire phase/source/cohort/seals; union canonical `original_md5` only
from acquired records, retaining unavailable slots if any. The prior audits
reported96 acquired/zero missing. Applying these byte exclusions can reduce
139/132: require96 again **before** selecting a new cohort; otherwise close.
COCO creator identities still remain UNKNOWN; byte/photographic disjointness
does not prove subject, event, author or unseen-pretraining independence.

### Minimal A/B proposal — conditional, not a frozen experiment

Independent audit proposes96 fresh slots (FIT32/CAL16/RESERVED48) **only if**
the actual feasibility gate qualifies capacity. No cohort exists yet. Learn A's
geometric coefficients on FIT; freeze them, then B learns only a nonnegative
native-HOI-margin coefficient. Same complete routes/banks/supports and no raw
object/person confidence or decayed HOI score in either arm. Assemble the whole
route before route-max and latent side-max; sides are not supervised anatomy.

Exact automatic endpoint-coordinate aliases have one group score (max over
aliases), determined without labels/scores and with all raw slots preserved.
The equal-image positive-set loss is `LSE(all groups)-LSE(observed positives)`
plus fixed L2. Unknown competitors are implicitly suppressed, not real negative
or OFF labels; this tests any-positive retrieval, not all interactions. FIT-only
image/group-balanced scale division without centering avoids introducing an
unrepresented bias on absent HOI routes. Constant/all-missing features have
zero coefficients, not arbitrary floors. Optimizer, initialization, limits,
stationarity check and CAL control must still be frozen before any FIT labels;
latent-max/multiple-positive fitting is nonconvex, with no global-optimum claim.

Proposed RESERVED gate: all48 fixed slots in the mean, positive paired delta,
one-sided exact sign p<=.05 and at least32 nonzero image differences. At32
informative images22 wins have null-tail.02505123; power is.84640537 **under an
assumed**.75 directional-win probability. This arithmetic is not actual power:
ties, access/matching failures and endpoint misses can make the study inadequate.
Uninformative support closes INCONCLUSIVE, never resampling or threshold rescue.
Claim scope remains annotated-holds person+object retrieval only.

One additional implementation gate precedes freezing that learner: the current
raw-linear core requires every active feature to be numerically supported. A
single unavailable hand root can therefore invalidate a score; zeroing only
features globally absent on FIT does not solve partial anatomy. Do not silently
impute it or train weights that change candidate support. An explicit small
availability-aware opt-in may use masked standardized values plus availability
indicators, while retaining raw NaNs and a fixed, arm-shared structural support
rule. An all-missing route is not a meaningful bias-only owner. This design and
its missingness tests are **not implemented or frozen** yet; old raw-linear
behavior and closed scorers remain unchanged.

Hard route/side/alias maxima also make the proposed objective nonsmooth. A tiny
gradient from one arbitrary tie subgradient is not a valid convergence proof.
Optimizer/termination must be defined and falsified on manufactured controls
before FIT references, not rescued with tolerances after a scientific failure.

The independent design audit suggests a minimal opt-in masked value design
`z=m*x/FIT_scale` plus learned numerical-availability indicators, with geometry
anchors fixed independently of coefficients. Raw arrays/NaNs stay untouched;
no anchor means unsupported, while missing HOI leaves the base in both arms.
A shared narrow geometric learner plus B's sole relational coefficient avoids
an extra model stack. A single fixed-budget deterministic subgradient schedule,
ties averaged over distinct branch vectors and best-FIT-loss iterate is a
candidate optimization procedure, **not an adopted recipe or convergence
certificate**. Validate gradients away from ties, directional derivatives at
ties, alpha0 parity, missing observations and full-bank/block/alias invariance
on tiny authored tests before fixing its iteration count/step/L2/support rules.
