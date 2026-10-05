# Fresh 128-image external attribution study — preregistered October 5

**Frozen before acquisition/inference/private geometry on this new cohort.**
The earlier15-slot study is permanently INCONCLUSIVE. It is not reevaluated or
expanded. This is a new disjoint sample, not a replacement for failed old images.
Rank definitions/model/hyperparameters are unchanged. No claim of independent
pretraining holdout, contact, true hand ownership, 3D quality or CARI4D victory.

## Cohort and integrity

Take ranks16 through143 inclusive (Python slice16:144) from the same complete
official test person-holds-nonperson ID universe sorted by
SHA256(`world_reward.oi_holds_runtime_v1/`+ImageID). All128 slots are frozen before
RGB/model/reference geometry. Their metadata census is76839B/SHA
bf576b143bf8304981eb8dddfa64726945262d79c24cff07601f14e20a4fd452.
Old16 IDs (one visual runtime QA plus15 closed pilot slots) are disjoint.

Keep creator rights, original-URL/size/base64-MD5 identity, unknown orientation
and source isolation rules from the earlier protocol. Maximum original file16MiB,
JPEG decoded area16Mi pixels; no alternate thumbnail/mirror/redirect or manual
orientation. Four CPU acquisition workers, timeout30s per exact request; missing
grants/downloads/orientation stay explicit in128-slot denominator without retry,
replacement or relaxing creator match. Heavy bytes stay Azure. No images are
inspected before predictions/reference metrics freeze.

Reuse original full HOI-DETR epoch5, strict1796-key weights_only registration,
non-EMA forward, native preprocessing1500-query/top1000/soft-NMS/retention as
qualified. Load once, all acquired image forwards, preserve semantic empties.
1800s native lifecycle plus60s cleanup. Original helper/math policy unchanged.
Every native observation file is sealed before reference geometry is decoded.
Predictor receives exact acquired JPEG leaves only, never reference metadata.

## Fixed methods and reference scope

A=lexicographically best negative hand-centre→closed-object-box distance then
centre→centre distance / original image diagonal. B=max original `z1-z0` margin
across exactly the same hand anchors. No additional confidence/weights/thresholds.
Alias slots retained then evaluator aggregates known reference instance once.

Use the earlier person-region proxy and unique IoU>=.5 object localization, same
geometry in both arms. Any containing group-person box censors a hand mapping;
any group-object match censors object mapping. No parent-class rewrite:
Person/Man/Woman/Boy/Girl are distinct annotation classes, not synonyms. All
annotations/relation endpoints/negatives remain evaluator-only. Manual positive
relation endpoints must bind to a unique same-class nongroup annotated box under
the frozen IoU>=.5 convention; multiple/no matches remain reference-unscorable,
not silently relabeled, added as a new box or snapped by hand. Retain their
positive-person slots as retrieval misses and report reference binding losses.

Absent-pair negatives ONLY for exact personclass/objectclass/holds triplets in
the official exhaustive vocabulary; unknown/off-vocabulary/unboxed/group slots
are unknown competitors, not false negatives/contact labels. Labels cannot
select a predictor target, exclude candidates or set its hand owner. Unique
hand-centre-in-person containment is merely an evaluator region proxy.

Primary conditional concordance: matched positive-vs-reliable-negative object
instance pairs within a positive person-region, half-credit ties; equal persons
within image then images. Preserve localization/hand/annotation-unscorable
denominators separately. Safety retrieval uses full automatic candidate bank,
known-instance aliases count once, unknown winners/ties contribute misses;
every annotated positive-person/image slot stays in denominator, including
no-hands/no-object/access/reference-unscorable. Equalpersons per image, all128
images (missing images score0 for both arms). No sampled best held object.

## Decision — prospectively scalable inference

The old pilot's exact all2^n sign-flip enumeration is infeasible for128 images.
**Before new observations**, replace only the statistical test for this new
study with the exact one-sided *paired binomial sign test*: among nonzero
image-concordance deltas, positives vs negatives; p=sum_{k=wins..n} C(n,k)/2^n.
Exact ties remain in the image mean/denominators but carry no directional vote.
This is not the old sign-flip test and does not estimate effect magnitude.

All criteria required to call B promising for this restricted component:
1. At least six distinct informative images (matched pos-vs-neg comparisons).
2. Strictly positive mean paired image delta (including zero deltas).
3. Exact one-sided sign-test p<=.05; no test switching after observations.
4. Full-bank all128-image retrieval B>=A, without tolerance relaxation.

Insufficient reference/informative scope => INCONCLUSIVE. Any safety regression
or failed statistical superiority => REJECT for adoption on this study; no
parameter tuning/retries/resampling. One frozen evaluation only. Report useful
aggregates/decision/pins; no test-instance labels/renders local. Even a PASS
requires separate continuous interaction/actor/3D validation before submission.
