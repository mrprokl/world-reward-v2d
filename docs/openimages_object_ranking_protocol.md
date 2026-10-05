# Fixed external object-attribution test — October 5

**Prospective before fresh RGB inference/reference geometry.** This evaluates
person-region conditional held-object retrieval, not hand ownership, contact,
temporal identity, 3D reconstruction or superiority to CARI4D. Pretraining overlap
with this external corpus is unknown; do not call it leakage-free performance.

## Fixed cohort and permissible input

Open Images official test manual person-`holds`-nonperson image IDs were ordered
by SHA256(`world_reward.oi_holds_runtime_v1/` + ImageID). The original first16
metadata-only cohort is frozen. Exclude runtime/visual QA image2333ac90234d7d50;
retain all remaining15 slots without replacements. Census10130B/SHA
b4333ea57bb408b0eafdbc5eee51d2b3ee3e6e963ef9ec269178bbd45480ff4a.

Verify each creator's exact Flickr ImageObject acquireLicensePage, CC-BY-2.0
license and author against publisher metadata before RGB acquisition. Require
exact original URL, bounded original byte count and publisher base64 MD5. Do not
use dynamic thumbnail URLs, mirrors or redirected alternate images. Unknown
publisher orientation is unscorable, not guessed from images/reference boxes.
No challenge records, matched FORM-HOI, source calibration or depth are inputs.

Actual CPU acquisition: eight originals15,110,779B stay Azure; five unavailable
grant/download slots and two unknown-orientation slots remain missing. Manifest
15090B/SHA e5856091d2a5b150baaa4ac2da2e8f39734d820510233c07a0536426ab10a0b4.
These are access/rights exclusions, **not prediction-dependent selection**. The
15-slot retrieval denominator remains fixed. No fresh image was inspected by a
human or passed through a model before this protocol.

## Freeze automatic observations before reference evaluation

Use the already qualified full original HOI-DETR epoch5, original non-EMA
forward fields, native EMA registration and strict1796-key weights_only load.
Same original resize/test preprocessing, all1500 query tokens/logits, native
top1000 soft-NMS(.5/min.3), raw-score retention.3, every surviving role0→role1
pair and both original logits. No model/source retuning, demo .6/.92 thresholds,
per-image prompts, selecting a lucky hand or required frame-zero pair. Load once,
run all eight exact originals and preserve legitimate empty outputs. Any runtime
error fails the whole run; never convert errors to empty predictions. Seven
missing slots stay explicit. Seal outputs/source/config/checkpoint pins before
opening reference geometry. GPU mounts contain exact JPEG files, not metadata,
relations, boxes or licenses. This pilot is a one-frame external observation test,
not a fabricated video trajectory.

## Two fixed shared-bank ranking arms

Eligible hands and objects are identical in both arms. No prediction-side
reference matching or filtering. Keep all role1 slots, including query aliases.

- **A, strong spatial baseline:** for a hand-box centre and object box, negative
  Euclidean distance from centre to closed box, then negative centre-to-centre
  distance, both divided by original image diagonal. Lexicographically rank the
  best *whole two-distance tuple* across eligible hands; no weights/confidence
  factors/independent minima. This avoids weakening A on large/elongated objects.
- **B, learned relation:** maximum original FP64 difference `z1-z0` across exactly
  the same hands. Original source labels1=interact and0=no-interact; not a contact
  probability. No temperature, normalization against other objects or cutoff.
- No eligible hand: unsupported/NaN, not a fake zero score. Exact score ties
  remain ties. No arbitrary query-ID winning vote.

## Independent evaluator-only region proxy

Use official test boxes and manual relations, never predictor mounts. The
Open Images exhaustive triplet catalogue is
https://storage.googleapis.com/openimages/v6/oidv6-relationship-triplets.csv
37869B/SHA c53e939aef2b72512026ab2d944b9356481c08e647a4086f00b17237d558d38e.
Only exact catalogue person-class/object-class/holds triplets support absent-pair
negatives; unannotated arbitrary categories are **unknown**, not false positives.
No parent-class aliases or extrapolation to contact/tool classes.

Fixed geometry conventions: original normalized reference boxes times original
image dimensions; no clipping/half-pixel/rotation fitting. Person proxy assigns a
native hand centre only if contained in exactly one nongroup annotated person
box; none/multiple is unknown, never true hand ownership. Native object box maps
only if IoU>=.5 with exactly one nongroup annotated object instance. Multiple
matches/unboxed/off-vocabulary/group instances remain unknown. Aggregate aliases
by best A tuple/max B margin, counting each reference object only once. Same
association convention for A and B, independent of their scores.

Primary: positive-versus-reliable-negative pairwise concordance within each
positive person-region, half-credit exact ties. Equal-weight eligible persons
within an image, then images. Positives are *all* annotated held objects, not a
selected target. Report conditional coverage/denominators separately; no negatives
or missed positives must be hidden by reference matching.

Safety retrieval: top-choice positive-hit on the **full automatic candidate bank**
for every annotated positive person-region, with fractional credit across exact
ties; unknown winning candidates and no-hand/no-object/missing-image slots count
as misses, not proven wrong/contact negatives. Equal-weight persons within image
then all15 image slots. Report missing/reference-unscorable slots separately.

## Predeclared decision and stopping

Only consider B promising for this restricted attribution component if:
1. At least six different images have positive-versus-negative comparisons.
2. Mean image-paired B−A concordance is strictly positive and the exact one-sided
   image-level sign-flip randomization test gives p<=.05 (enumerate all2^n signs;
   zero differences retained; no resampling/tuning).
3. Full-bank all-slot retrieval does not regress (B>=A, no tolerance relaxation).

Insufficient informative images/annotation scope is **INCONCLUSIVE**, never PASS
or justification to replace images. A quality-gate regression closes this cohort;
do not retune thresholds/rankings and reevaluate it. Record useful aggregate
results and decisions only. No challenge adoption follows from this pilot alone.

Primary semantic sources: Open Images factsfigures_v7.html#visual_relationships
(exhaustive *eligible* triplets), download_v7.html (metadata rotation and CSV),
HOI-DETR@1b367292f3833afd64a204bd4d9d84519541d035 interaction_head.py and
co_dino_head_w_interaction.py (native class1 label). Independent audit established
these constraints before inference; negative rows are implicit and require all
annotated box instances, not just objects appearing in the relation CSV.
