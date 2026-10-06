# Background selection: next empirical decision

Independent source-only checkpoint, 6 October 2026; literature cutoff
30 September 2026. No real references, predictions or media were inspected.
**Current evidence: 48 complete endpoint banks / 331 retained people and
48 complete HOI banks / 1398 native hand→object pairs. Full48 all-person pose,
fitted selection and demonstrated background rejection remain unverified.**
The original HOI model completed48 forwards in52.57s; the saved-only audit is
[`vcoco_full_hoi_observations_v1_actual.json`](../results/audits/vcoco_full_hoi_observations_v1_actual.json).
These are complete observations, not1398 correct ownership labels.

## Two shortcuts are verified, not solutions

- [CARI4D at71fa7cb, lines69–80](https://github.com/NVlabs/CARI4D/blob/71fa7cbe46081467edadd11ab534b0c14aa9d913/prep/run_sam3_masks.py#L69-L80)
  explicitly returns `masks.any(axis=0)`: all instances for a text concept become
  one mask. This loses within-concept identity; it does not establish which
  person manipulates which object. This claim concerns that upstream helper,
  not every CARI4D configuration or the separately supplied V2D target prompts.
- [4DAnyone at9cc2aa23, lines195–209](https://github.com/ant-research/4DAnyone/blob/9cc2aa230fe5d364da2c5634ec3dabadadcb0d60/fdanyone/motion/gvhmr.py#L195-L209)
  calls `get_one_track`; an empty track may produce a repeated full-image box.
  Its actual GVHMR gitlink is6ec3ca39. [Tracker64–81](https://github.com/zju3dv/GVHMR/blob/6ec3ca39336c50492c0fae65fba2fb831fc7d866/hmr4d/utils/preproc/tracker.py#L64-L81)
  ranks by **sum of normalized box areas**, then takes the first track. The
  function also computes track length, but that is not the final ranking.
  A large, persistent bystander can win without interaction evidence.

Re-fetched primary source identities, no execution: CARI helper10349B/SHA256
`7d74990996e13df4ac8f6d8e6c53b37d0d9a37db88bb26d0b515ecb93057d8d9`;
4DAnyone wrapper13196B/
`13ef1744bff6b5046ca11d38e686e271b7a9311e18783520ddec07d6d7cdbd3a`;
GVHMR tracker3830B/
`0b32eeaddbdd7425a262266761968c50d8ebe7da4d1b5019a8a84c34f773e6f8`.
Primary Git tree independently confirms the gitlink; 4DAnyone commit date is
23 September2026 (CARI source17 August; HOI source29 June). Our
`actor_selection` has its own top-confident-object then
person-affinity vulnerability; neither source audit validates that alternative.

## Mechanism to test, not another model stack

1. **Person-conditioned anatomy:** run the existing original DWPose callable
   on every retained person; retain133 points, wrists9/10 and roots91/112,
   raw scores, missingness and original IDs. Crop attachment is only a proposed
   person–hand relation: overlapping people can still give the wrong hand.
2. **Complete coherent relations:** combine each side's own anatomy/hand-box
   evidence, its native HOI-DETR hand→direct-object margin and the same
   direct→generic-object bridge **before** marginalization. Keep all people,
   both sides,3600 OWL patches and all native K; no independent distance/logit
   extrema, nearest-person label or physical-identity inference from aliases.
   Existing packed marginal primitives already implement duplicate-neutral
   latent geometry/margin priors. Empty HOI leaves a base proposal, not OFF.
3. **Temporal memory only afterwards:** preserve the proposed pair and separate
   competing instance tracks across occlusions. Appearance persistence cannot
   repair an incorrectly initialized owner. Missing/ambiguous evidence is
   UNKNOWN; OFF/contact/release require their own supervision.

Relevant primary mechanisms already audited: [SingleQuery-BHOI](https://arxiv.org/html/2609.12155v1)
(person-centric bimanual relations), [HOI-DETR](https://arxiv.org/html/2606.17384v1)
(learned hand/object links), [DAM4SAM](https://arxiv.org/html/2411.17576v2)
(distractor-aware memory after initialization). SingleQuery's [project page](https://lgecto-ail-vil.github.io/SingleQuery-BHOI/)
still says code coming soon; no ready checkpoint is asserted. Its explicit
annotation license is CC-BY-NC4, not a blanket ban on research competitions but
an unresolved use-scope question. [SAM3.1's March27 release](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/RELEASE_SAM3p1.md)
provides multiplexed multi-instance memory, not ownership. These are design
support, not measurements of our method; no new dependency is needed now.
Paper content relies on the existing primary audits: this checkpoint re-fetched
tiny code/project/license/release texts, not paper bodies or model/data assets.

## Minimum useful experiment: complete the existing fresh48

**Before FIT references:** freeze one recipe from
`vcoco_selector_learning_design.md`, the all48 endpoint/pose/HOI banks, complete
raw IDs/support, image-wise memory/cost bound and evaluator. No new census,
transport family or validation-data shopping is justified first. Invalid raw
boxes/missing pose must fail or remain explicitly unavailable under a
pre-frozen source-safe interface, never be clipped, pruned or repaired after QA.

**FIT32 only:** A learns17 geometry/availability coefficients; B freezes A and
learns only nonnegative alpha multiplying its own native relation margin.
Alpha0 must be the identical A path. Use the proposed bounded FIT-only CV,
regularization and convergence controls, not the old512-step failed-cost recipe.
Pre-register the existing informativeness proposals (six jointly covered,
competing images per fold;24 FIT images with useful B derivative). Failure is
INCONCLUSIVE on the fixed population, not permission to replace images. These
counts do not prove parameter identifiability. Incomplete-positive MIL makes
unknown alternatives compete; it does **not** label them verified negatives.

**Once-only CAL16:** preserve the planned macro pairRecall@3 and diagnostic
any-positiveRecall@1, using exact automatic alias groups and unique continuous
IoU>=.5 endpoint matching against *all* same-image instances, including crowd
ambiguity. Retain all positive pairs, all16 denominators, detector misses and
unsupported outcomes. Before labels, freeze an advancement gate such as strict
positive paired B−A primary gain with no Recall@1 regression; this is a proposal,
not an adopted/post-result threshold. Zero/negative gain falsifies this narrow
benefit; insufficient coverage/support cannot become conditional-only success.
Report the endpoint ceiling separately. A geometry win over largest/nearest
baselines is not evidence that B's relation logits help.

CAL is development evidence, not final victory. Freeze learned artifacts before
any reserved evaluation; the unopened pilot8 RESERVED can be a small independent
check, not an adequately powered crowd/temporal claim. A later fresh licensed or
consented multiperson video cohort must test wrong-pair persistence, crossing,
occlusion and a larger bystander on full timelines; static roles cannot establish
hand ownership, task choice, OFF or shared-frame3D reconstruction.

V-COCO repository-wide MIT is accepted here only as the documented private
research interpretation; COCO publisher CC-BY2 image attribution is separate,
creator identity remains unknown. DWPose/HOI/OWL COCO-family or web pretraining
means photo-/challenge-overlap is not certified absent. 4DAnyone/CARI4D and
tracking dependencies have separate research/noncommercial/custom terms.
Neither photo-disjointness, source PASS nor a proxy gain is a CARI4D/leaderboard
win or a competition redistribution clearance.
