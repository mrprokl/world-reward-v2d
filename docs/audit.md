# Track 1 audit — 2026-10-02

## Authoritative inputs and constraints

1. [Challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/): monocular
   RGB → full 4D human/body/hands, object shape/pose, consistent metric frame.
2. [Official FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf),
   SHA-256 `a1e569e1c83eeac2584de34e9503f1226e20ab369d53d2c67c2e243deb1a4d7a`:
   **Track 2 Tier 1 assets cannot provide Track 1 meshes or camera parameters**.
   External pretrained models/data are allowed. Continuous trajectories during
   occlusion are required. Winning code release requested under Apache-2.0.
3. [Pinned dataset card](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/5f68335f3acc802033d1e80728c1633197521de8/track_1/README.md):
   30 episodes, 16,563 frames, 30 fps, 1536×1152, static RGB camera per episode,
   10 objects × 3 episodes, 22 tasks. No pose/mesh/intrinsics/visibility labels.
   LeRobot calls this `train`, but it is **not labeled development ground truth**.
   Source sequence IDs/camera labels are metadata, not permission to retrieve MV.
4. [Submission kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip),
   SHA-256 pinned in config. Use its unmodified packer, exact sample row IDs and
   code commit URL. One Parquet for five Kaggle competitions; code shared with
   `Nvidia-V2D-Challenge`. Rules acceptance is verified below; final code access still required.

## Evaluation contract inspected in code

- **One** human-correspondence Sim(3) at the **first scored** frame, applied globally
  to human/object across the whole episode. First scored frame often is not frame 0.
  Neither per-frame ICP nor independent object alignment is permitted.
- MHR `pose[T,136]`, shared `scales[68]`, `shape[45]`; body model output converted
  from native centimeters by `diag(1,-1,-1)/100`.
- Object mesh fixed over clip; `p_scene = object_scale * R * p_mesh + t`, `R∈SO(3)`.
- Official sample: 740,780 rows; 9,877 scored frames across 30 episodes. Scored
  frames are nonuniform contact spans, with gaps; never infer predictions from
  the provided sample values or alter the reconstructed video trajectory by span.
- Each episode has 4,096 mesh vertices and 4,096 faces after official budgeter.
- CD-H/CD-O symmetric surface Chamfer, centimeters. ACC-H: 22 joints; ACC-O:
  object translation; second differences **without fps²**, only within contiguous
  scored stretches. They measure error against reference, not zero acceleration.
- PEN: submitted hand points against submitted mesh, mean over 512 points with
  outside points zero, scaled by alignment. This is **not** a contact-accuracy
  target and is **not** license to shrink/delete objects or detach hands.
- All five lower-is-better; challenge describes accuracy/plausibility axes equally
  weighted. Exact aggregate award formula/ranking tie-breaks still require audit.
- CARI4D baseline snapshot: CD-H 15.57527; CD-O 23.16246; ACC-H 1.97387;
  ACC-O 0.72622; PEN 0.00423. These are challenge scores, not paper Table 3.

## Submission calendar and prerequisites

- Current official page says leaderboard live (supersedes older email about
  opening after one week). Freeze: **2026-11-04 17:00 EST = 23:00 Europe/Paris**.
- Maximum five submissions per track/tier per week; unlimited final three days.
  Count one frozen five-metric file as one logical submission, but inspect actual
  competition quotas too. Failed scoring costs quota; unknown upload state must
  be inspected before retry. Final-three-days exact boundary/week reset need rules.
- Oct 7 webinar time TBD; Nov 12 announcement. Registration and Kaggle rule
  text/acceptance must be checked before uploading; no acceptance bypass.
- Technical report + agreed code release required for awards. Upstream code and
  model weights retain their own licenses; do not redistribute gated SAM weights.

## Deployment/research implications

- CARI4D public MHR wrapper is **not RGB-only**: mesh input is required and its
  metric scale is preserved. A provided scan would violate Track 1. Reconstruct
  own geometry from released RGB, estimate scale relative to human/depth, and
  test generated-vs-procedural/multiframe candidates using image evidence.
- Native CARI4D/SAM human controls are not the kit's 136+68 submission controls.
  Convert/fit with clip-shared identity and Apache-licensed MHR model, validate
  roundtrip in meters; cannot rely on matching dimensions alone.
- Upstream GT oracle modes exist. Explicitly prohibit/disable them, including
  `--no-first-usable-frame-gt-rotation-oracle` when alternate FoundationPose
  registration is used; no GT depth alignment or source calibration.
- Heavy runtime/downloads only on Azure; do not clone heavy repos or datasets on
  the local tether. Local official kit audit was small; disposable archive/data
  samples will be removed after the constraints/hashes are retained.

## Kaggle full rules audit (read-only browser, 2026-10-02)

[Rules](https://www.kaggle.com/competitions/v2d-challenge-track1-cd-h/rules):
max team size 10; 5/week controls despite platform 5/day cap; up to **two** final
submissions; identical file across metrics; all scored rows public leaderboard,
confidential references remain hidden and later broader evaluation is off-platform.
Single-command reproducibility, actual producing commit and weights required on
verification. Register separately once for the whole challenge. See the later
entry receipt below; the separate NVIDIA registration is not yet verified.

**Foundational 4.b prohibits hand labeling/human prediction of validation/test
records.** Therefore official example's SAM2 manual prompt GUI is **not adopted**:
all masks/keypoints/geometries must be generated automatically, with global
algorithmic fixes only. Manually reading test frames is not a labeling pipeline.

Foundational 6.c requires OSI-approved source licenses permitting commercial use;
specific 2.6 allows pretrained data/model licenses not to be rereleased Apache.
Use Apache/MIT/BSD code where possible; do not copy original NC CARI4D, HaWoR,
WiLoR or MANO code into the submission. SAM inference source/model terms need
specific compatibility review; access approval alone does not resolve license.
If source license exceptions remain ambiguous, ask organizers before final use.
Foundational rules declare precedence over competition-specific text, and public
code sharing must also benefit participants through Kaggle forums/notebooks.

## Kaggle entry receipt — 2026-10-02

User explicitly authorized acceptance and signed in using their existing account.
The authenticated browser confirmed **“You have accepted the rules for this
competition. Good luck!”**, plus Submit Prediction, on each Track1 rules page:
CD-H, CD-O, ACC-H, ACC-O and PEN. Each acceptance was read again after page load
(or reload); no prediction/file upload and no quota consumption. No account
creation, token disclosure, other-track entry or license exception.

User confirmed the final public-name change. **World Reward** was saved and
verified after reload on ALL FIVE Track1 Team pages; no profile name, invitation
or team member was changed. The authenticated CLI read-only check independently
confirms entered=True, used_today=0 and remaining_today=5 on each metric. NVIDIA's separate
once-per-team registration, producing Git commit accessibility and source-license
compatibility remain independent prerequisites; Kaggle acceptance does not prove
any of them.

## Public code access — 2026-10-03

Own-source hygiene audit found no tracked/history model, media, challenge GT or
credential material; this is practical inspection, not a complete originality/
SBOM/legal guarantee. Created public own-code repository
[mrprokl/world-reward-v2d](https://github.com/mrprokl/world-reward-v2d) and pushed
the existing producer history without squash/rebase/amend. `main` initially
points toce0519fb99125ee0b847cd89c9f37ea6a684738a; subsequent source changes must
also be pushed before claiming their commit is accessible. No gated assets,
third-party source, private synthesis truth or runtime credentials published.
Unauthenticated HTTPS exact commit page200, full producing hash present; remote
`refs/heads/main` equalsce0519f. No authorization/cookie was sent for this check.
Final producing commit, rules-required public sharing and registration/licensing/
reproducibility/quota checks still precede any upload.

The public [once-per-team registration form](https://docs.google.com/forms/d/e/1FAIpQLSdZJYNsEPPGDeIRH2yb_Dui-lWcIxWRF2CON7UOIijzCw8zyA/viewform)
was inspected read-only: required primary contact email, skill set and approach;
member name/email/affiliation may also be listed. No form submitted, signed-in
email inferred, affiliation/experience invented or track2/3 selected. Thomas
Gomez is authorized as the personal name; contact email and truthful affiliation/
skill-set details remain to confirm if registration is not already completed.

## Target designation re-audit — 2026-10-07

Re-read the current official challenge page, FAQ and Track1 card/metadata, with
an independent primary-source audit. No media/model/GT acquisition, GPU job,
reference evaluation or change to a frozen study follows from this review.

Independent exact Track1 text pins at the same HF revision:
README2814B/`9a0a7cf3faff0c413ef72248f87c4965f46ad75abfd67852870cfe33e187fdd6`,
episode metadata7597B/`cce105292e2b670f32b49e0504b0ece67b8aa860ea69dc9e728fce97531b53fe`,
tasks4493B/`1d12b398b55ee9587265fb11f674214f233fd955aeb6acefff09f4c889bd9f23`.
Root re-read challenge/FAQ through web text; no new raw-byte hash is asserted
for those pages or the FAQ.

**Track1 is target-conditioned, not unrestricted interaction discovery.** The
[official Track1 card](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/5f68335f3acc802033d1e80728c1633197521de8/track_1/README.md)
explicitly supplies `object`, `object_prompt` and the original action description
(resolved from episode/task metadata). These identify what object to track;
they do not supply its pixel location or designate a human through a person ID.
The six-key episode metadata has no bbox, mask, point prompt or human identity.
Source sequence/camera identifiers still do not authorize source calibration,
other camera views or sequence-matched FORM-HOI acquisition.

The required output is one reconstructed human and the target object's fixed
geometry and continuous poses per episode, in a shared metric frame. It is not
an exhaustive reconstruction of every background person/object. Accuracy and
physical-plausibility metrics score the final reconstruction, not a separate
detection leaderboard. The official kit remains authoritative for serialization;
the FAQ requires MHR and continuous object trajectories through occlusion.

**Association is necessary; a separate pre-detection module is not mandated.**
An implementation may automatically ground the supplied description, infer the
associated actor over time, then reconstruct, or infer association and geometry
jointly. Retaining competing proposals is our engineering strategy, not an
official obligation to reconstruct all of them. Semantic designation must not
be confused with guaranteed visual-instance disambiguation. If several actors
and instances equally satisfy the description/action, inspected sources provide
no explicit actor-ID tie-break: ask organizers, never manually assign test IDs.

The [public task descriptions](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/5f68335f3acc802033d1e80728c1633197521de8/track_1/meta/tasks.jsonl)
include pushing a desk with a foot and sitting/rotating on a stool. A hand-only
interaction selector therefore cannot represent the complete Track1 objective.
The action text is conditioning evidence, not a prescribed pose/contact timeline:
infer the actual motion from RGB rather than synthesizing what the text says.

The old `infra/automatic_masks.py` already consumes the official `object_prompt`
with the fixed `person.` query; it does **not** consume the action description.
Its top-object/person-affinity vulnerability remains rejected. The prospective
generic V-COCO A/B pilot is a limited relation-observation diagnostic, not the
whole challenge target-selection requirement or an adoption gate for foot/body
interactions. Do not rewrite its frozen population, metrics or results.

The [V2D CARI4D input workflow](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_cari4d/README.md)
requires already prepared human/object masks and a mesh; its SAM2 GUI creates
geometric prompts. Those are baseline implementation inputs, not distributed
Track1 annotations or a waiver of the existing no-hand-labeling contract.

Decision: scope the next architecture to automatic **video + supplied object
description + action → target instance/associated actor → full-T 4D HOI**.
Use a shared task-conditioned mechanism covering hands, feet/body and release,
not hand-authored per-episode prompts, nearest-hand rules or an unconditioned
universal selector.
Simple automatic grounding/tracking is a valid hypothesis to compare; neither
this review nor a static role-pair gain establishes its quality/generalization.

## Evaluation-first and visual QA checkpoints — 2026-10-07

User requests concise progress and lightweight visual involvement at meaningful
decisions. Synthetic benchmarking remains deferred. Do not start another broad
FIT/component campaign just to accumulate technical PASS counts.

Next useful comparison: frozen existing baseline versus one automatic,
object/action-conditioned temporal association hypothesis, with identical
downstream reconstruction and full original timelines. Before execution, declare
the external development/held-out scope, exact configurations, runtime budget,
metrics and abandonment rule; existing closed studies stay unchanged. Static
V-COCO retrieval cannot replace temporal/full-4D reference evaluation.

Show source RGB and automatic masks/instance IDs before expensive reconstruction;
then paired baseline/candidate mesh overlays and a short timeline excerpt after
the first complete comparison. Include fixed time-spaced views plus automatically
flagged failure/uncertainty views; never show only favourable frames. Render on
Azure; return only authorized low-resolution previews and concise scalars.

Human review identifies failure classes and whether evidence warrants the next
experiment. On challenge videos it is QA only, not target-ID/box/point/contact
annotation, episode-specific settings or winner selection based on human test
predictions. Algorithm changes are global and must be checked on lawful external
development data; once-opened held-out data cannot be relabelled as fresh test.

Report three distinct evidence levels: reference-backed external quality,
unlabelled-video consistency diagnostics, and human visual QA. Reprojection,
silhouette fit, apparent smoothness and physical residuals alone cannot certify
the correct target or 3D truth. Hidden official metric gains remain unknown
until scoring; do not turn a visual approval into a CARI4D superiority claim.
