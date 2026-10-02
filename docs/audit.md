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
   `Nvidia-V2D-Challenge`. Account/rules acceptance requires participant interaction.

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
verification. Register separately once for the whole challenge. No registration
or rule acceptance was performed by the agent.

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
