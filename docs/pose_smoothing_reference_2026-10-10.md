# Fast pose-smoothing ablation — preregistration

## Diagnosis and selected test

The latest native joint extension freezes object rotation. It cannot remove
angular noise already present in its initializer. The viewer replays poses;
there is no rigid-body simulator whose friction coefficient could fix this.
Temporal regularization already exists in our sequence fitter, but smoothing
and accurate interaction reconstruction are different claims.

Test a fixed, cheap cached-pose reference before another model/initializer run:
local cubic polynomial regression of the fixed-mesh centroid and local SO(3)
increments, nine samples at 30 Hz (declared 0.3-second sample-count window).
Use unilateral full-window endpoint fits, preserve time zero and every original
frame. No parameter sweep or challenge-based coefficient selection. This is an
authored local-chart reference, **not** an exact implementation of the cited
geometric Savitzky–Golay solver or an established SOTA reconstruction system.

Two predeclared scenarios share this one proposal:

1. **Object only:** change object translation and rotation, leave the human
   untouched. This directly tests object jitter but can break contacts.
2. **Common rigid correction:** apply a proper rigid correction identically to
   the full human, joints and object. This preserves relative geometry up to
   numerical error, but can move the person incorrectly or violate the fixed
   virtual floor. It tests common-mode jitter, not arbitrary interaction noise.

Freeze both predictions before candidate QA. Use cached full episodes 9 and 14
(415 and 442 frames) from the original random cohort [9, 1, 14, 7]. Preserve the
other two records; their scale-initialization recovery is a distinct scheduled
experiment, not substituted success. Baseline and source geometry stay sealed.

## Evaluation and execution contract

- Azure-only CPU, no models, GPU lease, video decoding or GT. Inclusive runtime
  cap: 300 seconds, four CPU cores; source and saved-input identities verified.
- Before challenge reads, execute a moving, noisy authored geometric DEV sanity
  test. This tests numerical behavior only, not external HOI generalization.
- Reproduce original A/B automatic QA. Compare the same image tracks, reserved
  automatic human keypoints and frozen anatomical IDs against complete original
  object triangles. Never choose a new witness to hide a broken contact.
- Report centroid/angular acceleration **proxies**, image motion increments,
  contact means/tails/per-witness regressions, motion extent and fixed virtual
  floor deviations. Virtual floor is inferred, not measured physical truth.
- No official penetration or silhouette improvement claimed by this small
  ablation. No native-control export or automatic production adoption. External
  CD/ACC/PEN validation remains a separate sealed-prediction protocol.

## Primary literature audited

- [1€ Filter, CHI 2012](https://gery.casiez.net/1euro/): inexpensive adaptive causal
  low-pass baseline; jitter/lag tradeoff and unit-dependent tuning make an
  offline centered reference preferable for this first diagnostic.
- [Geometric SG, IROS 2022](https://doi.org/10.1109/IROS47612.2022.9981409),
  [author code](https://github.com/MaartenJongeneel/Paper-Savitzky-Golay-filtering-on-SO3):
  rotation-aware polynomial filtering rather than Euler/component averaging.
  The author implementation's endpoint truncation is not compatible with our
  required complete trajectory; no author-code checkpoint is imported.
- [SmoothNet, ECCV 2022](https://arxiv.org/abs/2112.13715): learned human-pose
  temporal refinement, not a guaranteed rigid-object/contact correction.
- [CARI4D, April 2026 revision](https://arxiv.org/html/2512.11988v3): joint contact,
  image, penetration and acceleration losses. Its acceleration regularizer is
  not acceleration error against ground truth; a smoother output can be wrong.
- [MOCHI, June 2026](https://arxiv.org/abs/2606.18243): noisy interaction refinement
  through grasp optimization and contact-aware diffusion. Author code remains
  announced rather than released; this is not a runnable fast drop-in fix.

Status at preregistration: primitive implemented and tiny numerical tests pass;
real cached-pose ablation has not yet completed. No reconstruction gain claimed.
