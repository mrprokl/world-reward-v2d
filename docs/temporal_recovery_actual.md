# Temporal pose and occlusion recovery — 2026-10-10

## Diagnosis / preserved controls

The comparison renderer replays exported rigid poses; it has no physics solver.
Observed jitter therefore originates upstream, not in simulated friction. The
initializer recomputes depth/partial-ICP poses per frame, so noisy monocular depth,
occlusion and rotation ambiguity can perturb an otherwise stationary object.

Original baseline `de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba`, SAM frontend
`b658881079871508c6b3ec14d001dc1299956996` and full4D numerical outputs
`052ba1554e9a573d566713a99a61d89a5f27681c` remain unchanged. Frozen random
cohort: [9, 1, 14, 7]. All observations are Track 1 RGB/automatic predictions;
no GT, source calibration, hand labels or sequence-matched external data.

## Pose ablation / decisions

Implementation: fixed-shape full-timeline robust SE(3) bundle, persistent material
RGB tracks, optional inferred camera-axis depth in the original shared gauge,
centroid/angular acceleration in seconds. No static constraint, floor snapping,
scale/shape/camera fitting, mesh deletion or per-frame alignment. Depth sampling
authenticates both masks and excludes human pixels before bilinear interpolation.

Parameters were frozen on separate manufactured DEV/RESERVED fixtures, not
challenge quality. These tests establish numerical contracts, not real-data
generalization. Episode 14 has insufficient KLT texture: never invent tracks.

RGB-only producer `09f516d9085f0d64d725b46b0ad6aa52fe7c0d84`, profile v2:
episode 9 RGB mean 8.7317 → 3.4310 px, but frame 155 nearest-human surface proxy
0.00307 → 0.27011 m. **Rejected** despite smoother motion and improved RGB fit.

RGBD producer `9d31ed96edd24ed3bae13df0627b1fef26164325`: same 33 fixed
tracks, 415 original frames, fixed geometry/gauge; 21 evaluations, converged,
30.43 s CPU runtime. RGB mean 8.7317 → 4.6531 px; mean axial depth residual
0.14192 → 0.10785 m. Frame 155 nearest-human proxy becomes 0.01183 m rather
than RGB-only 0.27011 m. This is **not anatomical contact truth**. Isolated
candidate saved; production and baseline exports are not replaced.

Complete motion diagnostics subsequently **reject RGBD for adoption**:
angular acceleration median 27.82 → 7.17 rad/s², but centroid acceleration
median 7.32 → 12.67 m/s², p95 21.39 → 44.84 m/s²; centroid path
4.63 → 6.11 m. These are prediction-only motion proxies, not motion truth.
Improved rotation/reprojection cannot excuse worsened translation jitter.
Inferred per-frame depth is therefore a suspect noise source, not authoritative
evidence. Next research gate: independent external DEV with common-mode,
temporally correlated depth errors/occlusion outliers, then reliability-aware
depth factors; never tune against these challenge outputs or smooth away true z
motion. The ideal 3 mm independent depth noise fixtures were insufficient.

Alternative contact-coupled RGB candidate is implemented separately; inferred
depth is disabled. Automatic native contact logits plus initial anatomical
proximity may activate fixed hand-to-object factors. A single predicted hand
surface witness is selected by one global algorithm before fitting; contact
cannot deactivate itself by moving away. The continuous triangle surface is
unchanged, with conservative spatial broadphase and full fallback for unsafe
bounds. Absent contact leaves the prior RGB/RGBD algorithms unchanged.

Before this alternative's challenge run, declare diagnostic rejection gates:
RGB reprojection must improve; centroid acceleration median/p95 and angular
acceleration median/p95 must not worsen; fixed active anatomical contact mean
must not worsen. Fresh manufactured moving/absent-contact cases must retain
motion. These gates are conservative QA, not a hidden truth score. No coefficient
selection or per-episode manual intervention is permitted from their results.

## Mask recovery / decisions

First reverse policy producer `f0530c76d575dba10fa076bf99b25f571f0726b9`
abstained: object ORB seed features 0 (episode 1) / 3 (episode 7). Requiring
texture to authorize native SAM recovery is unsuitable for thin/untextured
objects; thresholds were not lowered to force acceptance.

Automatic two-view Gemini 3.5 Flash producer
`b97542acb3c11067b5a7d869ac132b881f3c9328`: reference RGB + official task +
one fixed latest-visible RGB, late SAM boxes hidden from the model. Both anchors
corroborated in 12.42 s; identity agreement is **not mask accuracy**. Credentials
removed. No human per-episode prompts or content rerolls.

SAM recovery producer `0826d17bd1a300220a986b961fcf942d0036e2b4`: opt-in
authenticated anchor ledger, one fresh inverse native pass, exact native logits
and masks, original forward/person masks preserved. Only originally empty object
frames may be filled. Semantic-anchor predictions explicitly remain
`reverse_native_semantic_anchor_unverified`, distinct from RGB-qualified recovery.
Native absence is not overridden; unresolved frames stay unknown observations,
not absent 3D objects or invented amodal masks. Latent-pose initialization is
implemented separately and is not a complete downstream reconstruction claim.

Actual complete runtime: 483.42 s native / 488.80 s host. Episode 1: **244/244**
empty masks now have native reverse predictions; episode 7: **85/128**; total
**329/372**, with 43 unresolved frames. Episodes 9/14 have no missing frames and
remain byte-exact. Nonempty recovered predictions do not establish visibility,
identity across every intervening occlusion, mask accuracy or complete 4D.
Visual QA and independent occlusion validation remain required before adoption.

Pinned Azure receipts: SAM host report 2,400 bytes / SHA256
`667227c17b2ce656e6c8c86a4bf7dc22bf7deea0823ac7e01037dee14cd44a64`;
SAM native report 23,843 bytes /
`029ac05ae0e411d4dfc5b21cd32dd73ed2c58b89549011dd011561191aa2cfa7`.
RGBD report 319,383 bytes /
`87f96748a6bb529cc81d5179b03c4fc62ce71a91f591e15cf1e22053a544892a`;
RGBD NPZ 233,604 bytes /
`8340fba63e5393c0a22f6ffa08de6f49d080e4c52692a2bd0661aac7e1d2993f`.

## Sources / scope

[GoTrack (2025)](https://arxiv.org/html/2506.07155v1) motivates persistent
frame-to-frame correspondences for stable, efficient pose tracking. We implement
the principle, not the pretrained GoTrack model.
[CARI4D v3](https://arxiv.org/html/2512.11988v3) describes forward/backward pose
recovery, human-subtracted silhouette comparison and contact-aware optimization.
Our inverse SAM pass is an adaptation, not a claim to reproduce that algorithm.

274 targeted local tests passed. Actual artifacts and rendering remain Azure;
only concise decisions/results are stored here. No held-out accuracy, overall
4D quality, leakage-free checkpoint or leaderboard improvement is claimed.
