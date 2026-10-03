# Track 1 literature audit — cutoff 2026-09-30

Primary papers, project pages, GitHub source/README and Hugging Face metadata were inspected on 2026-10-02. No Track2/3 assets, FORM-HOI GT, or challenge multiview data were accessed. Paper benchmark numbers are not challenge numbers. This is a targeted, not exhaustive, SOTA survey.

## Priority order

1. Reproduce frozen CARI4D baseline. It is a full-body category-agnostic metric 4D reference, not just a per-image PA-aligned method. It explicitly lacks detailed finger articulation and cannot fix major FoundationPose flips; first-frame object visibility is assumed.
2. Body branch: compare GEM-X/SOMA temporal output and SAM3D Body/MHR framewise output against original NLF on permitted non-overlapping external validation. Native SOMA reduces challenge export impedance. Couple a fixed sequence identity and image reprojected evidence; do not replace metric/global evaluation by PA-only.
3. Object branch: multi-keyframe/seed shape candidates, SAM3D vs Hunyuan, score on withheld *same monocular video* frames for visible silhouette, photometry and robust inferred depth. Shared shape+scale. Do-as-I-Do guided SAM3D shape-fixed tracking is a deployable alternate to FP initialization failures.
4. Symmetry-aware multi-hypothesis sequence optimization: include 180° hypotheses explicitly, scoring visible evidence + short-term continuity + hand contact. Select global sequence paths rather than greedy top1. Do not average symmetry-equivalent rotations or smooth through genuine fast motion. AgentSTAR is a useful difficult-case procedural shape/pose teacher.
5. Fine hands: first activate the already available SAM3D Body hand decoder;
   fit its articulation proposals under fixed full-clip identity. Do not add
   HaMeR/WiLoR/Dyn-HaMR/MANO source with unresolved NC/redistribution eligibility.
   C2Dex supplies canonical-contact ideas; its implementation is not released.
6. Metric depth: MoGe2 for stable global scale, compare MoGe3 detail as alternate; its fine detail gains do not guarantee better metric calibration. Cross-frame shared scale and focal consensus, calibrating from RGB/allowed priors only.
7. Physics RL refinement RePHO only after above, not default: original results trade worse 3D accuracy for physics and score successful rollout frames only. All challenge frames must remain reconstructed.

## Primary references and evidence

### CARI4D
- Paper https://arxiv.org/abs/2512.11988 (v1 2025-12-12; camera-ready v3 2026-04-19), https://arxiv.org/html/2512.11988v3
- Code https://github.com/NVlabs/CARI4D ; cutoff HEAD 71fa7cbe46081467edadd11ab534b0c14aa9d913 (2026-08-17)
- README announces inference release 2026-02-28 and custom-video/training 2026-04-04.
- CoCoNet weights https://huggingface.co/nvidia/CARI4D step031397.pth ; model revision 75e6f3d47123f7808afedcf83720911fada8bf27, nongated, lastModified 2026-04-05.
- Hunyuan3D first-frame mesh + UniDepthV2 metric depth + coarse-to-fine FP scale search + RGB/RGBD pose hypothesis IoU filtering, forward/backward jumps + NLF human depth alignment + render/compare CoCoNet temporal attention + joint contact/2D joints/mask/penetration/acceleration optimization.
- v3 BEHAVE metrics: CD-human 7.74cm, CD-object 12.05cm, CD-combined 9.23cm, Acc-human 1.14, Acc-object .35. First-frame human alignment applied to full clip. InterCap full-video combined CD 12.88 vs 20.17 VisTracker. Not directly comparable with PA methods.
- 96-frame network window; reported 45min/300 frames A100-80GB; training 38h/8 A100-80GB. Fingers are not explicitly regressed. CoCoNet cannot repair large 180° initialization flips.
- Camera-ready HTML says 35% BEHAVE improvement; website/abstract metadata say 38%. Cite version and exact table, not stale marketing number.

### SAM3D Body / Objects
- Body paper https://arxiv.org/abs/2602.15989 (2026-02-17); code+checkpoint announcement 2025-11-19.
- https://github.com/facebookresearch/sam-3d-body HEAD b5c765a0d89d789985e186d396315e7590887b94 (2026-02-19).
- Body/feet/hands, promptable mask/keypoints, MHR representation; framewise model, no temporal/world claim by itself.
- https://huggingface.co/facebook/sam-3d-body-dinov3 manually gated, revision 11aaa346c7204874a1cbafe3d39a979080b2c55a, model.ckpt and assets/mhr_model.pt; license/access acceptance required.
- Objects paper https://arxiv.org/abs/2511.16624 v1 2025-11-20 / v2 2026-06-02; release announcement 2025-11-19.
- https://github.com/facebookresearch/sam-3d-objects HEAD f91db411c50efee93d8db7aeb323885650f6f722 (2026-06-02).
- https://huggingface.co/facebook/sam-3d-objects manually gated, revision 2e73555018d2741ccd486e56c24fac41155a1dc6 (2026-06-12). Strong in-context shape/layout prior but inferred back surfaces/thickness remain hallucinations; no metric correctness guarantee.
- Official body/object alignment example: https://github.com/facebookresearch/sam-3d-objects/blob/main/notebook/demo_3db_mesh_alignment.ipynb
- MHR https://github.com/facebookresearch/MHR and tools/mhr_smpl_conversion provide conversion routines; conversion can add error and requires target model licences.

### GEM / GEM-X and GVHMR
- GENMO paper https://arxiv.org/abs/2505.01425 (2025-05-02), ICCV2025; renamed GEM December2025; GEM-SMPL demo release March2026.
- https://github.com/NVlabs/GENMO HEAD 16bebf402d8893184249ee206d957b8248cd8310 (2026-06-23), research noncommercial license.
- https://github.com/NVlabs/GEM-X HEAD 32992550dba114c62243fb55e361311972dce8f9 (2026-04-27); repo creation 2026-03-03. README news dates May/June2025 are inconsistent with repository creation; use verified March/April2026 availability instead.
- GEM-X temporal regression architecture ~520M, SAM3D Body features, 77 SOMA 2D joints, body/hands/face, camera/global output. Code Apache2.0, model NVIDIA Open Model license; no independent same-protocol challenge superiority demonstrated.
- https://huggingface.co/nvidia/GEM-X nongated, created 2026-03-11, revision 5ccf5ca3746c3620aa4016114f069a5f6ae399cd (2026-06-23), gem_soma.ckpt / gem_smpl.ckpt and supporting model assets.
- Follow-up source feasibility audit,2026-10-02: automatic YOLOX/ByteTrack
  boxes, VitPose77 and actual SAM Body pose-token1024 feed temporal regression.
  Static-camera mode uses identity camera, but demo K uses `max(W,H)` whereas
  SAM defaults to `hypot(W,H)`; a fair comparison must fix the same RGB-derived
  prior explicitly, not introduce source calibration. Require primary tokens,
  never the fallback padded204 controls. Freeze `postproc=False` for the first
  test; native global-scale clamp[.7,1.] and optional contact/IK remain possible
  metric biases, not evidence of accuracy. SOMA returns metres with clip-shared
  identity45/scales69 and77 articulated joints; it is **not** kit MHR136/68.
  Released MHR→SOMA wrapping is the opposite direction to submission export;
  no verified inverse exists here. A surface-quality experiment is possible,
  but adopting it requires an independently checked MHR fit, not renaming arrays.
  Mandatory SAM custom-license source leaves eligibility unresolved.
  VitPose also constructs a DINOv3 backbone via Torch Hub; DINOv3 code/weights
  have custom terms, so suppressing SAM image features alone is not an OSI
  closure fix. Do not recursively install optional SMPL/retargeting assets.
  Pinned `gem_soma.ckpt`541758499bytes, SHA4c1f85ca…ee298e, plus
  VitPose3388483384bytes would be new Azure assets; Body2109129346bytes
  SHA b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf
  and reference MHR are exact existing assets, not new/different checkpoints.
  SOMA assets pin d281db2d01553f7230c56e764cec00fe27ef23f7.
  No acquisition/benchmark/adoption performed. Defer until the cheaper J1
  grounding test shows which human/object depth error needs a new temporal prior.
- GVHMR https://arxiv.org/abs/2409.06662 (2024-09-10), https://github.com/zju3dv/GVHMR ; gravity-view coordinates, temporal world body recovery, static-camera mode; useful alternate prior, not detailed hands/objects.

### Do as I Do — deployable alternate object tracker
- https://arxiv.org/abs/2606.19333 (2026-06-17), https://github.com/malik-group/do-as-i-do ; HEAD 824591b808c342b20079c3b4198a8c2bdf88c74e (2026-08-02), MIT first-party code, upstream licenses remain.
- Complete reconstruction pipeline public: SAM3 segmentation -> SAM3D shape -> MoGe pointmap -> HaWoR hand -> BootsTAPIR -> shape-fixed guided SAM3D pose diffusion -> translation/scale optimization.
- Adaptive pose prior guided by 20 point tracks; sample candidate poses, clustering selection comparable to likelihood 30x faster. Reported DexYCB CD .66 vs FP .89, HOI4D .49 tied FP .49; controlled shared remaining pipeline. 150-video human preference 67% vs FP18%,15% ties. These are hand/object metrics, not full-body/world challenge.
- Requires >=32GB GPU, SAM3/SAM3D licensed gated access and MANO; current segmentation code click GUI can be replaced with automatic prompts. HaWoR fork CC-BY-NC-ND requires careful license review before modifying/redistributing.
- Assumes rigid object and semi-accurate monocular metric depth; only hand+object, not full-body/environment.

### AgentSTAR — deployable hard-case teacher
- https://arxiv.org/abs/2609.24487 (2026-09-21), https://agenticstar.github.io/, https://github.com/makezur/agenticSTAR ; HEAD a8eac9b1a4e8423bc148839e88a930f5f6aab1e5 (2026-09-23), MIT code.
- Coding VLM generates Blender primitive/kinematic shape, global shared scale; VLM picks bounded pose search, numerical renderer optimizes hand-occlusion-aware IoU; mandatory temporal diagnostics with selective smoothing. Cameras from Pi3X, not privileged calibration.
- HOT3D 93 trajectories/15 keyframes each: mean translation 3.04cm vs SAM3D Tracker6.41 and FP+VGGT7.56, mean rotation37.6deg. Evaluation uses global pose gauge alignment and GT depth *for scale alignment*. That alignment must never become challenge inference. High compute hours/tokens; not whole-body.

### MoGe2 / MoGe3, ViPE / Pi3X
- MoGe2 https://arxiv.org/abs/2507.02546 (2025-07-03); https://github.com/microsoft/MoGe ; HF Ruicheng/moge-2-vitl nongated 39c4d5e957afe587e04eec59dc2bcc3be5ecd968.
- MoGe3 paper https://arxiv.org/abs/2607.17967 v1 2026-07-20, v2 2026-07-21; release announcement 2026-08-18; code HEAD 74fbce054ebed49800de42d0ad0e83495065719a (2026-08-19).
- HF Ruicheng/moge-3-vitg revision 6ef26c5a4b4148dab5ccaacdb08b72dc66380475, vitl184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be, both nongated created2026-08-18.
- Sparse volumetric refinement helps fine/edge geometry; metric point-map global results can be worse than MoGe2 despite finer local geometry. Boundary pixel ambiguity/fly-points persist. Do not replace every global scale estimate by V3 blindly.
- ViPE https://arxiv.org/abs/2508.10934 (2025-08-12), https://github.com/nv-tlabs/vipe HEAD8c9f36144e08d8f8c8cf60d20ad70c100d9cff07 (2026-09-10), Aug2025 release, near-metric depth+intrinsics+camera, Sept2026 memory-bounded long sequence mode. For dynamic camera only if RGB evidence indicates motion; static challenge views should not induce artificial motion.
- Pi3X engineering release2025-12-28 https://github.com/yyfz/Pi3 ; RGB feedforward geometry+camera, approximate metric scale, confidence, optional conditions. Weights CC-BY-NC4, code BSD3. Never inject Track2 camera/depth.

### C2Dex / GraspHOI — ideas, not available full implementations
- C2Dex https://arxiv.org/abs/2608.07045 v1 2026-08-07 /v2 2026-09-06; https://github.com/K-Jie/C2Dex_code is README/assets only, TODO reconstruction/retargeting source.
- Canonical object contacts from hand/object silhouette overlap -> raycast surface -> opposite normals filtering -> locally stable phases -> DBSCAN dominant cluster/medoid per hand vertex -> sequence hand articulation/global fit, object motion fixed, penetration SDF+regularization. Good hypothesis, not contact proof: overlap can be occlusion; allow uncertain/noncontact/slip/regrasp states.
- GraspHOI https://arxiv.org/abs/2608.28386 v1 2026-08-28 /v2 2026-08-31; full code only promised, no GitHub GraspHOI repo found.
- SAM3D Body->SMPLH + WiLoR + Hunyuan2.1 + depth registration + palmar correspondence + finger/arm optimization. Thin thickness/shape hallucination is explicitly a failure. Static grasp-centric only; fixed body can penetrate large objects. Ablation reports some SAM3D non-watertight failures and Hunyuan mesh distortions, not universal winner.
- RRTrack https://arxiv.org/abs/2607.23669 (2026-07-26), https://github.com/7kevin24/RRTrack README/demo-only with source TODO. Useful VOS + 6D geometric memory verification and offline/online DINO recovery idea; RGBD reported results not monocular proof.

### RePHO — deferred physics branch
- https://arxiv.org/abs/2606.05359 (2026-06-03), CVPR2026 Highlight, https://github.com/dingbang777/RePHO source MIT; needs IsaacGym and InterMimic checkpoint, per-clip RL.
- Noisy kinematics -> adaptive reliable-frame sampling + independent forward/backward policies updating kinematic target; reconstructs physical trajectories.
- BEHAVE VisTracker CDh5.39/CDo8.73 -> RePHO6.82/11.06; penetration6.64->3.91, object float .30->.10. These are 10-frame PA-aligned CD, successful rollout frames only; full-sequence success51.4%. Not directly comparable with CARI4D numbers or a complete challenge submission.

### Other screened
- MILO https://arxiv.org/abs/2608.27407 (2026-08-27), https://github.com/ac5113/MILO HEADf58643f0a9b5967b7eb2142f5c7a759a00fd6e98 (2026-08-28), MIT complete code. Combined LRM human/object scaffold, virtual renders+triangulated body/hands, SMPLH fitting + object segmentation. Metric is PA-CD; single-frame generative human/object shape error coupled. Optional object template path must not use challenge Track2 GT meshes. Candidate initialization/scaffold only.
- EgoInfinity https://arxiv.org/abs/2606.17385 (v1 2026-06-16 UTC), https://github.com/Rice-RobotPI-Lab/EgoInfinity HEADde59610531c08010d63d0493d6a72973eabd402e (2026-06-18); deployable static-camera hand/object pipeline using MoGe2/WiLoR/MEMFOF/SAM3D/flow+PnP and contact refinement, useful source implementation reference. Mixed licensing WiLoR NC-ND, MANO NC, YOLO AGPL.
- HAT4D https://arxiv.org/abs/2606.28215 (June26 /Sept5); annotation toolkit released Sept4 but required special SV4D2 encoder checkpoint pending public release. Not reproducible complete branch.
- CoGS https://arxiv.org/abs/2606.28820 (June27): compositional Gaussian human/object/scene photometry, code promised upon publication. Rendering-oriented no verified full-code availability.
- InfiniHand https://arxiv.org/abs/2609.35743 (Sept28), https://github.com/infinihand/InfiniHand README/assets only, no released inference/weights verified. Egocentric, not priority Track1.
- Guiding Image-to-3D Generation with Test-Time Partial Observations https://arxiv.org/abs/2609.10531 (Sept9): occupancy + free-space posterior guidance into SAM3D, code not located. Useful future RGB-derived multiframe geometry guidance; paper real partial geometry may be unavailable/noisy from RGB.

## Fail-fast gates (proposed, not measured results)

- Lock a small external development set with object/subject-disjoint held-out subset and all-frame metric evaluation. Keep challenge clips unsupervised. Do not use leaderboard as hyperparameter validation.
- 2-4 representative clips smoke: normal object, thin/symmetric object, fast motion, bimanual occlusion. Reject stack if it cannot produce correct canonical pose+units+timestamps and valid all-frame outputs in first budgeted test.
- Require >=5% paired external metric improvement with no >5% mean regression on the opposite axis before full-dataset investment; report per-sequence failures not just mean/success cases. Threshold is a planning choice.
- Body/hands adoption: require reprojection gain plus metric/world/acceleration/penetration not worsened. Stop if only PA-only gain or neutral fingers artifacts moved to elbows.
- Shape adoption: reject scale/thickness candidates that win single-frame IoU but worsen heldout-view/time residuals or contact feasibility. Mask erosion for inferred-depth boundary bias; no mesh shrink solely to reduce penetration.
- Tracker adoption: measure catastrophic flip/reacquisition count, metric translation, surface CD and true acceleration error where external GT exists. Stop fixed smoothing if acceleration goes down but accuracy/dynamic fidelity worsens.
- Physics adoption: discard if coverage <100%, visual/CD regression overwhelms physics gain, or mass/friction tuning invents unsupported motion. Preserve measured base trajectory fallback for all frames.

## Second-pass prioritized hypotheses — 2026-10-02

### O1: one fixed-topology shape latent from all monocular frames

Implement our own low-dimensional cage/coarse-graph deformation with ARAP or
Laplacian regularization, alternating a **clip-constant** shape and proper per-frame
rigid poses. Keep verified K and the one human-anchored depth gauge fixed. Loss:
visible silhouette, robust visible camera-Z depth and reliable free-space;
actor-occluded regions are unknown, not negative evidence. Select complementary
views automatically, check temporally excluded frames. No new model/source
stack is needed beyond existing SAM (custom terms), Torch/PyTorch3D (BSD).

Fail fast on our own procedural asymmetric/thin/symmetric objects with fast
motion and occlusion: compare fixed-mesh ICP against shape fitting using metric
surface/pose/scale errors **without oracle input/alignment**, held-out silhouettes,
exact topology, closure and orientation. Reject a silhouette-only gain caused
by shrinking or a regression on the opposite geometry/pose axis.

### O2: MV-SAM3D multi-view latent fusion, adapted to a moving object

- [MV-SAM3D v2](https://arxiv.org/html/2603.11633v2), 2026-04-09;
  [released code pin abb04b5](https://github.com/devinli123/MV-SAM3D/tree/abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd),
  2026-05-30, `multi_view_utils.py:94–186`. SAM custom license, not Apache;
  reuses existing SAM weights, no new checkpoint.
- Entropy/visibility-weighted generative velocities share geometry; the exact
  released API keeps pose outputs from view0. GSO-30 reported CD42.0 single-view vs20.2 two-view/17.3 five-view;
  different normalized protocol, **not Track1 metric evidence**.
- Static exocamera does not mean static object: if estimated T_i maps object to
  camera, use effective w2c=T_i and c2w=T_i^-1. Do not pass static scene/DA3
  cameras as if the object were stationary. Test perturbation robustness from
  RGB-derived poses before challenge deployment.
- FlexiCubes topology can change during generation; final mesh is clip-constant,
  not fixed-topology across decoded latents. O1 needs a frozen template/cage or
  explicit projection, never invented vertex correspondences.

The September [partial-observation guidance paper](https://arxiv.org/html/2609.10531)
uses GT-rendered RGB-D in its benchmark and independent normalization/ICP in
evaluation; no released implementation verified. Useful occupancy/free-space
idea, not proof of a legal deployable monocular metric solution.

### H1: native hand proposals, not a full-output identity shortcut

Exact available runtime [sam3d_body.py at the audited NVIDIA pin](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/meta_arch/sam3d_body.py):
body mode returns before hand decoder1217–1219; full mode derives automatic
hand crops1227–1247 and fuses108 controls/wrists1471–1556. Existing crop/keypoint/
angle guards1311–1382 are not accuracy evidence. No MANO model is needed.

Full mode also changes scale8/9, scale18:+ and shape40:+ (1558–1593); it cannot
be exported as a sequence identity unchanged. The minimal current H1 experiment
transfers fingers **after** official conversion, keeping identity/body/wrists fixed
and never reconverting that proposal. `pred_pose_raw` becomes
zero1621–1623: re-forward the original body133/hand108/global3/shape45/scale28
blocks, never decode that placeholder. FreiHAND PA-MPJPE5.5mm is local PA
evidence, not camera-metric fingers under object occlusion. First sparse run is
an engineering gate; adoption still needs our own procedural native-MHR hand
validation (small/occluded/bimanual/free vs grasping) without GT input.

### H2: uncertain canonical contact, independent native-MHR optimization

[C2Dex v2](https://arxiv.org/html/2608.07045v2), 2026-09-06;
[code pin eae9248](https://github.com/K-Jie/C2Dex_code/tree/eae9248accaedd7a61dccad562f8048bb9e6c36f)
is README-only, not runnable. Its MPJPE28.55mm remains worse than HOLD22.13mm
despite better PA/physics quantities: do not equate contact plausibility and
metric accuracy. Implement independent native-MHR articulation/wrist/limited-arm
fit using H1 keypoints, priors, non-penetration and only confident contacts.
Allow free/contact/slip/regrasp states; no attraction from silhouette overlap
alone. Test noncontact foreground/background crossing, slip and release first.
Reject artificial hand displacement or over-smoothing masquerading as physics.

Priority: **H1 and O1**, then O2 if pose-conditioned fusion is robust; H2 only
after object geometry/poses are credible. Gates above remain proposed until
measured. No challenge GT/manual test labels or new large local assets accessed.

## Follow-up after O1 falsification — 2026-10-02

The first rendered five-DOF fitter is rejected (experiment R35): unchanged shape
controls regressed despite lower visible-depth loss. Do not retune these controls.
Next inexpensive measurement-model gate uses CUDA
[point_face_distance at PyTorch3D pin33824be3](https://github.com/facebookresearch/pytorch3d/blob/33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba/pytorch3d/loss/point_mesh_distance.py#L28-L93)
for observed-to-continuous-triangle distance only. The high-level symmetric
point_mesh_face_distance adds hidden-face-to-observation attraction and is not
appropriate here. Its default min_triangle_area=.005m² also treats our small
nondegenerate triangles as edges/points; use0 only after analytic interior/edge/
vertex and gradient gates. This removes a discretization hypothesis, not pose
bias. Any new fitter needs newly frozen shapes/paths/seeds and uncertainty-aware
pose handling; the old18-condition report remains failed.

MV-SAM3D exact pin audit: run_multi_view accepts RGB, masks and CHW pointmaps,
**no direct pose/camera arguments**. POSE_KEYS retains view0; all_view_states_storage
remainsNone, so the released all_view_poses_decoded branch is not a usable tracker.
compute_pointmap expects OpenCV XYZ then applies(-X,-Y,Z), despite a conflicting
PyTorch3D docstring: supply our already-scaled MoGe2 points without pre-flipping.
Supplying all pointmaps skips DA3; do not use its unpinned automatic download.
Visibility cameras require a custom callback: in camera0 coordinates,
w2c_i=T_i@inv(T_0), c2w_i=T_0@inv(T_i). The provided DDA callback handles self-
occlusion, not human/mask occlusion. Entropy-only1/2/3-view shape generation is a
possible separate gate with existing SAM weights, not a complete moving-object
pipeline or fixed-topology latent decoder. Same SAM license ambiguity persists.

### Next O1 hypothesis: nested pose-only / shape+pose with abstention

[BundleSDF](https://arxiv.org/html/2303.14158) (2023-03-24, CVPR2023,
sections3.3–3.4) alternates pose graph/SE3 and continuous shape, distinguishing
uncertain occlusion regions. Measured RGB-D is not monocular MoGe2 evidence.
[Pinned code licence](https://github.com/NVlabs/BundleSDF/blob/ffa67d425240b5b76d2e387a7dd3d3735a7cf1a1/LICENSE.txt)
is noncommercial: conceptual inspiration only, no code reuse.
[CamP](https://arxiv.org/abs/2308.10902) (2023-08-21/30) supplies a projection-
Jacobian preconditioner idea; [Apache2 source](https://github.com/google-research/google-research/blob/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/camp_zipnerf/internal/camera_delta.py)
does not make pose covariance calibrated or shape identifiable.

Compare M0=fixed initial shape+SE3 corrections with M1=sharedSPD5+identical SE3
corrections. Initial poses must come from generic image/depth hypotheses, never
synthetic truth plus noise. Continuous visible-point distance, isotropic robust
metric loss, equal frame weights, same pose budgets/priors; no hidden-surface
attraction, extra scale or intrinsics fitting. Pose dispersion may come from
automatic pixel-block bootstrap, labelled uncalibrated. Profile reserved-frame
poses on pixel blocks A, then evaluate on disjoint B; fitting and selecting on
the same held-out pixels would launder pose/shape error. Default M0: require
validation gain above block uncertainty, nonregressing silhouette, nonsaturated
bounds and pose-marginal shape identifiability before proposing M1. This remains
a design hypothesis, not an implemented or validated replacement fitter.

Use new calibration/test cohorts, untouched by the18failed cases: correct shape,
representable deformation, weak-view/symmetric geometry, correlated depth noise,
occlusion, and out-of-family scale/local deformation. Truth only manufactures
observations/scores, never initializes inference. Score canonical and camera-
posed geometry and pose separately, without GT realignment. False shape adoption
on correct or unidentifiable controls falsifies the hypothesis.

### Independent real validation acquisition screen

[CORE4D-Real V2](https://github.com/leolyliu/CORE4D-Instructions/tree/96b9084b9516af3ec4382a65d79e892d3e5c22b9)
announces CC-BY4.0 and provides object poses, human vertices/joints and camera
metadata. Its HF card declares MIT; resolve scope before acquisition, audit
challenge overlap without retrieving FORM-HOI, and avoid restricted SMPL-X
model/source by evaluating supplied geometry only if its terms permit. Two-person
Kinect15fps scenes are a domain shift, not automatically a Track1 substitute.
Future validator would expose one RGB view only to inference, isolate reference
assets until output hashes freeze, and report raw geometry plus one shared human/
object Sim3 over the clip. **No CORE4D or BEHAVE data downloaded; no real metric
validation or certified overlap clearance yet.**

2026-10-02 follow-up blocks automatic CORE4D acquisition: the official
[website source](https://github.com/core4d/core4d.github.io/blob/d3203b6cdf827705557282b99f39ce165d1f9ad3/index.html)
dataset JSON-LD specifies CC-BY-NC4.0, conflicting with instructions CC-BY4.0
and the tiny HF MIT card. Website footer licensing is not dataset licensing.
Bundled SMPL-X terms also extend to model-derived meshes/animations; no such
source/model is imported and supplied arrays do not establish a separate grant.
Conditional owner-provided validation clips, selected from `test_unseen_obj`
without examining outputs: `20231002/009`, `20231003_1/029`, absent the documented
training split. This is not proof of person disjointness or model-training absence.
Current HF concatenated RGB shards total103,867,863,822bytes and V2 motion batches
39,621,177,728bytes; no audited per-clip URLs/batch mapping. Seek written rights
and two hashed one-view minibundles before acquisition, not speculative bulk data.

### Camera-gauge audit and next shared-K hypothesis — 2026-10-02

Priority is **RGB-inferred, clip-shared intrinsics with the existing MoGe2 and
SAMBody weights**, not a new shape fitter or an assumed MoGe3 calibration gain.

1. [MoGe2 native source](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py#L195-L297),
   pin `925b8ed835a7a9cdb7578ba15c658a0afc969030`, MIT; existing
   `Ruicheng/moge-2-vitl-normal` weights pin
   `b135031bae30b5ac2ae141a0e68717795ce38340`.
   `infer(fov_x=None, force_projection=True)` estimates focal and Z-shift from
   the predicted pointmap, constructs centered square-pixel K, reprojects depth,
   then multiplies points and depth by the positive predicted `metric_scale`
   once. Focal recovery solves `min ||f*XY/(Z+d)-UV||²` on a 64×64 sample;
   returned focal is relative to half the image diagonal, hence
   `f_pixels = f_normalized*sqrt(W²+H²)/2`. With supplied horizontal FOV it still
   re-estimates `d` at fixed focal. **Changing K while retaining the old shifted
   depth is not equivalent to native fixed-FOV inference.** The scale head is
   independent of this postprocessing; `infer` does not expose its scalar.
   Geometry helpers assume centered principal point, no distortion and
   isometric X/Y. Fewer than two valid downsampled samples silently give focal1,
   shift0: detect that unsupported branch rather than accepting a guessed K.
   SciPy optimizer status is not exposed, and positive finite focal alone does
   not prove calibration. The [paper](https://arxiv.org/abs/2507.02546),
   2025-07-03, explicitly motivates the focal/distance ambiguity.
2. [SAMBody camera head](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/heads/camera_head.py#L61-L105),
   official source pin `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`.
   After native weak-camera sign conversion,
   `bs=bbox_size*s*default_scale_factor+1e-8`, `tz=2*K[0,0]/bs`; bbox-center
   corrections supply lateral translation. Explicit K also changes normalized
   bbox conditioning and ray embeddings, not merely this translation formula.
   Therefore rerun Body with the **same actual K** used by MoGe2; do not rescale
   saved `pred_cam_t` and call it new inference. The single-image estimator's
   actual `cam_int` path expects a Torch tensor despite its NumPy annotation.
   The underlying SAM custom-source licensing question remains unresolved;
   MoGe's MIT license does not clear the complete runtime.
3. [MoGe3 paper](https://arxiv.org/html/2607.17967v2), 2026-07-20/21;
   source pin `74fbce054ebed49800de42d0ad0e83495065719a`, 2026-08-19, MIT.
   Table1 improves average absolute metric-depth Rel from MoGe2 15.6 to
   MoGe3-L 15.0, but G is15.8. Appendix metric-point Rel is **8.65 for MoGe2
   versus9.61 for L and10.4 for G**; absolute metric pointmaps do not universally
   improve. No separate focal-accuracy result was found in the paper. Local or
   affine-aligned point improvements are not evidence of a resolved metric
   gauge. New MoGe3 weights are not required for the first camera experiment.

Proposed **J2**, not implemented/measured here: reuse J1's exact nine RGBs,
automatic masks, prediction firewall, score definitions and frozen gates. First
infer MoGe2 focal from RGB alone on each frame; take the median positive pixel
focal per clip. Then rerun native MoGe2 with
`fov_x=2*atan(W/(2*f_clip))` in degrees and `force_projection=True`, and rerun
SAMBody with that identical centered K. Keep one existing human-derived scale
per clip, applied once to both depth and XYZ; no per-frame scales, GT choice of K,
offset fitting or private camera feedback. Compare fixed-K1280 and shared-K
pipelines with all nine frames scored, including hidden true-focal960/1600
clips. Preserve median clip CD gain≥5% and no clip regression>5%; report raw
camera CD/Z-bias and shared-scale results separately, with human Sim3 only the
existing disclosed private diagnostic. Even a fixed-K baseline failure remains
useful mechanism evidence, not permission to omit difficult clips or retune.
Clip aggregation reduces jitter, not systematic focal/depth ambiguity: coherent
projection is testable, **better calibration or metric accuracy is not assumed**.

### External real camera/depth validation — 2026-10-02

Use [TUD-L RGB-D](https://bop.felk.cvut.cz/datasets/#TUD-L) as **object-only**
sensor evaluation, not human/HOI ground truth or V2D score. Reduced BOP19 ZIP
372464733bytes plus models/base is374952356bytes, pinned HF6527f7d4. Three
scenes have200 selected nonadjacent RGBframes each, native640×480; no temporal
adjacency or accurate acceleration truth assumed. Calibration and visible masks
are private evaluation only. BOP backprojection uses integer(x,y), while native
MoGe uses(x+.5,y+.5); unchanged XYZ is equivalent under cx,cy−.5 convention
conversion. Never erase focal error by reconstructing predicted Z with GTK.
Sensor depth noise and manually initialized/ICP-propagated mesh poses remain
limitations; prefer the measured sensor surface over a claim of perfect meshGT.

MoGe2 paper §4/A.2 and
[trainv2 at925b8ed](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/configs/train/v2.json)
declare **24** datasets (16synthetic/3LiDAR/5SfM), not46. TUD-L/BOP/YCB are not
listed; [evaluation10datasets](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/configs/eval/all_benchmarks.json)
also excludes TUD-L. Indoor ARKitScenes/Taskonomy/ScanNet++/Hypersim/IRS/
Structured3D and Objaverse overlap at domain/object level may still matter.
Exact normal checkpoint card b135031 contains only MIT terms, no frame-level
attestation; retain `training_overlap_excluded=False`. A paired RGB-only native
focal-vs-fixed test is useful real mechanism evidence, not verified Track1 gain.

### D76: isolate metric depth instead of rebuilding a camera solver

Primary source audit through2026-09-30: UniDepthV2/UniK3D and Pi3X weights
are noncommercial; PerspectiveFields uses Adobe noncommercial terms. GeoCalib
advertises Apache2 code/CC-BY4 weights and reports FoV errors3.21/4.90/4.46/3.03
degrees, not metric-depth gains ([paper](https://arxiv.org/html/2409.06704v2)).
Its actually imported `perspective_fields.py` attributes an adaptation from
PerspectiveFields; a specific commercial grant was not found. Do not claim
full source eligibility or acquire/adopt that path. A source-separated learned
frontend plus independently derived pinhole solver is technically possible,
but would require new solver validation; defer until calibration failure is
measured, rather than assume it. Random Hamburger bases even in eval also need
seed/replay checks. Primary training OpenPano lists HDRMaps/PolyHaven/Laval,
not TUD-L, but this does not establish checkpoint/backbone non-overlap.

Prioritize [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE/tree/4010e39f3634a45bc60553321fb49fb760bd594e),
HFpin4010e39f3634a45bc60553321fb49fb760bd594e, source
[3d835ec1a5802d64a8b8b15f817a1ab54809bfe4](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4),
both primaryApache2. Weight1336734448bytes/SHA
bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776;
config847bytes/SHAa336f3e76fe375aaae17a9aed9130c9f2aa061535d317ec57dcb2f1f02e1dd53.
Paper table11 reports metric AbsRel.070NYUv2/.086KITTI/.104ETH3D/.105SUNRGBD/
.128DIODE; neither proves RGB-only camera estimation or superiority to MoGe2.
General API imports GPL `evo` and unused exporters: use native minimal config,
model and input/output processors only. Audited closure27Python192452bytes,
no high-level API/evo/COLMAP/GS execution. DA3Nested/Giant NC weights excluded.

Nine D76 single-image forwards use the **same frozen original K800** as the
existing fixed MoGe branch, not another MoGe inference or private calibration.
Native resize518 upper_bound_resize maps640x480→518x388→518x392. Actual float32
processedK approximately fx647.5/fy653.3333, cx259/cy196; canonicalZ is multiplied
once by its **actual processed mean focal/300**, nominal2.1680555556, not800/300.
OutputProcessor does not resize/scale. Our declared bilinearalign_cornersFalse
upsamples metricZ only to640x480; reconstruct XYZ with originalK800 and+.5 rays
to compare depth methods on identical cameras. Not a native DA3 pointmap (native
unprojection uses integer pixels), no GTK/oracle rays/median scale fitting.
Preserve native sky correction.3/quantile.99, seed CPU/CUDA before each call;
allpositive finite pixels and allnine images required. SFcheckpoint header only
was inspected via48864byteHTTPRange:406F32 keys all `model.`; full tensors are
not acquired yet. Strict fullstate inclbuffers after one prefix removal, no
partial matching. Paired independent quality gates stay≥5% median scene CD gain,
no scene regression>5%, coverage≥95%; original MoGe queue unchanged. No accuracy
or model inference claimed until actual receipts and paired evaluation complete.

## D78 following paired quality evidence (2026-10-02)

DA3Metric is19.4160%betterthanMoGe fixedK onTUD-L, yet absoluteerrors58.95–75.33cm
remain; actualMoGelearnedcamera excelsTUD-L but pairedhumanJ2regresses22.88%.
Do not conflate domain-biased camera, depth and objectshape. Next ownhypothesis
is ray-preserving affinecameraZ referenced onlytoestimatedvisibleMHR:
oneαperclip/βperframe, independentlyheldoutspatialhumanpatches, conditioningfrom
**within-frame**depthspread (personmeanmotioncannotidentifyαwithfreeβ). This
adapts humananchoring motivatedby [DoasIDo](https://arxiv.org/html/2606.19333)
without its NC HaWoR/MANO dependencies or an off-ray3Dtranslation. [MoGe2](https://arxiv.org/html/2507.02546)
explains underlyingaffine/focalgauge. NewJ3privatecohort/predeclaredgates in
experiments.md, no privatecalibration/inference/manualtestlabels.
Lowerprioritysafe-separatedGeoCalibfront-end+ourcamera math and discreteSAM
mesh selection need separate rights/runtime audit; fullPerspectiveFields-adapted
package is not cleared by top-levelApache or CC-BY weights alone.

### D78 falsification and camera independence (2026-10-02)
FreshJ3affinefits reduce predictedhuman spatialheldout relativeerror from
[.01205,.01292,.02920] to[.00207,.00251,.00408], but objectcameraCDhalf
regresses6.6042%median. αapproximately[.350,.388,.364] and β2.6–5.5m show
localhuman-consistency collapses depthcontrast without establishing wider
scenegeometry; do not tune βclipcaps/conditioning on theseprivateanswers.
Relativehuman-objectcentroiderrors improve all3 yet rawcameraCD fails; neither
metric alone certifies fullHOI. AffineJ3 is rejected under its predeclared gate.
Pinned SAMBody's cam_int=None is NOT an independentcamera estimator: current
SAM3DBodyEstimator is built without fov_estimator, so prepare_batch defaults
f=hypot(W,H). Optionalnative FOVEstimator(name=moge2) calls the sameMoGe2
focalinference; it would not provide independent evidence againstJ2regression.
Next independentcamera evidence can come from perspective/gravity frontend or
constrained backgroundvanishingpoints, with mandatoryweak-sceneabstention and
freshnonchallengequality validation. GeoCalibwholepackage remainsuncleared
because PerspectiveFieldsadaptedsource; audit commerciallycompatible frontend
subset before acquisition and derive ownpinholemath ratherthancopyrestrictedLM.

### Independent geometric calibration fallback audit (2026-10-02)
OpenCV4.12.0 (2025-07-02), commit49486f61fb25722cbcf586b7f4320921d46fb38e,
LSDcodeSHA54e8d04a4da08d787c1beaba4cff6f3711a982746847f084fae3004d6184732d;
Apache2repo/BSDLSDheader and restoredMITNFA50807c32233ecbd584fb17afaa475d81c7d67ee7.
[VP-Estimation-with-Prior-Gravity](https://github.com/cvg/VP-Estimation-with-Prior-Gravity/tree/fc154dd36e82b5ef8cc8a75d4d7e0b7ae2e2392c)
MIT/ICCV2023 is a geometryreference, notlatestSOTA. UseNO suppliedgravity;
its 220solver obeys f²=−(v1−c)·(v2−c) fororthogonalfiniteVPs andfixedc.
Homogeneousgeneralform divides by w1w2; nearinfiniteVPs lackfocalinformation
andmustabstain, notclamporpickpositivef. TwoVPs imposeorthogonality ratherthan
proveManhattan: require3independentfamilies, heldoutsegments, agreementbetween
all3pairfocals and groupededgebootstrap uncertainty; weak/nonManhattan scene
acceptance explicitly tested. This remainsa conditionalproposal, notcameraGT.
LSDOpenCVpixelcoordinates needcare: usec=(W/2−.5,H/2−.5), thentranslateboth
VPs andc +.5 toWorldRewardcellcentres; verifywithownsteprasterbeforeuse.
Nocrop/anisotropicresize unless exactH/K/VP transformtracked. Bootstraprelative
focalCI>10%, missing3families, rankfailure orf²≤0 ->abstain. Proposedfresh36RGB
12strong/12rotated/12weak camera-onlycohort: strongcoverage≥90%, medianferror≤3%,
worst≤10%, weakfalseaccept≤5%. SubsequentfreshcoupledBody+depth+human/object
pairedvalidation stillrequired; camera-onlygain insufficientforadoption.
NoVPcode/model/benchmark acquired/executed/adopted atthisaudit.

### Independent camera frontend: exact GeoCalib safe-subset audit

GeoCalib pin97b8968e7798a66bf04fcf791fb535624241bda7 (2024, not a2026
SOTAclaim) supports an independent calibration hypothesis unlike optionalSAM
MoGeFOV. Wholepackageclearance NOTestablished because PerspectiveFields includes
AdobeNCadaptation. Runtimecandidate ONLY `geocalib/modules.py`18900B
SHA222f28ef570fbda9bd46bbd7c6089ccff8b9d96072177dd54179b6444fd48de6
plus GeoCalibclasses18–89 (LowLevelEncoder/UpDecoder/LatitudeDecoder/
PerspectiveDecoder)2569B SHA95b63c6917f9922e0859c6e2ce268bd25915ef965aa7646ede6cfd40d9bb7414.
These useTorch/typing only; owncontainer MSCAN+ll_enc+perspective_decoder excludes
original LMOptimizer/camera/PerspectiveFields imports. GeoCalibApache2 + SegNeXt
Apache2 (d46ffa737980ec7a9f5b9465c78f254449163509); retainlicenses/credits/changes.

[Publisher pinhole v1.0 weight](https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar)
116074121B published2024-09-08, READMECCBY4. PublisherSHAabsent; actualAzure
weightSHA/header/keyshapes/strictload still UNVERIFIED and acquisitionpending.
Require `checkpoint['model']`, explicitcollisionfreefullstatemapping includingBN
buffers/finitevalues, strict=True (officialloaderpermissiveness notadopted).
RGB[0,1] input; MSCAN internallyRGB→BGR×255. UpB2HW, latitudeB1HWrad, two
confidenceBHW; Hamburger randominitialbases eveneval requireCPU/CUDAseedreplay.

Ownmathematics: n=((u−cx)/fx,(v−cy)/fy,1), sinlatitude=g·n/||n||,
up=normalize(g_xy−g_z*n_xy), **normalized-camera-plane** notpixelvectorunits;
||g||=1, f=exp(a)>0. Integerpixelgridnative, distinctfromourcellcentres.
Nativepreprocess shortedge320/bilinearantialias thencentercrop32multiple: e.g.
1024×768→426×320→416×320; trackexact sx/sy/crophomography inK, noaveragefx/fy
or silentcoordinateparity. Ownconfidenceweightedfit/rank+reservedspatialblocks
onlymeasureconsistency, notaccuracy. Initialfrontendsmoke180s/offline twoNEW
procedural320×416RGB(noresize), strictload+deterministicreplay; thenpredeclared
freshcameraablation incluninformativecontrols andlaterfullhuman-objectvalidation
beforeanyadoption. No fittedcalibration/GT challengeinput orCARIscoreclaim.

## D87 contingency — native clip identity (not measured/adopted)

First test robust **geometric medoid** of uniformly sampled paired native
shape45/scale28 proposals: decode each in the same neutral pose, select the
actual candidate minimizing median neutral-surface distance. Keep the pair
intact; do not average PCA/shape coefficients, animated vertices or poses.
Equal weights, original validity guards; estimator's supplied-mask score1
is not identity confidence and its projected keypoints are not independent
observations. No guarantee against a common systematic bias. Fresh scenes/
occlusion order, raw/first/medoid paired metrics on all frames, identity
permutation-invariance control, predeclare before new GPU work. Fifteen neutral
decodes/no new checkpoint is the cheap gate, not permission to retune D87.

Second, only after evidence of observable morphology: optimize73 native
shape45/scale28 with poses/hands/K/translations frozen, robust automatic
visible silhouettes (object-occluded unknown), anchored identity regularizer.
Frames0/2/4 fit,1/3 excluded from objective; another independent cohort,120s
ceiling. Stop if silhouette alone improves, camera bias dominates, conditioning
is poor, or opposite-axis metrics regress. No GT-derived K or same-cohort
threshold adjustment, no pass directly to96frames.

Evidence: [SAM3D Body §§6.2–6.3](https://arxiv.org/html/2602.15989v1),
2026-02-17, describes shape/skeleton regularization but is not proof of this
variant; [native MHRHead at NVIDIA pin7c0d3b94](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py)
provides45/28→68; [SLAHMR CVPR2023 MIT source](https://github.com/vye16/slahmr/tree/58518fec991877bc4911e260776589185b828fe9)
uses shared shape/reprojection, not a license grant for its SMPL stack.
[CARI4D v3](https://arxiv.org/html/2512.11988v3),2026-04-19, motivates the
human/object depth-gauge issue; better identity does not guarantee its solution.
Existing SAM/custom/source-license restrictions remain unresolved.

## Real FULL-HOI validation priority — rights/provenance blocked, 2026-10-03

Prioritize two authorized HODome minibundles over another synthetic variation or new GEM-X stack; no benchmark authorized, acquired or preregistered here.
[NeuralDome CVPR2023](https://arxiv.org/abs/2212.07626) captures10subjects/23objects; [toolbox cutoff pin f6b869d](https://github.com/Juzezhang/NeuralDome_Toolbox/tree/f6b869dcc39f5cfd3bb7bf6b85721a534803a45a),2026-07-12, documents one-view26 (`data27.mp4`) evaluation.
The May26,2026 MHR announcement is present in the [immutable same-day README](https://github.com/Juzezhang/NeuralDome_Toolbox/blob/29a2435a825c0738a5ceb6950782d08ee85df830/README.md#L23),11633B/SHA256`13481cd8c107d5a29597d238e622c0ae512395bc6f1067ae4c153fee8b58f394`; this establishes a publisher announcement, not independently verified annotation quality.
[Native export source](https://github.com/Juzezhang/NeuralDome_Toolbox/blob/f6b869dcc39f5cfd3bb7bf6b85721a534803a45a/scripts/forward_mhr_to_npz.py) reads model/identity/face JSON, forwards MHR and exports18439vertices/36874faces in metres; objects and humans require matching `calibration_ground` world→camera.
**MHR supervision provenance is UNKNOWN:** the HODome files inspected only consume existing JSON/vertices; no annotation-generation optimizer or proof of independent multiview fitting was found. Do not treat potentially monocular SAMBody-derived reconstructions as independent human GT. HOI-M3's separate multiview-fit claim does not establish HODome provenance.
**Dataset NC blocker:** the [actual bundled LICENSE.md](https://drive.google.com/uc?export=download&id=1DvVJV1OmmqQz1tpuo4uTXHUs2D4u3W_z),5654B/SHA256`083e79c192726bad6048009a89792ea3e4ec312a768eddd635b9ec37f896f8e2`, allows only non-commercial research, prohibits commercial artefacts and redistribution without written permission; Apache toolbox/website footer do not override it. Its introductory name says InterHuman, but NeuralDome citation/commercial contacts match the bundle; ask owners to clarify scope.
Written minigrant request to `wangjingya@shanghaitech.edu.cn`/`xulan1@shanghaitech.edu.cn`: competition/commercial-solution private evaluation, permission to publish code/scalar results, and two separately hashed single-view RGB minibundles plus private MHR geometry/object mesh/poses/calibration; no restricted SMPL/MANO source/model.
Request exact MHR asset/version/units, annotation method and original sensors, identity consistency, reconstruction uncertainty, permission for pre-exported meshes, and possible SAM/CARI/FORM/challenge overlap; absence of evidence is not an unseen-data claim.
Proposed small pilot only after clearance: two distinct subjects/objects,96contiguous frames each at30Hz, identifiers fixed before outputs; public RGB-only inference, private reference/calibration isolated until all predictions hash-freeze. Raw camera errors plus explicitly disclosed one human-initialized clip-shared Sim3, never per-frame GT alignment. Quality gates still need preregistration before acquisition/compute.
**Downloader pitfall:** [download_hodome.py](https://github.com/Juzezhang/NeuralDome_Toolbox/blob/f6b869dcc39f5cfd3bb7bf6b85721a534803a45a/scripts/download_hodome.py) accepts `--modalities` but `gdown_folder` ignores it and downloads the whole folder; use verified individual URLs/minibundles, not this purported subset mode. Stop if rights, reference independence or bounded acquisition remain unresolved.
Alternative real human metric validation: [HUMAN4D MIT tools](https://github.com/tofis/human4d_dataset/tree/e4d752f68dd180d7d94d6b67e811fd4bf0323f9c),2020 RGB-D/Vicon; [Zenodo subject1 metadata](https://zenodo.org/api/records/4473009) is restricted with no verified data licence/files, so request permission/minibundle. No full rigid-object GT established.
[HOT3D official terms](https://www.projectaria.com/datasets/hot3d/license/) make both UmeTrack and MANO hand annotations NC-SA; sequence/object-only access is not commercially cleared full-body HOI. BEHAVE/InterCap remain NC; CORE4D's prior conflicting grants remain unresolved.
D84's [native Kornia resize](https://github.com/cvg/GeoCalib/blob/97b8968e7798a66bf04fcf791fb535624241bda7/geocalib/utils.py#L69-L140) differs from our Torch antialias; RGB/0..1 match. Mismatch is source-proven, causality unmeasured; preserve nine rejections and test only on a new authorized cohort without relaxing support gates.
GEM-X temporal SOMA still lacks a verified inverse to kit MHR136/68 and does not itself resolve metric depth/camera gauge; defer new assets until a real paired quality pilot is feasible. No RGB, depth, meshes, poses or GT downloaded during this audit; only primary textual source/metadata/licence read in memory.

### ContactPose — conditional real hand/object validation, not current bottleneck

- [ContactPose ECCV2020](https://arxiv.org/html/2007.09545v1), [source pin89cec790f2c7bdb3c81f4d9e98b337387ce6dda8](https://github.com/facebookresearch/ContactPose/tree/89cec790f2c7bdb3c81f4d9e98b337387ce6dda8),2025-05-07: [README licensing](https://github.com/facebookresearch/ContactPose/blob/89cec790f2c7bdb3c81f4d9e98b337387ce6dda8/README.md#licensing) explicitly grants code/all non-mesh data MIT; LICENSE.txt1078B/SHA256`8e54a940864e4631be3848cad23a7e73b208aab0b6287d05aeae6c557405f9e1`. Each3Dmodel has separate README.txt/licenses.json terms, unverified here; no universal mesh/commercial clearance or training-overlap exclusion.
- Paper§3.2 estimates21joints from multiview/temporal OpenPose observations, Optitrack object poses and grasp-rigidity assumptions; MANO is fitted **after** those joints. [Reader lines76–124](https://github.com/facebookresearch/ContactPose/blob/89cec790f2c7bdb3c81f4d9e98b337387ce6dda8/utilities/dataset.py#L76-L124) exposes JSON joints/object transforms/K directly, so own NumPy evaluation need not import MANO. This is estimated supervision, not perfect mocap hand GT; rendered hand masks/depth can depend on MANO and are not independent labels.
- [Publisher sample](https://drive.google.com/file/d/1paUAxXgHp6wDFBFw9MI1mxGElEl2KPew/view), linked in May2025, has Drive metadata1979376914B (~1.98GB); exact inventory/sha/two-record feasibility unverified. Any later authorized acquisition stays bounded/Azure-only, audits archive/asset licences before metadata-only selection, and separates raw one-view RGB from private joints/depth/K/object poses. Old Dropbox links are broken; no speculative full DataPort download, GT crop or per-frame alignment.
- D89 currently attributes99.65–99.89% squared error to centroid, with centered-body RMS~5–6cm: **do not prioritize hands or download ContactPose now**. It is a conditional real wrist/object-relative diagnostic only if that becomes a measured bottleneck; quasi-static grasps/blue printed objects are not full-body/dynamic Track1 evidence. No ContactPose images, depth, meshes, annotations or models acquired; overlap and per-mesh rights remain unknown.

### Coupled native identity/pose fitting — independent RGB keypoint candidate

[DWPose ICCV2023](https://arxiv.org/abs/2307.15880), [Apache source3dca5db79d9f9ffdd378753ddf6ec66535aace88](https://github.com/IDEA-Research/DWPose/tree/3dca5db79d9f9ffdd378753ddf6ec66535aace88): [publisher HF1a7144101628d69ee7a3768d1ee3a094070dc388](https://huggingface.co/yzd-v/DWPose/tree/1a7144101628d69ee7a3768d1ee3a094070dc388) declares Apache weights; `dw-ll_ucoco_384.onnx`134399116B/SHA256`724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843`. Publisher terms are not complete clearance of COCO/UBody/teacher rights or overlap; no weights acquired.
[Standalone onnxpose.py](https://github.com/IDEA-Research/DWPose/blob/3dca5db79d9f9ffdd378753ddf6ec66535aace88/ControlNet-v1-1-nightly/annotator/dwpose/onnxpose.py),11608B/SHA256`16fb69ab54f5e1ce8a5ad186e92f357da0162fc2ca2eeec4ccf1db72949291a2`, imports only typing/cv2/NumPy/ONNXRuntime; reuse actual automatic actor boxes, reject absent/invalid boxes before native full-image fallback. Avoid ControlNet/OpenPose drawing imports and new detector models.
Native low-level output is133 COCO joints, not134: the higher Wholebody wrapper inserts a synthetic neck/remaps OpenPose. Freeze original pixel coordinates/scores; scores are not calibrated visibility. RGB-conditioned2D predictions differ from [SAMBody's projected latent3D joints](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/meta_arch/sam3d_body.py#L473-L483), not independent measurements made by that same reconstruction.
[SLAHMR CVPR2023](https://arxiv.org/abs/2302.12827), [MIT58518fec991877bc4911e260776589185b828fe9](https://github.com/vye16/slahmr/tree/58518fec991877bc4911e260776589185b828fe9), supports shared shape/per-frame poses/translations with robust external-keypoint reprojection. Borrow the architecture, not SMPL/VPoser/HuMoR code/models; own nativeMHR forward/export must preserve ABI/bounds and one clip identity.
First hypothesis fixes existing K/metric prior, jointly refits translation/root then pose/identity with independent2D/silhouette evidence and regularization, instead of substituting identity into fixed poses. Joint2D alone cannot identify focal/body scale/depth; no per-frame camera to erase body error, no GT calibration. Future protocol/gates not yet preregistered; no accuracy/adoption or complete stack-licence claim, no download/install/job.

### Independent2D runtime prerequisite — pinned CPU ORT, not acquired

[PyPI ONNXRuntime1.30.0 metadata](https://pypi.org/pypi/onnxruntime/1.30.0/json)
reports the compatible LinuxCPython3.11 wheel uploaded2026-09-10T16:31:11Z,
latest stable compatible version identified before the2026-09-30 cutoff.
`onnxruntime-1.30.0-cp311-cp311-manylinux_2_28_x86_64.whl`23561046B,
SHA256`fd54b314ea385bcecac69ab431f020ba503e3878dad4ebb645fec5a24b041242`.
[Sourcev1.30.0](https://github.com/microsoft/onnxruntime/tree/f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7)
is MIT, LICENSE1073B/SHA256`2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c`;
embedded third-party notices still require audit. Nativeimage Python3.11.10
has noORT. Dependencies flatbuffers,numpy>=1.21.6,packaging,protobuf>=4.25.8
and glibc>=2.28 must be checked before an isolated Azure-only offline
`--no-index --no-deps --target` install; no Torch/CUDA/cuDNN or global resolver
changes. ExplicitCPUExecutionProvider/version/importpath mandatory; CPU cost
not measured. This is a runtime plan, no wheel/model install or inference.

Azure actualCPUimage check2026-10-03: glibc2.35,Python3.11.10; NumPy1.26.3,
packaging24.1,protobuf7.36.2 present, **flatbuffers absent**. FutureORT install
therefore additionally needs its own pinned flatbuffers wheel/source notice;
no install attempted and no existing image modified. GPU0%,111MiB while this
metadata-only check ran, no competing inference.

DWPose low-level channel contract clarified by author/publisher `yzd-v`,
2023-08-29: [«The input is rgb.»](https://github.com/IDEA-Research/DWPose/issues/25#issuecomment-1696943489).
The unchangedBGR `cv2.imread` example conflicts with this; use decodedRGB or
ONE explicitBGR→RGB conversion, never two. [Author export pointer](https://github.com/IDEA-Research/DWPose/issues/15#issuecomment-1683218589)
and [MMDeploy source](https://github.com/open-mmlab/mmdeploy/blob/6cd29e2152d6935bde2f9252b47170724bac20ac/mmdeploy/apis/pytorch2onnx.py#L63-L67)
execute preprocessing before export, consistent with normalizedRGB input.
No actual publisherONNX graph audited yet; this is primary intent/source
evidence, not a graph-proof or image-accuracy claim.
Missingdependency [Flatbuffers25.12.19](https://pypi.org/pypi/flatbuffers/25.12.19/json),
wheel26661B/SHA256`7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4`,
published2025-12-19, no Requires-Dist. [SourceApacheLICENSE](https://github.com/google/flatbuffers/blob/7e163021e59cca4f8e1e35a7c828b5c6b7915953/LICENSE)
11358B/SHA256`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.

Native low-level source refreshed at pin/SHA unchanged: `preprocess(img,
out_bbox,input_size=(192,256))` returns three lists (normalizedHWC crops,
centres,scales); `inference_pose(session,out_bbox,oriImg)` returns133joint
keypoints[B,K,2]/scores[B,K]. Critical runtime gate detail: source usesNumPy
float64 means/std, **no float32 cast**, then supplies a Pythonlist of one
CHWarray toORT. Actual list→tensor conversion must be verified rather than
pretending native code already castsF32. Invalidscore<=0 sets a−1 coordinate
BEFORE SimCC/2 and crop→image transformation, so finalinvalidcoords are not
necessarily−1; keep rawcoords/scores, validity from finitepositivescores.
No runtime/inference or embeddedwheelmetadata verification yet implied.

AcquisitionD94 actual packaging audit revealed noLICENSE in the exact
Flatbuffers25.12.19 wheel14members; ORT1.30.0 wheel353members embedsLICENSE
andThirdPartyNotices, with allversion/tags/dependencies matching metadata.
A separately pinned matching-source FlatbuffersApacheLICENSE exists; preserve
the failed original embedded-notice contract, and explicitly audit external
primary-notice binding in a new read-only namespace. This is not a blanket
waiver of third-party rights or wheel/binary eligibility.

NextD95 ABI plan (not yet executable/passed): offlineCPU isolatedtemporary
ORT+Flatbuffers install only afterD94v3integrityPASS; old synthetic public
RGB clip_00_frame_000/clip_01_frame_000 and automaticSAM2humanmasks, noprivate
geometry/K/depth/modelassets. Signedmanifest2199B/SHA
`2c584ea633a958c737520d53c68c12b1429b8358f182c07bf46e53627b8f8267`;
maskreceipt19818B/SHA`aa1c8346cfd7609a58d055f71762060aca238c216100bd8aa990ca1de79ea909`.
BBox fromnonemptyautomaticmaskmin/max+1, nofullimagefallback. TwofreshCPUsessions
with4threads, each2images; actualnames/types/133SimCCoutputs, nativeRGB
preprocess/postprocess, rawscorevalidity and exactreplay,180smaximumincluding
isolatedinstall. Existingcohort alreadyprivatelyevaluated: this is ABI/replay
only, neverfresh independentquality or D87retuning. Nativefloat64list feed
conversion mustbe observed, not silentlyadvertisedasnativefloat32casting.

### Verified independentCOCO→nativeMHR keypoint semantics (2026-10-03)

[NVIDIA nativeMHR70 names at7c0d](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/metadata/mhr70.py),
26326B/SHA`695c2c7d472e32757c480114fdb054d54ee4af53f69b2a6e040b00a55b270dc9`,
and DWPoseCOCOwholebody names agree body17 DWPose0..16→MHRkeypoints
`[0,1,2,3,4,5,6,7,8,62,41,9,10,11,12,13,14]`. Wrists are62/41,
not raw127joint indices. [NativeMHRHead](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py),
14148B/SHA`62af48b1f33462bc445d7f342009fcbceabd221f2a6e9bcd1d3739d582bc9fd6`,
`mhr_forward(return_keypoints=True,return_joint_coords=True)` returns vertices,
**308keypoints**,127joints; fixedkeypoint_mapping[308,18566] applied after
metre conversion. ModelHEAD.forward alone slices keypoints[:70] and flipsYZ.
Use first70nativekeypoints with YZflip once+pred_cam_t once for our camera;
not joint-skeleton ordinal guesses or projectedSAMtargets as observations.
Future cheap2Dbody fit can hold K,metricZ,shape/scales/hands fixed, optimize
XY/rootrotation against these independentRGB17points with fixedconfidence
policy/poseprior and reservedkeypoints/frames. No proof2Didentifiesdepth/
3Dshape/contact; nativeautograd/runtime and hypothesis gates still needed.
Feet17..22 map native15..20. Handroots91→62/112→41 and fingertips are
semanticallyverified; intermediateproximal/distal conventions and fullface68
mapping remainunverified, so do not guess a full133→70correspondence.

MHR70 itself warns some proximalfinger `third_joint` landmarks «doesnt match
with wholebody»; avoid a full21hand map inferred from anatomical ordinal
names. Verified body17 is enough for the first independentreprojection trial,
with face/handdetail deferred rather than pretending those uncertainlabels
are exact. FreshCPUimage metadata confirmed pip24.2, cv2runtime4.11.0 with
both opencv-python/opencv-contrib-python4.11.0.86; pinned-image provenance
records this coexistence, not an assumed exclusiveopencvinstallation.

ActualD95v2 AzureCPU runtime established on2026-10-03: exactpublisherONNX
declares symbolicoutputdimensiontokens, notfixed133tails inmetadata; actual
fourinferences nevertheless returnfloat32[1,133,576]/[1,133,768], decoded
float64[1,133,2]/float32scores[1,133]. Nativefloat64-list feed accepted without
owncast, byte-identicaltwo-session replay, .060s/call on4CPUthreads; internal
ORTconversion/channelgraph/semanticaccuracy notproven. ImmutablePASSreceipt
`d96eb5c8c4030bf2e16924093aef03a9636f12cc2f3d3a8b7475731fe49d9a79`.
This removes runtime uncertainty for independent2D observations, not their
trainingrights/challengeoverlap uncertainty or monocular3D ambiguity.
Native root3 are EulerZYXcontrols (`roma.rotmat_to_euler("ZYX",...)` inhead),
so futureoptimization should preserve nativeMHRforward and not assume an
arbitrary rigidSE3increment converts back to those controls unchanged.

### Native bounds are not guaranteed by learned decoding (2026-10-03)

After the strict D96 initial-feasibility failure, refresh primary source rather
than clip observations or bypass the preregistered bounds. Pinned native head
above expands68 scales as `scale_mean + scale_params @ scale_comps` (L226),
without projecting to physiological limits. [MHR body conversion](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/modules/mhr_utils.py),
15,157B/SHA`40d33308c83516c15e14889d148419b8c610a8c33aca57b3922354aa4987ec6e`,
uses6D-to-Euler and atan2 for compact body controls (L318–344). Wrist branch
selection minimizes original/alternative Euler limit violation; it does not
hard-clip every decoded articulation (L65–95).
[Momentum ParameterLimits](https://github.com/facebookresearch/momentum/blob/c54b9def6df15b7b807e6c3010f09bd7c3988ecb/pymomentum/torch/parameter_limits.py),
22,562B/SHA`21da39297684b8f2a27ae1b58ba9659ff478328c666e67773c05b3924e638819`,
explicitly calls its LimitErrorFunction a **soft constraint** (L17–19);
`evaluate_minmax_error` returns weighted violations (L366–386), not a projection.
These sources explain why a valid learned forward can still violate limits;
they do NOT identify the actual failed control, certify the exact TorchScript
getter semantics or authorize relaxing D96. Inspect frozen249 controls and
native metadata separately, without optimization or held-out quality reads.

### Translation-first follow-up and official validity boundary

The source-hashed public Track1 kit rebuilds pose136/scales68/shape45 through
parameter-transform/FK/skinning and rejects nonfinite inputs/geometry; no
`get_parameter_limits` check occurs in the five scorers, submission loader or
preflight. D96's dense249 hard-limit guard was an extra preregistered research
policy, not a verified official schema rule. Its failure is retained unchanged.

The native head/caller adds camera translation after decoding and YZ conversion.
Therefore external translation-only fitting is exact additive geometry with no
new native forward or reinterpretation of articulation limits. Analytic RGB
projection provides a cheap observation-only rank3 test. [SLAHMR source](https://github.com/vye16/slahmr/tree/58518fec991877bc4911e260776589185b828fe9)
initializes translation from2D evidence before orientation/pose optimization
(`base_scene.py:130–136`, `optimizers.py:25–28,329`). This supports graduated
optimization, NOT a SOTA translation-only accuracy claim. It cannot repair
wrong focal/identity/articulation; held-out metric/interaction gates are required.

Actual CPU inspection of exact696,110,248B MHR reference on2026-10-03:
`get_parameter_limits` returns a cached tensor; all37[0,0] dense rows exist in
the198-entry `character_torch.parameter_limits.minmax_parameter_index`, and
the sparse min/max values exactly match the dense rows. Missing-sparse-entry
sentinel interpretation is excluded for these rows. Combined with Momentum's
soft-limit implementation, this is metadata evidence, not proof the forward
hard-enforces these values or every learned prediction is physiologically invalid.
Immutable inspection receipt `c78b5ec438f1ff91685bd0cf53b76658d718504e4ac25e36c26597ef3d7c713a`;
zero forward/optimizer/GT, all source/input/model hashes rechecked.

### After H97: discriminate observation bias from model misspecification

Prospective, before H98 private scores. H97's translation-only RGB loss gain
and full-rank Jacobian did not prevent a32.26% median camera-PVE degradation.
H98 fixed-depth native orientation addresses one nuisance variable, not wrong
limb flexion, body proportions, detector bias or metric-depth calibration.
If it fails scientifically, prefer one untouched factorial observation diagnostic
(2 new morphologies ×2 appearances ×2 occlusion states ×3 articulation frames,
24 RGBs) over another camera optimizer. Freeze ordinary Body+DWPose first;
private evaluation only then compares projected landmark errors, centroid-Z,
centered3D errors and proportions across all8groups. No private-gradient update,
label-guided prompt selection, fitted candidate choice or challenge GT.

A concrete next mechanism, conditional on the diagnostic showing articulation
error with useful external2D evidence: prompt the existing trained SAMBody
with two automatically detected elbows, specified before observing scores.
[Native keypoint-prompt API at pinned7c0d](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/meta_arch/sam3d_body.py#L1649-L1681)
uses its original crop affine and one additional decoder pass; do not normalize
image pixels directly as crop coordinates. Decode fresh body articulation under
unchanged root/translation/Z/K, clip identity, hands and object proxy. Preserve
full-frame coverage and the same no-regression/metric/silhouette gates on a
NEW cohort, followed by independent confirmation if promising.

[SAM3D Body §6.2/AppendixB](https://arxiv.org/html/2602.15989v1) supplies source
motivation for learned prompt-conditioned correction, not proof that DWPose
prompts improve3D. Noisy prompts can worsen results; the paper's most-erroneous
landmark selection uses reference evidence and is NOT a permitted test-time
procedure here. [SLAHMR](https://arxiv.org/abs/2302.12827) motivates articulated
priors with video evidence; [CARI4D v3](https://arxiv.org/html/2512.11988v3)
motivates coupled temporal/depth/interaction constraints. GEM-X remains a
higher-cost temporal alternative until SOMA→MHR export and source eligibility
are resolved. These are hypotheses, no execution/adoption/Track1 gain claims.
