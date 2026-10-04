# Automatic point-association pose pilot — preregistered, not executed

BootsTAPIR passed the three-video RoboTAP 2D diagnostic, but its initial queries
were annotated oracles and its static negative control was not a motion method.
That result does not authorize a Track1 pose replacement. The next hypothesis
is that automatic RGB correspondences improve selection of rigid SE(3) candidates
relative to our native dynamic baseline. No model-training overlap is verified.

## Fixed operator

`world_reward.point_pose_cost` fixes a16×12 centre grid, first32 admissible points
in raster order, minimum8. Automatic object mask, valid inferred depth and the
first visible ray hit on the clip-constant predicted surface must all support
each query. Queries use original continuous pixel centres; no replacement after
tracking, manual prompt, GT mesh, calibration or annotated initial point.

The caller raycasts the reconstructed surface under the native initial pose to
obtain one canonical point per query, once per clip. Boots keeps its native256
tracks/visibility and all original frames. Added unary cost is exactly
`0.1 * mean(min(1, reprojection_error_256 / 8))` over natively visible queries.
No visible query gives zero with an explicit missing-evidence flag, not a static
track or invisible3D truth. Any behind-camera witness invalidates its candidate;
an entire frame without an admissible candidate fails. Geometry, K and scale
remain constant; all candidates must be finite proper SE(3).

The independently tested `world_reward.point_surface_queries` helper now
raycasts all original triangles at192 fixed pixel centres, selecting the first
32 supported hits, minimum8. It retains original face barycentrics and rejects
ambiguous supported rays explicitly; it does not repair/clip geometry or infer
surface correctness. No real-video raycast or model integration has run.

One shared native24+1 candidate pool must be generated and frozen for both
branches. Baseline uses current `1-IoU` and Viterbi translation1/rotation0.1;
candidate adds only the point unary with identical pool/transitions. The helper
does not yet generate candidates, run Boots or alter the challenge
pipeline. Its numerical arrays do not certify provenance or metric units.

## Proposed independent real-data gate

Before acquisition or inference, freeze exact source revisions/licenses/bytes
for the **full contiguous YCB-Video/BOP test** under publisher MIT terms. Do not
substitute the sparse BOP19 archive. Select the first three scene directories
lexicographically and first96 contiguous RGB filenames, before reading labels.
All288 RGB, automatic masks, MoGe and reconstructed clip geometry stay Azure;
the predictor receives no sensor depth, private K, object meshes or poses.
Acquisition CPU is separate; total GPU budget3600s, one frozen experiment only.
Neither an already exposed RGBD cohort nor a rejected hand pilot is a holdout.

The frozen `configs/ycbv_point_protocol.json` binds full-test and base archives,
publisher MIT README/license, embedded dataset license and BOP format revisions.
The CPU-only acquisition verifies complete archive SHA/layout/CRC before
retaining288 untouched RGB images. All instance masks, sensor depth and camera/
pose metadata are retained separately under evaluator-only private parents;
none are prediction inputs. Disposable source archives are removed only after
the retained-byte proof. No acquisition execution is inferred from these tests.

Freeze both full trajectories before CPU-only private evaluation. Match the
initial automatic mask to the sole GT object with at least95% mask purity using
the preregistered overlap rule, never whichever yields the best pose error.
Primary error is material-point displacement under `Tpred(t)*inverse(Tpred(0))`
versus `TGT(t)*inverse(TGT(0))`, for every original frame. Initial truth points
are evaluator-only. This measures relative camera-frame motion without an
optimized alignment, not absolute CD, shape or global-scale accuracy.

Accept only median of three mean-error gains≥10%, no scene regression>5%,
100% trajectory coverage, proper rotations, and report absolute displacement,
rotation and reacquisition errors. Missing candidates/queries/purity or budget
failure stops the whole pilot without frame deletion, fallback or retuning.
Current native `observed<40` fails before candidates: ranking alone cannot
solve fully hidden EP4. Any common occlusion bridge is a distinct experiment.

No acquisition/native comparison has run. Even future PASS would demonstrate
limited external rigid camera-motion gain, not articulated hand-object contact,
unseen benchmark generalization, full-HOI accuracy or victory over CARI4D.
