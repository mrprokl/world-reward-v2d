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

`world_reward.point_candidate_pool` now reproduces the native24+1 generation
loop as a sequential pure operator. The25th seed is the previous native greedy
winner, never point/Viterbi feedback. Exact original observation sampling,
default ICP, image/residual acceptance and tie order are tested independently.
One shared native24+1 candidate pool must be generated and frozen for both
branches. Baseline uses current `1-IoU` and Viterbi translation1/rotation0.1;
candidate adds only the point unary with identical pool/transitions. The helper
has not generated a real-video pool, run Boots or altered the challenge
pipeline. Its numerical arrays do not certify provenance or metric units.

For an attributable comparison, any canonical witness that is behind the camera
for an originally valid native candidate stops the **whole pilot**, rather than
silently pruning one branch or refitting a shared pool. Both rankings therefore
use exactly the original frozen valid-candidate mask; their only difference is
the preregistered point unary. No-visible frames keep the native zero added cost.
`world_reward.point_pose_comparison` now tests this two-ranking contract against
an independent manual unary/native Viterbi reference. It is not an executed
tracker or a validated improvement.

Initialization preflights are frozen before execution: three native MoGe2
frame-zero calls (≤300s) retain native validity/XYZ/K without a human scalar;
three fixed `object.` detections followed by SAM2's full96-frame propagation
(≤600s) use no actor/person or per-scene label prompt. These times accumulate
into the3600s total GPU budget. The proven SAM2/Grounding runtime on VM02 avoids
moving288 RGBs across VMs; only three initialization RGB/mask/XYZ/K bundles need
to reach the existing Objects runtime on VM01, and three fixed meshes return.
PNG staging preserves original bytes and indices under numeric filenames.

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
rotation errors. Reacquisition lacks a frozen definition and is explicitly
unsupported; do not invent it after reading validation errors.
Missing candidates/queries/purity or budget
failure stops the whole pilot without frame deletion, fallback or retuning.
Current native `observed<40` fails before candidates: ranking alone cannot
solve fully hidden EP4. Any common occlusion bridge is a distinct experiment.

The independently tested `world_reward.point_motion_evaluation` CPU operator
selects the sole95%-pure initial instance before selecting its GT trajectory.
All initial matched visible sensor-Z pixels contribute, minimum8; BOP uses
integer pixel coordinates (verified `misc.py` at the frozen format revision).
Both branches retain every original frame and share one initial frame gauge.
The three-distinct-sequence gate rejects zero-baseline unidentified gains and
uses only one float64 ULP on error ratios to avoid threshold cancellation.
Truth points/poses are not output, and no private evaluation has run.

The original download-only acquisition failed its900s deadline; one explicitly
bounded technical continuation is running. No native comparison has run.
Even future PASS would demonstrate
limited external rigid camera-motion gain, not articulated hand-object contact,
unseen benchmark generalization, full-HOI accuracy or victory over CARI4D.


The original CPU acquisition900s download-only timeout was independently sealed
before any ZIP/annotation interpretation. Exactly one technical continuation
keeps all source/license/archive/cohort identities fixed and changes only the
CPU acquisition deadline to3600s plus180s absolute cleanup grace. Archive the
complete original report/CID directory via Linux NOREPLACE only after original
unit/PID/cgroup/container inactivity and full disposable cleanup are proved.
Actual transition/report/source pins authorize the single continuation; occupied
archive/new-cohort namespaces forbid a third run. This is not a GPU/scientific
budget change or a validation-selected method. Compact generic stdlib atomic
primitives avoid carrying unrelated challenge controllers into acquisition.
Root134focused acquisition/transition/BOP/private-motion tests PASS2.31s plus
one Linux-only skip; scientific function AST and all non-time protocol fields
match the original frozen producer. Actual original FAIL archive and its source/report pins independently pass;
the one3600s CPU acquisition is active. No third attempt is authorized.


The external Objects/track implementation retains **raw native triangles**, not
Track1's fixed4096 budget geometry. It is a native24+1 motion baseline on the
same raw predicted mesh in both branches, not a packed Track1/CARI4D baseline.
Packing/topology/fidelity/submission eligibility remain separate adoption gates.
The pixel K is exactly diag(W,H,1) times original normalized FP32 intrinsics,
not rounded to the theoretical800 focal prior. Open/negative-volume surfaces
are audited without repair; finite nondegenerate triangles, proper transforms
and unambiguous supported rays are mandatory. All three final prediction NPZs
are rehashed after the final source/model/input postchecks, including failures.

The separately tested private BOP adapter binds each original instance by its
unique positive obj_id across all96 frames; repeated/missing IDs fail instead
of pose-error matching. Original uint16 cameraZ×declared depth_scale/1000 and
GT translation/1000 are the only millimetre-to-metre conversions. Exact all-frame
K/depth units and original initial mask order are mandatory. Full private
inventory is authenticated before any private values are decoded; all3proper
frozen prediction trajectories and automatic initial masks precede truth access.
CPU report explicitly requires a separately verified host-wrapper postreceipt;
a container PASS alone cannot authorize a scientific decision. No real private
labels or held-out3D quality has been evaluated. Root192combined Objects/track/
evaluator/BOP/private-motion tests PASS0.91s; later full-suite check follows.

Existing Objects prerequisite inventory is CPU-only: exact installed-source
fingerprints, original model/card links and a four-field image projection remain
on Azure. Raw Docker configuration is never copied to pins/submissions. Measured
installed bytes are not verified source-commit parity, model overlap clearance
or competition-license eligibility. Input and raw geometry bundles will travel
privately between Azure VMs; no heavy local transit or full runtime clone.

Minimal private initializer transport is frozen and tested:13original public
files VM02→VM01 (only RGB0/mask0/nativeNPZ0 and original metadata reports),
13raw Objects outputs/report VM01→VM02 plus6actual source/config files and
2original producer markers. Replica provenance is explicitly declared, never
reconstructed or attributed to a new execution. Fresh independently pinned
phases use only10.0.0.4→10.0.0.9:2222 with forced commands/expiring listeners.
All member hashes precede publication; originals are never merged/overwritten.
No dataset/model/private annotation payload moves locally or in this bridge.
Future actual input/report/archive pins must precede transport; no transfer or
3D inference is inferred from the172tiny contract tests (1.26s).
