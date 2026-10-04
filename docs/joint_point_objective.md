# Persistent points in the native joint objective — limited runtime qualification

`world_reward.joint_point_objective` adds one continuous object-point term to
the existing joint HOI objective. It does not select poses, track images, estimate
identity, infer metric scale, recover geometry, or establish reconstruction gain.
The unchanged historical native/default producers remain separate.

## Frozen observations and equation

`FixedTriangleTracks` owns immutable copies of the **actual native F32 loaded
mesh**, original triangle IDs/barycentrics, RGB-derived OpenCV K, original image
dimensions, positional `arange(T)`, distinct native source frame IDs, and every
ordered native bundle frame name/automatic query/XY256 track/visibility slot. Attachments must be
selected before observing future tracks. Source references are required metadata;
the pure module cannot certify their provenance or the tracker checkpoint rights.
No future reselection, refill, reweighting by confidence, clipping, or identities
are introduced. Coordinates outside the native grid reject input, including
unsupported coordinates; nonfinite placeholders are not converted to evidence.

`canonical_surface_queries` now retains the original selected F64 barycentrics
alongside its five unchanged outputs. `joint_point_evidence.bind_joint_point_evidence`
binds these exact first-frame attachments to existing `PointTrackEvidence` and
the actual native F32 loaded mesh. No barycentric recomputation or approximate
frame matching is permitted. A selector run on a different pre-alignment mesh
must generate fresh attachments **before** tracking, not refit them afterward.

For a fixed triangle attachment `p_q = sum_i bary[q,i] * V[F[q,i]]`,
`u_hat = [256/W,256/H] * pi_K(R_t p_q + t_t)` and `r = (u_hat-u)/s`.
The robust isotropic term is `rho(r)=sqrt(1+||r||²)-1`, computed in a stable
rationalized form. `L_points = weight * mean_q(mean_supported_t rho(r_tq))`.
Scale `s` and weight have **no defaults**: an explicit calibration reference is
required, but its actual independent receipt must be authenticated by the caller.
This smooth loss is not the old hard-saturated NumPy candidate-pool unary.

Forced time0 observations are excluded. Every original frame remains present;
occluded frames contribute no invented residual. Each track needs at least one
post-initializer observation for positive-weight fitting; a zero-support track
rejects the experiment rather than silently changing the track population.
All projections must remain finite with positive Z, including occluded tracks.
Projected geometry may leave the image and incurs the same residual; it is not
dropped. Weight0 returns the original native total/metrics objects exactly.

## Executable native extension and scope

`run_joint_point_refinement(native, bundle, V, F, cfg, mhr_layer=...,
evidence=..., point_config=...)` constructs an explicit subclass of the original
`MHRParityPostOptimizer`. It calls `super().loss(...)` and adds the point term;
the original `run`, Adam groups, scheduler, 301 updates, contact activation,
human decoder, penetration, priors and output creation remain inherited.
No native source, globals, classes, geometry or camera are monkeypatched.

The full native source is bound to NVIDIA V2D
`7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`,
`learning/training/mhr_opt_refineout.py`, SHA256
`84e0e818a3bc0935bb30b75fcd82fd7c5e3730ed812864594cd759697ddb406b`.
Every live class method's bytecode is compared with that authenticated source.
This is source compatibility, not authentication of all transitive dependencies.
Caller still must verify helper/model/config/input receipts and the execution
image, then supply the actual imported module. No source import is fabricated.

Only explicit full-batch/full-T config, fixed object rotation and fixed internal
body translations are supported. Native body rotational controls and object
translation remain the only optimized parameters. Human changes are coupled
through the existing contact/penetration objective; no hand/body observation is
invented. Rotation unlocking is a separate confound and is not implemented here.
Output records the new objective and must not pass as unchanged-baseline parity.

## Before any experiment

Local tiny tests cover metadata, immutable identity and manufactured native
source contracts. Ten actual Torch2.5.1 controls pass on Azure, including
translation gradients/gradcheck, equal-track occlusion, initializer exclusion,
invalid geometry/timeline/support rejection and unit similarity. The original
source-bound class's object state, inherited loss and301-update loop also pass
with manufactured state and a fake body callback; zero-weight output is exactly
the baseline and synthetic point residual decreases. Constructor, actual MHR,
contact/render kernels and upstream scheduler construction are **not qualified**.
No installation/GPU/model/private inputs enter these controls. A licensed
calibration/held-out cohort and real-model integration remain prerequisites.
No production deployment, probability calibration, HOI/3D gain or adoption is claimed.
Compare same-input native objective versus native-plus-points with rotation fixed
in **both** arms, all full trajectories and unchanged shape/gauge. Require actual
3D object/relative-HOI benefit without human, penetration, acceleration,
wrong-identity or coverage regressions. RGB residual improvement alone is not
independent 3D validation. Full-HOI reference rights remain unresolved.

## Actual native qualification protocol

The real-model pair binds the full563-frame EP21 input/forward receipts, original
MHR decoder/contact assets and optimizer source. A is the original class; B is
the explicit zero-weight subclass. Both reset Python/NumPy/Torch/CUDA seed0,
construct the real native optimizer/scheduler, probe original losses/gradients at
steps0 and181, then inherit all301 updates. Initial states/probes and complete
results/history must be byte-exact; only B's declared point metadata is excluded.
No tolerance or replacement renderer is allowed.

First-frame attachments use the original **full-resolution automatic PNG** bound
through the frozen object-pose receipt, inferred aligned depth/K, actual native
F32 mesh and A's constructed object pose. Future support is entirely false:
this is a numerical dead-branch control, **not tracker predictions**. Scale1 and
weight0 are runtime-only values, not positive-weight calibration.

The first launch failed before any container/model/update because systemd
expanded a controller FD9 path containing `$$`. Its original failure is retained.
A scheduling-only replay uses `/proc/self/fd/9`, the **same frozen9d4badf driver**
and a fresh unit/log. That same frozen replay then **failed** in the original
constructor/query-attachment phase after22.004624s: fewer than8 fixed automatic
queries, with no refill or fallback. No loss probe, optimizer run,301-update
pair or positive-weight term executed. The immutable8727B report is SHA256
`92a0286d43479ef5fe5685916f366cf75c7199d99706ba4684dab2e1ba7cb450`.
Independent source/input/mask/runtime posthash and exact owned-CID absence checks
passed. The later strict cleanup fix does not mutate that historical job.

This closes that qualification version, not a successful native parity claim.
Sparse-grid availability versus shape/pose/K/mask disagreement is not yet
identified. A [new mask-conditioned operator protocol](mask_query_quantile_protocol.md)
uses fresh analytic fixtures; it does not rerun EP21 or relax the old min8 gate.
Actual paired optimizer qualification, licensed calibration and positive-weight
held-out accuracy remain separate prerequisites.
