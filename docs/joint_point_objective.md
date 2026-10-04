# Persistent points in the native joint objective — prepared, not qualified

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
source contracts. Real Torch gradient/gradcheck tests run only when Torch is
already installed; no installation/model/GPU/private data is part of this work.
An actual source-bound native CPU/autograd compatibility gate and independently
licensed calibration/held-out cohort are prerequisites. No adapter deployment,
runtime qualification, probability calibration, HOI/3D gain or adoption is claimed.
Compare same-input native objective versus native-plus-points with rotation fixed
in **both** arms, all full trajectories and unchanged shape/gauge. Require actual
3D object/relative-HOI benefit without human, penetration, acceleration,
wrong-identity or coverage regressions. RGB residual improvement alone is not
independent 3D validation. Full-HOI reference rights remain unresolved.
