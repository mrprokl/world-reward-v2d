# New analytic object-only RGBD holdout

This is a new, preregistered procedural reference, **not a rescue or rescore** of
the failed triangulated authored cohort. There are three new constant ellipsoids,
new material seeds and four uniform rigid-motion instants per shape. No challenge
assets, labels, calibration, models, meshes or human geometry are consumed.

The whole recipe is byte-pinned in the renderer before manufacture. Camera rays
are the fixed 640×480, K800, `index+0.5` convention. FP64 stable quadratic roots
select the first positive ellipsoid hit before a finite physical textured plane.
RGB, visibility and camera-Z use the **same analytic hit**; RGB uses fixed seeded
object/plane sinusoidal materials and ambient-plus-Lambert lighting.

Before any public manifest: positive convex geometry, conservative whole-solid
frustum bounds, positive complete depth, nearest entering root, independent
completed-square silhouette/root and implicit-residual checks, and nine fixed
70-digit Decimal probes per frame must pass. The normalized discriminant margin
must exceed `1e-12`; ambiguity stops the entire cohort rather than dropping pixels.
FP32 camera-Z cast error must be at most `1e-6` metres. Per-scene geometry hashes
must remain constant. These bounds are mathematical contracts, not model-score
selection criteria. No image preview or depth prediction selects any recipe.

Manufacture is CPU-only on the sealed Azure image, network disabled, with 360 s
inner / 363 s outer budgets and an owned uniquely named container. Only the new
output namespace is writable; failure removes its arrays/PNGs and retains a
read-only failure receipt. No local images, heavy arrays or checkpoints are made.

Public exposure is exactly twelve PNGs and an RGB-only manifest, under
`validation/analytic_rgbd_holdout_v1/inputs`. Private truth contains twelve camera
depth/visibility/pose NPZs and three constant geometry/physical-plane NPZs, with no
face indices. Separate blind inference must receive only public inputs, produce
and independently freeze all predictions before private quality evaluation.

The downstream method/gate is unchanged: fixed 10% peripheral background proxy,
at least 1,024 paired pixels and 95% border coverage, one median-of-four scene
anchor, unchanged MoGe validity, no offset/alignment/per-frame correction. Private
evaluation uses up to 8,192 visible-object samples, seed0 without replacement,
95% object coverage, median relative camera-CD gain ≥5% and no scene regression
>5%. Passing would establish **new synthetic object camera-depth transfer only**,
not photorealism, real-data transfer, human/contact/HOI quality, licensing/overlap
clearance or a verified victory over CARI4D. No manufacture/quality run is claimed
until an authoritative Azure receipt exists.
