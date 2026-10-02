# Multiview research status (2026-10-02)

Native preprocessing, six-modality SS dynamics and complete single/three-view
SS+SLAT+mesh decoding executed on Azure. Full integration V2 passed78.868s,
including49.280s generation. Those inputs used disclosed procedural oracle
pointmaps/masks: **not accuracy evidence or candidate adoption**.

The original decoded proposals did not retain raw arrays, so no inversion of
GLB axes is accepted as raw-decoder parity. Source-bound exact order is
`Vglb @ A.T -> component-scale -> row-Q -> translation -> XY-flip`, with
native decoded scale explicitly isotropic. SSI/downsample are already applied.
New public object generation stores actual raw vertices/faces and verifies
native export, serialization and imported compose_transform against this order.

Completed decisive cohort: two new asymmetric closed procedural objects, six rendered
RGBs each. Private geometry/cameras/visibility remain isolated. Only RGB views
0/2/4 enter automatic color-background segmentation and pretrained MoGe2;
held-out pixels are not mounted. No GT mask, depth, scale or pose is an inference
input. Single and three-view use equal native50/25 steps and seed42. Predictions
freeze before private anchor-camera Chamfer evaluation, without GT alignment or
fitting. All four proposals passed139.118s. Camera CD median gain was-0.18784%,
not the required5%; this RGB-only fusion hypothesis is **rejected**. Large raw
camera errors combine metric depth, pose and shape; first diagnose those factors,
not retune the same fixtures. No claim about held-out pose tracking, full V2D
metrics, photorealism, embedding, or final mesh budget follows.

Completed source-backed tracker hypothesis: own OpenCV LK
forward/backward correspondences attached to predicted-mesh raster barycentrics,
PnP/RANSAC plus reserved-track reprojection. Shape/scale frozen; abstain on weak
support and retain measured baseline. A separate three-object/eight-frame RGB
cohort completed tracking22.398s and private evaluation44.716s. Camera CD median
gain0.131468%<5%: **rejected**, not adopted. Relative-motion diagnostic improves
on two objects but large inherited anchor depth/shape bias dominates absolute
reconstruction. These six wide-baseline views were not reused as LK validation.
Private truth scores all frames only after inference freezes, without alignment.
Next priority is independently validated shared human/object metric grounding,
not additional flow weights or claiming reprojection as pose accuracy.

Source: [MV-SAM3D abb04b5](https://github.com/devinli123/MV-SAM3D/tree/abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd).
SAM custom licence/competition eligibility remains unresolved. Detailed useful
results, thresholds and decisions are retained in experiments.md; heavyweight
artifacts stay exclusively Azure-resident.
