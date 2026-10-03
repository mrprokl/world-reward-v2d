# Authored RGBD transfer holdout v1 — frozen before manufacture

This is a **new synthetic, object-only camera-depth transfer test**, not a
rescore of D88, real-data generalization, photorealism, human–object or temporal
accuracy, a CARI4D victory, or pretrained-model license/overlap clearance.
There are no downloaded shapes, challenge inputs, human fixtures, or old failed
synthetic branches. Twelve RGB images are produced once on Azure only.

The complete recipe is `configs/authored_rgbd_protocol.json`, SHA-256
`5d53806c3737aa1b15288628ba2fc364bbeb9dc95731075c2f3821ef5a2c053e`.
The renderer verifies this hash **before JSON parsing and after manufacture**.
The root launcher freezes/publishes the producer commit before GPU execution.
Changing the recipe after viewing model outputs would require a separately
declared experiment; it cannot rescue this cohort.

## Manufacture and independent gates

- Three distinct ellipsoids, semiaxes in metres: `(0.25, 0.18, 0.22)`,
  `(0.19, 0.30, 0.16)`, `(0.32, 0.15, 0.24)`. Each is a positive affine image of
  an inscribed 24-latitude/48-longitude sphere triangulation: 1,106 vertices and
  2,208 triangles. Shape and scale are fixed throughout each scene.
- Four uniformly spaced rigid-motion instants, `t = 0, 1/3, 2/3, 1`, use a
  fixed rotation axis and formulaic moving translation. No individual fitting.
- Native 640×480 RGB, centred square focal 800, OpenCV metre camera frame,
  pixel rays through `(x+0.5, y+0.5)`. No resize or alignment.
- Seeded PCG64 continuous object-space bands, continuous physical background
  plane texture, and one fixed Lambertian point light per scene are evaluated
  at perspective-correct surface points. The background is a finite-depth
  plane, not a constant RGB fill with undefined depth.
- Every mesh must have active finite vertices, nondegenerate triangles,
  two-face edges, consistent winding, Euler characteristic 2, positive signed
  volume and outward convex support planes. No flip, repair, union or deletion.
- Before any public manifest, every frame must contain the whole object and
  valid positive finite FP32 camera-Z over **all 307,200 pixels**.
- Independent NumPy float64 centre-ray/triangle-plane intersections verify
  every raster depth to ≤`2e-5 m`; reference barycentrics must be ≥`-2e-4`.
  All object selections must be entering/front-facing. For the independently
  verified convex solid this selects the first surface, not its rear face.
- An independently computed projected convex hull rejects missed object
  pixels and extra object pixels. Only the preregistered `2e-4 pixel` boundary
  tie band is uncertain; all depth/face checks still apply there. The visible
  object must occupy at least 1,024 pixels in each frame.

The renderer has a hard 360-second budget and requires actual Linux/CUDA,
network `none`, Torch `2.5.1+cu124`, PyTorch3D `0.7.9`, NumPy `1.26.3`, and image
`sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3`.
It independently binds its source, wrapper, and protocol to the exact immutable
producer namespace. Failure removes partial arrays/RGB, retains one failure
receipt, and does not publish a manifest. No runtime fallback or rescue.

## Paths and interfaces

Container invocation: `python CODE/infra/authored_rgbd_render.py --protocol
CODE/configs/authored_rgbd_protocol.json`. The root-owned wrapper supplies
`WR_ROOT`, `WR_CODE_REVISION`, actual `WR_IMAGE_ID`, and
`WR_RENDER_OUTPUT_RESERVED=1`; code must be
`ROOT/jobs/REV/run_authored_rgbd_render/code`. The reserved destination must
already be a new empty directory, never a resumed or overwritten cohort.

All outputs remain under Azure
`ROOT/validation/authored_rgbd_holdout_v1`:

- **`inputs/`**: exactly 12 read-only (0444) PNG files named
  `scene_000001_frame_000000.png` through `scene_000003_frame_000003.png`, and
  `manifest.json` (0444). The manifest has exactly `schema` and `images`;
  schema `world_reward.authored_rgbd_public.v1`; each ordered entry has exactly
  `scene_id`, `frame_id`, `file`, `sha256`, `width`, `height`. Public data has no
  shape, pose, depth, visibility, camera intrinsics or private receipt.
- **`eval_private/`** (0700): 3 fixed `scene_00000N_mesh.npz` files and 12
  `scene_00000N_frame_00000F.npz` files (all 0400), plus
  `render-report.json` (0400). Frame NPZ keys: `depth` FP32 `(480,640)`,
  `visible` bool `(480,640)`, `K` FP64 `(3,3)`, scalar int64 `scene_id` and
  `frame_id`, FP64 `camera_R` and `camera_t`, and int64 `face_indices`.
  Mesh keys: `vertices`, `faces`, `semiaxes`, `material_seed`,
  `background_vertices`, `background_faces`.

The report schema is `world_reward.authored_rgbd_manufacture.v1`, stage
`authored_object_rgbd_manufacture`, `status=pass` / `phase=complete` only when
all gates pass. It records `producer_revision`, `image_id`, `script_sha256`,
`protocol_identity`, actual source/wrapper identities in `source_helpers`,
runtime versions, scalar per-frame checks, and exact `public_files` (13) /
`private_files` (15) basename → `{bytes,sha256}` maps. The report itself is not
self-hashed. Prediction input pins carry its separately computed identity;
the private evaluator verifies the report and inventories before opening truth.
Private paths and recipe are **never mounted into the blind predictor**.

## Unchanged hypothesis and held-out score

Freeze/hash public inputs and predictions before private evaluation. The
method chooses its own fixed K800 from the prior method contract, not private
calibration. Its border is `x<64 or x>=576 or y<48 or y>=432`, a background
**proxy**, not a claimed semantic exclusion. Every image needs ≥1,024 valid
pairs and ≥95% border coverage. One scene coefficient is the median of the
four image-median native MoGe-Z/native DA3-Z ratios. No offset, frame coefficient,
alignment or validity changes; candidate validity equals MoGe exactly.

Evaluation uses all selected objects' visible pixel union and the same +0.5
camera rays. Sample all visible pixels up to 8,192; above that, use sorted
PCG64 seed-0 choice without replacement. Require at least 32 visible truth
pixels and ≥95% baseline/candidate coverage in each frame. Compute paired
camera Chamfer-half centimetres, average the four frames per scene, then
relative scene gain. **Accept only median gain ≥5% and no scene regression
>5%**. Synthetic acceptance does not authorize production replacement or
claim cross-dataset real-world generalization.

## Verification at source creation

Thirty tiny NumPy-only tests pass (`0.19 s`): exact full-recipe hash, three
closed convex fixtures, unchanged rigid geometry/camera margins, invalid
mesh rejection, independent centre-ray depth versus Euclidean distance,
wrong/back/missing/outside face rejection, independent silhouette missing/extra
pixel rejection, and exact public manifest completeness/privacy. These are
code contracts, **not actual CUDA manufacture or model-performance evidence**.
