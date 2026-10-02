# O2 camera-local gate and next native step

Research only. No challenge GT, multiview recording, extra weights, GPU execution
or accuracy result is supplied by these helpers. SAM custom licence remains
unresolved for competition/source-release eligibility; do not label it Apache or
submission eligible. [Paper v2, 2026-04-09](https://arxiv.org/html/2603.11633v2),
[source commit abb04b5, 2026-05-30](https://github.com/devinli123/MV-SAM3D/tree/abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd).

## Pure gate available now

`validate_camera_pointmap(CHW_XYZ, K, valid_bool, frame_index=original_index)` checks
OpenCV camera-local rays against `(column+.5,row+.5)`. It does not infer valid
pixels or establish metric depth. Invalid coordinates outside explicit support
are not filled. `to_pytorch3d_once` illustrates the single XY flip and rejects a
second conversion; **do not feed its result into MV**: the native API flips itself.
`validate_ssi_roundtrip` checks supplied native `metric = normalized*scale+shift`,
not the normalizer's estimator. `validate_joint_pixel_registration` checks shared
explicit source-pixel gathers for crop/nearest sampling only. It does not certify
native bilinear RGB interpolation, antialiasing or photometric normalization.

Frame data must have matching original grid/provenance before native crop. Supply
already human-scaled MoGe2 OpenCV pointmaps without another alpha or `Tref@inv(Ti)`.
Coordinate-only rotation/translation cannot be presented as aligned RGB. Along-
ray scale ambiguity remains even when the registration test passes. Moving-object
and virtual-camera equivalence holds for object pixels only, not person/background.

## Azure-only acquisition/bootstrap recipe (not executed)

Run acquisition through the existing approved Azure immutable acquisition path,
not inside the inference container. No source/weights transit through local RGB or
large payloads. Local-to-Azure own driver/control archive must stay <100,000 encoded
bytes; vendor source is fetched directly on Azure and is not that payload.

```bash
set -eu
PIN=abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd
DEST=/srv/scenesmith/world-reward/vendor/mv-sam3d/$PIN
test ! -e "$DEST"
mkdir -p "$DEST"
git -C "$DEST" init -q
git -C "$DEST" remote add origin https://github.com/devinli123/MV-SAM3D.git
git -C "$DEST" config core.sparseCheckout true
printf '/sam3d_objects/\n/LICENSE\n/README.md\n' > "$DEST/.git/info/sparse-checkout"
git -C "$DEST" fetch --depth=1 origin "$PIN"
git -C "$DEST" checkout --detach FETCH_HEAD
test "$(git -C "$DEST" rev-parse HEAD)" = "$PIN"
test -z "$(git -C "$DEST" status --porcelain)"
```

Record the exact commit, LICENSE hash and every imported source hash in a frozen
acquisition manifest. No `pip install`, submodule init or README setup (DA3 is not
needed). Full `sam3d_objects` source is intentional: even unweighted native Stage1
imports `multi_view_weighted` and its `latent_weighting` dependency. The four core
files total180,862 bytes; replacing them with fabricated stubs is not a native test.

Reuse the existing image **digest**, installed dependencies, acquired Objects HF
revision `2e73555018d2741ccd486e56c24fac41155a1dc6`, local config/checkpoint hashes and
DINOv2 revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Run a fresh process under
Linux `--network none`; mounts: own frozen code+vendor+weights read-only, a unique
results directory read/write. Set HF/Transformers offline flags and the same strict
local Torch Hub allowlist as `infra/object_smoke.py`. Put vendor parent first on
PYTHONPATH; assert every `sam3d_objects.__file__` resolves inside the pinned checkout,
not the installed package. Do not merge two package trees or patch live sources.

Load the already acquired SAM config; preserve all generator/embedder/decoder and
preprocessor settings and absolute local weight paths. Override only inference
class to pinned `InferencePipelinePointMap`, and `depth_model` to a fail-on-call
object; all external maps supplied. Record resolved config, condition input keys,
`normalize_pointmap` and transforms. Config/Hydra target incompatibility, missing
import/dependency or unexpected weight request is a **stop**, not a silent fallback
or new package/model installation. Constructor/API compatibility remains untested.

## First actual native gate: no decoded shape yet

Use own procedural object-camera points/masks/RGB only. Tiny crop geometry is fine;
native preprocessing still uses its configured model input resolution. Confirm:

1. `compute_pointmap` applies `(-X,-Y,Z)` exactly once (pointmap pipeline260–308),
   takes the same original grid with no implicit resize288–293 and no depth-model
   call. Stored output rays invert to supplied K; inferred intrinsics326–331 must
   not overwrite our authoritative camera. Registration is structural, not depth
   accuracy. Native clipping, invalid tokens and unsupported-point fallback are
   recorded; reject missing finite object support.
2. Observe `_process_image_mask_pointmap_mess` before/after its normalization and
   joint crop (`preprocessor.py:81–100`). Check native SSI inverse where normalization
   is enabled; when disabled, require metric points unchanged instead. Use native
   transform instances on synthetic coordinate-ID channels to audit crop/resize
   sampling and align modes separately for image/mask/pointmap. The pure nearest
   helper alone cannot validate a bilinear native transform. Confirm actual
   condition embedder consumes configured pointmap keys; default `['image']` is not
   evidence that the checkpoint uses or ignores XYZ. Stage2 inputs are image-only.
3. After embedding, freeze the same `x_t`, timestep, conditions, model eval state,
   CFG settings and CPU/CUDA RNG state. Call native Stage1 `_generate_dynamics`
   through `inject_generator_multi_view` for **one dynamics call**, not full shape
   decoding or differing integration schedules. Hash states and report tensor
   shape/dtype plus scalar differences only; preserve/restore methods in `finally`.

Native equivalence expectations:

- Unweighted identical conditions repeated three times: shape is arithmetic mean
  of the three per-view predictions; pose is first prediction only. Equal to one
  prediction only if repeated native forwards are deterministic under the actual
  eval/RNG behavior. Measure replay differences first; no bitwise promise in BF16.
- `[anchor,b,c]` versus `[anchor,c,b]`, same fixed latent/RNG: unweighted shape is
  permutation invariant mathematically, up to numerical summation differences;
  pose anchor is unchanged. Replacing anchor is **not** a pose invariance test.
  Stochastic mode is not this gate.
- Weighted fusion: verify collected conditional attention for all views, normalized
  finite weights, correctly permuted weight/view axes, and retained anchor pose.
  Duplicate conditioning gives the one-view result only when weights sum to1 and
  predictions agree. Unequal view weights are not expected to equal plain mean.
  Final decoded shape or a single-view `run` need not match a multi-view pipeline
  with different warmup, RNG, sparse-threshold/downsampling or integration paths.

Hooks at source pin: `multi_view_utils.py:94–186` contains shared latent, view slicing,
per-view dynamics and mean/weighted/POSE_KEYS rules; `inference_pipeline.py:1091–1146`
contains SS warmup, RNG save/restore, attention collection and **silent fallback to
plain mean**. Record that fallback explicitly and do not call it entropy fusion.
Stage2 collector/fallback is in `sample_slat_multi_view_weighted` and
`multi_view_weighted.py`; assert real attention before a later entropy experiment.
Pose decoding uses view0 SSI1710–1717 and scale downsample factor1733–1735 exactly
once. `all_view_states_storage` remainsNone: no native per-view pose tracker claim.

Predeclare numerical tolerances from independent deterministic replay/analytic
controls before real inputs; record absolute and relative differences and stop on
failed slicing, fusion, RNG restoration or representation—not tune to T15.

## Only after native gate: three-view proposal

Use `select_object_views` on automatic full-clip quality proxies and predicted R/t,
anchor first; its coverage abstention must block generation. Direct
`run_multi_view`: seed42, SS50/SLAT25, multidiffusion, SS entropy layer9/alpha30/
warmup1, SLAT entropy layer6/alpha30/step0/min_weight.001, no visibility callback,
physics/layout/texture baking. These are pinned CLI defaults2514–2529, not tuned
challenge thresholds. External CHW maps are already scaled; no DA3/new weights.

Compare anchor single-view and three-view proposals using reserved monocular
frames and equal pose-fitting budget. New canonical mesh cannot inherit old T15
without a verified gauge conversion. Hold scale/K fixed; no hidden-face attraction,
GT alignment or inferred-contact truth claim. Mesh validity and reserved image
evidence gates precede any adoption. No proxy result establishes superiority over
CARI4D. Five views are a later hypothesis, not an automatic expensive retry.
