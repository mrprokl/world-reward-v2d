# End-to-end external evaluation — 2026-10-10

## Decision

Use the **already frozen, never evaluated FORM-HOI DEV/RESERVED cohort** for
real-video full human/object reference evaluation. The assets are available,
but not yet acquired/qualified: do not confuse this with absence of GT or with
an inference-ready benchmark. References are **multiview reconstructed pseudo-GT**,
not verified motion-capture truth. Prefer this direct domain match over inventing
another synthetic benchmark. Keep existing synthetic controls as diagnostics.

### Exact frozen split

`configs/form_hoi_insight_v1.json`, dataset revision
`c63db107e84c7f74bb4929ef643b67b5c8bcc00e`, CC-BY-4.0:

| DEV (first four; 3,261,132,800 archive bytes total) | RESERVED (unopened) |
|---|---|
| `2026-06-03_17-02-20_beige_bin_ground_desk_03` | `2026-06-08_15-00-38_snack_box_squat_05` |
| `2026-05-22_16-37-56_rolling_luggage_pick_and_place_03` | `2026-03-17_15-30-58_monitor_slide_away_03` |
| `2026-03-20_11-52-56_cyan_water_bottle_roll_ground_01` | `2026-04-29_10-48-10_woven_basket_right_foot_push_08` |
| `2026-03-17_16-56-57_soup_can_flip_hand_04` | `2026-05-21_15-45-47_guitar_floor_raise_01` |

All IDs, archive sizes/SHA256 and selection rule are pinned in that config.
Current split is **sequence/provisional filename-family disjoint**, not verified
object/person held-out. Resolve authoritative IDs before claiming object/person
generalization; do not silently reassign cases after seeing predictions.
All seven admission gates are currently false, intentionally.

## Shortest runnable acquisition

New CPU-only wrapper `infra/run_form_hoi_external_acquire.sh inventory_first`:

- On Azure (VM02 is sufficient), download **only first DEV archive** (1,093,089,280
  bytes), verify publisher SHA256, inventory bounded tar headers and native
  metadata keys/safe text/IDs. No mask, pose, depth, mesh or calibration decoding.
- Output `/srv/world-reward-data/form_hoi_external_v1/inventory_first-<REV>/`.
  Each sequence gets an immutable receipt; top-level report includes member
  names and native metadata schema. The authenticated archive is kept for
  subsequent extraction, avoiding another 1 GB download.
- Source entry `run_form_hoi_external_acquire`; immutable source path
  `/srv/scenesmith/world-reward/jobs/<REV>/run_form_hoi_external_acquire/code`.
  600 s inclusive / 660 s wrapper cap, 2 GiB virtual-memory cap, no GPU.
- After layout/text qualification, acquire **four DEV only** and retain exactly
  front-left RGB plus sanitized native `object.prompt`/action in predictor
  inputs. Never invent labels from filenames or use reference-derived boxes.
  Reference masks/MHR/poses/mesh/EDEX/QC stay in evaluator-only quarantine.
  Delete original archives only after useful extracted bytes are verified.
- RESERVED archives stay entirely untouched until algorithm, hyperparameters,
  candidate/baseline code and evaluation protocol are frozen on DEV.

This first job is acquisition/schema qualification, **not** an accuracy test.
It must lead directly to the external input adapter and paired inference, not
another indefinite metadata audit. No remote job was launched in this audit.

## Qualification and end-to-end comparison

1. Reuse `world_reward.form_hoi_protocol` for exact-ID/recording/object-slug
   exclusion and public-input allowlist. Published metadata audit has zero
   intersections with all 30 Track1 records; this is **not a content/alias proof**.
   Before inference, compare eligible RGB hashes/content fingerprints against
   existing permitted Track1 RGB on Azure. Fail if an unresolved recording alias
   exists. No matched FORM-HOI assets or challenge multiview are legitimate.
2. Source camera is exactly `front_stereo_camera_left`, no substitution. Validate
   fps, length and original frame grid by RGB-only probe. Read native prompt/action
   from dataset metadata only. Ground plane, source K/extrinsics, reference masks,
   trajectories/meshes, depth and QC are forbidden predictor inputs.
3. The old frozen grid `0,14,29` tests **localization**, not continuous motion.
   Preserve that study. Before full 4D evaluation, freeze a separate full-T or
   contiguous prefix protocol (e.g. first 96 original frames, declared before
   prediction; all 96 or abstain). No acceleration metric on sparse frames.
4. Run unchanged automatic baseline and candidate joint pipeline on identical
   public packages, original indices, model/mesh assets and clip budget. Seal
   predictions/config/input/source hashes **before** mounting private references.
   Candidate may change general joint refinement, not object size/topology to
   game PEN or individual sequence labels. Record all failures; no best-case
   replacement or denominator filtering.
5. Evaluator-only schema qualification: MHR parameters `mhr_params_mv.pt`, rigid
   object `poses.npy`, metric `output_aligned.glb`, EDEX camera frame and QC.
   Validate the native MHR/world-to-selected-camera transform by reference-only
   rendering/roundtrip before scoring. Do not assume our CARI camera convention
   is identical. Use existing pinned official Track1 metric operators on full
   paired geometry after conversion, not a new approximate depth-Chamfer score.
6. Report every sequence: CD-H, CD-O, ACC-H, ACC-O, PEN; raw shared-camera
   geometry/relative human-object distance diagnostics; contact gaps and image
   consistency; actual latency, missing rate. Publish all frames and the same
   predeclared QC-unflagged subset with exclusion counts/reasons. QC flags are
   fallible; never use them to tune inference or manufacture contacts.

**Adoption:** paired DEV gains require geometry/true-motion nonregression, not
smoothness alone. Freeze any numeric tolerance before opening references.
Use RESERVED once after DEV selection; then it is no longer reusable as an
unseen test. FORM cannot establish a leaderboard victory or model-training
independence: CARI4D is explicitly trained on FORM-like data and checkpoint
overlap remains unverified. Report this limitation rather than claiming clean
pretrained holdout performance.

## Track1 metric caveats (pinned kit audit, not paper metric)

`docs/baseline.md` and `docs/audit.md` already audit the official operators:

- One **first-scored-frame human-only Sim3**, applied unchanged to human AND
  object over the clip. No per-frame alignment; initial relative-object error
  counts. Also retain unaligned shared-camera errors as diagnostics.
- CD is the **sum** of directed surface means, centimetres, not half/squared
  depth-point Chamfer. ACC is second-difference **error against reference** on
  contiguous scored stretches, not zero-motion reward or an fps² quantity.
- ACC-H uses 22 joints; ACC-O object centroid. PEN averages 512 sampled hand
  points including outside=0 against the submitted object mesh. Original kit
  body64/hand512 sampling and episode/dataset weighting must remain intact.
- Packing requires full original trajectories, shared gauge, clip-constant
  shape/scales/object surface, valid poses and <=4096 mesh rows per budget.
  A schema-passing Parquet is not reconstruction-quality validation. Re-audit
  the **updated** kit explicitly before submission; do not silently swap the old
  pinned scoring/packing runtime during this experiment.

## Other existing banks: useful, but not replacements

| Existing bank | What is actually referenced | Honest role now |
|---|---|---|
| authored/analytic RGBD, 3 objects x 4 frames | synthetic object/depth/camera, no human | depth/shape DEV only |
| `validation/object_motion_v1`, 3 objects x 8 frames | synthetic rigid geometry/pose, no human | object-motion DEV |
| `validation/joint_rgb_v1`, 3 x 3; `validation/root5_rgb_v1`, 3 x 5 | authored MHR human+bottle geometry | already evaluated synthetic joint DEV, not unseen held-out |
| TUM RGBD, frozen single-view frames | real registered sensor depth, no complete human/object surface GT | depth-only; prior study already closed REJECT |
| HO-Cap subject5, clips `20231027_112303`/`20231027_113202`, camera `105322251564`, 1493 frames | real segmentation; hand/object reference schema possible but not yet qualified by our evaluator | already used semantic-point DEV; full-body reference/contact GT explicitly unavailable |

Reusing the existing evaluators for those scopes is preferable to another generic
"end-to-end" evaluator that would silently compare incompatible metrics. No new
accuracy evaluator is implemented before the actual FORM reference schema is
known. This avoids synthetic truth being relabelled as real full-HOI ground truth.

## Sources and checks

- [Pinned FORM card](https://huggingface.co/datasets/nvidia/form-hoi/raw/c63db107e84c7f74bb4929ef643b67b5c8bcc00e/README.md): 4772 bytes,
  SHA256 `5463a86d774c9f995f68d0613f7916d13c72f43acc212a8aa5f5a3dac7f233b5`,
  revalidated from primary source in memory on 2026-10-10; commercial/noncommercial
  CC-BY4, 4135 cleaned sequences, MHR/object metric reconstructed annotations.
- Pinned native pipeline guide in `configs/form_hoi_insight_v1.json`: documents
  `hoi_metadata.yaml` -> `object.prompt`, ignores manual `object.bbox`.
- `results/audits/form_hoi_insight_v1_metadata.json`, `docs/track1-inputs-2026-10-08.md`:
  exact cohort/admission limitations; no past FORM image evaluation.
- New acquisition + existing admission tests: **43 passed, 0.13 s**;
  wrapper `bash -n` passed. These validate control flow/safety, not external accuracy.
