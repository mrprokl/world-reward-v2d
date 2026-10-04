# World Reward — V2D Track 1

Goal: a correct monocular human-object reconstruction submission that improves
on CARI4D, **not an evaluation exploit**. Track1 RGB/public metadata only: no
Track2/3 assets, challenge multiview, hidden meshes/trajectories, source camera
calibration or sequence-matched FORM-HOI. All masks/identity/geometry must be
inferred automatically; no hand-labeling of challenge records.

## Status — 2026-10-04

**No final Parquet, submission or verified CARI4D improvement yet.**

- **Engineering:** all30 original videos pass byte/metadata readiness. Episodes
  **1,2,3,5,6,8,15** pass full native shared preparation, forward,301-update
  refinement, direct export and original official packing. These are seven
  complete engineering checks, not held-out accuracy; scratch Parquets deleted.
  EP0's legacy conversion failure stays separate. EP4 missing masks, EP7 empty
  anchor, EP9 invalid geometry and EP10/EP11 actor identity failures remain closed.
  EP11 stops before SAM2/later frontends. EP0/full790 and EP12/full405 now pass
  native301-update refinement and direct export; EP0 official packing also passes,
  EP12 packing dispatched only. EP0's
  exact historical metadata omission is authenticated, not a relaxed legacy
  gate or a reversal of its old conversion failure. EP13 full425 inputs and
  shared preparation and full native forward pass; collected-predecessor scheduling failures remain
  separate. Original controls/reference replay and whole source-chain pins are
  frozen before each next stage; no prediction is inferred from dispatch ACKs.
- **Depth:** D106 gains56.2007% on twelve external TUD-L frames, but the same
  development scenes/objects and33–38cm absolute errors prevent a generalization
  claim. D107's independent TUM test **REJECTS** the recipe:4.9046% < frozen5%.
  No threshold/scene rescue or adoption.
- **Hands/temporal research:** the four-anchor learned RGB pilot passes frontend
  execution but **fails hand-support QA** despite passing silhouette IoUs;
  no192-frame extension or occlusion-bridge adoption. RoboTAP2D execution passes,
  but oracle initial queries, static negative control and unknown overlap make
  it neither automatic3D validation nor a CARI4D comparison.
- **YCBv2:** corrected acquisition and native initial depth pass; automatic object
  identity abstains on ambiguous detections (two detector calls, zeroSAM2).
  No private3D values/projections, Objects trajectory or quality evaluation;
  cohort closed without prompt/margin retuning.
- **Framework foundation:** `shared_scene` provides readonly byte-exact adapters
  from existing Track1Episode/Reconstruction, with original indices, explicit
  camera/gauge, constant geometry/identity and missing-observation support.
  `temporal_identity` provides raw globally associated path costs over supplied
  actor/object hypotheses. It does not infer/calibrate observations or accept an
  identity. `relational_motion` extracts background-compensated 2D movement
  features from supplied automatic tracks, with raw support/degeneracy diagnostics.
  `automatic_candidate_bank` batches all retained automatic hand/object proposals
  through model callbacks, preserving empty masks and outside-union background.
  These are tiny-tested primitives, **not an operational new framework,
  integrated learned method or measured improvement**.
- **Automatic identity micro-test:** twelve external DexYCB clips/872original
  RGB frames acquired only onAzure, with opaque annotations separated. The real
  generic hand-grounding bank **REJECTS** atclip0 in13.13s (no detected hand).
  No tracking or private annotation values read, no prompt/threshold rescue.
  Specialist [hand methods](docs/hand_specialists_audit.md) and
  [MediaPipe](docs/mediapipe_hands_audit.md) have independent feasibility audits;
  neither is adopted or presumed robust to object occlusion.
- **Fresh hand diagnostic:** three distinct Dex03 full72-frame clips are sealed
  before evaluation. CPU scan216 calls/9.31s, private2D diagnostic3.70s:
  171/177 annotated-positive frames uniquely associated, six misses, conditional
  17-joint EPE9.65px (2907/3009 valid joints scored). Weak bbox association,
  39 unlabelled frames and unknown pretraining overlap prevent broader claims.
  No prompt/threshold tuning, contact/identity/3D proof or challenge adoption.

Detailed receipts, producing revisions and decisions live in
[experiments](docs/experiments.md). [Compute](docs/compute.md) is the compact
operational snapshot; do not treat stale status prose as a live job monitor.
See [architecture](docs/framework_architecture.md), [constraints/audit](docs/audit.md),
[literature](docs/literature.md), [4DAnyone audit](docs/4danyone_audit.md),
[baseline](docs/baseline.md), [point-pose protocol](docs/point_pose_protocol.md),
[identity association](docs/identity_association_protocol.md) and
[licenses](docs/licenses.md). Proxy/packing/test PASS never establishes victory.

## Repository and data flow

- `src/world_reward/`: reusable numerical contracts, scene adapters and research
  operators; model/GT/provenance verification is not implied by array validation.
- `infra/`: isolated Azure executors and existing native adapters; migrate
  incrementally rather than mutate historical producer code.
- `configs/`: frozen protocols/source/model/artifact pins, not per-episode labels
  or quality-selected constants. `configs/sources.json` pins official inputs.
- `tests/`: tiny procedural correctness/equivalence fixtures, no heavy media.
- `docs/`: concise research contracts, sources, results and decisions.

Heavy data, checkpoints, renders and inference/training stay **on Azure**. Only
code, small reproducibility pins and useful summaries stay here. Human and object
share one camera frame; shape/scales/geometry are clip-constant and trajectories
retain every original frame, including occlusion. No deletion/shrink/static
trajectory or per-frame alignment to evade metrics.

## Development

```sh
rtk uv sync --extra dev
rtk uv run pytest -q --basetemp=/tmp/world-reward-tests-unique-run
```

Choose a different exclusive basetemp for concurrent runs. Optional official-kit
parity uses `WR_KIT_ROOT=/path/to/v2d_submission_kit`; template I/O reads row IDs
alone, never sample prediction values. Tiny tests are not accuracy validation.

Download/setup **on remote Linux only**, never the tethered laptop:

```sh
uv sync --extra data --extra dev
uv run wr-data --config configs/sources.json --root /data/world-reward/data \
  --manifest /data/world-reward/results/input-manifest.json
```

Model access, actual image/source/assets and protocol gates must pass before
inference. Use immutable committed code, fresh output namespaces, bounded stages
and the existing GPU scheduler; secrets never enter version control or logs.

## Submission and eligibility

Own code is Apache-2.0; external source/models/data retain their separate terms.
**Training overlap and source/checkpoint eligibility remain unverified** where
[licenses](docs/licenses.md) says so. Never equate granted model access with licence
clearance or claim leakage-free weights without evidence.

Use the original pinned official packer/uploader after numerical/geometry,
full-trajectory, quality and rules checks. One frozen Parquet goes to all five
competitions under **World Reward**; Thomas Gomez only where registration needs
an individual name. Verify accessible exact producing GitHub commit and weekly
quota before upload. Kaggle team name and all five rule acceptances were verified
on2026-10-02; recheck submission requirements before uploading, without duplicating
registration. NVIDIA's separate registration remains to verify.

Credentials use ignored secret storage, never printed. Own-code history is public
at [mrprokl/world-reward-v2d](https://github.com/mrprokl/world-reward-v2d). No upload
or licence waiver is implied by repository visibility or current engineering PASS.
