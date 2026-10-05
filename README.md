# World Reward — V2D Track 1

Goal: a correct monocular human-object reconstruction submission that improves
on CARI4D, **not an evaluation exploit**. Track1 RGB/public metadata only: no
Track2/3 assets, challenge multiview, hidden meshes/trajectories, source camera
calibration or sequence-matched FORM-HOI. All masks/identity/geometry must be
inferred automatically; no hand-labeling of challenge records.

## Status — 2026-10-06

**No final Parquet, submission or verified CARI4D improvement yet.**

**Visual QA changes the priority:** EP8/EP9 select a background person/object at
automatic initialization. Its complete engineering PASS is not a correct
interaction. A read-only census of all14 exports/42 fixed views is complete;
mask/mesh agreement cannot certify semantic identity. Prioritize a genuinely
automatic multi-instance observation bank and external persistent-ID validation,
not manual episode reassignment or another downstream fit of the wrong pair.

**Current research:** CARI4D's concept-mask union and 4DAnyone's largest summed
track-area selection do not resolve ownership. The new coherent-route core
preserves every person/side/object and combines one native relation before
reduction. Its masked variant and pure FIT-only learner prototype pass numerical
controls, but no real selector has been fitted or adopted. The fixed-step
hard-max recipe fails its frozen cost gate. A separate normalized latent
reference, compact factors/CSR and CPU analytic gradients pass tiny independent
controls. The first tiny GPU control closes before Torch/container creation:
its B47 image was addressed on the wrong Docker-store worker. A new explicit
VM01 execution profile now passes sealed tiny GPU arithmetic checks in7.24s
(8fixtures/20controls/288FD calls), independently audited with unchanged gates.
The new full-bank score/VJP control also passes: all1,843,200 procedural routes,
28,800 native rows and14,400 grouped rows agree with CPU FP64; host84.79s,
peak GPU reservation644MB. This is measured arithmetic cost, not an objective,
optimizer, FIT or selection-quality result. Fresh V-COCO/COCO2014 metadata and
its exclusion-safe role census pass independent audits:199 eligible VAL photos
and362 TEST, after432 historical photo exclusions. Next is a separately frozen
8DEV/8reserved unfitted pilot; no RGB, learned selector or accuracy result yet.
The first strict VG/COCO-linked reference census
completes but yields0 eligible photos: this path is closed without RGB, relaxed
joins or retries. Fresh external validation remains necessary.
MMHOI's complete Azure-only metadata census passes an independent audit; its
supplement still does not qualify anatomical indices, sentinels or RGB alignment.
The explicit corrected OpenImages metadata gate finds132 independent source
slots above the unchanged96 gate; unsupported identities are excluded, not
repaired, and the old failure stays immutable. Independent rederivation passes.
All96 historical COCO original-byte identities are now excluded. A fresh96-slot
cohort (32 FIT/16 CAL/48 RESERVED) is frozen in a new SHA order before HTTP;
an independent saved-only audit reproduces selection and full source/input pins.
The subsequent single original acquisition is **closed INCONCLUSIVE**:
68/96 (FIT23/CAL10/RESERVED35), below the frozen24/12/36 availability gates.
Independent integrity checks pass, not quality. No bank inference, fitting,
resampling or retries on this closed study. Generic caller work stays dormant;
data-free cost checks and a separately justified reference search continue.
See [next ownership decision](docs/ownership_next_decision.md). No manual labels
or episode-specific rescue are used.

- **Actual automatic observation bank:** two external HO-Cap clips retain all
  1493 original RGB frames and all287 native class-agnostic AMG proposals at ten
  fixed anchors, yielding3133 automatic point queries. Execution/source/cleanup
  checks pass. Both full-T native tracker outputs are independently authenticated.
  The [frozen semantic-retention test](docs/hocap_point_retention_protocol.md)
  is now **closed REJECT**: correct support increases, but wrong-object assignment
  regresses on one external clip (3.28%→7.74%). Do not tune this cohort or adopt
  this bare bank as qualified identity evidence. Learned relational observations
  are the next separate hypothesis; actor/contact/3D accuracy remain unverified.

- **Learned relationship runtime:** original HOI-DETR/MMCV CUDA operators and
  the full detector now pass independent H100 execution audits. All1796 native
  checkpoint fields load strictly; one procedural image forward completes in
  the45.29s host gate. A separate licensed external image now executes both
  native hand→object pairs in45.96s, with all proposal/query evidence retained.
  This qualifies pair-head execution, not semantic quality. The separately
  preregistered128-slot attribution study is now **closed INCONCLUSIVE**:
  61 exact licensed originals predicted,67 explicit misses, only1 informative
  image versus6 required. Full-bank retrieval15.625%→17.1875% is descriptive,
  not a verified improvement; no component adoption or tuning on this cohort.
  Prioritize existing full-native coverage and a genuinely informative external
  interaction validation, not another expansion of this closed comparison.
  Benchmark training overlap and submission eligibility remain unresolved.

- **New generic operators:** [surface identity/LOD](docs/surface_lod_protocol.md)
  separates lawful open surfaces from our optional closed-solid backend; no
  historical failure is relabeled. Under-budget identity and
  [mask-conditioned queries](docs/mask_query_quantile_protocol.md) pass289 tiny
  tests. Three fixed open-surface controls now pass an independent actual
  loader/mesh-budget audit; the original host cleanup FAIL is retained. Whole
  Parquet and real-object pose quality remain separate.
  Two fresh over-budget open-surface QSlim controls pass in17.725s (4224 actual
  contractions), with source/result receipts independently authenticated; their
  separate host cleanup FAIL is preserved. No full-surface accuracy/adoption.
  The [surface consumer seam](docs/surface_consumer_protocol.md) now connects
  the same CPU proposal, inert geometry reader, ICP/Viterbi and shared-stage
  profile without changing pose weights. Its fresh96-frame native operator
  control stops in historical-receipt preflight before any native geometry;
  its technical reader fix preserves the old failure and all numerical gates.
  The separate corrected native control passes all96 poses/six negatives in
  3.561s, with an independent saved-proof audit and no QEM. No quality claim.
  EP21 real native point qualification stops before optimization (<8 queries);
  no replay, positive-weight fit or accuracy gain. EP09 full415 preparation,
  forward,301-update refinement and directexport pass independent source/full-
  bundle audits; original official packing also passes its independent audit.
- **Engineering:** all30 original videos pass byte/metadata readiness. Episodes
  **0,1,2,3,5,6,8,9,12,13,14,15,16,17,19,21,26** pass full native shared preparation, forward,301-update
  refinement, direct export and original official packing. These are seventeen
  complete engineering checks, not held-out accuracy; scratch Parquets deleted.
  EP0's legacy conversion failure stays separate. EP4 missing masks, EP7 empty
  anchor, EP9's original topology-budget failure and EP10/EP11 actor identity failures remain closed.
  EP11 stops before SAM2/later frontends. EP0's exact historical metadata
  omission is authenticated, not a relaxed legacy gate or a reversal of its old
  conversion failure. EP13 full425 complete chain passes. EP14 full442 inputs, shared preparation
  and native forward plus301-update refinement, export and official packing pass with independent source-chain pins.
  EP16/EP17 close on topology-budget failure; EP18 rejects a boundary/nonmanifold
  source before object pose. EP19 closes on the same topology-budget gate;
  EP20 also rejects all eight topology-budget candidates; no mesh repair.
  Collected-predecessor scheduling failures
  remain separate. Original controls/reference replay and whole source-chain pins are
  frozen before each next stage; no prediction is inferred from dispatch ACKs.
  EP26 preserves the original open surface and all399 frames through the complete
  independently audited chain; its earlier host preparation failure stays separate.
- **Coverage continuation:** the existing qualified open-surface route passes
  independently audited CPU proposals for EP16,17,19,20; EP18's nonmanifold
  source remains CLOSED FAIL. EP16,17,19 now pass the complete independently
  audited native chain and official packing; EP20's proposal does not qualify
  its remaining chain. Original topology failures remain separate. Complete
  engineering coverage is17/30,
  not an accuracy result. Lightweight QA openly shows EP8/9 wrong background
  identity; engineering packing success does not resolve that scientific defect.
- **Temporal research:** MASA's isolated author-runtime build passes independent
  source/52-wheel/CPU-import/image/cleanup checks. The separate native H100
  contract stops on an operator RuntimeError before checkpoint/model work;
  that technical FAIL is retained, not described as successful inference.
  Appearance persistence is not physical ownership or a validated HOI gain.
- **Geometry compiler:** exact F32/weld predicate controls pass, but the frozen
  real-collapse comparison **fails** on the thin cavity: both original and new
  native queues exhaust before the budget. Capsule control not executed; no
  rescaling, fixture reroll, production repair or adoption. A separate
  [fixed-chart numerical hypothesis](docs/mesh_conditioned_qem_protocol.md)
  passes four fresh procedural controls (max CD/diagonal0.006118, shell-volume
  error0.001015). Its [cached implementation](docs/mesh_conditioned_cache_protocol.md)
  passes eight byte-exact paired arms and is about2× faster on those controls;
  the qualified binary stays on Azure. First production source rejects an
  unqualified component arrangement before any native call, after its sole
  packaging-only replay. Neither source nor gates are repaired to force PASS.
  A distinct whole-solid chart-v2 compiler now passes four frozen procedural
  controls and the first real predicted EP9 source: nine complete components,
  one QEM/eight exact queries/six fidelity stages, 65.14 s host. Its original
  tuple/list packaging FAIL is preserved; the new producer changes no numerical
  operator or acceptance gate. The same fixed geometry now passes full415-frame
  pose execution; automatic mask medianIoU0.244 is not accuracy evidence.
  No HOI gain, licence clearance or adoption is inferred.
- **Unified fitting:** [persistent point reprojection](docs/joint_point_objective.md)
  now augments the unchanged native joint objective through an explicit subclass.
  Original first-frame triangle attachments connect to full-T tracks without
  refitting. Ten real Torch controls and the original301-update loop on
  manufactured state pass; real-body/contact execution and independent HOI
  validation remain required. No challenge-tuned loss weights or measured gain.
  A [fresh three-frame real-MHR runtime pair](docs/joint_point_authored_runtime_protocol.md)
  passes manufacture/32 attaches/real-kernel execution but **fails exact parity**:
  A301 updates, B0; full602-update qualification remains unachieved.
  A distinct two-original-optimizer A/A also **fails exact repeat parity**;
  this does not identify an extension defect. Both controls remain closed.
  A new [single-instance zero-weight delegation](docs/joint_point_zero_delegate_protocol.md)
  passes on H100:301updates/303exact native return-object pairs, real contact/
  renderer/penetration execution, full9-artifact saved-file audit. This verifies
  semantic delegation only, not independent bit parity, tracking or HOI gain.
  A [six-scene articulated component preflight](docs/authored_point_study_protocol.md)
  **fails before decoding**: the frozen recipe violates actual rig limits.
  Zero decoded/rendered frames or fits; cohort closed, no bounds clipping/rescue.
  [Balanced exact sums](docs/balanced_solid_sum.md) pass15 fresh native controls
  on the authenticated existing CPU runtime; four fresh QEM/query composition
  controls also pass, with the original QEM unchanged and query proofs separate.
  Technical production replay remains distinct from reconstruction accuracy.
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
- **Fresh paired hand-mask ablation:** distinct Dex04/full218 frames, all masks
  frozen before private segmentation. Dice0.367→0.376, but one clip regresses
  and object-pixel contamination rises: preregistered gate **REJECTS**. Missing
  proposals affect82/175 annotated positives; no tuning or tracking extension.
- **Temporal hand ablation:** distinct Dex05/full220 frames, all masks frozen
  before segmentation. Native memory improves Dice0.547→0.642 and reduces
  positive empty unions50→0, but object-label contamination rises40086→48203:
  preregistered gate **REJECTS**. No anchor/reseed tuning or challenge adoption.

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
The experimental CGAL solid-query executable combines Apache glue with GPL-3.0-
or-later PMP/Side (or commercial CGAL terms); its binary is **not Apache-only**.
No submission eligibility or Apache rerelease exception is presumed for it.
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
