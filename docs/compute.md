# Azure runtime — operational snapshot, 2026-10-04

This is a compact operational summary, **not a live monitor**. A dispatch ACK
proves launch acceptance, never completion or quality. Detailed immutable
receipts, source revisions, timings and prior failures belong in
[experiments](experiments.md); scientific contracts are in [baseline](baseline.md),
[framework architecture](framework_architecture.md) and [licenses](licenses.md).

## Verified coverage and unresolved work

All30 original Track1 videos pass byte/metadata readiness: 16,563 original
frames, 30Hz, 1536×1152. This readiness does not decode or score predictions.

| Episode | Full original frames | Last verified pipeline state |
|---|---:|---|
| 1 | 668 | Shared preparation → native forward →301-update refinement → direct export → official packing PASS. |
| 2 | 866 | Same complete chain PASS; earlier frontend timing failure remains separate. |
| 3 | 592 | Same complete chain PASS; historical queue/cache and two export failures retained. |
| 5 | 668 | Same complete chain PASS on qualified fixed volume geometry. |
| 6 | 816 | Same complete chain PASS on qualified fixed volume geometry. |
| 8 | 634 | Same complete chain PASS; VM02 full pose/result return byte-sealed before VM01 assembly. |
| 15 | 501 | Same complete chain PASS. |
| 0 | 790 | Full native refinement and direct export PASS42.687162s, whole source chain/four outputs pinned. Official packing PASS16.896468s/all790 frames, scratch removed; old conversion failure separate. |
| 4 | 747 | Tracking stops at18 empty automatic object masks; full trajectory unavailable. |
| 7 | — | Empty automatic fixed anchor; closed without reroll. |
| 9 | 415 | Geometry/topology failures before pose; no face deletion or threshold rescue. |
| 10 | — | Automatic actor identity support FAIL5/16; no substitution by the better-covered bystander. |
| 11 | — | Actor identity feasibility FAIL2/15 valid observations, fragmented tracks; no SAM2 or later frontends. |
| 12 | 405 | Full native refinement and direct export PASS31.726037s, whole source chain/four outputs pinned. Official packing PASS13.521699s/all405, scratch removed. |
| 13 | 425 | Original full inputs and shared preparation PASS18.417917s/all425 frames; full source/geometry lineage pinned. Original collected-predecessor scheduling FAIL remains separate. Native forward PASS109.129234s/all425 and301-update refinement PASS220.562431s, both independently frozen. Export queued behind original EP16; ACK only. |
| 14 | 442 | Original full frontend/input PASS3319.999303s; independent15-input pins sealed. Shared preparation queued; ACK only, no reconstruction/accuracy PASS. |

Nine complete packed episodes are engineering/fidelity evidence only. Scratch
single-episode Parquets were deleted. No final all-episode Parquet, submission or
verified CARI4D superiority exists. Other episodes are not presumed ready.

VM02 YCBv2 acquisition and three native MoGe2 initial calls PASS; automatic
object identity then FAILS on ambiguous detections: two detector calls, zero
SAM2 calls. No Objects/Boots trajectory or private3D quality evaluation followed.
Private annotation values/projections were not consulted. This pilot is closed,
not a reason to change its prompts/margin. **No active VM02 GPU job is reported
at this snapshot.** EP11 is closed FAIL at08:46:36UTC; its original log and
producer remain unchanged. EP12 original CPU inputs PASS3035.001663s at10:17:31UTC;
its collected unit is not the proof of success. EP13 queue under `fc10a6e`
failed at10:17:34UTC after that predecessor was collected, with no child inference.
Root independently checked all EP13 targets absent before direct replay under
the unchanged `814179a` source. Actual full CPU inputs now PASS/pinned, followed
by shared preparation; later stage status is in the table, not a live unit claim.

VM02 external identity research : DexYCB original two-subject downloadv2
PASS224.717368s,24,416,459,511B entirely on the owned Azure data disk. SHA values
are caller-measured, not publisher checksums. Full gzip/header/extraction
PASS329.644207s,12clips/872originalRGB with private opaque bytes kept separate.
Actual generic hand-bank inference REJECT13.134305s atclip0 (zero hands), sources
rehashed and own container removed. No tracking or private values interpreted.
Candidate-bank, full-clip tracker and CPU calibration stages are implemented;
initial-identity micro-gate is frozen before private labels, not a3D victory.
D107 independent TUM depth validation is REJECT4.904632756% < frozen5%; the
four-anchor hand-support pilot is also closed REJECT. Neither is adopted.

Fresh Dex03 hand-only pilot is complete:216 original RGB frames/three72-frame
clips,9.314269s CPU native scan and3.698009s separate private2D diagnostic.
All outputs freeze first;171/177 positives uniquely associated,6 misses,
conditional17-joint EPE9.648746px (2907/3009 valid joints scored). No GPU,
identity/contact/3D gain, threshold tuning or adoption. Detailed limitations and
pins stay in experiments; source/task eligibility and overlap remain unresolved.

## Azure identities and storage

| Worker | Azure resource group | Private address | Role |
|---|---|---|---|
| `scenesmith-ncc-h100-01` | `SCENESMITH-H100` | `10.0.0.4` | Original native Track1 and existing Objects runtime. |
| `world-reward-ncc-h100-02` | `WORLD-REWARD-RESEARCH` | `10.0.0.9` | Isolated second H100, qualified narrow frontends/pose work and external validation. |

Both are West Europe H100NVL workers (~95,830MiB). Do not confuse VM02's name
or resource group with VM01 defaults. Existing unrelated SceneSmith services,
system Docker/containerd and files remain untouched.

Runtime ABI root: `/srv/scenesmith/world-reward`. VM02's separately owned1TB
managed data disk mounts `/srv/world-reward-data`, ext4 UUID
`24df126a-5f5f-41d8-801c-9ddaa7a582d8`. Never format, relocate or replace existing
storage implicitly. Space/readiness must be checked before a new heavy run.
Private Docker uses `unix:///srv/scenesmith/world-reward/docker.sock`, with owned
runtime storage. No global Docker, bridge or iptables changes are authorized.

| Azure-only directory | Contents |
|---|---|
| `data/` | SHA-bound Track1 RGB/public metadata, never Track2/3 or source calibration. |
| `vendor/`, `weights/` | Audited inference/kit source and externally licensed models. |
| `outputs/` | Full-index predictions, masks, constant meshes and native exports. |
| `validation/` | Licensed external public inputs; private evaluator artifacts remain separately inaccessible to predictors. |
| `results/` | Reproducibility inventories, measurements, decisions and useful failure evidence. |
| `jobs/<revision>/<entrypoint>/code` | Immutable code-only dispatch closure and producer markers. |

Only tiny source/config/pins and summaries cross the laptop connection. No
videos, PNGs, meshes, arrays, renders, model weights or full caches transit locally.
Private peer transfers use exact allowlists and byte pins, not full-disk/runtime
cloning. Expiring forced SSH listeners/client keys/owned NSG allowances are
removed after independently verified transport; no standing peer exposure assumed.

## Runtime identities, not tag-based assurances

| Runtime | Pinned identity / qualification |
|---|---|
| VM01 CARI/Body | `sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7` original OCI identity. |
| VM02 classic CARI/Body | `sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3`; audited44 ordered rootfs layers from the original image. |
| VM01 Grounding/SAM2 | `sha256:53b33bc4b60e0e3e8f83b401775b4701b18eef54408fd585fbe3a5d376c042e1`. |
| VM02 narrow Grounding/SAM2 child | `sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252`; independent CPU/CUDA kernel gates. |
| VM01 Objects | `sha256:eb389b26358c49778a14303b5875c66d887824011388ce9f8666ed7cc1841ce5`; measured37 layers and selected runtime inventory, not commit-parity proof. |
| Official CPU packer child | `configs/official_pack_runtime_pins.json`, exact image/build/wheel/source pins. |
| BootsTAPIR CPU-verified child | `configs/bootstapir_runtime_verify_pins.json`; original incorrect API probe FAIL retained, no rebuild/retag. |

An OCI index is not the classic platform/config ID. Tags, installed-source
fingerprints and layer equality do not establish licence eligibility, leakage-free
training, whole-runtime parity or reconstruction quality. Validate actual image,
source and selected assets against the producing stage's independent pins.

## Scheduler, publication and reuse

- One cooperating GPU job per VM; acquire the existing canonical lock and check
  actual compute applications before model work. CPU assembly/evaluation releases
  the GPU lock and may overlap disjoint work within explicit CPU/disk budgets.
- Serialize Azure Run Command calls. Inspect exact unit/PID/owned container and
  bound report; collected/missing units, provisioning success and observation
  timeouts never authorize restart or imply PASS.
- `infra/azure_job.py` archives a verified committed import closure. Default
  requires a clean tree; strict `--revision <40hex>` supports disjoint edits.
  Source-only XZ/base64 is capped256KB. Long payloads use exclusive staged chunks,
  full reassembly SHA checks and atomic publication; every phase needs its exact ACK.
- Explicit `--reuse-published` performs metadata-only verification of the exact
  existing bytes/modes/markers/canonical closure, then dispatches a new unique
  unit/log. No upload, chmod, republishing, retry or result reuse is implied.
- `run_track1_frontends_queued --after-terminal <unit>` starts a distinct untouched
  clip only after a loaded, coherent terminal predecessor (success or retained
  failure). Source/targets/lock/GPU are rechecked; missing/unknown units fail.
  Bounded12h wait, unchanged fixed_all16 child, no retry or failure reclassification.
- `--after-gpu-lock` is an explicit scheduling-only alternative for fresh clips:
  no unit query or claim of predecessor success, same existing readonly lock,
  bounded12h wait, full source/target/GPU checks and unchanged child reacquisition.
  Collected units do not block this mode; they still never prove prior success.
- Fresh outputs only; reuse successful artifacts solely after their full lineage
  checks. Authenticate historical producer helpers, not current consumer hashes.
- Budgets are predeclared per stage. CPU assembly retains7200s; no longer timeout
  or smaller validation set silently converts an incomplete run to success.

## Evidence, replay and cleanup

Predictions are frozen before private values are decoded. Models see only allowed
public inputs and narrow assets; inference runs offline. Technical replays require
explicit authorization, demonstrated cause, fresh namespace and retained original
FAIL. Scientific failures remain closed: no frame mining, per-episode labels,
prompt/threshold retuning, static fallback or missing-frame deletion.

Local tests are tiny/data-free; concurrent pytest uses distinct exclusive
`--basetemp` paths. Keep concise asserted results, not disposable logs/fixtures.
Secrets stay in ignored secret storage, never stdout/raw environment/config dumps.
Source/model licence and overlap checks, NVIDIA registration, producing commit
accessibility, rules/quota and one frozen Parquet to all five **World Reward**
competitions remain submission gates; see [README](../README.md).

The optional `azure_job --github-source` mode now passes an exact-byte remote source publication audit: Azure
fetches the exact public source closure, verifies every file and original TAR/XZ
hash, then uses the same atomic publisher. One90s bounded phase; no redirects,
credentials, retry or fallback. It is not a scientific/runtime qualification.
