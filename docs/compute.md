# Azure runtime — current state 2026-10-02

## Isolation and data flow

VM `scenesmith-ncc-h100-01`, resource group `SCENESMITH-H100`, West Europe,
H100 NVL95,830MiB. User stopped competing jobs and authorized exclusive GPU use.
Persistent root `/srv/scenesmith/world-reward` on the existing1TB managed disk;
all previous SceneSmith files, system Docker/containerd and resources untouched.

| Remote directory | Role |
|---|---|
| `data/` | SHA-bound official Track1 RGB and public metadata only |
| `vendor/` | Pinned official kit and required inference source; no GT datasets |
| `weights/` | Pinned externally licensed models; never committed or copied locally |
| `outputs/` | Predictions, meshes, masks and native exports; remote only |
| `results/` | Artifact manifests, scalar reports and useful failure evidence |
| `jobs/<commit>/<entrypoint>/code` | Read-only committed runtime import closure |
| `cache/`, `docker/`, `containerd/` | Task-isolated dependencies/runtime storage |

Azure Run Command controls jobs; SSH tests failed and temporary22/443 ingress
rules/listener were removed. Extra VM creation attempts failed before compute;
task-created extra resource group/network resources were deleted. No new VM needed.
Private Docker socket `unix:///srv/scenesmith/world-reward/docker.sock`, with
private dockerd/containerd data and state on managed disk. No bridge or iptables
mutation. Inference uses `--network none`; acquisition downloads directly toAzure.
**No videos, weights, meshes or renders transit the tethered laptop.**

## Reproducible jobs

`infra/azure_job.py` requires a clean committed worktree, sends only source import
closure plus small package/config as SHA-verified XZ (<100KB base64), extracts
read-only, and creates one unique systemd unit/log. Never replace active readers,
restart on an observation timeout, or reuse a failed result path. Earlier legacy
snapshots under `jobs/<commit>/code` remain unchanged. Status observations alone
do not demonstrate final job success; require the frozen validated result report.

Serialise `az vm run-command invoke` calls. Its output is tail-limited; retrieve
compact scalar summaries, not full checkpoint provenance. Poll exact units, and
work on independent CPU/source tasks between observations. General native wrappers
support strict `--episode0..29` and explicit `--wait-for world-reward-<unit>`;
non15 clips never implicitly wait for episode15. Missing/failed dependencies fail
fast; active waits are bounded12h. Never bypass stage provenance gates.

## Verified runtime and credentials

- `world-reward/grounding:0.1`: pinned GroundingDINO/SAM2 offline inference.
- `world-reward/cari4d-source:0.1`: official source plus six precisely restored
  Body Python `data/` files; PyTorch3D0.7.9/CUDA and original MHR assets.
- `world-reward/sam3d-runtime:0.1`: Objects with required libusb/Open3D dependency.
- GPU KNN, Kaolin, FlashAttention and nvdiffrast kernels passed. EGL failed and is
  optional, not silently considered supported. Our renderer uses BSD PyTorch3D.
- HF grants checked after user approval; token mode600, never logged. Kaggle API
  authenticated using ignored mode600 `.env`. No rule acceptance bypass or upload.
- Transitive image/dependency pins and full release SBOM still need final freezing.
  SAM/native CARI/nvdiffrast source eligibility remains unresolved: see licenses.md.

## Current engineering gates

Active dispatch2026-10-02 16:18UTC, source
`baba81be965dff878d7c16c9f132f85dcf712bd7`:
`world-reward-episode0-volume-native-v2` runs the existing full790-frame object
pose→native prepare→forward→conversion→schema chain, using the qualified frozen
volume mesh without resimplification/second scale. Source closure93340encoded
bytes only. The original empty legacy failed-pose directory was removed with
`rmdir`; original failure log remains. Initial v1preflight log remains unchanged.
Active state is observed, **not completion or validated accuracy**.

`world-reward-joint-rgb-grounding` (same source,82444encoded bytes) is queued
behind that exact unit, bounded12h. It independently renders a new nine-image
human/object RGB cohort, produces automatic masks and Body/MoGe2 predictions,
then scores frozen predictions against private synthesis truth. No native
challenge predictions are J1 inputs. GPU steps remain serial; quality evaluation
is CPU-only. No success/adoption follows from dispatch.

Source-only throughput audit (not a measured speedup): the current native
`MHRDepthH5Writer.write_frame` waits for a single encoding future despite
`encoding_workers=8`, then flushes HDF5 twice. Raw/aligned PNGs use level9+
optimize; exhaustive validation decodes/reencodes their canonical bytes.
See [pinned writer](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py#L817-L903).
Future minimal test: profile phases, then synchronous `write_frames` batches8
of unchanged `DepthFrameRecord(index,raw,aligned,scale,shift,valid_count)`;
one future per frame, same byte encoding, two flushes per batch. Keep complete
canonical-byte validation, ordering and metadata. Active frozen chain unchanged.
Preparation uses CPU-only PyAV/OpenCV/Pillow/HDF5; its immediate GPU forward
handoff means background GPU research still requires explicit serialization.

Episode15 is a predeclared engineering clip, **not a labeled validation split**.
All results below are execution/representation checks, not challenge accuracy.

| Stage | Verified result |
|---|---|
| Automatic masks |501frames,0empty,52.00s |
| SAM Body initializer |501frames,172.99s, source/geometry/projection pass |
| Official Body shared identity |106.61s,.9719mm mean to input predictions,1.4861mm worst frame |
| MoGe2 depth |501frames,258.30s, fixed RGB-only K/cameraZ |
| Human/depth gauge | One equal-frame clip scalar; no GT calibration |
| Grounded object generation |46.99s, fixed closed inferred geometry |
| Packed object budget |4080real faces,4096official rows; cavity/orientation retained |
| Sparse object pose |3frames,24generic orientations+ICP image gate,30.49s |
| Native Body adapter |501frames,8.42s, original decoder roundtrip |
| Native checkpoint load |7.12s, strict finite offline state; **no forward** |
| Native full hand proposal |3frames,34.38s, fresh block-decode rotations; no accuracy claim |
| Raster batch parity | Exact masks/Z, batch8 median1.7832×; not wholeICP speed |
| O1 rendered shape test |18conditions,27.45s;6controls fail, **adoption rejected** |

Completed producer `world-reward-object-pose-full`, revision
`ca6b23464bed6b2dbced81f13123c620d64d487c`, output
`outputs/episode_000015/object_pose_full`;
completed501frames in3929.56s (verified11:26UTC). Final Viterbi poses are in
`geometry_and_poses.npz`; per-frame `selected` is a greedy diagnostic, not final
trajectory. **Do not restart or overwrite.**

Completed chain: `world-reward-cari-prepare-v2` (`5d4f5db`, remote EXDEV copy fix)
→ `world-reward-cari-forward` → `world-reward-cari-converter` (both`414aac2`).
Preparation passed501frames3582.24s; actual native forward68.85s and official
conversion127.96s passed501frames at12:29UTC. Worst per-frame mean conversion
fidelity1.78963mm to native predictions, not GT accuracy. Final schema/H1 consumers
initially rejected missing legacy report fields; exact source/hash compatibility
is now separately gated, without rewriting original reports or claiming a
collected transient systemd launch was observed. See R50/R51 in experiments.md.
A separate `world-reward-masks-episode0` from`a867e23` tests routing/general automatic
mask initialization; final reporting failed from a shadowed provenance variable.
Failed outputs quarantined; `masks-episode0-v2` from`c344eb3` passed full coverage
in72.67s with zero empty masks. No manual labels or hyperparameter adjustments.

Continuous small-triangle PyTorch3D gate failed in.816s; backend not adopted.
Alternative Kaolin import failed before numerical evaluation (unwritable `/.cache`).
Exact infrastructure-only retry passed in2.874s after task-isolated HOME/cache fix:
distance error1.2899e-11m², gradients and rigid/duplicate/singleton contracts pass.
Frozen original failure remains; no fixture or tolerance change. Explicit transitive
noncommercial-import uncertainty remains; no fitter adoption from QA.

Two immutable procedural throughput gates failed their total elapsed budgets:
whole24-candidate scalar/batch8 passed exact JSON parity but exceeded120s;
canonical KDtree passed all72discrete/NN/trim/1e-6 comparisons but exceeded60s.
Neither finished3timing trials or proved>=1.3× median speedup. Own synthetic inputs
are not challenge validation; cached-ICP seeds are synthetic-oracle-derived.
Neither solver/schedule is adopted. Independent own float64 continuous geometry
passed the same analytic fixture plus its fixed runtime budget (f5feba2):1.554s
total, median.05682s forward/backward on2048points/5120faces,90.55MB peak allocated.
Distance error5.42e-20m², point/interior triangle gradients0error. This validates
the primitive only, not any new fitter or challenge shape accuracy.

New serial bundle `world-reward-episode0-initializers` (`6b0020c`) started11:58UTC:
sparse body/depth→scale→grounded object→full body/depth→native adapter. Requires
all seven target outputs absent; no automatic reuse/overwrite. Full object poses
and native forward are deliberately not part of this short generalization bundle.
Sparse body31.12s/depth11.14s/scale3.61s/grounded object36.19s passed on episode0;
full body790frames passed302.89s, full depth407.27s and native adapter8.97s;
the complete initializer bundle passed at12:11UTC. Full object producer on0
failed its fixed topology-preserving budget before fitting any trajectories.
Investigate a different simplifier without relaxing topology gates. These are representation
checks only. New cached whole-candidate gate passed parity but median1.2794× is
below1.3; not adopted. Continuous fixed-pose shape8condition test executed15.12s
but both correct/biased-pose controls failed; fitter remains rejected.

Results/decisions and useful failure causes are retained in experiments.md.
Temporary redundant masks/download archives may be cleaned **after** validated
export; preserve immutable reports/source hashes and never prune shared user data.


## Independent RGB validation (2026-10-02)

All RGB/meshes/checkpoints remain Azure-resident. Two immutable serial entrypoints
now separate render-only private truth, public inference and private evaluation:
`infra/run_hand_synthetic_pipeline.sh` and `infra/run_object_synthetic_pipeline.sh`.
Each stage reserves its own output; failures stop the chain and are never
implicitly restarted/overwritten. Runtime closures are25files/~85KB encoded and
21files/~63KB respectively, below100KB control-only transfer limit. H100 work
remains serial, while disjoint source audits/tiny local tests run in parallel.
Reference cold TorchScript replay requires unoptimized execution; fresh-process
diagnosis passed17.546s. Strict hand semantics V4 passed8.801s/6calls. Six RGB
hand cases passed automatic paired inference/official conversion; quality gain
0.8233%<5%, so finger-transfer hypothesis rejected. Four RGB-only object
single/three-view proposals passed139.118s; median camera CD gain-0.18784%<5%,
so fusion hypothesis rejected. Both private evaluations execute only after
prediction hashes freeze; neither establishes challenge accuracy or adoption.
Latest full tiny suite3450PASS32.89s. No new submission or production mesh.
Fixed-anchor RGB motion tracking completed24frames but quality gain0.131468%<5%
rejects adoption. CPU guarded-QEM replay identifies39of46 cavity volume errors
over5%, while global CD/net volume/embedding pass. Frozen failures are retained;
no thresholds, source scale or camera trajectories are changed for mesh QA.

2026-10-02 17:12:33UTC: H100 exclusive, sequential GPU queue remains
`episode0-volume-native-v2` → `joint-rgb-grounding` →
`cari-native-final-refinement-v2`. First unit's full-object pose is650/790;
J1 is independent research, final refinement reuses already-frozen episode15
inputs. None restarts or mutates existing predictions. New producing revision
`9f2a4ec4cf8980364262c2b1ee62fad06b4cfc85`, immutable22-file closure54432encoded
bytes; acquisition of173282bytes directly on Azure only. CPU native-import
preflight passed without GPU/context; expensive refinement not yet executed.
Whole refinement bound7200s is exploratory budget, not a measured ETA.
First preflight failure/cache fix retained as concise R89/R90; cache changes
use writable container `/tmp`, no persistent vendor or dependency mutation.

2026-10-02 17:39:33UTC: épisode0 pose complète790frames en3712.869s;
préparation CPU native100/790 observée, pas fin de chaîne. File GPU exacte:
`episode0-volume-native-v2` → `joint-rgb-grounding` →
`cari-native-final-refinement-v2` → `joint-rgb-learned-camera`.
J2 producteur `20714cf21436259ab8e6abc1993a90f97da1c2d5`,15files54864bytes,
réutilise les9RGB/masques publics J1, sorties neuves; comparaison privée CPU
des deux bundles complets seulement après gel des prédictions. Aucun résultat
J1/J2 ou raffinement observé. Benchmark compactCPU distinct D71 peut coexister
sans GPU; charge hôte partagée explicitement exclue d'une conclusion de débit
préparation isolé. Ni les sources ni les budgets des lecteurs gelés ne changent.

2026-10-02 18:22UTC: préparation0 depth450/790 activeCPU; J1/refinement/J2
restent en queue, suivis du nouveau `world-reward-tudl-real-camera`.
Ce dernier producteur `2a18859543b3538d755c8a47c1961ac125b4e7ff`,11files25580bytes,
attend le prédécesseur exact J2; il ne lit aucune prédiction challenge. Son
acquisition indépendanteCPU est terminée en15.318s: troisZIP374952356bytes
directAzure→neufRGB publics et41files privés/licences hashés, ZIP supprimés.
Pas encore qualité réelle ni nouveau modèle acquis. Benchmark compactCPU R92
PASS3.963827× writer-only, validation inchangée; ancien R88FAIL conservé,
aucune mutation du prepare0 actif. AssetsBody15/MHRreference hash identiques
352e271a…7377bc vérifiés. Réserve GPU Azure familleNCCads2023:40cores utilisés
sur80; pas quota de la familleNCadsH100 (0/0), ni secondeVM provisionnée.

SecondGPU option audited, not provisioned: existing confidential VMI exact
`cgpu-NCC-2204-base-image/versions/2204.20260615.0`, SecureBoot/vTPM, zone1.
Do not use vanilla Ubuntu/generic GPU extension or clone/restart running Docker
state. Future new VM needs independent NIC/NSG/no ingress, same pinned VMI,
SHA-checked image export/import into a fresh private Docker root, task-only
artifact transfer within Azure, explicit target support in azure_job.py, new
unit/output namespace and actual CUDA/replay checks. The launcher now accepts
validated `--resource-group` and `--vm-name`; defaults and frozen remote script
remain identical to VM01. No automatic migration or root override is allowed.
Quota permits one more40core SKU nominally; capacity/setup duration unverified.
No snapshot, user-data copy, second VM or reader migration has been performed.

2026-10-02 18:40:51UTC: actual-depth-v2CPU termine en deadlineFAIL240s,
parité première paire et reversebatch exactes mais dernière validation incomplète.
Temp H5 supprimés; **aucune adoption batch dans la préparation native**. R96
retient timings/digests, pas un median validé. Source active episode0 toujours
600/790 depth; indépendants J1→native15refine→J2→TUD-L restent en queue GPU.
Fulltiny suite3889PASS/1optional-trimeshSKIP41.40s; worktree/code propres.

2026-10-02 D74/D75: launcher target tests63PASS and full tiny suite3915PASS/
1optional-trimeshSKIP39.39s. At18:51UTC prepare0 is700/790; GPU queue unchanged.
Predeclare a bounded independent H100 VM02 in `WORLD-REWARD-RESEARCH`, same
confidential VMI/SKU/zone/SecureBoot/vTPM, fresh OS and private Docker state.
Use a new NIC/NSG on the existing subnet without modifying VM01 policies;
no VM public IP. Temporary SSH ingress only10.0.0.4/32→VM02 TCP22 for task-only
TAR/SHA transfer executed on VM01, never laptop remote-to-remote SCP. SSH private
key stays mode600 on Azure and is revoked/deleted after transfer; host key is
verified through Run Command, not TOFU. New NSG then denies all ingress.
No whole SceneSmith disk/OS snapshot or live Docker-state copying. Stage only
the pinned CARI image, MoGe snapshot/blob/receipt and independent validation
fixture files; preserve HF symlinks. Budget setup≤1h conditional on allocation,
agent/egress, image and CUDA gates; failure stops only new resources. No second
VM result, GPU adoption or migration is claimed before those gates complete.

19:04UTC user explicitly authorizes community-image terms. CLI previously
failed before any deployment because its publisher EULA URL is blank; no terms
were bypassed. Actual creation now accepted with `--accept-term`: deployment
`vm_deploy_PPKbqHbQ5jcAYWvCSXRhz9kVKKgbo1UE` running, VM
`world-reward-ncc-h100-02` creating, private10.0.0.9/no publicIP, new OS128GB,
new NSG denies ingress except temporary10.0.0.4/32→22. No existing network policy
modified. Fresh runtime/bootstrap and task-only archive helpers have56 tiny
tests PASS22.48s; no runtime/GPU success yet. Original prepare0 still750/790 at
19:06UTC, active without errors; no duplicate GPU job or reader restart.

19:08UTC VM02 provisioning **PASS**, actualH100NVL95830MiB/driver595.71.05,
CC ON, Docker29.5.3/containerd2.2.4/NVIDIA runtime1.19.1; HF HTTPS200 egress.
19:11UTC initial runtime launch failed209/STDOUT **before script execution**:
new immutable launcher creates jobs/root but assumed `results` existed. No
private Docker/containers started. Repair only fresh task-root ownership/results
directory, keep failed unit untouched, dispatch runtime-v2 separately; don't
restart or weaken bootstrap checks. Image/assets export runs CPU-only onVM01
at low scheduling/I/O priority; source queue still750/790, no GPU contention.

19:15UTC fresh private runtime **PASS12s** onVM02: explicit private containerd,
Docker29.5.3/overlay2/NVIDIA runtime, zero images/containers, no system-daemon
or network changes. Runtime producing0365d057435e019ebdfb3cdc592c8172c9d01a70;
receipt/source hashes remote. VM01 export **PASS** producing671d10c:
assets1337763840bytes SHA9b876f95c80e1b68a9f7695960ee324d97b22d44575047730a017fcdb87e6027,
image14565534720bytes SHA203af62c8c03931919acd2fab28b1fa53a802be73d0f04d99febf60c26321b4e.
No whole user-disk/live Docker copy; private SSH transfer initiated only onVM01.
GPU0 observed89%/4342MiB, full prepare finished writing and native forward active;
not yet conversion/schema or J1 quality. No model/CUDA import result onVM02 yet.

19:23UTC original GPUqueue idle. Episode0forward complete→conversionFAIL2mm
fidelity (not adopted). J1/J2/TUD quality complete; externalTUD27nativeMoGe calls
13.539s and CPU1.919s pass91.5848%median scene gain/no regressions/coveragePASS.
No waitingTUD job remains to migrate and no duplicate inference will be launched.
Freeze/transfers existingTUDpredictions for DA3 comparison. Native15refinement
had pre-GPU queueFAIL because completedJ1 transientunit collected; no optimizer
or output ran. Explicit newcontinuation source/assets checks without reacquisition
or oldunitrestart. VM02import active, DA3publicacquisition dispatched; all data
remain Azure. Full tiny suite4093PASS/1optional-trimeshSKIP65.55s.

19:29UTC VM02 import image active, elapsed7+min and disk grows as verified
14.6GB image layers unpack; do not mistake quiet import for stalled/restart.
Independent DA3 acquisitionPASS35.157s, source+weights all exactpublisherpins.
Frozensensor prediction transfer68.086MBPASS, no duplicateMoGe. Native15
refinement newunitv3 dispatchedsource2286fbd, pendingactualoptimizerstatus.
J2 **paired**camera change againstJ1 is−22.8774%median, failing adoption;
its raw→scaled gain51.09% is a different question, not a camera improvement.
Opposite realTUD gain91.5848% supports domain-dependent calibration, not global
replacement. Launcher now sanitizes failed dispatch exceptions (no base64
payload echo/no automaticretry); legacytarget/sourcecommands unchanged.

19:34UTC native501refinement v3 **PASS258.872s**, 301 effective nativeupdates
for300steps; conversion and finalschema501 alsoPASS. No held-out HOI score.
VM02image import v1 terminated **FAIL after image load, before CUDA**: source
containerd image store reports OCI-index b47e4450…380a7, whereas fresh classic
Docker overlay2 reports platform-config 7ebfff18…c6d3. The sealed TAR explicitly
contains that OCI-index→linux/amd64 platform→config graph; it is not an image
change. Do not relax arbitrary image IDs or reimport/rebuild/re-extract. New
continuation verifies full TAR SHA203af62c…21b4e, each small graph digest/size,
exact pinned config, legacy manifest and every ordered rootfs diff-ID before
CUDA smoke and a fresh import receipt. Failedv1 log/unit and old snapshots stay
unchanged. GPU runtime/inference still unverified onVM02.
