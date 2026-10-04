# Azure runtime — current state 2026-10-04

## Current jobs and gates

Episodes15/1/2 full501/668/866 native shared preparation, forward,301-update
refinement, direct export and official CPU packing all pass in immutable
namespaces. These are engineering/fidelity results, not independent challenge
accuracy. All30 original RGB/hash/container-metadata readiness passes:
16,563frames,30Hz/1536x1152, no frame/model/GT decode in that readiness audit.

Episode3 full592 native refinement and export now pass after independent
historical-source authentication; the outer queue/cache and two earlier export
FAILs remain separate. Its independently audited CPU consumer and original
official packing pass; temporary Parquet deleted. Episode4 full747 preserves
all25 components but tracking stopped at18 empty automatic masks; no full-chain
PASS. Episode5 full668 initializers and a separately gated three-shell/cavity
volume proposal pass; full tracking now passes668/668 in3605.685147s. One GPU job
per VM; scoped locks release before CPU preparation for disjoint overlap.

EP5's original CPU input assembly passes all668frames; shared preparation now
passes all668frames in21.814658s; the GPU lock is released. EP6's default topology
reduction failed
before any pose; an independently qualified volume proposal and authenticated
empty-directory archive preserve that failure. The unchanged full816-frame
volume tracker passes all816frames in4794.561042s; CPU CARI input assembly is
active. EP7's empty fixed
anchor is closed without reroll; EP8's complete634-frame initializers and minimal
private archive pass. Its first private transfer stopped on the existing NSG
deny before any bytes; a temporary exact private-pair TCP2222 rule fixed that
transport barrier, not the prediction algorithm. The complete private transfer
passes, but the unchanged VM02 tracker fails topology before any pose. Its
isolated empty failed output is preserved; no canonical pose is promoted. The
temporary ingress/listener and both newly generated private keys are removed.
A separately qualified volume proposal and new full634-frame archive/transfer
pass independently; its unchanged volume-based native tracker is active on
VM02. New transport ingress/listener/keys are also removed after receive sealing.
All masks/frames/shape/scale contracts remain unchanged.

VM02 Boots native CPU verification, full three-video inference and preregistered
CPU evaluation pass: meanAJ0.6679 versus static0.1778. The earlier API/format/
receipt-access FAILs remain preserved; oracle initial queries, a static negative
control and unknown training overlap preclude a3D/generalization/CARI4D claim.
Full-contiguous YCBV CPU acquisition fails its900s download deadline, with
complete disposable cleanup and no ZIP/private-label interpretation. The authenticated first FAIL
is atomically archived; the single technical3600s continuation now fails
archive inventory in1701.101840s with original source/cleanup PASS and zero
retained public/private inputs. Cause is not yet proved; the independently
verified tiny base ZIP layout matches exactly. No third acquisition is authorized;
only a bounded public header-only diagnostic is next;
automatic3D point-pose validation has not run. Its three-anchor depth and
automatic-mask preflights and identical-pool comparison are separately gated.

D107 real-depth independent validation is closedREJECT: median4.904632756%
gain is below frozen5%; no retuning or rounded success. Native offline temporal
depth is a new hypothesis, not acquired/adopted; OpenLORIS registration is not
yet proved exact color-camera Z. Fullsuiteabe793f **11837PASS5optionalSKIP425.23s**; later EP5
inventory/scheduling checks281PASS1platformSKIP48.19s.
No final Parquet or verified CARI4D
improvement. Heavy models/RGB/arrays stayAzure.
Frozen receipts, producing commits, budgets and decisions are in
[experiments](experiments.md). No final submission/CARI4D superiority yet.
Older completed engineering records below are historical, not active jobs.

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

Azure Run Command controls jobs; earlier SSH tests failed and temporary22/443
ingress rules/listener were removed. Separate owned VM02 now exists on Azure
with no publicIP, a byte-sealed equivalent Body runtime and1TB owned data disk
mounted `/srv/world-reward-data`. Native model/data work remains remote. A
19.91GB frontend asset-only archive passes independent audit, private transfer
and extraction on VM02; no local checkpoint traffic. Selected Body/DINO source,
offline Grounding v6 CPU imports and independent SAM2 CUDA operator gates pass.
Model readiness, historical-image parity and final licence/train-overlap
eligibility remain unverified. Earlier failed empty resources
were cleaned; preserve all unrelated SceneSmith infrastructure.
Private Docker socket `unix:///srv/scenesmith/world-reward/docker.sock`, with
private dockerd/containerd data and state on managed disk. No bridge or iptables
mutation. Inference uses `--network none`; acquisition downloads directly toAzure.
**No videos, weights, meshes or renders transit the tethered laptop.**

## Reproducible jobs

`infra/azure_job.py` defaults to a clean committed worktree; optional strict
40-hex `--revision` archives only that verified commit during disjoint edits.
It sends only source import
closure plus small package/config as SHA-verified XZ (≤256KB base64), extracts
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
- `world-reward/official-pack-cpu:0.1`: separate pinned CPU child1a04b193,
  verified Arrow19.0.1 addition only, original CARI parent b47e unchanged.
  CPU build passes68.37s; first official pack smoke rejected14.38s for exact
  source-vs-packed surface mismatch. Readonly audit proves original GLB surface
  unchanged: source trajectory uses native FP32, GLB/packer FP64. New dual-exact
  geometry/source gate passes actualv2 official packing14.43s:501fullframes,
  356originalscoredframes,26,031rows, unchanged4,080triangles. Scratch deleted;
  receipt only, no all30/finalsubmission or accuracy claim.
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

19:40UTC VM02 import continuation **PASS**: sealedindex/platform/config graph
and all44orderedrootfs layers identical, actualCUDAeye-matmulPASS/H100NVL.
Effective classicimageconfig7ebfff18…c6d3 explicitlyrecorded vsoriginalOCIindex
b47e4450…380a7; no rebuild/reimport. Native501refinementreceiptSHA
22b3c20a2f8b948630e2f5cbc30dbb4ef9d713501037a2796475ab9bffd1891c,
conversion63d7d54974dc2bcae0bae285bb0de651b20de894283810a70d120eb27d7c236b,
finalschema18d0b18e7c68e4244b70c2bd337ced5bd6faf5d168b3e0458a6a72aba12e8d8f.
Conversion132.557s/mean.826410/worst1.787483mm all501PASS2mm unchanged.
TemporaryprivateSSHkey/authorizedkey removed and newNSGallowrule deleted;
newVM02 nowonlyexplicitdeny-all-inbound4096, unrelatedkeys/networkuntouched.
DA3chain9735385 dispatched(runtimeCPU→frozenbaselineimport→9actualGPUcalls→
pairedCPUprivatequality). Native0dispatch initiallyConflict whileVM01keycleanup
RunCommandpending: nojobstarted, inspectabsence beforeexplicitnewdispatch.

19:43UTC DA3validationv1 terminates beforeGPU atactualimportmissingaddict,
expectedfail-fastCPUgate. AddexactauditedMITwheelzipimportRO, noimagechange/
sourcepatch/requirementsresolver. OldCPUgatev1 remains; v2newnamespace. Native0
notstarted verifiedunitnot-found/logabsent afterRunCommandConflict; nextexplicit
launch legitimate afterserialization. No duplicatedinference/GPU reader.

19:47:52UTC native0v1 actualGPU100%17738MiB/100of300steps, noforwardduplication.
VM02D76chainactualimports+frozenbaseline9numeric+DA39callsPASS11.290s, paired
CPU2.339sPASS19.4160%medianindependentscene gain atidenticalK800. Human/camera
accuracyandV2Dscoreunverified. Source8632e50 sameoriginalprotocol/codepins.
4172tinytestsPASS/1optionalSKIP64.23s beforenewaffinecoreimplementation.

20:11UTC J3prepare actualPASS: independent18RGB reference synthesis7.533s,
36automaticDINO/SAM2masks17.367s; source2867ab1. Privatecamera/GT nevermounted
for masks. Renderreceiptf594c4b24e857c3b42836bf6e03d6ef6540a03d88ccd198caa7e5b0a8807cf78,
maskreceipt51fc4b61113520728c39e80a489f1a85e8e3af32462892e99376595496e35a7e.
Exactcommittedbundleprepare51788B/validation68192B/diagnostic52700B allunder
100KBcode-onlycontrol, no localheavytransit. Fullsuite4297PASS/1optionalSKIP.
VM02verifiedcompletedimage/assets/TUDpredictionTARs removed15,971,384,320B
ONLYaftersealedimport/CUDA/sourceimage/frozenbaseline/DA3prediction/quality
receiptscheckedPASS; sourceassetmodels/privateeval/frozenoutputs unchanged.
Cleanupreceipt65659f3f0707f47dc6e2137194c181a7d994c8aa5a1ea3299578519e92fd5791,
VM02disknow35Gused89Gfree. Receiptsretained; originaltransferSSHauthrevoked.

20:20UTC sourceVM01sameverified3completedTARs removed15,971,384,320B after
transferdone receipts+exactSHA/bytecounts and independentVM02import/quality/
cleanup confirmation. Cleanupreceipt56f8652ff5da9338ce2eabcd2a9167564d45e1b22f5a41a0e06af977dd47db91;
sourceoriginalmodels/inputs/outputs/notices unchanged, alltransferreceipts kept.
TotaltransientTARcleanup31,942,768,640B acrossAzure; nothing downloadedtoMac.
Sealedconverterrefined0diagnosticcb75806 active, no simultaneousVM01GPUjob.
J3fullcohort hypothesisREJECT−6.6042%median, engineeringallstagesPASS.


2026-10-03 readonly re-inventory: VM02 is stillrunning in
WORLD-REWARD-RESEARCH (notVM01resourcegroup), private10.0.0.9/H100NVL95830MiB,
GPUempty. ExistingprivateDocker/runtime/import unchanged; noVMcreation/rebuild.
Runtime1424B/dea57d51f7edb1b9313a430578f28b2821cb3e4810b2cfa0c859226160d5e0dc;
import4272B/c98fb4ec995f2cbaecaeabecb0e1b35a5192e6a828efb5710f599197fd67b549;
imageidentity3778B/9bf4d946b2ad116c57c2fb88cdb37f92535df5b3238a069deec1a592cffb80af.
Importproducer43d7c4f8fd77f32b9a701b6fac2ba274994d231a proves originalOCIindex
b47e4450→platformde690d04→config7ebfff18, all44rootfslayers and actualCUDA.
Never pretend classicconfig is originalindexID. MHR/v2receipt/CARIsource/Track1
inputs absent; cannot run originalVM01neutraljob there without distinctproper
binding/assettransfer. NSGonlyexplicitdeny-all-inbound4096; originalSSHrevoked.
No transfers/network/keychanges made; 93,890,007,040B free,128GBOSdisk.

12:29:39UTC VM01episode2failedatobject-topology preflight after866body/depth;
GPUempty/scopedlockfree; episode1CPUdepth350/668active. NoCPUbatchrewrite,
wholepipelinePASS or native668pins yet. Predeclare one separate report-only
neutralexactwitnessGPUdiagnostic and one disjointCPUvolumeproposal afterfreeze;
heavyassets stayAzure, no duplicatepredictions. Publicexactc37efc71351fe5e8c8536f87143b2ff8769b6cfe
verifiedunauthHTTP200; no finalsubmission or codeeligibilityclaim.

2026-10-03 actualneutraldiagnostic/CPUvolumeproposal bothinactiveexit0;
originalproducer3242cb23486726c9636ca66241020f6aa276be1b. ONE neutralGPUcall,
3.378421s/all8exactcrosswitnesses, no geometrypayload; disjointCPUproposal
5.397416s/18originalcomponents/4096faces/fidelitygatesPASS. GPUemptyafterward,
episode1scalarCPUdepth450/668active. NoVM02assettransfer or native668job yet.


2026-10-03 19:07UTC readonly actual production observation: EP3pinnedvolume
full592tracking stillactive, MainPID895234, one895323GPUworker1886MiB; same
18:41:43start, no stagePASS/inputassembly/nativechain completion yet. No duplicate
GPUjob or failedunitrestart. Separate VM02analyticreferenceCPUgatePASS but
blinddepthsupportFAIL13.608s; no containers/GPUworkers left19:04:53UTC.

FreshVM01image inventory for possible Azure-only frontendworkerclone:
world-reward/grounding:0.1 actualimage53b33bc4b60e0e3e8f83b401775b4701b18eef54408fd585fbe3a5d376c042e1,
DockerSize10914551643B; world-reward/sam3d-runtime:0.1 image
eb389b26358c49778a14303b5875c66d887824011388ce9f8666ed7cc1841ce5,
DockerSize10141699500B. These are inspected IDs/size only, NOT sealed export
TAR identities/classicconfig equivalence/licenseeligibility/CUDAproof onVM02.
Firstrootfsprint exceeded Azure4KBtail; boundedsecondreadonlyinventory returned
actual IDs+size. No image exports, data/modeltransfers, keys/network changes or
storageattachment performed. VM02existing93GBfree is not yet validated for full
frontendreplica capacity; minimumasset/auditedsourceclosure remains prerequisite.


2026-10-03 frontendclone prerequisiteaudit: lib_mhr package __init__ transitively
imports17files160540B, not onlybody_pose/mhr_layer/rotations; prep10files165589B
plusbehave_data/toolsinitializers+video_reader/pipeline_timing11867B. SAMBodyall
Python+LICENSE incl6dataPython code actualinstalledparity. No CARIcommercialckpt
neededforfrontendsendingcari_inputs, but Body2.109GB+bundledMHR.696GB, Objects
6ckpts12.449GB/7YAML, groundingSAM2nine1.832GB, DINOv2reg4twoactualpinnedckpts,
DINOv2/v3cleanpublicsources, internalMoGe1 (constructedevenalignedpointmap)
1.256GB atad326bfb61facd6c52b5a825bc1e34d7c97d9672/modelSHAda96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f.
Primarysize/hash doesnotproveactualVM01HFsnapshot/XETlinkbinding. Needreadonly
actualwhitelistinventory, sealedimagesOCIplatformconfig+rootfs proof, newVM02
cameraCUDAgate andnewscopedwrapper (oldposemountswholevalidationprivateTUDL)
beforeclonefrontends. Nofilteredsource/weights/imageexports orAzurepeertransferyet.

User-authorizedstorage preparation: newowned1TBPremium_LRSdisk firstcreation
world-reward-vm02-data-v1 succeeded but attachFAILED because VM02zone1 anddisk
hadnozone. Noexistingdiskformat/mount/replacement. Immediatedelete ofownempty
unattachedwrongzone disk was refused due pendingattachoperation; preserveuntil
providerstateconsistent. Newseparatezone1diskworld-reward-vm02-data-z1-v1
creationPASS, subsequentVMshowprovesattachedLUN0/manageddiskid, cachingNone;
OS/unrelatedresourcesuntouched. Stillmust inspectactualguestLUN/size/blank
filesystem beforeformat/mount atcanonicalnewpath; no disk-successinference from
CLIquietoutput. NoexistingrootDockerrelocation/symlink/credentials/networkchange.

19:19:26UTC actual gueststoragePASS: newzone1/LUN0 device /dev/sdc,
serial60022480b2eeb99d9fd9b5d85692cc3a, exact1099511627776B andnochildren/
filesystem/signatures/mounts checkedbeforeformat. Newext4 UUID
24df126a-5f5f-41d8-801c-9ddaa7a582d8 mounted /srv/world-reward-data rw,noatime;
actualavailable1026108792832B. Ownedreceiptresults/vm02-data-z1-v1.json0444,
UUIDfstabpersisted. ExistingOS/resource/data neverformatted, runtimenotrelocated.
Wrongzoneownedunattacheddisk recheckedSucceeded/nullmanagedBy beforedeletion;
provider deletionpending, notclaimedyet. No heavydata downloadedlocally.
Wrongzoneemptyowned disk deletionexit0 and independentResourceNotFound confirm
cleanupDONE; no extraunusedmanageddisk retained. Only actualzone1/LUN0data disk
wasformatted/attached. No VM02image/assets transfers or frontendclonePASS yet.

D106 newTUDL acquisition8.860s, nativepublicdepth25.025s andprivateCPUquality
3.301s allactualPASS in separatefrozenAzureVM02namespaces; originalmodelsand
privateGPUisolation retained. No data/checkpoint/rendersdownloadedtoMac, only
bounded13input/predictionpins andqualitydecision summaries. VM01full592EP3
tracking stillactive19:32:51UTC,500framescomplete/one1886MiB895323worker; no
finaltracking/assemblyPASSyet. EP4frontendchildrenabsent, no duplicateGPUjob.

Nextclone prerequisite inventory frozenreport-only: actualminimalBody/Objects/
Grounding/DINOv2/v3 publicsource+learnedweights+MoGe1 exactsinglesnapshot/relative
blobgraph required. Hoststdlib/nice15/ionice3/600s, noDocker/GPUmutation/export/
transfer/data/outputs/validationread. Fullindividualmanifest staysAzure, stdout
bounded4KB. Rawimagebuildreceipts evidenceONLY/nevertransfereligible; projected
actualID/platform/rootfs only, configgraph remainsunsealed, nolicense/CUDAclaim.
Agent35tinyPASS0.32s/bash-n; no actualinventory or replicaPASS yet.

ActualreplicainventoryFAIL21.181993407s, producerf895e4a/43files38936encodedB:
incorrectrequiredkitv2dlb/__init__.py absent. Receipt771B/
3fc2a05902c88164023bac39639e96f7699580dc7d70b53cde1d214547dfd848.
Readonlyactualkitlisting confirms namespace package with11genuinePythonfiles,
notregularpackage; nonecreated/modified. Newdistinctinventory corrects only
sourcecontract, rejectsfabricatedinitializer, retainsoriginal2KBmesh_budgetpin.
36tinyPASS0.23s; noimage/assets/credentialtransfer/clonePASSclaimed.

EP3objecttracking actualPASS592frames3360.15873205s; sameproducer6242389,
originalwholegeometry/fulltimeline preserved; nowCPUcari_inputdepthassembly
active50/592 19:48:34UTC. EP4frontenddispatch initiallyrejectedbyAzureRCConflict
beforeunit/log/snapshot; readonlycount0/allabsent establishedbeforelaterdispatch.
ActualfirstEP4frontendproducerf895e4a/83files139692encodedB ran919457 andFAILED
automatic_masks19:43:55UTC, actoridentityclosest2/3observationscoverage2/3 failed
unchangedgate. Nomanualidentity/label/thresholdrelax/restart. Preservefailureand
diagnostics, auditgeneralalgorithm onnonchallengefixtures beforeanynewmethod.

Actualreplicainventoryv2 PASS23.102188225s; producer40fdc278/44files46700encodedB,
actualpublicsourceHTTP200, VM01inactiveexit0 19:53:22UTC. Receipt149203B/
95d09454133e481324380e9253d235ba3bbdd3657e1fac25f97e8f7c4f1741f1;
entrydigestbdae24745ec4d49e61557f6930ba81e21a4b62ce940d042b7f858fb55e28177f.
Independentactualcaller/hash/sourcefilepin+allentriesrehashed19:56:34UTC:
448files+3links19910803804B,3rawbuildreceiptsevidenceONLY/nontransfereligible.
MoGe1exactad326snapshot→2relativebloblinks→actualflat22/22786b099760587050342c6bbafb8d6f2608c36deee2aa793e1fbafebe31cb6a,
1256823446B/SHAda96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f.
Actualimageindex/rootfs counts body44/grounding23/Objects37 intact. Noimageexport/
transfer/CUDAcamera/installedsourceparity/licenseclearance/replicaready yet; next
sealableownedimage+assetexport canuse thisauditedwhitelist, neverwholeweights/cache.

TUMv1acquisition actualFAIL0.882454892s beforeanyarchivebody/publicRGB/depth;
receipt2412B/3cf80bcc5fd992c64765971084c40a50a5134136af95a6437bb95c981c424e85.
ActualreadonlyHEAD+GET(noBODY)20:01:29UTC proves soletransportcontractdefect:
publisher originalURL redirects to exactwebshare.cvg.cit.tum.de/g/rgbd/dataset
suffix, all200/ContentLength344011403/ETag/LastModified identicalfrozenHEAD.
No changedbytes/labels/predictions. Preservev1FAIL/licenses/emptyinputs; distinct
transportv2namespace andhardmappedofficialredirect willretainALL24sourcepins/
scientific/licence/model/support/evalgates. DoNOTreinterpretv1 assuccess.

Fullsuite9806PASS/2optionalSKIP/2FAIL266.14s: genuinehumanphotometric/official
packercodeclosures now162312/163116encodedB duefullyretainedconfigs, notmissing
dependencies/numericfailure. Raiseonlylocalcode-controlcap160→256KB, preserve
ALLcommittedconfig/provenance andboundedoversize-rejection tests; no heavydata
transport change. ActualCLIalreadyaccepted146812B forfrontends, capnotclaimed
Azureuniversalhardlimit. Focused202PASS6.29s; newfullsuite recheckpending.


20:19:31UTC actualD107transportv2acquisitionPASS62.429337204s: producer
70bf74e18b0989bc0573f71d4e121caa54a8ab30/46files45824encodedB publicHTTP200.
VM02inactiveexit0/GPUempty; complete24 selectedbytes retained, original4.029GB
archives removed. Receipt21050B/
616090b11060005ad28476964de36d4b82894f5b27f61e45b86481b95870d8e4 0400.
Originalv1FAILreceipt/publicempty preserved before/after. Nativeinfer/private
qualityNOTyetrun; independentactualaudit/inputpin freeze precedesanyGPU.

20:13:44UTC VM01EP3CPUassemblyactive300/592/noGPU; EP4originalunitfailedexit1
MainPID0 andONLYautomatic_masks/seed-diagnostics.json. Independentactual
diagnostic8017B/675c2629f6cf786f51be576188301b7f7863bc34cfc0c9f267d96d4f06d4b19c
andoriginalunitlog2927B/90f705e8b766b290158b60adecc6f5b53022b05263d76ba75e63d0b3cc161970
pinned. 20:18:14UTC actual8selectedsourcefiles allmatch oldGitf895e4a exactly;
inputmanifest13081B/3df960ce0f594b8f51675b21bb070925de7aa87a583332674eb89b0e90fc6263
and selectedoriginalEP4RGB/two publicmetadata hashesverified. No failurearchive
ornewmaskrunyet. Archival CPU contract preservesoriginalFAIL/log/diagnostic via
LinuxatomicNOREPLACE rename, never deletes/overwrites/reinterprets.


ActualEP3structuralmetadata20:43:52UTC: originalpublicRGB1152×1536, length592;
20:45:21UTC originalcamera labelleft_stereo_camera_left (hashverifiedpublicmeta).
NativeCARI export intentionallynames its sole camerastream MHR_CAMERA_NAMES[0]
(front_stereo_camera_left), independentofphysicalcamera label; do notconfuse
structuralname withGTcalibration orrelabeltheoriginalvideo. The actualpublic
exportformat needsfront_stereo_camera_left in clipinputspec AFTERreceipt.
EP3CPUassemblystillactive550/592, noinputreportyet; EP4initializersactiveone
939915GPUworker4420MiB. SchedulerdoesNOTstartEP3nativeGPUuntilEP4GPUisidle.


2026-10-03 21:56:56UTC VM02 CPU preflight: workspace/data mounts UID1000
mode755, /run/sshd UID0/755 and runtime UID0/700; no peer unit. Private
sshd control will use new /run/world-reward-frontend-peer-v1, not a
UID1000-owned ancestor (OpenSSH StrictModes); existing parents unchanged.
VM01 initial client-key setup failed before key generation because its
runtime parent does not exist. Fresh transfer/frontend-peer-client-v1
actually created 21:52:48UTC: private600/public400, only public identity
113B/2e2365faeb3b41f01ecd4657b6520617026874e2255c8dd698eab8181fee948b
returned. No private bytes logged or copied.

VM01 installed SAM2 source is VCS2b90b9f5ceec907a1c18123530e92e794ad901a4
(actual distribution metadata, not guessed Git HEAD). Grounding environment
uses transformers4.53.3/tokenizers0.21.4/safetensors0.6.2/hub0.36.2,
numpy2.1.2 and OpenCV5.0.0.93; VM02 Body has transformers5.3.0/
tokenizers0.22.2/safetensors0.8.0 and lacks SAM2/Objects. A new scoped
Grounding child must pin dependencies and validate ABI; no current image
parity or licensed submission eligibility inferred.


22:04:09UTC EP4 whole747tracking stillactive962210/single9623011886MiB;
EP3queuednative965420 waits on existing lock, no reports. Failedpeer1
transport usedzero incomingbytes, so sharedGPUresearch continued untouched.
A diagnostic AzureRC was rejectedConflict before any execution while
sender-dispatchRCstillrunning; later read-only dispatch waited for that
client tocomplete. No unit/namespace blindlyrestarted.

Existing VM01 Objects CPU runtime inventory951868e0 actualPASS19.230776s.
Independent full asset/report/image rehash passes; restricted root/cap-drop
CPU probe reads181narrow selected assets/source leaves and verifies actual
image_to_mesh signature includes stage1_only, without models/CUDA. The59KB
runtime staysAzure, referenced by tiny measured pins; no image/weights copied.
Installed180PY fingerprints and37actualLayers are measured runtime facts only,
not upstream commit parity, training-overlap exclusion or source-license waiver.
