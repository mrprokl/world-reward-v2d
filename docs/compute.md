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
Latest full tiny suite3323PASS33.34s. No new submission or production mesh.
