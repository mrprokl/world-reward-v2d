# Azure runtime

Persistent working root: `/srv/scenesmith/world-reward` on the existing 1 TB
managed disk, **separate** from all prior SceneSmith work (do not modify it).
Current GPU: NVIDIA H100 NVL, 95,830 MiB, Azure West Europe.

- `code/`: mirrored project code only.
- `data/`: pinned Track 1 RGB/index/metadata inputs only.
- `vendor/`: pinned official kit and sparse reconstruction upstream source.
- `weights/`, `outputs/`: models and predictions, never copied locally or into Git.
- `results/`: input/artifact manifests, scalar results and decision evidence.
- `cache/`: reusable dependency/Hub cache; clean only World Reward disposable files.

`infra/bootstrap.sh` is a remote-only bootstrap: verifies exact official kit/sample
hashes, whitelist downloads, structural Parquet schema and all frame indices. It
does not train or claim benchmark superiority. Local tests do not need the dataset.

Access check 2026-10-02 (token securely obtained from Modal): SAM3D Body and Objects
artifact HEAD returns 200. CARI4D initially returned 403; after user granted access,
its exact pinned `2026-08-25-09-35-57/manifest.json` returned 200. Token securely
configured remotely, mode 600; no value printed or committed. Kaggle CLI installed
and API token in ignored mode-600 `.env` authenticated successfully. All five
competitions still require user rule acceptance in browser (entry check verified).

VM power is running, GPU confirmed through Azure Run Command. SSH port 22 blocked
from current connection despite correct /32 NSG allow and healthy daemon; Azure
Run Command works. Temporary SSH-on-443 test failed; listener stopped and its NSG
rule removed. Keep existing ingress/jobs unchanged; use Azure Run Command control.

Verified bootstrap complete (2026-10-02): systemd service exited 0; 490 MB Track 1
data and 21 MB pinned sparse reconstruction source/kit on Azure only. Downloader
validated 30 episodes, 16,563 frames, structural-only Parquet and SHA-256 manifests.
No video, mesh, weights or rendered frames were transferred back locally. Current
source code has lightweight tests; GPU inference has **not** been validated yet.

Existing VM was externally deallocated at 07:32 UTC. User confirmed other work
stopped and re-authorized exclusive H100 use; VM restarted, H100 reverified. Extra
VM attempts failed before compute creation; task-created RG/network resources are
were deleted (resource-group existence check: false). Existing user resources
remain intact. No additional VM needed.

Docker storage isolated at `/srv/scenesmith/world-reward/docker`, socket
`/srv/scenesmith/world-reward/docker.sock`; separate dockerd uses no bridge or
iptables mutation, builds/runs use host networking. This avoids filling the old
root disk or deleting prior Docker images. Model acquisition and runtime build
are systemd jobs; poll those exact units and never restart after mere observation
timeouts. Initial acquisition exited 127: launcher assumed a private `bin/uv`,
while bootstrap reused `/usr/local/bin/uv`. Fixed by executing the already-built
private Python environment directly. Initial build exited 1 before Docker work:
root Git ownership protection for the scenesmith-owned checkout. Fixed using a
single-command scoped `safe.directory`, without changing global Git settings.

Principal assets stage subsequently exited 0: 21 GB remote weights and verified
CARI4D/MHR hashes. This is not yet an offline-ready full baseline: FoundationPose
and DINO torch-hub/code/checkpoints are a separate auxiliary stage. CUDA extension,
EGL, model forward and MHR geometry roundtrip remain required. Runtime Docker build
is active; base image resolved digest is
`sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014`.
Transitive mutable Git dependencies still require freezing before final release.

Docker 29's containerd image store initially used the system containerd despite
the separate Docker data root, consuming root space. Stopped only World Reward's
jobs, then launched private `world-reward-containerd` with both root/state under
the managed disk and explicitly bound the private dockerd to that socket.
The original user containerd/dockerd remain untouched; no original images pruned.
Verified both processes use separate sockets/roots. Restarted the terminal task
build after this storage fix; root free 40 GB, managed disk free 667 GB.

First automatic-mask gate uses Apache GroundingDINO base (pinned HF
`12bdfa3120f3e7ec7b434d90674b3396eccf88eb`) with exact public object metadata,
16 candidate frames, confidence/ambiguity rejection, then full-frame SAM2.
Episode 15 is a predeclared engineering smoke, not a tuning/GT validation split.
`world-reward-masks-smoke` waits for the exact SAM2 image. A preceding synthetic
MHR forward/converter gate must pass 0.01 mm mean-per-frame residual. Neither
this threshold nor mask proxies establish challenge reconstruction performance.
