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
artifact HEAD returns 200; `nvidia/cari4d_commercial` returns 403. User must accept
the repository's license/access gate. No token value was printed or committed.
Kaggle authentication and individual competition rule acceptance remain prerequisites.

VM power is running, GPU confirmed through Azure Run Command. SSH port 22 blocked
from current connection despite correct /32 NSG allow and healthy daemon; Azure
Run Command works. A temporary SSH listener on 443 is being tested; remove it and
its task-created NSG rule if ineffective. Do not alter existing ingress or jobs.
