# World Reward — V2D Track 1

Read `/Users/thomasgomez/.codex/RTK.md` when it exists. Prefix shell commands with `rtk`
on the local host; remote commands may use the remote runtime directly.

## Non-negotiable research contracts

- This repository is **Track 1 only**. Never fetch/use Track 2/3 assets, challenge
  multiview recordings, ground-truth trajectories/meshes, source camera calibration,
  or sequence-matched FORM-HOI data. Even if publicly downloadable, those are not
  legitimate Track 1 inputs (official FAQ).
- All heavy data, models, inference, training and rendering stay on Azure. Local
  workspace holds code, tiny tests, concise results/decisions only. Do not transfer
  videos/checkpoints/renders back to the user's tethered connection.
- External pretrained models/data are allowed, but verify source, license and
  challenge overlap. Never claim an unverified checkpoint is leakage-free.
- Explicitly disable any upstream GT/oracle mode. Validate provenance before run.
- No score gaming: no object deletion/shrinking to evade penetration, no static
  trajectories to evade acceleration, no sample values used as predictions, no
  per-frame evaluation alignment. Reconstruct the actual interaction.
- Keep original frame indices and full trajectories, including occlusion. Human
  and object share one frame; shape/scales/object geometry are clip-constant.
- Choose hyperparameters on non-challenge validation; video-only self-supervised
  adaptation is distinct from held-out performance validation. Never call proxy
  improvements a verified victory over CARI4D.
- Use subagents for independent literature/baseline audits or disjoint experiments
  when useful. Do not duplicate GPU jobs or writes to the same files.
- Keep secrets out of terminal output, logs, files under version control and
  submissions. Reuse credentials securely. Never print environment variables.
- Record only reproducibility manifests, experiment results and decisions. Fail
  fast with predeclared gates; clean disposable downloads and failed run noise.
- Submit one frozen Parquet to all five competitions under **World Reward**;
  Thomas Gomez only where registration requires a personal name. Check rule
  acceptance, GitHub commit accessibility and weekly quota before upload.

