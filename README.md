# World Reward — V2D Track 1

Goal: a valid monocular human-object reconstruction submission that improves on
CARI4D, **not** an evaluation exploit. No superiority claim until held-out scores
verify it. See [research audit](docs/audit.md), [literature](docs/literature.md),
[baseline contracts](docs/baseline.md) and [experiment gates](docs/experiments.md).
Third-party source/model terms and unresolved eligibility are documented in
[licenses](docs/licenses.md); own code is Apache-2.0, not the external assets.

## Research status — 2026-10-03

No final Parquet or verified CARI4D improvement yet. Two independent synthetic
RGB pilots rejected root-only correction despite valid execution/replay:
translation XYZ worsened median camera PVE32.26%; fixed-depth native XY/Euler
worsened2.19% and failed the silhouette safeguard. All frames were retained;
private labels were used only after frozen predictions, never for fitting.

The untouched24-case factorial diagnostic now rejects external DWPose prompts:
Body2D error2.34px vsDWPose5.11px, worse in all8groups. Appearance changes
increased centered geometry error despite identical reference geometry.
A separate scoped-native photometric execution test now passes exact SHAM,
after two preserved runtime failures. It validates reproducible mechanism on
one fixture only; the medoid kept the original image prediction. The subsequent
untouched24-image human-only comparison under clip-constant identity also
rejects gamma-medoid:0% median gain, two changed predictions slightly worse.
No retuning/rescore. The native96 shared-identity preparation now passes;
the actual HOI forward and saved-output validation now pass after an exact
signed-zero ABI correction, without changing predictions. Native301-update
refinement and direct export now pass on96frames. Full501 shared preparation
also passes, as do the fresh full501 HOI forward,301-update native
refinement and full direct export. The complete pipeline now passes on one
public video, including the source-bound CPU episode consumer. All30 original
videos now pass the metadata/hash readiness audit; the first structurally clean
new episode is running through the automatic frontends. The original official
packer also passes on the full501-frame source, with exact GLB-surface and native
FP32 fidelity checks; its temporary one-episode Parquet is deleted after QA.
The next episode's object trajectory passes all668originalframes; CPU input
assembly is ongoing. The following episode failed the original fast-QEM mesh
budget gate, then passed one independent volume-preserving proposal retaining
all18components; its full866-frame tracking is now running from frozen pins.
A fresh native near-grasp manufacture route remains stopped at neutral
self-embedding; exact rational witnesses confirm eight flagged crossings.
No geometry or thresholds are relaxed to rescue it. One separate authored
full-human reference feasibility gate is predeclared; it is not a new prediction
method or a quality result. All30 reconstruction and independent real/fullHOI accuracy
validation remain separate.
Detailed receipts and decisions: [experiments](docs/experiments.md).
Heavy data and all GPU work remain onAzure.

## Layout

- `src/world_reward/`: input firewall and strict reconstruction contracts.
- `infra/`: Azure-only acquisition, isolated image builds and GPU smoke gates.
- `configs/sources.json`: official sources, pinned revisions and baseline scores.
- `tests/`: lightweight, data-free correctness checks.
- `docs/`: constraints, literature, experiment results and decisions only.
- Runtime data, weights, upstream checkouts and outputs live **on Azure**, outside Git.

## Development

```sh
rtk uv sync --extra dev
rtk uv run pytest -q
```

Optional parity checks against the already-audited official kit use
`WR_KIT_ROOT=/path/to/v2d_submission_kit`; only source helpers are imported and
template I/O reads `row_id` alone, never sample prediction values.

Remote Linux download (never run on the tethered local host):

```sh
uv sync --extra data --extra dev
uv run wr-data --config configs/sources.json --root /data/world-reward/data \
  --manifest /data/world-reward/results/input-manifest.json
```

Final submissions use the original pinned official packer/uploader. Independent
schema assembly/roundtrip tests guard its input/output contracts; they are not a
replacement metric or evidence of reconstruction accuracy.
One frozen file goes to all five metric competitions after validation, quota check,
rule acceptance, team identity **World Reward**, and accessible producing GitHub commit.
Kaggle credentials are configured in an ignored secret file; browser acceptance
of all five competition rules and team name was verified on 2026-10-02;
the own-code producer history is now public at
[mrprokl/world-reward-v2d](https://github.com/mrprokl/world-reward-v2d).
NVIDIA’s separate registration, source-license eligibility and the final exact
producing commit/reproduction still require verification before the first upload.
