# World Reward — V2D Track 1

Goal: a valid monocular human-object reconstruction submission that improves on
CARI4D, **not** an evaluation exploit. No superiority claim until held-out scores
verify it. See [research audit](docs/audit.md), [literature](docs/literature.md),
[baseline contracts](docs/baseline.md) and [experiment gates](docs/experiments.md).
Third-party source/model terms and unresolved eligibility are documented in
[licenses](docs/licenses.md); own code is Apache-2.0, not the external assets.

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
NVIDIA’s separate registration, source-license eligibility and producing GitHub
commit accessibility still require resolution before the first upload.
