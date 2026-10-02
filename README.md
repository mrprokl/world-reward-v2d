# World Reward — V2D Track 1

Goal: a valid monocular human-object reconstruction submission that improves on
CARI4D, **not** an evaluation exploit. No superiority claim until held-out scores
verify it. See [research audit](docs/audit.md), [literature](docs/literature.md),
[baseline contracts](docs/baseline.md) and [experiment gates](docs/experiments.md).

## Layout

- `src/world_reward/`: input firewall and strict reconstruction contracts.
- `configs/sources.json`: official sources, pinned revisions and baseline scores.
- `tests/`: lightweight, data-free correctness checks.
- `docs/`: constraints, literature, experiment results and decisions only.
- Runtime data, weights, upstream checkouts and outputs live **on Azure**, outside Git.

## Development

```sh
rtk uv sync --extra dev
rtk uv run pytest -q
```

Remote Linux download (never run on the tethered local host):

```sh
uv sync --extra data --extra dev
uv run wr-data --config configs/sources.json --root /data/world-reward/data \
  --manifest /data/world-reward/results/input-manifest.json
```

Submissions use the original pinned official packer/uploader, not a reimplementation.
One frozen file goes to all five metric competitions after validation, quota check,
rule acceptance, team identity **World Reward**, and accessible producing GitHub commit.
No Kaggle credentials are currently configured in this workspace.
