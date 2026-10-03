# World Reward — V2D Track 1

Goal: a valid monocular human-object reconstruction submission that improves on
CARI4D, **not** an evaluation exploit. No superiority claim until held-out scores
verify it. See [research audit](docs/audit.md), [literature](docs/literature.md),
[baseline contracts](docs/baseline.md) and [experiment gates](docs/experiments.md).
Third-party source/model terms and unresolved eligibility are documented in
[licenses](docs/licenses.md); own code is Apache-2.0, not the external assets.

## Research status — 2026-10-04

No final Parquet, submission or verified CARI4D improvement yet.

- **Engineering:** all 30 original Track 1 videos pass byte/metadata readiness.
  The full 501-frame episode passes native preparation, forward, refinement,
  direct export and the original official packer; its temporary one-episode
  Parquet was deleted after QA. The next full 668-frame episode now passes
  preparation, forward, 301-update refinement, direct export and the original
  official packer.
  A further 866-frame episode has verified complete inputs, but its frontend
  continuation exceeded the original time budget: provenance PASS is not a
  timing PASS. Its full native preparation, forward, refinement and export
  and original official packing now pass independently. The next 592-frame
  episode has independently verified preparation, forward and native refinement;
  the outer queue failed its source-cache postcheck and is not reclassified.
  After a separate source audit, two export attempts stopped before model
  execution: first on historical-source binding, then on private-proof access.
  The explicit authenticated readonly-source access fix now passes a full native
  export of all 592 frames. Its independent export inventory now passes;
  CPU consumer validation and official packing remain pending. The 747-frame
  episode keeps all 25 object components but tracking stopped at 18 frames
  with empty automatic object masks. The next 668-frame episode has complete
  automatic masks and Body/depth initializers; its default topology reduction failed before object tracking. A separately
  gated volume-constrained proposal retains its three shells, including two
  cavities; full tracking is running and remains unverified. No frames, components or
  trajectories were dropped to rescue either episode.
- **Research:** a frozen DA3 depth hypothesis, anchored to MoGe by one
  scene-constant median ratio on the fixed 10% image border, gains **33.42%**
  median visible-object camera Chamfer on 12 external TUD-L frames. This is
  a background **proxy**, not semantic exclusion. The same three scenes/objects
  were previously used for development; absolute errors remain large.
  Independent generalization and full human/object/contact/motion accuracy
  are unverified. This is **not** a CARI4D victory or deployment authorization.
  Earlier root correction, DWPose prompts, gamma-medoid and global DA3
  replacement hypotheses were rejected; no post-score retuning/rescoring.
  The independent T-LESS acquisition stopped at its frozen layout/licence
  gates before heavy data or predictions; no generalization score exists.
  A new authored object-only RGBD fixture also stopped at its independent
  CUDA depth/ray gate before publishing RGB or running models; no score
  exists and the failed reference was not relaxed or rerun.
- **Eligibility:** upstream source/checkpoint licenses, training overlap and
  NVIDIA's separate registration remain unresolved before any submission.
  World Reward and all five Kaggle rule acceptances were verified on October 2.

A verified 19.91 GB asset-only transfer and extraction between Azure VMs
completed without local checkpoint traffic. The second runtime is not yet
ready: the selected Body/DINO source binding passes, and the minimal Grounding
image compiles offline and its thin Python import gate passes. The independent
SAM2 CUDA connectivity/hole-filling operator gate also passes, without models
or historical image-parity/eligibility claims. Replica model readiness is unverified.
The full frozen `3bee2c7` lightweight suite passed **10,570 tests, 2 optional skips**.
Earlier extraction-fixture failures were corrected using authenticated historical
Git bytes; the historical producer and production pins remain unchanged.

Heavy data, models and computation remain on Azure. Only reproducibility pins,
results and decisions are kept here; see [experiments](docs/experiments.md).

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
