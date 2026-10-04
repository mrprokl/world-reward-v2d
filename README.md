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
  export of all 592 frames. Its independent export inventory, CPU consumer
  validation and original official packing now pass; scratch Parquet deleted.
  The 747-frame
  episode keeps all 25 object components but tracking stopped at 18 frames
  with empty automatic object masks. The next 668-frame episode has complete
  automatic masks and Body/depth initializers; its default topology reduction failed before object tracking. A separately
  gated volume-constrained proposal retains its three shells, including two
  cavities; full tracking now passes on all 668 frames (3,605.69 s). Its CPU
  CARI input assembly is running; the native reconstruction/export remains pending.
  No frames, components or
  trajectories were dropped to rescue either episode.
- **Depth research:** D106 anchors DA3 to MoGe with one scene-constant median
  ratio over whole native-valid support. It gains **56.2007%** median
  visible-object camera Chamfer on 12 newly selected external TUD-L frames,
  but these are the **same three development scenes/objects**; absolute errors
  remain 33–38 cm. This is a narrow proxy, not independent generalization.
  D107 tests the unchanged recipe on three independent TUM recordings:
  **REJECT**, median relative AbsRel improvement **4.9046% < 5%**, with large
  absolute errors. Both cohorts are closed without retuning or frame mining.
  Neither establishes human/contact/temporal accuracy or a CARI4D victory.
- **New authored RGB pilot:** four fresh anchors pass the independent CPU
  triangle-ray rendering gate (`5536090`, 16.472 s). Automatic masks pass
  (`001aa0f`, 14.529 s): eight Grounding and eight SAM2 calls with fixed
  `person.`/`bottle.` queries, no manual prompts or model access to private
  authored geometry. Four full Body/MoGe predictions pass (`14ebc28`, 30.926 s),
  including native control replay and point-map projection. This is
  pipeline readiness, **not** reconstruction accuracy. The frozen predicted-
  geometry QA **REJECTS** the pilot: all four silhouettes pass, but one left
  hand lacks the required automatic-mask support. No threshold/seed change,
  192-frame run or bridge adoption; this cohort is closed. Earlier hypotheses and failed
  authored references remain closed; the new pilot does not turn them into PASS.
- **Eligibility:** upstream source/checkpoint licenses, training overlap and
  NVIDIA's separate registration remain unresolved before any submission.
  World Reward and all five Kaggle rule acceptances were verified on October 2.

A verified 19.91 GB asset-only transfer and extraction between Azure VMs
completed without local checkpoint traffic. The second runtime is not yet
ready: the selected Body/DINO source binding passes, and the minimal Grounding
image compiles offline and its thin Python import gate passes. The independent
SAM2 CUDA connectivity/hole-filling operator gate also passes, without models
or historical image-parity/eligibility claims. The new four-frame frontend pilot
passes; complete replica and reconstruction accuracy remain unverified.
The full `a06b703` lightweight source/test suite passed **10,944 tests,
3 optional skips** in 399.72 s. The synchronous GPU-lock queue retains unchanged
native stages. Opaque RoboTAP/BootsTAPIR acquisition passes on Azure; its known-
benchmark initial-query diagnostic remains pending and training overlap is
unverified. A separate native CPU check is required after the dependency build's
incorrect `einshape.torch` probe failed; that original failure is preserved.
None of these engineering gates is a reconstruction-quality result.
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
