# Track 1 input and evaluation re-audit — 2026-10-08

## Decision

Keep initial localization task-conditioned and short: official single-view RGB
+ object_prompt + action → principal person/object boxes → SAM image masks.
No full-clip proposal bank or tracking is required to test this first stage.
The object identity is already provided; matching its visible instance and actor
is the real problem. This engineering simplification is not an organizer rule.

Track 1 contains NO provided masks, boxes, keypoints, depth, calibration, poses
or meshes. The public Parquet is six indexing columns, not reconstruction GT.
CARI4D requires prepared masks; its manual SAM2 demo is neither a supplied Track1
annotation nor permission for manual test labeling. Keep fully automatic prompts.

## Verified sources

- Track1 main revision `5f68335f3acc802033d1e80728c1633197521de8`:
  https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/5f68335f3acc802033d1e80728c1633197521de8/track_1/README.md
  episodes_metadata.jsonl SHA256
  `cce105292e2b670f32b49e0504b0ece67b8aa860ea69dc9e728fce97531b53fe`.
- Official FAQ (pages2–3):
  https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf
  88947 bytes, SHA256
  `6ec86e8e32e817b325fca4c0800932a421b49df951045568d46a91a561d67de8`.
  Track1 only; no Track2 meshes/calibration. External models/data allowed.
  MHR, continuous trajectories through occlusion; no reference pose/mesh access.
- NVIDIA code revision `12c36fb129c1a739a383c64f682d8d0a4b7e354d`:
  reconstruction/modules/v2d_cari4d/README.md; reconstruction/docs/mv_hoi_local_pipeline.md;
  reconstruction/modules/v2d_sam2/lib/mv_videos_to_masks.py.
  Their automatic object initialization is GroundingDINO(description)→SAM2,
  actor Detectron2 tracks→principal actor→SAM2. A best-confidence seed need not
  be frame0. This is a reference recipe, not evidence of superior Track1 quality.

## FORM-HOI insight evaluation, NOT matched challenge annotations

Card revision `c63db107e84c7f74bb4929ef643b67b5c8bcc00e`:
https://huggingface.co/datasets/nvidia/form-hoi/blob/c63db107e84c7f74bb4929ef643b67b5c8bcc00e/README.md
CC BY-4.0. Current cleaned release has4135 sequences (not4315 from the slide),
excluding held-out object sequences. It supplies four-view human/object masks,
SOMA/MHR trajectories, rigid object poses, metric mesh, depth/calibration,
symmetry/ground plane and failure_segments.json. These are reconstructed
pseudo-GT with imperfect automated/human QC, not perfect motion capture.

Metadata-only audit: archives.parquet272115 bytes SHA256
`856e297f6c4f70a2c9b6041f327598c94c1bd935a1745cdb5056dc2973bd36ef`;
4135 FORM IDs vs30 public Track1 IDs: exact-ID intersection0, recording-timestamp
intersection0, none of the10 Track1 object slugs present as sequence prefix.
This allows a disjoint external cohort, NOT proof of pretrained-model independence.
Content/alias duplicate guards remain required before any model-facing inference.

Single-camera RGB + automatic official-style text ONLY mounted for inference;
FORM masks/pose/mesh/depth/calibration/QC stay exclusively in an isolated evaluator.
Start with frames0/14/29 and quantify pair localization/mask overlap, abstentions,
latency/cost. Report all cases and separate unflagged-QC subset; group splits
by sequence/object/person where IDs permit. No per-episode parameter tuning.
Never use matched challenge FORM assets. Never call these scores a verified
CARI4D leaderboard win. All heavy extraction/inference/evaluation stays Azure.

## Official submission kit changed — explicit pre-submission gate

Current archive:698314 bytes, ETag6ac7186d-aa7ca. Public Track1 packer11338 bytes
SHA256 `e0fb5749bc9753e0fd522934bb51c246164c9cdee526584ecb0b1a0b5b1bf585`.
Track1 README confirms sample rows/layout only, GT retained by organizers;
new packer adds mesh quality reporting. Old pinned runtime is not silently
replaced: re-audit the new kit explicitly before frozen Parquet upload.
No challenge GT/multiview, sample values, other-track assets or heavy media were
acquired in this re-audit (only small public sources/metadata in memory).

## Implemented external test admission (not executed image evaluation)

`configs/form_hoi_insight_v1.json` freezes8 deterministically SHA-selected
sequences,4development and4reserved/unopened provisional filename families.
Metadata protocol rejects challenge IDs/recording aliases/object prefixes and
unsafe or annotation-bearing inference packages. 4108/4135 admitted;27 legacy
names rejected conservatively.18 tiny tests pass. All7 qualification gates
remain false until eligible Azure extraction establishes original camera/text,
true grouping, content duplicate guard, reference isolation and frame timeline.
No fabricated prompts, inferred labels or claim of qualified held-out performance.
Cohort frozen now in Git before any image inference. Actual annotation-based
FORM scoring has NOT run yet; it is the next external evaluation task.
