# Frozen random full-4D visual diagnostic

Uniform sample drawn once, before outputs, from all 30 official Track1 episodes:
Python `random.Random(20261008).sample(range(30), 4)` → **9, 1, 14, 7**.
No failed clip is replaced. This overlaps development data and is **not held-out
evaluation**, independent accuracy validation or a leaderboard claim.

Pipeline: unchanged generic Qwen3-VL task grounding → full-T SAM2 → SAM3D Body,
MoGe2 and one shared inferred metric scalar → SAM3D Objects → fixed whole-surface
identity/QSlim budget route → full-T body/depth → rigid ICP/Viterbi object poses
→ shared MHR identity → native CARI forward → native joint refinement (300 steps)
→ frozen full-T export → synchronized RGB / full scene video.

All episodes use the same source, model settings and geometry policy. No manual
boxes, source camera calibration, multiview data, ground truth, matched FORM-HOI,
per-frame recentering or scale fitting. Pretrained-model challenge overlap remains
unverified. Runtime pins are immutable **producer-derived continuity receipts**,
not independent historical measurements or correctness certificates.

Each stage fails closed on its existing provenance/numerical gates. Costs are
frozen in `configs/full4d_sample_v1.json`; clips continue independently after a
clip-specific failure. In particular, the current rigid initializer cannot bridge
frames with fewer than 40 valid object-depth points: no static fallback is added.

Outputs are isolated on Azure under `experiments/full4d-v1-<commit>`. Baseline
outputs are not writable/mounted. Videos preserve all original frames at 30 fps,
with one inferred camera and unchanged shared scale. A fixed background-only
virtual floor and 0.5 m grid aid QA; horizontal-camera/scale assumptions are
explicit, and predicted geometry is not snapped to the ground or hidden by it.
MP4s ≤2 MB/clip stay remote; only tiny posters/results may be transferred locally.

Human QA gate: inspect target identity, body/hand motion, object shape/orientation,
contact plausibility, drift, occlusions and inferred scale. Observations guide
general algorithms, never manual prediction of individual challenge records.

## 2026-10-08 infrastructure continuation

The frozen `de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba` producer completed all
415 object poses for episode 9 (2,460.6 s), but its input consumer rejected the
unsealed publication directory. Seal only the three verified files' permissions;
reuse their exact bytes, source and full timeline. Preserve the interrupted
original report and write a separate source-bound continuation receipt.

Episode 9 scouts inputs through final rendering before new expensive tracking
on the remaining frozen clips. Then two episode workers share one GPU lease;
learned stages remain serialized. Focused continuation/preview tests: **87 pass**.
The historical 32 MiB pose-report consumer would reject longer clips. A named
64 MiB capacity adapter now accepts only the canonical, sealed full-pose JSON;
all other roles and geometry/model calculations are unchanged. Exact source
parity permits only three import/ledger substitutions. New input producers carry
their real continuation revision; later shared producers remain original.
Existing PASS native stages/videos are rehashed and reused, never overwritten.
Focused capacity/reuse checks: **117 pass, 1 skip** (no GPU execution locally).
