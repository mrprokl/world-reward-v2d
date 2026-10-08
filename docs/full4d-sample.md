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
