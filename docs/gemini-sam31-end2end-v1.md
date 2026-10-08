# Gemini → SAM3.1 → SAM3D/CARI full-4D causal frontend diagnostic

Frozen before new inference, 2026-10-08. Reuse the original uniform random draw
from30 Track1 episodes: random.Random(20261008).sample(range(30),4)=[9,1,14,7].
These are development QA, not independent held-out validation. No replacement
of failed episode7 and no selection of pretty outputs. Baseline preserved.

Saved originalRGB + official object_prompt/action → Gemini3.5Flash independent
boxes fromframes0/14/29 (producerb6f0a89, already recovered) → fixed-ID native
SAM3.1 PVS person/object fullT → refreshed SAM3D Body + SAM3D Objects → one
shared inferred scale/K, fixed shape → unchanged rigid ICP/Viterbi → native
CARI forward/refinement/export → original versus reconstructed fixedfloor.

The fixed-ID route uses native point labels2/3 for box corners; it is NOT the
concept/exemplar route that detects all category instances. Its native extra
not-a-point token is retained, so no equality claim to image box encoder.
Three automatic initial observations condition the same two IDs; no human
prompts, clipping/copying labels or picking better responses. Prefix30 transport/
identity gates precede longer tracking. FullT native absence/presence remains
explicit. PVS alone is not detector-guided semantic reacquisition through every
occlusion; measure drift/absences instead of assuming first boxes solve it.

Causal question: does improved automatic identity/segmentation repair the wrong
object proposal and final motion? Do NOT claim it resolves resting-object jitter:
old depth-reset ICP and fixed-rotation native refinement remain limitations.
Refresh mask-conditioned human/shape stages; cache only source-identical RGB-only
MoGe2 outputs with original producer provenance, never fictitious new inference.
Two clip CPU workers share an H100 learned-stage mutex, firstclip9scout then
remaining independent clips. Same models/numerical settings; native300steps.

All heavyweight masks/videos/arrays/meshes stay on Azure. Tiny JPEGs only local;
private bounded compressed comparison videos stream on demand without caching
or exposing credentials. Fixedfloor is an uncertain display background, not
measured calibration/physics or an optimization constraint. No scene snapping,
per-frame centering/alignment, shrink/delete meshes or manufactured static motion.

## Primary-source insights by September2026

- SAM3.1 release: https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/RELEASE_SAM3p1.md
  Prompt routing: sam3/model/sam3_multiplex_tracking.py; native box-corner labels:
  sam3/sam/prompt_encoder.py. Custom SAMLicense; training overlap unknown.
- OAMVOS April20: https://arxiv.org/html/2604.22837v1
  Preserve reliable anchor memory; only bounded independent-confirmation recovery.
  Its GT-initialized benchmark cannot be called our automatic performance.
- CARI4D v3 April19: https://arxiv.org/html/2512.11988v3
  RGB/RGBD pose hypotheses, bidirectional occlusion recovery, pixel/temporal/contact
  refinement. Our current ICP is NOT its FoundationPose tracking algorithm.
- GRC-Pose September30: https://arxiv.org/html/2609.39116v1
  Persistent object-scene correspondences and inlier memory support pose stability.
  Paper-only insight; no verified official code, variable Sim3/perframe scale and
  calibrated benchmark do not meet our clip-constant RGB-only contract directly.

Defer VideoMaMa(alpha matting, NC), BundleSDF(realRGBD/NC), KVTracker(uncleared
Imperial license), human-generatedviews4DAnyone, FastSAM3D(backbone acceleration
without orientation fix). No new checkpoints acquired on speculation.

Predeclared quality inspection: target identity, full body, object shape/handle,
orientation, motion/contact, disappearance/reappearance and scale. Report source
coverage, empty masks, consistency/flow and stage cost as PROXIES, never trueGT
accuracy or leaderboard victory. FORM-HOI disjoint insight cohort remains separate
and not yet scored; no challenge-matched assets are allowed in this experiment.

### Native integration qualification

The first two immutable integration runs failed before full-frame propagation:
`7cad945` compared padded/BF16 tokens to bare box tokens under different rounding
contracts; `69869ff` passed the corrected FP32/equal-shape proof but hit native
multi-object interactive gap-fill (`1` pointer versus `2` mux entries). The
source-backed instance path uses one singleton tracker state per fixed ID,
sharing the same image/backbone feature cache. This is not semantic redetection
or a SAM2 checkpoint fallback. All six automatic boxes and full original frame
indices are retained; native output accuracy is still unverified.

### Completed tracking and next isolated recovery test

Producer `b658881079871508c6b3ec14d001dc1299956996` completed all 2302 frames
in 442.03 seconds. Episodes 9/14 have no empty object observations; episodes
1/7 have 244/128. These counts cannot distinguish true occlusion from dropout.
The pinned native source keeps all three conditioning frames (attention limit
four); shared visual-cache eviction does not erase identity memory. Native
presence logits suppress absent masks. No seed-loss bug was identified.

Do not increase memory limits or fill masks blindly. A future short external
diagnostic compares current forward tracking with an independent reverse pass
using the same seeds plus one automatic fixed final-frame anchor. Abstain if
that anchor is unavailable. Same-state native reverse also recomputes ordinary
frames; a fresh state is an experimental control, not a replay-bug workaround.
Maximum two passes and one additional VLM call. Use independent manufactured
occlusion/distractor scenes, hidden evaluator labels, complete timelines, and
identity/J&F/reappearance/false-visibility/time gates before adoption. Four scenes
are a diagnostic, not a generalization claim. Persist raw native presence logits
for diagnosis, without changing the model threshold.

### Early saved-only 3D inspection

`run_full4d_video_initialization` observes original frame zero from already
completed body/scale/object stages, in a separate publication namespace. CPU-only
Azure rendering shows RGB, new masks, and the exact initial human/object geometry
on an assumed floor. It never fits, modifies, rescales or replaces predictions.
This is explicitly **initial 3D**, not a complete 4D video or motion validation.
The numerical job and baseline remain untouched; its proper full-T export/video
is the later temporal QA deliverable.
