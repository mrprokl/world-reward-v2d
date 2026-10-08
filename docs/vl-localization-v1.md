# Independent VL localization before segmentation

Frozen2026-10-08 before inference. Preserve the original9/1/14/7 random cohort,
including failed7; no new clip selected from output quality. This is a development
visual/transport diagnostic, not a held-out 3D or identity score.

The baseline's nine images share one autoregressive conversation. Ep14 returns
the same wrong boxes for all9views; the frame0 object box is above the pan, and
SAM2 follows that background. No coordinate bug is demonstrated: Qwen normalized
coordinates map once to original RGB dimensions, as confirmed by its primary
documentation. Do not invent pixel offsets, resize corrections or pan prompts.

## One attributable change

One original full-frame RGB, original frame index and unchanged task/object
conditioning per independent conversation. Keep8B-Instruct revision, BF16/SDPA,
native processor pixel bounds and1536 output capacity, seed0 and greedy decoding.
No previous boxes, context shared between questions, judge, crop, retry, box
interpolation, nearest-hand rule or new checkpoint. Invalid/null answers remain
explicit independent failures, rather than corrupting the entire clip.

Decode once the sorted union of frames0..29 and nine uniformly spaced full-clip
views. Retain RGB identities. Four30-frame sheets are exhaustive and ordered;
the nine full-clip views expose movement/occlusion that the first second cannot.
All source media and native processing remain Azure-only.

## Batch correctness and efficiency gate

Load the existing verified model once. Native left-padded batches of8 independent
questions retain the same image resolution. Before inference, first0/29 frames
from each of the four clips form one mixed-task control batch. Compare real
tokens, image grids and each packed processed-pixel segment against singleton
encoding exactly. Validate image-pad counts, still-image grids and original
episode/frame/output row binding. Any encoding mismatch rejects the experiment.

Compare EOS-trimmed greedy tokens between singleton and batch on those8control
images. If any differ, use singleton execution **globally**, reusing the singleton
controls; no per-image choice of a visually better output. Record both actual
control results and scheduler. This guards batching, not semantic correctness.
GPU budget1800s inclusive; fail rather than lower resolution/output capacity or
reroll. Models/files rehashed before/after; one shared H100 lease, offline mounts,
no baseline writes. CPU decoder/previews run independently of the GPU when useful.

## Handoff and QA

Prepare literal per-frame original-resolution person/object boxes in a separate
proposed SAM-input JSON. Missing boxes remain missing. **SAM is not run by this
VL-only ablation**: captions must say proposed, not successful segmentation.
Do not promote first_both/nonempty/all-valid JSON to identity certification.
Show uncropped miniature RGB, cyan person/orange object boxes, index and explicit
abstentions. Display-only zooms/compression can improve legibility; they never
change inference inputs. ≤180kB/JPEG, private Azure publication, SHA/ETag-bound
tiny local transfer. QA can reject a general algorithm, not label individual
challenge test records. No superiority/generalization claim from these clips.

SAM3.1 is a **later separate experiment**: detect concept instances independently,
associate to actor over time, then detector-guided re-prompting/reacquisition.
All matches are candidates; static objects can be legitimate interactions. Custom
SAM terms, actual checkpoint acquisition and overlap require separate checks.

Primary batch instructions:
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL#batch-inference),
[Transformers5.3.0 processor](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_vl/processing_qwen3_vl.py),
[Qwen original-size coordinates](https://github.com/QwenLM/Qwen3-VL/issues/1486).
