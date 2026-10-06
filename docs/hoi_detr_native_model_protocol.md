# Neutral original HOI-DETR constructor seam

`build_original_hoi(torch, np, runtime, deadline, source_root, checkpoint_path,
native_config, checkpoint_buffers)` is keyword-only and returns a frozen
`OriginalHOIModel(model, operations, provenance)`. The model/device tensors remain
mutable native objects; the frozen container is not an immutable weight proof.

It factors the original `hoi_detr_model_qualify.gpu_model` construction sequence,
reusing **unchanged** `configuration_policy`,
`register_original_checkpoint_buffers` and `strict_checkpoint`. Only original
`load_from=None` and `model.train_cfg=None` overrides are applied. Original test
configuration and registry side effects remain. Register exactly898 native
states plus898 EMA backup buffers, then weights-only CPU-load every1796 tensor
with strict key/shape/dtype/finite checks. No EMA update/swap, checkpoint-key drop,
teacher/model swap, new threshold or dataset/train API is used. FP32 CUDA eval,
CPU/CUDA autocast disabled and TF32 off are required. No AMP is introduced.

Original `LoadImageFromFile` becomes `LoadImageFromWebcam`; subsequent RGB is
copied into BGR **once**, then processed by the untouched original Compose.
Original collate/scatter/native bbox transform and batched NMS are exposed to
unchanged `infer_hoi_detr_frame`. All1500 query logits/boxes/tokens, top1000/raw
score distinction and all native H→O/O→target pairs remain in that adapter;
this constructor neither postprocesses nor selects a target. No RGB is supplied
to construction and no supervised `interaction_targets` are manufactured.

The caller must authenticate the exact author source/config/hook, weights,
qualified MMCV extension and FairScale/terminaltables overlay, original runtime
image and helper bytes **before and after**; initialize fixed seeds/import paths,
bind the deadline and keep sources/weights readonly. This module does not clone,
download, install, mutate producer profiles, import training APIs, choose an
image tag or qualify a runtime. Existing five qualification producers remain
byte-unchanged. The qualified HOI image is the original5fa… image on VM02;
B47/7eb image equality or installed compatibility must never be guessed.

Provenance describes operations, with source/runtime/weight qualification false:
it is not a replacement for independent receipts, license/creator verification,
pretraining/challenge-overlap review or ownership/quality evidence. Manufactured
fake-author tests cover call order, strict schema, EMA registration and RGB/BGR
handling only. No real Torch/CUDA/model/data run or new accuracy/FIT/adoption
claim is made by this seam.

Qualification of this source only:28 dedicated fake-author controls and163
combined tests PASS0.77s; independent root review repeats163 PASS0.79s. Original
producer bytes remain unchanged and owned tiny fixtures were removed. Real
model construction on the qualified VM02 image is a separate future gate.
