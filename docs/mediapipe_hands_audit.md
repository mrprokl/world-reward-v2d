# MediaPipe Hands: independent feasibility audit

Evidence cutoff: **2026-09-30**. Audit performed 2026-10-04 using small primary
source texts, package/object metadata, and the model card (parsed in memory).
No model, image, or video was downloaded; no inference was run. The mutable
official overview marked updated 2026-10-01 is not cutoff evidence.

## Decision

**Worth one cheap, preregistered CPU coverage pilot, not an assumed HOI fix.**
Keep the closed GroundingDINO bank failure unchanged: clip 0, zero hands, two
objects, two detector calls/one SAM2 call, 13.134 s; no tracking or private-label
evaluation. MediaPipe is a different specialist method requiring a **new,
previously untouched external validation cohort**, not a prompt/threshold retry
of that closed cohort.

The primary model card explicitly lists **occlusions, including hands holding
objects**, and counting hands in a crowd as out of scope (p. 3). This is directly
relevant to manipulation. Older source documentation claims partial-visibility
and self-occlusion robustness, but does not establish object-occlusion accuracy.
No verified DexYCB, challenge, or CARI4D quality comparison exists here.

## Verified released artifacts

- [MediaPipe v0.10.21 release](https://github.com/google-ai-edge/mediapipe/releases/tag/v0.10.21),
  published 2025-02-07; source commit
  `cad7f3ab99ebf175947e40c5252c642612aae927`.
  [Source license](https://github.com/google-ai-edge/mediapipe/blob/cad7f3ab99ebf175947e40c5252c642612aae927/LICENSE): Apache-2.0.
- [Official version-1 task object metadata](https://storage.googleapis.com/storage/v1/b/mediapipe-models/o/hand_landmarker%2Fhand_landmarker%2Ffloat16%2F1%2Fhand_landmarker.task):
  `hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, 7,819,105 bytes,
  created 2023-04-26; generation `1682480004222387`, MD5
  `15318430ea3851670fe9914116a9cfad`. No published SHA-256 in this metadata;
  measure it during future Azure acquisition, never invent it or use `latest`.
- [Official Oct-2021 Lite/Full model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf):
  358,044 bytes, SHA-256
  `43127ff8a92e22717f0d8c30dab3efef4641a29cbf50b15ae3de001be94ab76b`;
  GCS generation `1683223215027175`, created 2023-05-04. It explicitly states
  Apache-2.0 (p. 2). This supports pipeline/model licensing; exact task-bundle
  constituent identity and embedded license still require the Azure asset audit.
- [PyPI 0.10.21 metadata](https://pypi.org/pypi/mediapipe/0.10.21/json):
  `mediapipe-0.10.21-cp311-cp311-manylinux_2_28_x86_64.whl`, 35,622,638 bytes,
  published 2025-02-06, published SHA-256
  `05dc4a9e593655a79558d05d6227d31018c2537a4bd3362b51e230cf22aecfe3`.
- [Zhang et al., MediaPipe Hands](https://arxiv.org/abs/2006.10214), first released
  2020-06-18; establishes the palm-detector/landmark-regressor design, not modern
  manipulated-hand validation.

## What the native method actually supplies

The full-image palm detector proposes native rotated crops; a hand regressor
produces 21 landmarks and projects them back to the **original, uncropped image**.
The [pinned Python Tasks API](https://github.com/google-ai-edge/mediapipe/blob/cad7f3ab99ebf175947e40c5252c642612aae927/mediapipe/tasks/python/vision/hand_landmarker.py)
returns image landmarks, world landmarks, and handedness only.

- Pixel conversion is `(x * width, y * height)`, without an invented half-pixel
  correction. Image `z` is wrist-relative; world landmarks have a hand-centred
  origin. Neither supplies shared human/object camera pose or metric translation.
- The public result exposes **neither palm detection confidence nor hand-presence
  confidence**. Handedness score means left/right classification, **not hand
  detection confidence, visibility, contact, or persistent actor identity**.
  Do not feed it into the current detector-score/NMS contract.
- Native regressor output is 21 three-coordinate landmarks, not per-joint
  visibility. Generic container visibility/presence fields do not make absent
  proto fields observed. Occluded coordinates may be predictions from priors.
- Native defaults are detection/presence/tracking thresholds 0.5 and
  `num_hands=1`. The latter is a hard capacity: the graph clips candidates and
  streaming detection may be skipped once the tracked count reaches capacity.
  Preserve every reported hand; preregister capacity outside the failed cohort
  and flag saturation. Stable sorting is not physical identity association.
- Prefer synchronous **IMAGE** mode for an initial frame-0 bank or complete
  per-image scan. VIDEO requires real monotonically increasing millisecond
  timestamps; missing DexYCB timestamps must not become fabricated 30 fps.
  LIVE_STREAM can drop busy frames and is unsuitable for a full-timeline claim.
- The older [pinned Hands documentation](https://github.com/google-ai-edge/mediapipe/blob/cad7f3ab99ebf175947e40c5252c642612aae927/docs/solutions/hands.md)
  assumes mirrored selfie images for handedness; retain raw output and disclose
  the input convention rather than using handedness to pick a target actor.

The [unchanged native graph](https://github.com/google-ai-edge/mediapipe/blob/cad7f3ab99ebf175947e40c5252c642612aae927/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_graph.cc)
also exports `PALM_DETECTIONS`/`PALM_RECTS`, which Python Tasks omits. A separately
qualified graph adapter could expose real palm scores, but palms are not whole
hand masks; this is not an already-tested public API integration.

## Provenance and runtime feasibility

Source documentation discloses approximately 30K manually annotated real images
plus synthetic hands/backgrounds. The card describes consented smartphone-AR
images and GHUM-derived synthetic 3D supervision. Its 700-image geographic and
420-image skin-tone/gender evaluations come from the same source as training
images, not an object-manipulation benchmark. Exact sample/dataset manifests are
absent: **DexYCB/challenge overlap remains unknown**, despite pre-challenge dates.

Use a separate pinned **Linux x86_64, Python 3.11, glibc >=2.28 CPU** environment.
Do not install into the qualified SAM2/Boots images: MediaPipe 0.10.21 brings
NumPy `<2`, protobuf `<5`, JAX/JAXlib, and OpenCV dependencies whose current
resolver choices can conflict. Pin the complete dependency closure and licenses.
All acquisition remains on Azure. H100 is unnecessary for this small seed model;
TFLite GPU delegates use OpenGL/OpenCL, not CUDA. Headless EGL/delegate support
on H100 is unverified, and phone latency is not an Azure runtime estimate.

## Minimal next experiment

Freeze the new external cohort, runtime budget, capacity, and native defaults
before its use. The later [full-timeline protocol](specialist_hand_observation_protocol.md)
supersedes the initial frame-zero-only proposal: measure every original frame
with IMAGE/CPU and preserve all hands, empty early frames and capacity saturation.
A missing frame-zero hand does not establish a visible-hand false negative.
Failure ends the pilot before Boots; do not rescue with another threshold/prompt.

Source/model byte acquisition is implemented separately; it does not install or
run MediaPipe. Its 300 s budget covers acquisition, archive checks, public artifact
sealing and post-hashes; the small receipt publishes once under an outer 320 s
deadline. A future CPU runtime should be a **new child** of the already present
VM02 classic image `sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3`,
with an isolated Python 3.11 venv, no system site packages, all declared/transitive
wheels pinned and installed offline, followed by `pip check`. Do not modify an
existing SAM2/Boots image, omit JAX, substitute OpenCV headless, or resolve latest
dependencies after a failure. Build proposal: 600 s; graph-load-only CPU smoke:
120 s, no `detect`, RGB or annotations. These are proposed prerequisite gates,
not an acquired dependency closure or operational runtime.

If coverage is adequate, separately validate an automatic 2D hand-seed adapter
(e.g. deterministic landmark-derived box or points) into unchanged SAM2; do not
claim landmarks are masks or automatically reuse an arbitrary expansion. Keep
the object candidate bank and all-instance/background tracking unchanged.
Freeze features before private external evaluation. Discriminating ablations:
specialist hand seeds versus GroundingDINO seeds on the **same fresh cohort**;
then relational motion versus an object-only baseline with the same candidate
bank. Missing/occluded/cap-saturated evidence must remain explicit, with no
nearest-object fallback, fabricated contact, or shared-3D claim.
