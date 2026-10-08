# Gemini3.5Flash initial-localization diagnostic

Frozen2026-10-08 before challenge-image API calls. Eight development episodes
9/1/14/7/28/13/18/6, exactly original frames0/14/29 (24 independent images).
No GT, multiview, calibration, FORM matching, labels or manual episode prompts.
No full video decode, new SAM inference/tracking, GPU job, or4D reconstruction.

## Question / comparisons

Does Gemini directly propose better principal-person/object initial boxes than
the current frontend? Gemini sees one untouched original RGB frame and official
description/action as quoted data. Null abstentions and invalid responses stay
in the denominator. The same generic instructions apply to every image.

Display original → prior independent Qwen boxes where available → saved native
SAM/Qwen-ID masks → Gemini boxes, on the exact same original frame/RGB SHA.
Prior SAM/Qwen-ID saw the full clip and native hotstart future context: explicitly
**retrospective**, not a matched causal ablation. Prior independent Qwen exists
only on9/1/14/7; other cells show NOT AVAILABLE, not an inferred prediction.
Boxes are proposals, not segmentation masks. Qualitative rejection/coverage and
latency, not GT accuracy, superiority claim or held-out validation.

## Official API contract / efficiency

Actual Vertex OAuth text and manufactured-image structured smoke bothHTTP200,
served modelVersion`gemini-3.5-flash`, endpointglobal. ModelGA; spatial grounding
featureexperimental. Single original RGB→losslessPNG in Azure RAM, image before
task text, strict `responseJsonSchema`, JSON, 4integersYXYX normalized0..1000;
floor lower/ceil upper to original dimensions, no clamp/repair/offsets.
LOWthinking, HIGHmedia,1536outputtokens, four concurrent independent requests,
60s/request,300s inclusive native stage. No retries or best-answer selection.
Omit temperature/topP/topK/thinkingBudget per model-specific guidance. No search,
function/code tools or shared conversation memory during predictions.

Alias can change: record actual modelVersion/responseId/finishReason/usage and
request/RGB/PNG SHA per call. Training/challenge overlap unknown; never assert
checkpoint leakage-free. Source videos and all request images remain Azure;
Google Vertex receives only the24images explicitly authorized for API inference.

Existing localgcloud credential is encrypted using one-shot Azure-held RSA-OAEP
public key; only ciphertext traverses Azure control transport. Decrypted OAuth
stays RAM/stdin, no credential logs/arguments/plaintext files/Git. Temporary key
and envelope removed after run. No permanent IAM/API-service changes made.
Private SHA/ETag-bound JPEG QA ≤180kB each; no videos returned to laptop.

Primary documentation (consultedOct8):
- [Vertex bounding boxes](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/bounding-box-detection)
- [Gemini3.5guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-5-flash)
- [Vertex images](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/image-understanding)
- [User Roboflow reference](https://playground.roboflow.com/models/google/gemini-3-5-flash)

## Actual result

Produceradfad8c6 completed24independent API calls,8episodes×frames0/14/29,
119.65s native/121.00s total stage (excluding first-time research/code/transport
setup).20valid pair-box responses;4socket/API TimeoutError at60s, not semantic
schema failures. No retries.20returned is not20correct. Successful-call median
3.28s;40776total reported tokens. OAuth envelope/private key removed.

Same-frame initial QA: Gemini generally selects the visible foreground person;
provides initial object proposals even where saved SAM object banks were empty
(e.g.7/6). This alone does not verify target identity, tight geometry or masks.
No demonstrated overall superiority over Qwen/SAM. Await user visual QA;
then test actual VLM-box→SAM image segmentation only, not full-video tracking.
Saved SAM/Qwen-ID references remain retrospective. Prior Qwen image-only
comparison available for4episodes. No manual box/episode correction or tuning.

Saved-only CPUzoom producer7b699c5: original box-driven context zooms, all24
successes/timeouts retained,0additional model calls. Code/tests committed;
external model overlap remains unknown. Heavy/source images never downloaded.
