# TUD-L: frozen whole-valid-support depth anchor

**Scope.** A new, automatically selected twelve-frame cohort tests one whole-valid
MoGe/DA3 anchor. It is **not** a new scene/object benchmark, a training-overlap
clearance, full HOI evaluation or a verified CARI4D victory. All heavy data stays
on Azure under `validation/tudl_whole_support_holdout_v1`.

## Preregistered acquisition

The entire `configs/tudl_whole_support_protocol.json` is hard-pinned in the new
driver before acquisition. Source revision, archive bytes/SHA, original licence
evidence, selection, method and evaluation gates are immutable for this run.
Genuine original TUD-L and holdout acquisition helpers remain byte-identical.

For each fixed scene 1–3, sort the original 200 RGB filenames, verify the nine
previous development frames at ranks 0/100/199 and the twelve earlier holdout
frames at ranks 40/80/120/160, then select **30/70/110/150**. All 21 exclusions
are `(scene, frame)` pairs. Complete filename selection precedes **every** private
annotation/archive-value read; no model output selects records.

Remote CPU acquisition has a 600-second internal budget, 603-second outer
TERM bound and 10-second kill grace. Downloads use the original exact archive
hashes and bounded ZIP layout/expansion checks. Fresh namespaces only: no
resume, replacement, earlier-cohort rewriting or licence/layout relaxation.
Disposable ZIPs are removed on success/failure; the failure receipt remains.

Public inference inputs expose only twelve original 640×480 RGB PNGs and an
attributed RGB-only manifest (`schema`, `revision`, `license`, `selection`,
`images`). Public files are 0444. Original private calibration, sensor depth,
all selected object-instance visibility masks, geometry and licence evidence
remain under 0700 evaluation-only directories, as 0400 files. The complete
acquisition receipt is sealed 0400 and records all public/private identities.

## Frozen hypothesis and decision

Later blind inference uses unchanged native backends, fixed K800, +0.5 pixel
rays, **all original MoGe-valid pixels**, ≥1,024 pairs and ≥25% full-grid native
valid coverage. No border/semantic restriction, support filtering or fallback.
Log-ratio quantiles, log-ratio IQR and support diagnostics are persisted before
the gate. Each scene gets one positive median of four equal-weight image-median
MoGe-Z/DA3-Z ratios. Baseline and candidate validity match exactly. No offset,
per-frame fitting, private prediction calibration or alignment.

Only after all predictions are frozen and independently pinned may the private
evaluator read labels: all selected instances, visible-mask union, original
sensor integer-pixel convention, 8,192 deterministic seed-0 samples, ≥95% object
coverage, ≥5% median scene gain and no scene regression >5%. Acquisition performs
no inference, evaluation or adoption. Failure invalidates this run; it does not
justify retuning against these records.
