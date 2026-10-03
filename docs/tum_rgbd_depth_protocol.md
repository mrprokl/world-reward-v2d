# TUM RGB-D depth transfer pilot — specification before acquisition

## Question and scope

Does the **unchanged D106 whole-native-support anchor** improve camera-Z depth on
new real recordings, rather than just the already observed TUD-L scenes? Freeze
this protocol before acquisition/prediction. No result, adoption or CARI4D victory
is established by these metadata checks. No D105 RGB/truth or old TUD-L frames
are reused. This is a 12-frame depth pilot, not full HOI or temporal validation.

The committed JSON is the complete reproducibility specification. Select sorted
RGB filename indices **40, 80, 120, 160**, zero-based, in `freiburg1_desk`,
`freiburg2_xyz`, and `freiburg3_long_office_household`. These span roughly four
early seconds per sequence and represent three original sensor groups; different
recordings do not prove that all office/object identities are disjoint.

## Primary terms and independent byte identity

[TUM's publisher](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) explicitly
grants **CC-BY-4.0 for benchmark data**, unless otherwise stated, and BSD-2-Clause
for its code. The exact 627-byte HTML licence section has SHA256
`04070bc137e1ebbb7fbb394fa7239403a5340ba30657990c38aa6a6bbd5f2d35`.
Pin that section, not unrelated mutable bibliography/page content. Inspect any
sequence-specific terms before use; conflicting/missing permission means STOP.
Retain Sturm et al., *A Benchmark for the Evaluation of RGB-D SLAM Systems*,
IROS 2012 attribution, source links and CC-BY notices; media is not Apache.

[Original download links](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
returned HTTP200 HEAD sizes 344,011,403 / 2,201,854,648 / 1,483,556,251 bytes,
with exact ETags/Last-Modified recorded in JSON. **HEAD and ETag are not SHA256.**
No independent original whole-archive checksum was found. Record the first actual
Azure archive SHA256 as reproducibility evidence, not a preexisting authenticity
pin. Original archive membership, filename selection and all selected bytes must
match independently pinned file identities before interpreting sensor values.

The third-party [HF mirror](https://huggingface.co/datasets/voviktyl/TUM_RGBD-SLAM/tree/76818471ae555dd6cd4e100b7cedcccda660448d)
at `76818471ae555dd6cd4e100b7cedcccda660448d` supplies **24 LFS SHA256/byte pins**
in JSON, gathered solely from filename/file metadata. Complete paginated listing
counts were RGB/depth 613/595, 3669/3666, and 2585/2509. The mirror is an
independent byte-reference, not the authority for data terms. Compare selected
original publisher archive files to those pins on Azure; any mismatch stops the
whole protocol without alternate frames or mirror-first label use. The selected
24 files total 7,358,364 bytes; heavy archives/media never transit locally.

Associate each selected RGB timestamp to the **unique nearest depth timestamp**
using exact Decimal arithmetic, maximum absolute difference **20 ms**. Ties,
reused depth files or missing support mean STOP. The twelve frozen offsets are
already derived from metadata, maximum 17.203 ms; no sensor values selected them.
Retain original timestamp names and verify associations against original archive
member lists before decoding. Hash all 24 selected files before any private depth
values; preserve 12 original RGB files plus a public-only manifest and 12 private
depth files/inventory. No calibration, trajectory or external-camera recordings
enter inference. Delete disposable archives after verified subset acquisition.

## Frozen inference and sensor evaluation

Use original native MoGe2 normal/DA3Metric-Large checkpoints/loaders and source
hashes from JSON, fixed **640×480/K800/+0.5 rays**, native focal callback, DA3
processed focal mean/300 conversion once and native sky correction unchanged.
No learned model, source-camera fitting or confidence weighting is added.

For each image use **all unchanged native MoGe-valid finite-positive pixels**:
at least 1,024 paired pixels and 25% of the whole grid. Persist counts, coverage,
ratio median, log-ratio q10/q50/q90 and IQR **before every gate**; dispersion is
diagnostic only. One positive equal-frame median of four MoGe-Z/DA3-Z medians
anchors each sequence. No offset, per-frame scalar, semantic mask, dropped
validity or private depth/camera enters prediction. Any failed support ends the
run; no substitute frames. Freeze all twelve complete predictions, report,
source/model/public inputs and three coefficients before private depth decoding.

[Publisher format documentation](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
says RGB and 16-bit depth PNGs are already registered 1:1; **sensor Z=raw/5000 m**,
raw zero means missing. The publisher already applied Freiburg depth corrections:
**do not multiply again by 1.035/1.031**. No depth clipping, manual ROI, outlier
discard or calibration-dependent reprojection is allowed.

Each private frame requires ≥1,024 sensor-valid pixels and ≥25% whole-grid sensor
coverage. Native MoGe validity must cover ≥95% of sensor-valid pixels. Baseline
and candidate use exactly the same intersection of sensor-valid and original
MoGe-valid pixels. All their evaluated Z values must be positive finite.
Primary metric is pixel camera-Z **AbsRel**, with RMSE metres and signed/absolute
error diagnostics. Equal mean of four frame AbsRel values defines each sequence;
continue only if median sequence relative improvement is **≥5%**, no sequence
regresses **>5%**, and every coverage gate passes. No scale/shift/alignment fit.
Zero baseline AbsRel means STOP, not a division workaround. Original inference
300/303 s and private CPU evaluation 180/183 s budgets remain unchanged.

## Limits and training screen

The pinned [MoGe2 train config](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/configs/train/v2.json)
does not name TUM. The exact normal checkpoint card contains MIT metadata only,
not an exclusion attestation. [DA3 §4.4/Table1](https://arxiv.org/html/2511.10647v1)
does not name TUM; its metric section says 14 datasets but enumerates 13, and the
checkpoint card says only public academic datasets. **Frame-level, backbone,
pseudo-label and challenge training overlap remain unverified.** Absence of a
name in these sources is not proof of leakage-free weights or eligibility.

Fixed K800 differs from the documented TUM cameras around focal525. Private K
must not enter inference or rescale predictions; scalar Z scoring avoids truth
reprojection but cannot remove the influence of an incorrect camera prior on
model depth. RGB/depth timestamp mismatch, sensor noise/missingness and possible
shared scene objects also remain limitations. A PASS would support this frozen
method on new real depth records only, not calibration accuracy, human mesh,
object trajectory, contact, acceleration, full-HOI gain or a CARI4D victory.
