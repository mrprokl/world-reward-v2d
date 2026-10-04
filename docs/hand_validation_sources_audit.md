# Fresh hand-coverage validation: source preflight

Audit 2026-10-04; literature/source cutoff **2026-09-30**. Only bounded primary
text, archived publisher text, and HEAD/public-viewer metadata were read. No
image, video, label file, model, or dataset archive was downloaded or executed.

## Recommendation

**Use one fresh DexYCB subject-03 original archive on Azure and a small,
preregistered set of complete clips for a CPU IMAGE-mode diagnostic.** It is not
the smallest transfer, but it is the only compared option with a verified current
publisher link, explicit data license, byte-size metadata, and already documented
RGB/annotation format. EgoHands would be cheaper if its authoritative hosting
were restored; current official archive links return HTML, not archives.

Freshness means no use of the closed subject-01/02 experiment. It does **not**
prove MediaPipe pretraining independence: no exhaustive training manifest exists,
and subject-03 appears in some official DexYCB training splits.

The old FAIL establishes **no required hand proposal at frame 0**, not a proven
false negative on a visible hand. No labels/images were examined to establish
visibility. Do not treat a specialist as a demonstrated rescue.

## Compared primary sources

| Source | Rights established | Data/split and transfer | Current feasibility |
|---|---|---|---|
| **DexYCB subject 03** | Publisher explicitly **CC-BY-NC-4.0**; attribution/non-commercial restrictions remain | `20200820-subject-03.tar.gz`: publisher rounded 12G, public Drive viewer size **12,197,037,343 bytes**. 100 sequences, eight camera streams; choose one camera only. `s0_test` subject 03 uses sequence indices `i % 5 == 4`; `s1_train` also includes subject 03 | Publisher and viewer accessible. No per-image original hosting verified. Original `.tar.gz` must be streamed/scanned on Azure; sparse retention does not make gzip random-access or remove transfer cost |
| **EgoHands (2015)** | Archived publisher says use is welcome **provided the ICCV paper is cited**. No named CC/OSI data license verified; archive README may add terms | 48 videos, 4,800 labeled 720x1280 JPEG frames/15,053 hand instances. **Labeled ZIP 1.3GB**, videos 2.2GB, all extracted frames 8.2GB. Native “main split” mentioned, exact membership not verified without archive README | Old official page and labeled/archive URLs now redirect to IU homepage `text/html`; newer paths checked return 404. An unverified Kaggle/HF mirror is not an authoritative license/identity substitute |
| **100DOH / 100K frames (2020)** | Archived publisher agreement: **non-commercial research only**, **no redistribution**, respect participants, no warranty/full liability. MIT detector source does not license the images | 100K YouTube frames/annotations; prepared raw/Pascal-VOC caches. Ego extension 56.4K frames from EPIC-Kitchens2018, EGTEA, CharadesEgo with original conditions. Published ego train/test frame-name lists exist; main exact split/cache byte size not verified here | Official HTTPS currently fails certificate verification; historical download/terms text recovered. No authenticated archive size or cheap image endpoint verified. Do not disable TLS verification or accept unknown cache identity |
| **“Grasping in the Wild” (Song et al., 2020)** | No explicit dataset license located on the primary project page; paper distribution terms are not dataset rights | Primary page: sample code/data 715MB, full data 75G. Sample ZIP HEAD: **716,185,103 bytes**, Last-Modified 2020-08-14, supports ranges. Split and human-hand annotations unverified | Accessible original sample, but this is low-cost-interface/robotic 6DoF grasp demonstration data, not an established human-hand coverage benchmark. Smaller transfer alone does not establish suitability or permission |

“The complexities of grasping in the wild” (Nakamura et al., 2017) is a distinct
work, not the Song dataset above. No authoritative licensed download/split/size
was verified for that candidate; do not conflate names or adopt a similarly named
third-party repository.

## Smallest useful diagnostic, before data

Proposed cohort: subject 03, one fixed camera `836212060125`, **three complete
sequences at lexicographic indices `[4, 39, 74]`**: three evenly spaced entries
within the official `s0_test` subject-03 sequence set. These are a metadata-only
proposal, not selected from visibility/results, and are not yet adopted or
downloaded. Confirm the actual original 100-sequence inventory and frame ranges
before inference; any mismatch fails rather than substituting clips.

1. Freeze cohort, IMAGE/CPU native defaults, hand capacity, wall-time budget,
   coordinate conversion, and scoring definition before acquisition/use. Acquire
   the one original archive on Azure; retain only chosen **full original RGB
   timelines** and corresponding evaluation-label bytes in separate sealed paths.
   No depth, other camera images, calibration, mesh, or MANO asset is needed.
2. Run synchronous full-original-T IMAGE observations, preserving every returned
   hand and every empty frame; no inferred 30fps, interpolation, or fake frame-0
   hand. A future tracking seed can use **automatic first appearance**, recording
   its true query time; its absence before that time stays unsupported. Detector
   result-slot order is not persistent identity.
3. Freeze outputs, then separately evaluate hand-presence/coverage on external
   annotations. The pinned toolkit documents hand pixels as `seg == 255`, and
   missing `joint_2d/3d == -1` can mean **no visible hand OR no annotation**. Thus
   valid positive hand annotations support recall; all-missing labels alone do
   not certify negative examples. Detection-count/nonempty rate alone is not
   accuracy, and hidden-joint predictions are not measured visibility.
4. Report runtime, proposal availability, capacity saturation, annotated-positive
   coverage and missing/unscorable counts. This is a diagnostic, not calibrated
   acceptance, manipulation/contact/identity performance, or SOTA proof. If
   coverage fails, close the cohort; do not tune thresholds or hand-box padding
   on it. No Boots/GPU work is necessary to falsify basic hand coverage.

## Audited evidence

- [DexYCB publisher](https://dex-ycb.github.io/), last updated 2022-05-21:
  14,026 B, SHA256 `09b12e4a37ccd0142e849101456a7389727b13a4d798434d0c35b686ccc9d008`;
  [original subject-03 Drive viewer](https://drive.google.com/file/d/1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq/view)
  advertises the exact filename/size above. Viewer HTML is session-dependent,
  not an archive checksum. Future acquisition must measure archive SHA256.
- [Pinned DexYCB split implementation](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/dex_ycb_toolkit/dex_ycb.py):
  8,713 B, SHA256 `f73074505bb822b01178dc7aae9778f5107efea479d43423f4fe2dc37224d8ad`;
  [format/absent-hand README](https://github.com/NVlabs/dex-ycb-toolkit/blob/64551b001d360ad83bc383157a559ec248fb9100/README.md):
  40,820 B, SHA256 `e19797f352bb5615b43b5a3ed4a6a193cee4eee2cd1cd1f5859615165081bb7c`.
  Toolkit source GPL-3.0 is separate from dataset CC-BY-NC-4.0.
- [Archived EgoHands publisher, 2025-01-11](https://web.archive.org/web/20250111161003id_/http://vision.soic.indiana.edu/projects/egohands/):
  33,318 B, SHA256 `b4c07c3effe1b80ea1f57d136e639f5ccb4ea9fb42a422ba34ce2b5b2b358d0f`.
  Archived text establishes historical offers, not current archive availability.
- [Archived 100DOH downloads, 2025-03-30](https://web.archive.org/web/20250330080628id_/https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/download.html):
  11,743 B, SHA256 `4ad3f87c9b8d694155118dce226752383339c1d10174d7e10bf7f01ef9a26669`;
  [agreement, 2023-05-24](https://web.archive.org/web/20230524132800id_/https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/agreement.html?pascal_voc_format):
  decompressed text 4,722 B, SHA256 `6ebe7d8f3d5b1b86994aab97764d5c01b67b667c9439be43c708afca4582c7f4`.
- [Pinned detector README](https://github.com/ddshan/hand_object_detector/blob/e6eec712a498ec7844b97893c8d012cea1a71e09/README.md),
  revision dated 2023-10-12: 9,794 B,
  SHA256 `15ade9c67fd2dcba18845cfb6b377f54b973d53e3ea5914167dddaba8d5e6801`.
- [Song et al. primary project](https://graspinwild.cs.columbia.edu/): 10,549 B,
  SHA256 `6bb40eead41a34be3fd3b62e5847924bac2e971605325d106f31d560556a8792`;
  [sample ZIP](https://graspinwild.cs.columbia.edu/giw_sampledata_code.zip) checked
  by HEAD only. No model-training overlap is cleared for any compared dataset.
