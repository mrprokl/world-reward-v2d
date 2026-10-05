# Source/model licensing audit — 2026-10-02

Scope: Track 1 submission eligibility, public winner-code release, and the pinned
organizer baseline. This records primary terms, **not legal advice or a permission
from the organizers**. Commercially usable weights, public source, an OSI-approved
source license, and challenge eligibility are four different checks.

## October4 source recheck

Official HEAD7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80 remains unchanged.
The exact84,452B FAQ SHAa1e569e1c83eeac2584de34e9503f1226e20ab369d53d2c67c2e243deb1a4d7a
is byte-identical. Closed/unmerged draftPR164 is not adopted policy; SAM
inference code, FoundationPose/nvdiffrast research restrictions and legacy
CARI notices still lack an explicit competition-source waiver verified here.
The actual Body backbone also imports **DINOv3**, distinct from Apache
**GroundingDINO** used by the candidate bank. Pinned DINOv3
[`6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md`](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md),
7503B SHA256 `25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e`,
is a custom agreement covering code, inference and weights, not an established
OSI licence. Add its source exception to the unsent organizer question, not a
new categorical disqualification or an inferred waiver from HF access.
Kaggle JavaScript shell is not a new rules reading. Contact remains
v2d_challenge@nvidia.com; existing question below is unsent. Eligibility is
separate from successful research execution; no all-clear inferred.

## 1. Governing challenge clauses

Source: [official CD-H Kaggle rules](https://www.kaggle.com/competitions/v2d-challenge-track1-cd-h/rules),
read in the browser on 2026-10-02; see the full operational audit in
[audit.md](audit.md). All five Track1 rules were later accepted with user
authorization; no organizer clarification message was sent.

- **Specific 2.6.a:** “Your winning Submission and the source code used to generate
  it must be publicly released under the Apache License, Version 2.0”. The sponsor
  receives only the rights available to everyone through that release; no separate
  exclusive license or assignment of ownership is required.
- **Specific 2.6.a exception:** “For generally commercially available software that
  you used to generate your Submission that is not owned by you, but that can be
  procured by the Competition Sponsor without undue expense, you do not need to
  release that software under the licence above. In the event that input data or
  pretrained models with an incompatible licence are used to generate your winning
  solution, you do not need to release that data or those models under the licence
  above.”
- **Foundational 6.c:** “Unless otherwise stated in the Specific Competition Rules
  above, if open source code is used in the model to generate the Submission, then
  you must only use open source code licensed under an Open Source Initiative-approved
  license […] that in no event limits commercial use of such code or model
  containing or depending on such code.”
- **Precedence:** “Any competition-specific rules […] are in addition […] in the
  case of any conflict or inconsistency, these Foundational Rules control and
  nullify contrary competition-specific rules.”
- **Specific 2.6.b and 2.9.c–e:** detailed methods, source for training/inference,
  computational environment, used weights and reproduction instructions are
  required for verification; Apache source-release duty is repeated.

**Established:** there is an express exception from *Apache rerelease* for certain
third-party software, input data and pretrained models. Do not relicense their
assets as World Reward Apache code. The text also expressly qualifies 6.c with
“Unless otherwise stated”.

**Unresolved:** whether those release exceptions also permit *non-OSI inference
source* necessary to use a pretrained model, and whether the organizer baseline
has a specific exception. The cited clauses/FAQ do **not explicitly name an
exemption for SAM Materials or FoundationPose source**. Do not treat their official
example use as proof of a waiver, or categorically declare them disallowed without
clarification. A model-weight exception alone does not establish a source exception.

## 2. Pinned baseline: component-by-component

Official source revision:
[`nvidia-isaac/video_to_data@7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`](https://github.com/nvidia-isaac/video_to_data/tree/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80).

| Component | Primary terms established | Submission/release status |
|---|---|---|
| Organizer-written wrappers | Root/reconstruction LICENSE assigns source Apache-2.0 and documentation/skills CC-BY-4.0; several wrappers have explicit SPDX Apache-2.0. | Apache source is compatible at the license level, with preserved notices. The parent license does not silently replace separately licensed vendored dependencies. |
| Native commercial MHR CARI4D checkpoint | The commercial model card designates NVIDIA Open Model Agreement and says commercial/noncommercial use. Exact weights below. | Pretrained-model rerelease exception is relevant. Do not substitute the original research checkpoint or assume its source has changed license. |
| Native CARI4D source in official release | Parent source license is Apache; e.g. wrapper `lib/__init__.py` explicitly SPDX Apache. Several native files retain proprietary notices requiring an express NVIDIA license; no separate CARI4D module LICENSE was found at this revision. | Scope of parent Apache over legacy native notices needs explicit confirmation; notices are not evidence of an additional commercial source grant by the weight card. This is not a categorical declaration that the parent Apache grant is ineffective. |
| SAM 3D Body / Objects source **and** weights | Custom SAM License, dated 2025-11-19, explicitly covers machine-learning model code, inference/training/fine-tuning code, weights and algorithms. | Not Apache; no OSI approval/competition source exemption established by this audit. Redistribution of SAM Materials/derivatives must remain under SAM terms. HF access approval does not answer eligibility. |
| DINOv3 actually used by Body | Custom DINO agreement at6876159 covers source/inference/weights; distinct from GroundingDINO Apache model. | Foundational6.c exception unverified; preserve its original terms, never relabel as Apache. |
| FoundationPose vendored NVlabs source | Custom NVIDIA license §3.3 limits use to research/evaluation noncommercially; NVIDIA/affiliates alone have a commercial exception. | Fails the commercial-use condition of 6.c if no competition exception applies. Cannot convert it to Apache by copying or wrapping it. |
| FoundationPose TAO/TensorRT models | Separate commercial ONNX models under NVIDIA Open Model License; HF card says commercial use. | This weight license does **not** relicense vendored NVlabs source. Switching the current wrapper to TensorRT does not eliminate its noncommercial source imports. |
| MHR reference model/source | MHR v1.0.1 release source license Apache-2.0. | Suitable license-level building block; preserve license/notices. MHR does not confer a license to SMPL/MANO conversion targets. |
| SAM 2.1 / MoGe wrappers and source | SAM2 code/model card Apache-2.0; MoGe pinned source MIT (dependency notices still apply). | No source-license conflict established for those first-party components; audit actual transitive dependencies and weight terms separately before release. |

### CARI4D: do not conflate three artifacts

1. [Original research source LICENSE](https://github.com/NVlabs/CARI4D/blob/71fa7cbe46081467edadd11ab534b0c14aa9d913/LICENSE)
   §3.3: “The Work and any derivative works thereof only may be used or intended
   for use non-commercially”; it defines this as “non-commercial scientific
   research purposes only”. Original code is not Apache just because the newer
   checkpoint is commercial.
2. Official MHR source parent
   [LICENSE](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/LICENSE)
   versus representative native
   [Utils.py notice](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_cari4d/lib/cari4d/Utils.py):
   “Any use, reproduction, disclosure or distribution […] without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.” Ask about
   scope; retain all existing notices, never strip them to resolve ambiguity.
3. Commercial weights: `nvidia/cari4d_commercial` artifact revision
   `1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9`,
   `2026-08-25-09-35-57/step200000.pth`, SHA-256
   `78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3`.
   The [official README](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_cari4d/README.md)
   pins that artifact. **There is no model card at that older artifact revision.**
   The [card at `6e064f8b261599a4e563d72cb8457256929d7455`](https://huggingface.co/nvidia/cari4d_commercial/blob/6e064f8b261599a4e563d72cb8457256929d7455/README.md)
   (2026-09-18) covers version `2026-08-25-09-35-57`, step 200,000 and declares the
   [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)
   governing. That agreement (release 2026-04-02, text version 2026-03-09) expressly
   says “Works are commercially usable”, permits derivative works, and requires
   license/notices on redistribution. Record the card revision separately from
   the weight revision, rather than falsely claiming the old revision has a card.

The commercial card reports 2,126 internal Daniel/FORM-HOI training sequences with
four cameras. This is **not a finding of challenge leakage**, nor proof of no
overlap. The organizer-provided pretrained model may be permitted, but do not
claim leakage-free training without organizer confirmation. Never download its
FORM-HOI training data or challenge-matched multiview/GT to investigate it.

### SAM Materials: exact scope and obligations

Pinned primary source licenses:
[Body `b5c765a0`](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/LICENSE),
[Objects `f91db411`](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/LICENSE),
and [official vendored Body copy](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/LICENSE).
Both upstream texts SHA-256:
`b3a5a0e2d973ab80e6610ccf1cffc40756050d0ace3cd4fec879b3ec290b2e9b`.

- §1.a grants use/copy/modify/derivative/distribution rights, not an NC-only grant.
  It is still a custom license with use conditions, not an Apache license.
- §1.b.i: SAM Materials and derivatives may be redistributed **only under the SAM
  Agreement**, with its copy. §1.b.ii requires acknowledgement in research publication.
- §1.b.iii–v contains laws/privacy/trade-control, reverse-engineering and prohibited
  end-use conditions. Do not assert unrestricted OSI status from the absence of NC.
- §8 permits amendments; archive exact accepted terms in the remote reproducibility
  manifest. Token/repository access does not replace terms review.

Pinned weights: Body `11aaa346c7204874a1cbafe3d39a979080b2c55a`;
Objects `2e73555018d2741ccd486e56c24fac41155a1dc6`. Keep upstream acquisition
separate; do not upload gated weights as World Reward Apache artifacts.

### FoundationPose: commercial weights do not fix source restrictions

[Vendored LICENSE](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/LICENSE)
§3.1 preserves the license/notices; §3.2 makes the §3.3 use limitation apply to
source derivatives too. License SHA-256:
`0de51abc6b73e24724858239efc42cfeada5d9e7c911cf7eff077bd9a23dec62`.

The [official downloader](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_foundation_pose/lib/download_weights.py)
separates NVlabs PyTorch/Google Drive research weights from TAO
`nvidia/tao/foundationpose:deployable_v1.0` ONNX models, requiring an explicit EULA
flag for the latter. [Commercial HF model card](https://huggingface.co/nvidia/FoundationPose/blob/18d8309afc9790cddc03a1d50bc69954dc058693/README.md)
covers the models, not an Apache release of the NVlabs code. Official CARI4D
[download_weights.py](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_cari4d/lib/download_weights.py)
selects `NVLABS_PYTORCH` explicitly.

Even the [TensorRT predictors](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_foundation_pose/lib/trt_predictors.py)
import vendored `learning.datasets`, `learning.training` and `Utils`;
[tracker](https://github.com/nvidia-isaac/video_to_data/blob/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_foundation_pose/lib/foundation_pose_tracker.py)
imports `estimater.FoundationPose`. Backend selection alone is **not a license
compatibility fix**. A separately licensed commercial runtime would need its own
source/dependency audit and technical equivalence validation.

## 3. Release decision and minimal organizer question

- Experimental topology tool: PyMeshLab `2025.7.post1` is GPL-3.0, an
  OSI-approved license permitting commercial use, but with copyleft obligations
  on redistribution. It is an external CPU dependency, not World Reward Apache
  source. Pin its wheel SHA-256
  `c3c1b01f101334b14469ace3b004382cd313b80a128f551a1da77e3053f09c30`;
  do not publish a combined image without its required notices/source offer and
  an actual dependency/release audit. License compatibility alone is neither a
  successful geometry gate nor a final competition-eligibility determination.

- First-party World Reward code should be Apache-2.0; root maintainer adds the
  license. Preserve third-party notices and identify their actual licenses.
- Separate own code, dependency pins, and externally obtained weights; document
  one-command reproduction plus lawful gated access prerequisites. Do not publish
  credentials or falsely promise fully anonymous downloads.
- **Final eligibility of a SAM/NC-FoundationPose-dependent solution is unresolved.**
  Seek a written public clarification before committing it as the award-eligible
  final stack. Continue useful data-free contracts/research, but do not represent
  official baseline use or a private inference process as an established waiver.
- An exception to release obligations does not remove an upstream licensor's
  restrictions. An organizer answer about challenge admissibility cannot itself
  grant Meta's or another licensor's rights.
- Do not use original NC CARI4D/HaWoR/WiLoR/MANO source as an Apache shortcut.
  This audit is not a complete transitive software bill of materials; perform that
  check on the frozen actual environment before submission.
- Additional transitive restriction: [nvdiffrast source license](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt)
  §3.3 permits only non-commercial research/evaluation with no direct/indirect
  monetary gain (except NVIDIA/affiliates). CUDA kernel functionality does not
  establish eligibility. Include this dependency in the organizer question and
  actual-image SBOM; prefer BSD PyTorch3D for our own commercially unrestricted
  synthetic renderer. Current runtime use is research, not an asserted waiver.

  Full native CARI inference does require this kernel, not merely visualization:
  pinned `tools/run_mhr_wild_inference.py` imports nvdiffrast, constructs
  RasterizeCudaContext and renders object geometry/texture through `Utils`.
  Switching World Reward's QA renderer to PyTorch3D does **not** eliminate that
  dependency from the native forward path. A separate source-compatible path
  or written eligibility clarification is still needed for final release.

**Draft only — not sent** to `v2d_challenge@nvidia.com` / the public Kaggle forum:

> For Track 1, can an award-eligible solution use the official baseline's SAM 3D
> inference code under the custom SAM License and vendored FoundationPose source
> under its research-only NVIDIA license? Foundational 6.c requires OSI-approved,
> commercially unrestricted source unless otherwise stated; Specific 2.6 exempts
> certain third-party software and pretrained models from Apache rerelease. Does
> that exception cover their inference source, or must it be replaced? Please also
> confirm whether native CARI4D’s mandatory research-only nvdiffrast runtime is
> admissible under an explicit source exception, whether parent Apache-2.0 covers
> native `v2d_cari4d/lib/cari4d` despite retained proprietary headers, and whether
> the provided commercial checkpoint
> `2026-08-25-09-35-57` is authorized for Track 1 despite its FORM-HOI training origin.

## 4. External real validation subset, not challenge labels

[TUD-L publisher](https://bop.felk.cvut.cz/datasets/#TUD-L) and
[HFcard6527f7d4b25d3e2e8dec84529284d9797b15f7b5](https://huggingface.co/datasets/bop-benchmark/tudl/blob/6527f7d4b25d3e2e8dec84529284d9797b15f7b5/README.md)
agree **CC-BY-SA4.0**, commercial use with attribution/share-alike obligations.
Base/models ZIPs have no LICENSE; retain primary source bytes/URLs separately,
not an invented embedded grant. HFREADME SHA f35ac7b3…31dcab; publisher447byte
TUD-L license section SHA aeccefac…20b31a (mutable page drift fails acquisition).
Attribution: TU Dresden Light, Hodaň, Michel et al., *BOP: Benchmark for 6D Object
Pose Estimation*, ECCV2018. Preserve original `dataset_info.md`, exact blob
hashes and modification notice for the selected subset. Never relicense media
as World Reward Apache. This validation does not generate challenge labels.

TUD-L is object-only RGB-D; no MHR/human quality guarantee. Main private sensor
evaluation can use external calibration/masks, never challenge calibration or
inference inputs. No declared MoGe2 train/eval entry names TUD-L, but checkpoint
training frames/backbone overlap are not independently certified.

Not acquired: CORE4D has conflicting CC-BY/NC metadata and SMPL-X-derived rights;
HOPE/HANDAL author sites retain NC contrary to HF/BOP cards; HOT3D hand labels
are NC with extra model restrictions. ContactPose nonmesh data/code MIT is
promising, but individual mesh terms and useful hashed minibundle remain
unaudited; do not import MANO fits or pretend it supplies full-body MHR truth.

## 5. D76 metric-depth candidate (pinned acquisition PASS; inference pending)

DA3METRIC-LARGE pin4010e39f3634a45bc60553321fb49fb760bd594e card declares
Apache2; matching source3d835ec1a5802d64a8b8b15f817a1ab54809bfe4 Apache2.
HF has no separate LICENSE file: retain its primary card and sourceLICENSE,
not an invented embedded grant. Minimal native depth runtime excludes API's
`evo` GPL and unused exporters; addon addict/einops MIT, OmegaConfBSD3 with
ANTLRBSD3/PyYAMLMIT, imageioBSD2, PillowMIT-CMU and tqdm MIT/MPL2 OSI file-level
terms. Record effective dependency versions and notices; this is not full image
SBOM clearance. NC DA3Nested/Giant checkpoints excluded. Exact training/backbone
frame provenance remains unverified, so training_overlap_excludedFalse.

GeoCalib top-levelApache2/weightCCBY4 declarations do not settle the provenance
of imported PerspectiveFields-adapted camera code (upstreamAdobeNC). No
full-package acquisition or commercial adoption. D82 acquired only the
audited standalone Apache module/four-class slice, exact retained Apache
GeoCalib/SegNeXt licenses and publisher README CC-BY-4.0 statement. This does
not clear PerspectiveFields/LM/full-package code or prove weight-training
overlap exclusion. Runtime excludes those imports; native strict load and
independent analytic calibration require separate engineering checks.


2026-10-03 primary-source recheck: officialmain remains7c0d3b94, releasev0.4.0
(Sep23); exactFAQa1e569e1c83eeac2584de34e9503f1226e20ab369d53d2c67c2e243deb1a4d7a
unchanged. No new explicit source exception for SAMcustomnonOSI, legacyCARI/
FoundationPose or mandatorynvdiffrastNC, nor FORMpretraining-overlap waiver.
NvdiffrastLICENSE latest1f95925cdad6e792961f0d4a077950bb14b785d8 unchanged;
commercialCARIcard remains6e064f8b261599a4e563d72cb8457256929d7455.
KaggleliveJSshell did not expose rules text; this check does NOT replace the
Oct2browser rule audit/acceptance. No written clarification received, outgoing
organizer question/registration notsent. Research engineering continues;
finalrelease/awardeligibility remains unresolved, not an asserted waiver.


### 2026-10-03 independent eligibility re-audit

Primary SAMBody/Objects customlicence againconfirmedcovers inference source,
not just weights; FoundationPose and nvdiffrast NC restrictions stillapply.
CurrentnvdiffrastLICENSE at253ac4fcea7de5f396371124af597e6cc957bfae retains
noncommercial research/evaluation and no direct/indirect monetarygain.
CARIforwardactuallyusesCUDA rasterized inputrenders; TAO/TRT trackingstillimports
NC FoundationPose/learning. Replacing a visualizer or weights alone cannotclose
this dependencygraph. Organizer clarification remainsunsent/unresolved; public
KaggleJSshell notclaimednewverifiedrules. No waiver/eligibility/overlap clearance.

MHR v1.0.1 [primaryrelease](https://api.github.com/repos/facebookresearch/MHR/releases/tags/v1.0.1)
explicitly states “Added LICENSE to assets”; primarysource licence isApache-2.0.
Publishedassets.zip digest e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94,
198943157B (metadataONLY, not downloadedlocally). Existingacquire_weights.py
retainsmhr_model.pt butnotassetsLICENSE. Preserveactuallicense+notices before
bundling/distribution; a manifestlabel alone isnotnoticeclosure. SAM-acquired
MHR checkpoint remains its originalSAMMaterials provenance, neverrelabeldedup
asApache. MANO/SMPL conversiontargets remainseparate. Continue research with
permissiveown/MHR/MoGe/SAM2/P3Dblocks whilefinallicencepath staysunverified.


Scope correction from actual ownsource: currentobject_pose_smoke.py doesNOTrun
FoundationPose; its producer is `own_ICP_Viterbi_not_FoundationPose`. FP libraries
in runtimeimage or documented possibleTAO route donotprove theirusebythisown
tracker. Keep the existing clean tracker. ActualCARIforwardnvdiffrast inputrender
remainsa separateunresolved NCdependency; SAMBody/Objects sourceexception and
legacyCARIgrant also stillunresolved. No blanketeligible/runtimeclearedclaim.

## October4 primary-source recheck

Official repository remains7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80; FAQ is
byte-identical84452B SHAa1e569e1c83eeac2584de34e9503f1226e20ab369d53d2c67c2e243deb1a4d7a.
No explicit non-OSI inference-source exception was found. PR164 is closed,
draft and unmerged, not adopted permission. Kaggle HTTP-only page is a JavaScript
shell, not a new authenticated rules audit. Contact remains
[v2d_challenge@nvidia.com](mailto:v2d_challenge@nvidia.com), verified on the
[official contact page](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/#contact).
The organizer clarification below remains unsent; eligibility is unresolved.

### October4 independent full-body validation alternatives

No commercial/competition-compatible real full-T HOI reference is newly cleared.
[HOI-M3 toolbox](https://github.com/Juzezhang/HOIM3_Toolbox/tree/0665e177c3f8285593e688c265fa56c44ae314ff)
and [publisher metadata](https://huggingface.co/datasets/JuzeZhang/HOI-M3/tree/1359843ef96929bbc96f6129ffbb8d60cfc89c2d)
make single-camera acquisition technically feasible, but code/site licenses do
not establish rights to captures, multiview reference fits or MHR assets. Mono
MHR estimates are not independent truth; frame pairing and the two documented
camera gauges must be clarified. Contacts `wangjingya@shanghaitech.edu.cn` and
`xulan1@shanghaitech.edu.cn`; no message or acquisition made.

[CORE4D license issue14](https://github.com/leolyliu/CORE4D-Instructions/issues/14)
remains unresolved: README CC-BY4, publisher MIT and dataset-site NC disagree.
[Object poses issue13](https://github.com/leolyliu/CORE4D-Instructions/issues/13)
and [synchronization issue9](https://github.com/leolyliu/CORE4D-Instructions/issues/9)
also prevent assuming a clean shared SE3 reference. HIMO is explicitly NC;
HUMOTO public animations are not verified synchronized real RGB. No model
overlap or derived-label rights are inferred from downloadable files.

### October5 — targeted BEHAVE/InterCap private-evaluation recheck

Primary text only, no registration, RGB/annotation/archive acquisition or email.
[InterCap licence](https://intercap.is.tue.mpg.de/license.html):14,569B,
SHA256 `1a13747439996bf95d55784511a5dbcfa977c839033d75483240429589cd9d7c`;
allows non-commercial scientific research/education/artistic projects and
restricts commercial artefacts/third-party disclosure. Download redirects to
login; registration asks email/password and licence/privacy acceptance. Contacts
`intercap@tue.mpg.de`; licence `ps-license@tue.mpg.de` differs from site's
`ps-licensing@tue.mpg.de` and must not be silently conflated.
[BEHAVE licence](https://virtualhumans.mpi-inf.mpg.de/behave/license.html):15,593B,
SHA256 `f5ba537b429c1f3e91ef95ce34577e0dbb7ff9b1ef1999c341251c170a1ffe14`;
allows non-commercial scientific research, restricts commercial artefacts and
third-party disclosure, requires face blurring in published images. Original
Date01–Date07 archives and RHOBIN evaluation packages are linked, not acquired.

Neither text categorically bans all competitions, but prize-bearing V2D method
selection/development is not explicitly covered. Private GT, no training and
term acceptance alone do not settle that intended use. Decision: defer
acquisition/adoption until targeted clarification; not a blanket illegality
claim. Body-model/reference rights and model training overlap remain separate.
Unsent question: authorize private Azure evaluation, monocular RGB-only predictor
with annotations isolated to metrics, to choose a prize-eligible World Reward
V2D method, without data redistribution/commercial exploitation, and publication
of aggregate results? Human approval for outbound email is still outstanding.

### October5 — two distinct full-HOI alternatives remain deferred

[ParaHome@535dada](https://github.com/snuvclab/ParaHome/tree/535dada556536a54d8b3a5185f2a153e6e3ccbca)
README5551B/SHA `ff0ac820b7539645a3131e05d525ffa36fdf5efe85d68c4c5b83e93892c85ef4`
grants CC-BY-NC-SA4, explicitly non-commercial academic use. The site licence
is not the dataset grant. No verified small synchronized RGB/GT bundle or prize
permission; separate SMPL-X rights/overlap remain unknown. No acquisition.
[CHOIS@8ec585a](https://github.com/lijiaman/chois_release/tree/8ec585aa0200fd2a890ffb12897bcf69ae719463)
README5562B/SHA `5266e8a7df6dc2e4c0a745e7806982cc5d1575df22e32bb9568c023439d96fcf`;
MIT LICENSE1066B/SHA `deb92c90a9b660c53ef267c7895f828ddc99783db191d55e26da4fee7883e5d6`
covers software/documentation, not an independent grant of its processed OMOMO
captures/scans. Motion generation is not a verified monocular reconstruction
benchmark. Neither route unlocks lawful real full-HOI validation today.

### October5 — newly identified hands–object validation route

[HO-Cap's dataset grant](https://irvlutd.github.io/HOCap/) is explicitly
CC-BY-4.0, distinct from its GPL toolkit. Public
[HPE/OPE splits@576c63e](https://github.com/IRVLUTD/HO-Cap/tree/576c63ebf3b84dfec8744ba0f021234213bf0dab/config)
identify subject_5 clips20231027_112303 and20231027_113202 as test-only in both
tasks; subject5's third clip is not OPE-heldout. Camera105322251564 is a possible
single-view input. Original subject archive3,427,758,110B plus labels, poses,
models and calibration totals about5.08GB; no acquisition performed. Decimated
benchmark indices are not full trajectories: qualify original inventory/frame
counts and isolate every annotation/mesh/calibration from prediction first.
The [primary paper](https://arxiv.org/html/2406.06843v1) describes coupled RGB-D
fits with MediaPipe triangulation/interpolation/MANO, not independent mocap.
Actual distributed-joint provenance, separate model rights and training overlap
remain unverified. This permits investigating a conditional hand/object 3D
diagnostic, not full-body validation, acceleration/contact truth or V2D clearance.

MMHOI's publisher DATA grant is CC-BY-SA4, but its93.68GB archive, sampled
benchmark frames and manually refined SMPL-X reference do not yet establish a
small continuous full-T independent holdout. OakInk2's CC-BY-SA4 dataset has
upper-body/hands-object coverage and separate body-model rights; lower-body
parameters are documented unused. Both remain deferred, with no acquisition.

Fresh metadata-only OakInk2 audit at HF21705616140d726607027e70d58b7837f442ffd8:
ungated/card CC-BY-SA4, exact program.tar3594240B/
67d234e396e8392c95234db5dd84b33163469419aa68117bf1c355ee7319d500.
Toolkit502a02809b50b7f5d0037f92a1ccd2a479051ad1 has no LICENSE in its complete
tree and no setup license; do not incorporate it. Task-object references can
avoid MANO/SMPL-X but require separate private geometric projection and do not
test bystander actor selection. No heavy acquisition justified at this stage.

MASA source c5472b9c7615f35abdf1188cb1a0c5408fe50d66 is Apache2; independent
R50 publisher card25ed372c47f2c46cf36fd446d1b657b656bc7ea9 declares Apache2.
Declared training SA1B500k, overlap unknown. Acquisition/runtime are prospective,
not submission clearance or an interaction-quality certificate.
October6 V-COCO delta: the original README says this repository hosts data and
code, with a root MIT grant and no data exclusion. The documented repo-wide
interpretation is accepted for private metadata research; annotation-specific
legal certainty is not claimed. Original role/COCO2014 byte acquisition and
the crowded-role census are independently qualified, without RGB or FIT.
Original COCO/Flickr individual photo grants, attribution/creator identity,
checkpoint overlap and deployment/submission eligibility remain separate.
See `vcoco_reference_feasibility.md` and `vcoco_role_census_protocol.md`.

## Experimental general solid certificate (October4, not adoption)

The independently written certified_solid_query.cpp glue is Apache-2.0.
CGAL6.0.1 PMP/Side headers are GPL-3.0-or-later OR commercial; the linked
executable is not Apache-only. Preserve upstream notices/corresponding-source
access and do not infer a challenge Apache-release exception. Primary release
library5077192B publisherSHA c752737f91d1af71fa96038f0e37945ce82a5f1fffb6200172cfcdd77755a356.
Pure oriented_solid_forest does not itself import CGAL/certify geometry.
The isolated Azure runtime is now compiled and passes fifteen fresh procedural
controls under253fc9d; actual image/binary/receipts are independently pinned in
certified_solid_qualification_pins.json. No production geometry or submission
eligibility is established by these controls.
