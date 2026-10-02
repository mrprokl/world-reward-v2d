# Source/model licensing audit — 2026-10-02

Scope: Track 1 submission eligibility, public winner-code release, and the pinned
organizer baseline. This records primary terms, **not legal advice or a permission
from the organizers**. Commercially usable weights, public source, an OSI-approved
source license, and challenge eligibility are four different checks.

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
