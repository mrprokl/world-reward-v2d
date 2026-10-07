# Frozen short pilot — 2026-10-07

Hypothesis: jointly grounding the supplied object description **and action** over
ordered RGB views reduces background-person/object selection failures compared
with our existing GroundingDINO/person-affinity frontend. This is not a novel
trained model or a claimed SOTA result. Use the native zero-shot
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/README.md)
spatial/multiframe interface, followed by the unchanged SAM2 full-video tracker.
The publisher declares Apache-2.0; training/challenge overlap is **unknown**.

Population: all three previously inspected stress clips **8, 9, 26**. These are
development visual diagnostics, deliberately **not** a random or held-out set.
No quantitative accuracy, generalization or victory over CARI4D can follow from
this test. External temporal reference evaluation remains necessary for adoption.
Earlier external studies stay closed; no synthetic benchmark is started here.

Freeze before inference: exact weights/file identities, code/global prompt,
9 ordered full-frame views `i*(T-1)//8`, original frame indices, native processor
pixel bounds 65,536–589,824, BF16/SDPA, greedy decoding, seed 0, at most 1,536 new
tokens. One model call per clip; no output repair, retry, prompt change, manual
boxes, resampling or alternate model. Official action text is evidence, not a
motion template. Missing or ambiguous boxes may be null. Choose the first view
with both boxes automatically; keep IDs 0/person and 1/object through SAM2.

Budgets: Azure acquisition ≤1,800s/17,545,914,364B, three model calls including
load/posthash ≤900s, all three full-T tracks ≤1,800s, preview ≤180s; outer cap
4,800s including overhead. Existing baseline outputs are read-only. No GPU job
runs concurrently on this VM. Input/output/model identities and actual elapsed
times are retained. Disposable partial downloads and owned containers are removed.

Gates: exact allowed RGB/metadata provenance and native runtime; strict complete
9-entry JSON/no inverted or out-of-bounds boxes; no retry if invalid; null/failed
clips remain in the denominator and visible in QA. Every successful track must
contain both mask streams at **all original T** indices. Empty areas/area jumps
are diagnostics, not accuracy gates; real occlusion must not be deleted. A
technical failure closes this run, without raising its budget. A scientific
failure/obvious identity drift rejects this candidate, not the episode.

Deliverable: small Azure-generated baseline/candidate overlays at fixed first,
middle and last indices, plus an automatically chosen uncertainty/area-change
view. Show every clip, including abstention/failure. Human review classifies
general defects and decides whether a subsequent experiment is warranted; it
must not select/correct challenge target IDs or supply per-clip prompts.

No expensive 4D rerun until this association screen has been reviewed. Approval
of masks does not validate 3D or submission metrics. Keep the frozen run and
its results; change only the general algorithm in a separately declared study.
