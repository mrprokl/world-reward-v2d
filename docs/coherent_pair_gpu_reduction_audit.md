# Coherent-pair GPU reductions: source audit

**2026-10-05 — prospective method, not native qualification.** No Torch import,
CUDA compilation, Azure execution, models, RGB, references or FIT were used.
Existing CPU scorer and its numerical contract remain unchanged.

## Primary source pin and findings

GitHub's tag API returned `refs/tags/v2.5.1` as commit
`a8d6afb511a69687bbb2b7e88a3cf67917e1697e`.
[Tag mapping](https://api.github.com/repos/pytorch/pytorch/git/ref/tags/v2.5.1).
This is a source pin, not proof of the installed B47 runtime. Raw-file SHA256s
were not computed; no file-hash assertion is made.

- [SegmentReduce.cu](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/native/cuda/SegmentReduce.cu):
  lines 80–143 assign one output segment/lane to a thread, looping over its
  offsets in order without atomics. Lines 438–521 dispatch **2D output** to that
  kernel, but **1D output** to CUB. Thus `[incidences,1]` selects the ordered
  path; CUB's reduction order and cross-runtime bit parity are not established
  by this audit. FP64 is supported by the dispatch at 415–419. SUM backward
  writes disjoint input cells (216–222). MAX/MIN tie backward divides only
  positive gradients (204–206): negative tied gradients are not divided.
- [SegmentReduce.cpp](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/native/SegmentReduce.cpp):
  offsets safety checks remain TODO at 399, even with `unsafe=False`.
  Lengths checks at 418–422 use `.item()` and synchronize GPU results.
  `indices` reduction is unsupported (381–383). CPU also contains the
  positive-only tie-gradient division at 258–260.
- [derivatives.yaml](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/tools/autograd/derivatives.yaml#L2586-L2587):
  ordinary autograd calls `_segment_reduce_backward`; it does not supply our
  hierarchical analytic VJP.
- [TensorAdvancedIndexing.cpp](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/native/TensorAdvancedIndexing.cpp#L1527-L1538)
  implements ordinary gather backward with `scatter_add_`.
  [ScatterGatherKernel.cu](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/native/cuda/ScatterGatherKernel.cu#L448-L454)
  identifies the ordinary CUDA scatter-add kernel as nondeterministic because
  of atomics. Deterministic-mode alternate dispatch is not qualified here.

## Decision: fixed CSR, FP64, explicit analytic VJP

Prefer `torch.segment_reduce(x.reshape(-1,1), "sum" or "max",
offsets=offsets, axis=0).reshape(-1)` under `no_grad`. Validate CSR **once**:
contiguous int64, correct device, 1D, start 0, nondecreasing, final offset equal
to incidence count. Preserve duplicate proposals, frozen ordering, support and
all original conditional priors. No padding, clipping, filtering or fake states.

For each nonempty segment: `m=max(x)`, `e=exp((x-m)/tau)`, `z=sum(e)`,
`LME=m+tau*(log(z)-log(n))`, posterior `e/z`. Use explicit empty-segment handling
(unsupported score NaN); never evaluate `-inf-(-inf)` or pretend zero evidence.
The stabilization maximum has **no autograd chain**. Apply posterior products
through distinct conditional margins, geometries, then equal-supported sides.

At `alpha=0`, A/B scores and geometric VJPs share computation exactly, while
the right alpha derivative remains the side/geometry posterior expectation of
each geometry's mean **distinct conditional margins**. It is not zero simply
because the alpha-zero forward branch is shared.

Precompute stable inverse CSR by factor/component ID, sum posterior adjoints
in that order, then contract the base/local/bridge factors into 17 gradients.
Avoid autograd gather fanout, `index_add`/`scatter_add`, and a global routes×17
feature tensor. An optimizer can consume explicit `loss_and_grad`; only final
loss/gradient scalars need transfer to its host callback.

## Qualification and limits

The width-one kernel serializes each segment; many target segments provide
parallel work, but long segments may be slow. Bounded padded chunks are a
separate prospective fallback, not global maximum-length padding. No speed or
determinism claim follows from source inspection.

Before FIT, freeze a native probe: actual image/Torch/CUDA/GPU identity, repeat
bit tests, CPU-oracle agreement, empty/support/tie/overflow checks, analytic FD
including alpha's right derivative, preparation plus full native/group VJP,
and peak memory. The existing tiny CPU tolerances are not evidence of GPU
agreement. **1200 s and 95 GiB are suggested future limits, not frozen or
achieved gates.** Keep theta/tau/alpha and any numerical tolerances explicit;
do not relax a failed frozen probe. This is computational feasibility only,
not ownership accuracy, model adoption, or a verified win over CARI4D.
