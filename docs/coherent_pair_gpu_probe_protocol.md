# Tiny native coherent-pair GPU control v1

Prospective, data-free numerical control, frozen before the first execution.
No model, RGB, actual bank, external/challenge references, FIT, optimizer,
hyperparameter search or quality metric. A PASS qualifies only tiny arithmetic
on the observed runtime, not full-bank cost, ownership, adoption or CARI4D.

## Frozen recipe

The source `manifest()` is published in `proof.json` before native imports or
fixture construction. Eight NEW procedural banks use (P,O,hand copies):
`(3,3,2), (3,3,3), (2,3,2), (2,2,0), (1,2,0), (0,2,2), (2,0,2), (0,0,0)`.
Copies yield all ordered native hand/direct-object pairs (K≤9); they are not
inferred labels. Variants cover complete banks, aliases/duplicate proposals,
missing and off-grid wrists, zero-area unsupported objects, absent HOI and
empty populations. No old test fixture or closed-study data is imported.
Original IDs/provenance and all native slots survive.

Original grid 48×64, frame 7. Tau `.8125`; alpha `0,.3125` are distinct fixed
controls, not a sweep. Theta j (j=0..16) is `(-1 if j%2 else 1)*(j+1)/64`;
12 scales are `(j+3)/8`, with all scale flags active. All fixtures must finish;
no repair, resampling, retry, tolerance relaxation or unavailable-case removal.

## Predeclared checks

- Source-authenticated packed CSR, CPU packed and enumerative NumPy oracles.
  Every score/analytic VJP: rtol=atol=`1e-12`, equal NaNs; Boolean support,
  identity and parameter fingerprint exact.
- Two complete device executions: dtype, shape and RAW bytes identical,
  including NaN payloads. At alpha zero A/B scores and geometry VJPs share
  buffers; the right alpha derivative is retained, not forced to zero.
- All 17 coefficient central differences, h=`1e-6`, rtol=atol=`1e-7`.
  Alpha zero uses second-order right difference `(-3f0+4fh-f2h)/(2h)`;
  positive alpha uses central difference. FD is explicitly inapplicable only
  when that fixture has no supported target; such fixtures remain required.
- Ordered 2D CUDA segment controls widths 1/9/6/2/17, sum/max, CSR holes and
  all-empty segments. Extreme mixture overflow must raise, never clip/fill.
- Hash ALL device snapshot buffers (`_arrays`, `_layouts` including inverse
  CSR, `_pairs`) before and after scoring/FD, plus immutable host tables/IDs.

## Native envelope / reproducibility

VM02 only, existing immutable B47 image
`sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7`;
observe its actual image/rootfs before/after. Require Torch `2.5.1+cu124`, CUDA
12.4, Python 3.11, one H100 compute capability 9.0. No install/network/model
load. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before Torch imports; TF32 off,
FP64 and deterministic algorithms on. The runtime policy is recorded, not a
claim of cross-machine bit replay.

Inclusive host deadline **1200 s** includes source proof, preparation,
controls, comparisons, FD, cleanup, hashes and sealed receipt publication.
Outer 1215 s timeout has kill grace solely for failure cleanup. GPU lease FD9
and idle check precede execution. Four CPUs; Docker host-memory limit 6 GiB
is distinct from GPU Torch-allocator limit 6 GiB. Set allocator fraction from
actual GPU total memory and require peak allocated AND reserved ≤6 GiB;
CUDA context/driver allocations are not falsely included in this metric.

Only exact readonly source leaves/markers and a fresh owned output are mounted;
no data/model/private tree. Host authenticates full source closure and fixed
helper byte pins, native checks mounted leaves, source/manifest rehashed after.
Receipts persist source/runtime/fixture/digest/check results, no array values.
Only the CID with exact image/name/owner label may be stopped/killed/removed;
absence is verified. `native.json` and `report.json` are sealed even on FAIL;
same owned FD demotes PASS if deadline is crossed during final publication.
Result namespace: `/srv/scenesmith/world-reward/results/coherent-pair-gpu-probe-v1`.
Any failure closes this one control; future changes require a new protocol.

## Pre-execution qualification, October6

Root262 combined tiny/source tests PASS2.79s, including38 probe controls and
15 device-prototype AST controls; no local Torch import or GPU computation.
Independent source review caught incomplete runtime/segment/FD receipt checks
and name-only cleanup proof. These were fixed before any execution: exact20
ordered segment records, actual runtime, four FD-applicable/four unsupported
cases and saved-CID absence are required. Source review is not native ABI proof.
The first run targets **VM02**, not VM01, and must use its exact public frozen
commit with a fresh result namespace. No full-bank recipe or FIT is authorized.

## Actual first attempt: closed image preflight, no numerical execution

Producera48aaec runs on VM02 and stops in `image()` before output creation,
Torch import or container/fixtures. B47's Docker OCI ID belongs to VM01; VM02
uses the separately documented classic config ID7ebfff for the same exported
graph. Do not add an arbitrary image-ID fallback or rebuild/retag. Unit exits1,
PID0; no proof/native/report/CID exists, zero GPU processes afterward.
Two observer mistakes (target identity, literal marker newline) remain separate
INCONCLUSIVE records. Root saved-only bounded diagnostics locate the image
inspection error, not a mathematical failure or full-source PASS.
Next: a new explicit **VM01** profile with identical fixture/math/FD/memory/time
gates and fresh namespace. Existing v1 source/result remain historical and closed.
