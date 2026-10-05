# Objective scalar-gradient indexing — technical ABI correction

The original `7281db2a3f7c7298a0d2a22e6ad7e7036fbe4e18` control is closed
FAIL: sealed native phase `objective_oracle`, error family `OtherError`, no
completed objective/FD/solver controls. Its source, receipts and gates remain
unchanged. The receipt does **not** record the exact exception or failing line.

Primary PyTorch **v2.5.1** source establishes a universal indexing incompatibility:

- [TensorIndexing.h](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/TensorIndexing.h),
  23,965 bytes, SHA256
  `44d5c6cebb19a4bd620391c64fa2e83009b15beb10a046d455fa787854258056`:
  lines312–319 record a non-scalar tensor index and advance the index dimension
  once; lines460–463 apply `None` by inserting a dimension before advanced
  boolean-mask expansion.
- [IndexingUtils.h](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/IndexingUtils.h),
  5,570 bytes, SHA256
  `2e85c9ea26e416dda38405b25c78af172d60cbb4a3d88aded224f8eeeeb68fe4`:
  lines30–35 require every boolean-mask dimension to match the intermediate
  tensor and otherwise raise an indexing error.

Consequently, for the complete group grid, `derivative[support, None]` first
forms an intermediate `[4,1,3600]`, incompatible with mask `[4,3600]`.
The NumPy-backed test double previously accepted NumPy's different semantics.
The narrow new-source correction selects first, then reshapes:
`derivative[support].reshape(-1,1)`. It changes no values, reduction order,
population, objective, parameters, tolerance, optimizer or scientific gate.

An independent tiny source-mirroring backend reproduced an accepted geometry
arm followed by scalar-arm `IndexError` before the CPU comparison; `IndexError`
maps to the original generic `OtherError`. This is consistent with the saved
failure, not a recovered native traceback or a GPU qualification. The regression
now explicitly rejects the old expression and checks both arms at alpha0 and
0.3125 with the corrected expression, without importing Torch. Any future native
execution needs a fresh namespace and authenticated corrected source; the old
FAIL is not converted into PASS.

## Prospective technical resumption, same scientific recipe

The maintained driver keeps its entrypoint but writes only the fresh
`coherent-pair-gpu-objective-probe-v2` namespace under a new producer. It pins
the corrected objective and host-authenticates all three immutable7281 failure
receipts plus the complete original source before/after. Only their declarations
cross to the child; old receipts/source are not mounted. No failure is prior PASS.
The same B47 image, fixtures, full population, scales, theta, tau, alpha, lambda,
tolerances,72 FD calls, solver options and1200s budget remain fixed. Root AST
regression proves `manifest`, `measure`, `solve`, `scipy_evidence`,
`validate_native` and `publish` identical to the failed producer. No duplicated
runtime/profile layer is added.304 combined tiny controls pass in1.75s, including
52 objective tests and explicit ABI/failure-source controls. No native objective,
SciPy, optimization, FIT or quality result follows before actual execution.

Actual producer8db0500a09589dd288ecd8494f00ad33f61983ec completes both
full-bank objective/CPU-GPU repeat-bit comparisons and all72 prescribed FD
calls, then fails at `scipy_native_inventory` with `ValueError` after86.108390s
host elapsed. No solver ran and no SciPy inventory was qualified. Thus the
complete control remains **FAIL**, while the recorded objective/FD subcontrols
passed on the same1,843,200-route bank. Source312files/317entries, original
failure lineage, image, sealed receipts and cleanup passed saved-only audit;
see `results/audits/coherent_pair_gpu_objective_probe_v2_actual.json`.
The receipt lacks the failing inventory condition; a separate bounded,
metadata-only diagnostic is needed before any runtime repair, not an objective
rerun or relaxed numerical/scientific gate. No FIT/adoption/quality claim.
