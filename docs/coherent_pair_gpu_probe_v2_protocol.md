# Tiny coherent-pair GPU control v2: technical host correction

V1's actual VM02 attempt closed in image preflight: Docker did not contain the
exact requested B47 image ID. It produced no `proof.json`, native receipt,
container, Torch tensors, GPU arithmetic or fixture. An acknowledged launcher
is not a native proof. Keep that failed producer/source and diagnostic intact.

V2 uses **VM01 `scenesmith-ncc-h100-01`** (Azure SCENESMITH-H100), where the
original B47 runtime is expected; it must still pass actual image inspection.
No VM02 fallback, image alias, transfer, rebuild or installation is allowed.
Image remains exactly
`sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7`.

One shared driver selects explicit frozen profiles before host/source checks:

| Profile | Entry | Output | Schema |
|---|---|---|---|
| v1 | `run_coherent_pair_gpu_probe` | `results/coherent-pair-gpu-probe-v1` | `world_reward.coherent_pair_gpu_probe.v1` |
| v2 | `run_coherent_pair_gpu_probe_v2` | `results/coherent-pair-gpu-probe-v2` | `world_reward.coherent_pair_gpu_probe.v2` |

Default v1 preserves its original host, wrapper, namespace and all numerical
gates. V2 requires its own immutable wrapper/source closure; host and native
CLI both receive `--profile v2`. The v1 wrapper and numerical helpers are not
modified. Historical executed sources remain frozen by their original Git pin.

All substantive v1 checks are unchanged: eight authored procedural cases,
P≤3/O≤3/K≤9, tau `.8125`, alpha `0,.3125`, original fixed 17 coefficients and
12 scales, full CSR/snapshots/IDs, two raw-byte repetitions including NaNs,
NumPy packed/enumerative agreement rtol=atol `1e-12`, FD17 and second-order
right-alpha-zero h=`1e-6`, rtol=atol `1e-7`, 20 ordered segment controls,
support/missing/empty/overflow behavior. No data/model/FIT/labels/optimizer.

Envelope unchanged: FD9 exclusive GPU lease and actual idle check, four CPUs,
1200 s inclusive / 1215 s outer failure-control timeout, Docker host-memory
6 GiB distinct from Torch allocated/reserved GPU limit 6 GiB, FP64, TF32 off,
deterministic algorithms and pre-import `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Only exact source leaves/markers plus fresh owned output mount; exact owned
CID cleanup and source/manifest/snapshot before-after verification. Same-FD
publication demotes late PASS and seals failure receipts. No retry or numerical
relaxation. PASS means tiny arithmetic only, not full-bank speed or accuracy.

This is a **technical preflight correction**, not a rerun of an executed
numerical evaluation: V1 never reached arithmetic. Its frozen source and failure
remain separate from the v2 result below.

Source qualification, October6: root171 combined tiny tests PASS1.24s;
independent AST/lifecycle review READY with unchanged v1 mathematical functions
and wrapper. No local Torch or native GPU evaluation entered these checks.

Actual producer `d3506722efb77d034ee329d04609c621188d6c0b`: sealed host/native
PASS, 7.240327705 s, H100 NVL / SM90 / Torch2.5.1+cu124. Eight fixtures,
20 segment controls and288 FD calls complete; peak Torch allocated33723392 B,
reserved35651584 B. Independent saved-only audit authenticates original302 Git
files/307 entries, source/image/manifest/snapshots/output before-after evidence,
exact owned CID/name/label absence and native driver absence, without replay.
The successful systemd unit was already collected: its independent exit status
is unavailable; default0/success fields are not proof. Qualification relies on
source-bound sealed native/host receipts and native-exit0 evidence. The first
unit-not-found observer remains INCONCLUSIVE, not native failure. See
`results/audits/coherent_pair_gpu_probe_v2_actual_v3.json`. Tiny arithmetic only;
full-bank cost, objective/optimizer, ownership and accuracy remain unqualified.
