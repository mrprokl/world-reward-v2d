# Full48 public byte replica v1 (prospective)

This is a new byte-transfer profile, not a replay of the closed/failed old16
exports. No model, GPU, RGB decode, numerical NPZ load, reference, selection,
training or quality evaluation is performed. All payloads stay on Azure.

## Frozen scope and independent inputs

Original endpoint producer `6837a74a36a29c7c0cc79dfecfb98fc03d569a3f`:
340 files / 345 source entries, closure
`dff99fe3f87febfed73162c3fc535d8c83ed11d44d1271a4c23564d35e6c1c4c`,
XZ SHA `a9c86bf2b00e48450b6a84e866987e8068e3bde726909f1446953534579cd47c`.
The driver freezes independently observed host/native/proof byte pins and the
9754-byte public manifest SHA. It verifies original full source, readonly Git
modes, unchanged reused helper bytes, successful source-bound saved declarations,
all900 person queries/all3600 OWL patches per image and every original file SHA.
The saved acquisition/runtime posthash is authenticated prior-producer evidence;
this transfer does **not** rerun acquisition context or runtime qualification.

Exactly98 transferable leaves:

- `inputs/manifest.json` and48 original `image_000000.jpg`…`image_000047.jpg`;
- `banks/image_000000.npz`…`banks/image_000047.npz` (complete original17 arrays);
- fresh `public-reference.json`, top-level exactly `schema,inputs,manifest_identity,banks`.

The bank projection contains only the original16 public bank fields. Old
host/native/proof receipts, private acquisition/context/grants, split/native
dataset IDs, role references, checkpoints, Git archives and CIDs are never tar
members. Original opaque image and native retained-person IDs remain unchanged.
The first independent manifest control adds one member:99 canonical USTAR regular
members total, mode0400, uid/gid/mtime0, no links/PAX/GNU aliases, sorted order,
zero padding/tail, streamed member SHA, maximum128MiB including TAR overhead.

## Execution, failure and installation

One300s inclusive attempt per export/import,315s shell outer cap,8GiB host virtual
memory limit. VM02 export / VM01 import are verified through existing private
Azure metadata mechanics. Managed-identity tokens stay in RAM. One immutable
BlockWriter commit uses a fresh revision-scoped Blob; no alternate URL, retry,
reseed, media copy locally or image installation. Fixed bounded failure-stage
enums distinguish source/peer/sender/pack/commit/HEAD from later publication;
exception text and private subprocess output are never serialized.

Import requires independently supplied export-receipt, archive and first-manifest
byte/SHA pins. Fresh destination is
`/srv/world-reward-data/vcoco_full_public_replica_v1`: root0700,
`inputs/` and `banks/`0500, all98 regular single-link leaves0400. Staging is fresh
and atomic `renameat2(NOREPLACE)`; no overwrite, source chmod, broad cleanup or
fallback. Partial cleanup only removes originally opened owned inode leaves.
Original/current source and every payload SHA/state are rechecked before/after.

Separate technical receipts live at
`ROOT/results/vcoco-full-public-replica-{export,import}-REV/`. Import technical
leaves are `report.json,manifest.json,export-receipt.json`; they are **not** native
mounts. PASS is published on one originally opened0400 report FD after sealing,
fsync, deadline and posthash. Only after installed bytes and receipt are sealed
does one DELETE with the exact original ETag occur;202 and one attempt are
required, followed by another source/replica check. Failure demotes that same FD;
no re-export or third-party Blob cleanup is implicit.

## Reusable receiving API

`authenticate_receiver(code, replica_revision, receipt_pin)` is host-only and
returns `images,banks,projection_path,projection_identity,files,states,
import_source,import_identity,original_source_declaration` plus explicit
`sender_source_live_verified=False,sender_runtime_live_verified=False`.
`files` includes the98 public leaves and3 technical receipts; consumers must
mount **only** public projection/inputs/banks and their own native source whitelist.
Native DWPose/HOI consumers must reopen and numerically verify all17 endpoint
arrays before the first model callback. Byte identity here does not qualify
native execution, ownership, relation accuracy, external generalization or
submission eligibility. Actual run and independent audit are pending.

## Actual v1 failure and envelope-only repair

Producer `984e419dd21727f8ba32b37f6ed741af909206b8` closed FAIL in1.3023s
at sender/ValueError; no qualified archive or ETag exists. Its sealed receipt
8334B/SHA256`eb32cbe957ae9b08f2e9f754799a34b09c649c579b49ddbdb4d4e4e9cbd9a0be`
and namespace remain unchanged. An uncapped saved-only sender diagnostic passed,
so that first diagnostic did not establish a historical cause.

A separate exact-envelope diagnostic reproduced the lifecycle gate failure:
sender line141 → endpoint.command line63; readonly Docker CID/name checks
returned exit2 under1GiB virtual address space and exit0 with empty output under
8GiB. Source, original RGB/endpoint bytes and failed receipt rehashed unchanged.
The CID stderr matched a fixed out-of-memory classification, not arbitrary
exception text. This establishes a current Go/Docker virtual-memory defect;
historical stderr was not captured and is not asserted bit-identical.
See the three compact `vcoco_full_public_replica_export_v1*` audits under results.

Only the wrapper address-space cap and its assertion change; data, sender gates,
300s deadline, payload and scientific settings are untouched. A new immutable
producer/fresh namespace is required. This is a technical repair, not a scientific
retry, data subset or relabeling of FAIL. Import and full48 pose remain pending.
Repair controls:83 transfer/publication tests PASS0.76s with the repository's
explicit `infra:src` test import path; shell syntax and whitespace PASS.

Qualification:61 dedicated controls PASS0.57s; independent83 including publisher
controls PASS0.53s; root127 transfer/pose/core controls PASS1.03s. Original6837
Git archive independently reconstructs340 files/XZ301068B and the declared SHA,
one original executable leaf. Authored lifecycle uses actual BlockWriter/download
interfaces, all98-leaf installation, exclusive commit, sameFD failed DELETE and
owned race/partial cleanup. One root test harness initially inherited macOS
gid0 while the process gid was20; test-only fixture group binding was corrected,
not the production root-owned contract. All exact owned fixtures removed.

## Actual export v2

Envelope-only producer `5c67b6a3a89529b45badef7014c2ebe5de18347a` completed
export PASS in2.6953s. Independent saved audit verifies original/current full
Git sources,98 public leaves and all816 native array descriptors before/after;
no RGB/NPZ numerical replay or model load. Actual sealed receipt8489B/SHA256
`8247e76345d88c50b3099392fb92d2fa7de104b06246d7f5d3fa4a7ed9a8a92c`.
Declared archive62,228,480B/SHA256
`92d5107d143dafeaa7e55ae7cb67100e8ba6482b935aa5b60d1f70145ef6e432`,
manifest12573B/SHA256
`2ef141f4763d6bb44c4450c2a7ef2cea96fd707278c00cf12fc85f13a18eadb1`.
The audit does not independently read the Blob; archive SHA/ETag derive from the
saved producer's streaming hash and HEAD. Collected unit exit is explicitly
unavailable, not inferred. Receiver must verify the actual downloaded bytes.
Import, pose, selection and quality remain pending. V1 FAIL is preserved.

## Actual import v2

VM01 producer `5c67b6a3a89529b45badef7014c2ebe5de18347a` completed import
PASS in2.8559s. The receiver verified the62,228,480-byte archive's full SHA and
canonical99-member USTAR before installing all98 leaves. Independent saved
observer rehashed the98 installed leaves plus3 technical receipts before/after,
including48 banks/816 array descriptors/331 retained people; no numerical replay.
Actual import receipt8748B/SHA256
`6c32778305d227d863d561275de569abeb2e5b90836b5fafca40e21f89890f37`.
Original exact ETag DELETE202/one attempt derives from sealed producer evidence,
not an observer Blob request. Archive/stage/producer absence verified; collected
unit exit unavailable. No media/checkpoints crossed the local connection. Full48
CPU pose is separately dispatched against this pinned receipt, not yet verified.
