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

One300s inclusive attempt per export/import,315s shell outer cap,1GiB host virtual
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

Qualification:61 dedicated controls PASS0.57s; independent83 including publisher
controls PASS0.53s; root127 transfer/pose/core controls PASS1.03s. Original6837
Git archive independently reconstructs340 files/XZ301068B and the declared SHA,
one original executable leaf. Authored lifecycle uses actual BlockWriter/download
interfaces, all98-leaf installation, exclusive commit, sameFD failed DELETE and
owned race/partial cleanup. One root test harness initially inherited macOS
gid0 while the process gid was20; test-only fixture group binding was corrected,
not the production root-owned contract. All exact owned fixtures removed.
