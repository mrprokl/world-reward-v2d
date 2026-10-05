# Independent V-COCO census source audit — 2026-10-06

**Source readiness only; no Azure/data/census/quality result.** The current
configuration validates exactly: 23 pinned inputs, six unchanged imported
helpers, 1200 s inclusive/1215 s cleanup, 8 GiB. No source edits by this auditor.

- Historical 432 photo identities are collected from metadata-only ledgers,
  including unavailable slots, before fresh annotation/action semantics. Embedded
  historical reference paths are not opened; known authors/MD5 are exclusions,
  not proof of independence from UNKNOWN new creators/bytes.
- First COCO2014 passes consult image/license/category catalogues and skip all
  annotations. Second passes decode only rights-eligible nonhistorical official
  VAL/TEST image rows, independent of `image_id` field order. Original annotation
  IDs, XYWH, area and crowd flags remain unchanged. Complete member SHA/CRC
  evidence binds both passes to the acquired original archive.
- The action-major projector authenticates each role-file open, freezes eligible
  IDs and validates the official split's structural image union before pass two.
  Excluded label/role scalars are lexically checked but not decoded. Selected
  flat roles remain role-major with original action/row slots. No missing-role0
  OFF, hand ownership or contact truth is manufactured.
- Capacity is narrowly ≥8 distinct photos in each official VAL/TEST partition,
  with ≥2 noncrowd positive-area persons, ≥2 nonperson objects and ≥1 localized
  positive role pair. Multiple positive agents are diagnostic, not an extra gate.
  No eligible-ID list/cohort/geometry values leave the census report.
- Alarm is installed before source/input authentication. Publication verifies
  original directory identity/UID/GID/mode, empty700→500 and the owned report400
  inode; late errors demote the same open FD. Preflight may legitimately fail
  before any output namespace exists; that is not a successful census.

Independent manufactured controls: three RAM-only old-row poison/duplicate/EOF
controls (≤154 bytes), plus one late-publication same-FD control in a unique
automatically cleaned temporary fixture, all PASS. Four earlier role projector
RAM controls (≤218 bytes) verified field-order safety, role-major slots, complete
byte changes and structural duplicates/truncation. No real annotation values,
HTTP, RGB, Torch, GPU or native replay were used.

No concrete remaining source blocker was found in this scope. Actual runtime,
capacity, photo access, endpoint recall, pair retrieval, ownership and model
adoption remain unqualified. Repository-wide MIT private research interpretation
and COCO photo rights are separate; pretraining/challenge overlap remains UNKNOWN.
