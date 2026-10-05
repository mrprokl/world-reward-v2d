# Visual Genome metadata acquisition V1

Prospective source freeze,2026-10-05. No archive acquired by this implementation
locally and no Azure execution yet. Parent owns commit and the single Azure run.

Scope: original-author UW `image_data.json.zip`, `objects.json.zip` and
`relationships.json.zip`; pinned `about.html` and release `api.html` retained.
HEAD-only observations independently confirm200, ZIP `application/zip`, text
`text/html; charset=UTF-8` and exact sizes1,780,854/55,323,929/77,904,473B plus
8984/8197B. These sizes are **not publisher cryptographic archive pins**. Original
archive SHA is measured on the first read, explicitly not called publisher SHA.
The catalog dates objects/relationships v1.4 and image metadata v1.2; there is
no inferred cross-version join guarantee.

The primary data readme names single members `image_data.json`, `objects.json`
and `relationships.json`. Actual central-directory names, expanded lengths
and contents are **unverified before the run**. V1 fixes these exact names and
fails closed on any additional/directory/special/encrypted member. No filename
adaptation after a frozen failure. STORED/DEFLATE only; original CRC and SHA of
the expanded byte stream checked without JSON parsing or disk extraction.

Inclusive600s,2workers,15s/request,TLS with no redirects/proxy/auth, one attempt
per asset,140MiB received total. Each member≤1GiB, aggregate≤2GiB reserved
**before** decompression. Host stdlib only, no Docker/GPU/models;16GiB virtual
memory cap, nice/idle I/O priority, outer615s with bounded termination. Native
SIGALRM and per-chunk checks close timeout without resampling or URL fallback.

Fresh `/srv/world-reward-data/visual_genome_metadata_v1` only. Raw archives and
notices retained0400 under0500; partial writes removed only when the opened
inode is still owned. CRC failure preserves the sealed original ZIP as evidence.
Original full runtime source/markers/helpers plus byte/mode/owner closure are
checked before/after. Report publication seals/fsyncs before declaring completion;
late deadline or fsync failure demotes via the already-owned report FD, not a
replacement file. Tiny stdout contains status/phase/budget only.

Preflight: root142 manufactured tests PASS0.72s, AST/bash-n; independent
read-only audit found a provisional-PASS interruption boundary. Root repaired
it before any execution, explicitly demoting every exception; an injected
interruption immediately after the PASS update tests the retained FAIL receipt.
Repaired root143 tests PASS0.74s and independent re-review confirms that
boundary closed. These are manufactured mechanics, not an actual acquisition.

PASS means metadata bytes and mechanical ZIP safety qualified. No annotation
rows, old references, rights-per-photo, predicate frequencies, cohort/split,
RGB, training, accuracy or challenge-overlap claim. CC-BY4 is the explicit
publisher metadata grant, not certification of underlying photographs. A
separate metadata census must exclude all432 historical slots and authenticate
creator/photo/known-MD5 identities before any fresh selection or RGB requests.
See `ownership_reference_next_audit.md` for primary text hashes and rights limits.

## Actual frozen acquisition and independent saved-only audit

Producer `caf8e6309636d46986f70b1be58c31f00e9dd145` completed PASS28.405586338s,
135026437B received on Azure. Report4623B SHA256
`b1c5d6a9e2212ca90751edd568f104e7f4d388d09604a36b7290b0fe82925717`.
Exact one-member names qualify; expanded17,612,822/349,437,266/743,673,397B
CRC-checked streams stay unparsed and are not written as expanded files.

Independent saved-only audit PASS0.347063498s authenticates full original
Git288files/293entries, XZ and modes/owners before/after, all five saved artifacts
and sealed output, ZIP central-directory identities and terminal unit exit0.
It **does not repeat decompression**: stream CRC/SHA is the producer's proof;
independent catalog and original compressed hashes bind it to the frozen bytes.
Audit3287B SHA256
`b54d27473fd2a214200ae58d3df8dbf3f40570b26f7123778ffd067c7e6f1dd9`
in `results/audits/visual_genome_metadata_v1_actual.json`. No RGB, annotation
values, old references, model or GPU was read. Metadata acquisition only is
qualified; census, photo rights, overlap and predictive quality remain separate.
