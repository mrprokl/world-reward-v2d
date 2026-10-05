# Data-free JSON parser cost control v1

Prospective, before any VG semantic census. One Azure VM02 stdlib CPU attempt,
600s inclusive, outer615s,16GiB virtual-memory cap, no GPU/network/model/media,
dataset values, selection or expanded files. Fresh owned namespace
`/srv/world-reward-data/metadata_json_cost_probe_v1`; original failures remain
closed. No retry, alternate fixture, parser warm-up or performance tuning.

Authenticate full immutable caller source/markers/modes and original parser
8322B/SHA742b10d5f92e3446d17d4953f1d088281f051577653899ddcc7081f61545bf53.
Use the unchanged stdlib parser in a single process on three generated streams,
each exactly64MiB, in order:

1. 256B-stride scalar-dense rows, **all excluded**.
2. 4096B-stride rows,15/16 excluded,1/16 strictly decoded.
3. 65536B-stride long-string rows, **all consulted**.

Stride includes comma; JSON object is stride−1 bytes. For total B and stride S,
rows=floor((B−1)/S); array contributes rows×S+1 bytes, completed by whitespace.
IDs are1000000000+row index; first/last top-level ID order alternates. Strings
contain authored UTF-8 and escaped controls. Excluded semantic rows contain
`OLD_POISON` and `1e999` and never receive whole-row semantic decoding. Exact
generated SHA is calculated before each timed stream using generation/hash only,
not a parser pass. Timed parsing includes streaming generation and observed SHA.
All counts, IDs, consulted fields and byte hashes must agree. Tiny malformed
JSON/depth/duplicate/nonfinite/UTF-8 and corrupt-CRC/truncated DEFLATE ZIP controls follow
timed cases, never warm them.

Predeclared throughput gate:
`max(case_seconds/67108864) × 1130711325 + 120 ≤ 600`.
All three cases, negative controls, source/mode posthash and sealed publication
must complete within the same600s. A late signal/fsync/deadline cannot retain PASS.
Save one400 JSON receipt under500 directory; no generated stream files.
Peak RSS is observed with stdlib `resource` (Linux KiB×1024), bounded16GiB.
The zero excluded whole-row semantic-decode count describes the structural
branch, not an instrumented profiler; tiny tests guard its callback path.

PASS is **data-free engineering cost feasibility**, not a proven upper bound on
real schema, number distribution, machine load or real-census completion. It is
not image fidelity, ownership, model quality, dataset eligibility or adoption.
Failure closes this control; any method/budget change needs a new frozen protocol
before actual reference rows—not continuation/retry of this namespace.
