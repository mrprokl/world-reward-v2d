# Marginal pair objective — numerical primitive, not a trained selector

`coherent_pair_marginal_objective` consumes full, source-bound marginal score
grids and unchanged published-positive masks. It neither selects examples nor
reads reference files. The caller owns reference eligibility, fixed population
and, for Torch, independent device-buffer snapshots before and after execution.

For each record with supported positives, the data term is
`logsumexp(all supported groups) - logsumexp(supported positive groups)`.
This positive-set surrogate rewards finding **a** published positive pair, not
every interaction. Unannotated alternatives compete, but are not certified
negative examples, OFF states, task targets, contacts or anatomical owners.
The caller must not describe a decrease as reconstructed-interaction accuracy.

Average over the original **fixed N**, never over successful records alone.
No-positive or entirely unsupported-positive records have an explicitly
undefined per-record term (`NaN`), zero data contribution and reported counts.
Partially supported positives remain reported; all-supported-positive grids
have zero data term. `partial`/`no_informative` are not qualified fitting results.
Apply L2 once to the active vector: geometry17 in arm A, scalar nonnegative
alpha in arm B with geometry frozen. Temperature and scoring parameters must
be identical across records. No weights, optimizer or fitting schedule are
chosen by this primitive.

The analytic VJP uses all-group minus positive-group expectations. Center both
expectations on the **same derivative row** to cancel large common derivatives
without subtracting two independently rounded large means. Supported scores
must be finite, unsupported scores raw NaN; overflow fails closed. Stable
logsumexp separates maximum differences from log-normalizers. Exact native IDs,
complete alias memberships and scorer parameter fingerprints are required;
coordinate aliases are not asserted to be physical identities.

CPU outputs are immutable. Torch is imported only on an explicit device call,
uses detached FP64 tensors and no autograd or in-place source writes. CPU/fake
Torch manufactured controls include missing positives, fixed-N regularization,
uniform temperature, common offsets and constant derivatives of magnitude
1e300. Independent source review found and verified the two numerical fixes.
Root317 combined tiny tests passed in1.86s, including51 objective tests.
This is **not** a native Torch objective, optimizer, FIT, held-out retrieval or
CARI4D victory qualification. The previously measured H100 full-bank scorer
remains a separate result; its source and receipts are unchanged.
