# Positive role-pair rank seam — prospective, source-only

`evaluate_vcoco_pair_rank_retrieval` reuses the unchanged validated full-bank
evaluator. References enter only this evaluation/positive-mask API, never the
predictor. The caller authenticates completeness, source, split and licensing.
No actual banks or role values were used for these controls.

The returned mask covers **every canonical unique person/object geometry group**
in lexicographic `np.unique` order, matching `PairRouteBank`. Original IDs remain
in `validation`; native-to-group maps and complete memberships are returned.
Matching is continuous pixel IoU>=.5, uniquely against **all** same-image instance
classes/crowds before category/validity censoring. The mask marks published
localized positive pairs only; a false entry is unknown, not a verified negative.
An unsupported but geometrically matched positive still has a true mask entry.

Recall@1/3/5 uses every unique localized native reference pair in the denominator,
including proposal/support misses. Exact-coordinate aliases have one rank slot;
different proposal geometries matching the same native reference pair may consume
several slots but contribute at most one positive hit. Duplicate action/role rows
never add reference mass. No-localized-positive images return explicit status and
`None` metrics: a cohort caller must not silently omit them.

Scores above a cutoff tie are all retrieved. From a tie of n candidate pairs,
b remaining rank slots are sampled uniformly without replacement. For an unseen
reference pair represented m times in that tie, its hit probability is
`1-C(n-m,b)/C(n,b)`, evaluated with at most five bounded factors. Already-retrieved
reference pairs have probability1. Unknown candidates occupy slots normally.
This is an **expected metric over tie selections**, not an invented deterministic
ID priority or a stored selected prediction. Any-positive@1 equals the original
top-tie positive fraction exactly. Rank budgets are metric views, never pruning
of the retained raw bank.

These are general published role associations, not physical ownership, contact,
anatomical-side truth, official V-COCO AP or a reconstruction-quality claim.

Qualification:39 dedicated and118 related role/retrieval controls PASS0.23s;
independent77 controls PASS0.21s plus6,561 exhaustive score/support/rank subset
checks PASS. Root77 controls PASS0.26s. The original12116-byte evaluator SHA256
`13aa086cd2a7a9ac68d3b1485fd5c5945a0dc33d6180802a0f344e4bc362f026`
remains byte-identical. All exact owned fixture trees removed; no real references
or predictions inspected. This qualifies arithmetic, not selector performance.
