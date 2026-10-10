# End-to-end actual results — 2026-10-10

## Frozen real-video ablation

Baseline `052ba1554e9a573d566713a99a61d89a5f27681c` preserved.
Candidate `ebcee73cc76e036b3c2ec9c32e9a72abc951ad78`: same raw native bundle,
300 steps /301 updates, full original frames, automatic RGB evidence, no GT.
Random cohort seed20261008: [9,1,14,7]; 1/7 remain upstream-pose unsupported,
never replaced or counted as reconstructed successes.

| Diagnostic (not reference accuracy) | Episode9 A→B | Episode14 A→B |
|---|---:|---:|
| object reprojection mean(px) |8.759245→6.813755|no material tracks; Q=0|
| reserved human reprojection(px) |3.888364→3.138316|2.210584→2.371458|
| human centroid acceleration p95(m/s²) |20.000171→11.493335|15.404654→8.792092|
| object acceleration p95(m/s²) |21.389428→22.053627|50.417902→50.368228|
| same anatomical contact mean(m) |0.005596→0.015923|0.000829→0.004523|
| contact p95(m) |0.016746→0.040796|0.003156→0.015741|

Both complete exit0, exact native decoded full geometry sealed. Neither meets
conservative contact/motion nonregression gates; neither adopted. Object rotation,
hands and human root rotation are fixed in this ablation: angular jitter cannot
be improved by this extension. Smaller human acceleration is not better ACC-H
against truth. No Kaggle/held-out victory claimed.

Actual report identities: EP9 163589B SHA256
`118276f5f81bf926b08906e2973ef74af213d634631b7cb283611a012db72200`;
EP14 171105B SHA256
`90d77958a2069922c9de025de414e762d0b2f11dd6d33390ac545bff4ea2e735`.

## Contact-memory alternative

Producer d8061c9f6f0a649e17e545af40e9da483fa59366, EP9 all415,
742.895s complete. RGB8.731688→3.786101px but same anatomical contact
0.007087→0.096114m. Not adopted. It changes associations but fixes the human,
so it cannot solve joint relative placement. This is an actual completed failed
hypothesis, not a claim that contacts are impossible.

## External evaluation execution

Four frozen FORM DEV, first96 original contiguous frames, RGB+native public
prompt/action only. References opaque in separate VM02 evaluator quarantine;
all4×A/B and external sampling sealed before any truth read. Reserved unopened.
Exact all30 Track1 RGB byte/first96-pixel alias bank frozen200330B, SHA256
`e43f5b5f3b2adcf14e8791ef86c2733e43e46f6368f3bf10ae881a3e72a8b5ed`.
Exact-ID/content checks do not prove near/reencoded alias absence or pretrained
checkpoint independence. FORM truth is reconstructed multiview pseudo-GT.

DEV extraction9b32d28 bootstrap failed before output allocation (HTTP403,
VM02 managed identity lacked private-container permission). Narrow container-scoped
Blob Data Contributor granted, with no public access/ACL/account-key changes;
retry uses a fresh producer, original archive retained. Acknowledged dispatches
are not completion. Prediction/evaluation code is tested, not yet an actual
reference score. Full three-column Azure preview rendering launched; publication
and playback remain pending actual completion.
