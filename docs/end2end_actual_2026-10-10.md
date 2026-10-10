# End-to-end actual results — 2026-10-10

## Frozen real-video ablation

Baseline `052ba1554e9a573d566713a99a61d89a5f27681c` preserved.
Candidate `ebcee73cc76e036b3c2ec9c32e9a72abc951ad78`: same raw native bundle,
300 steps /301 updates, full original frames, automatic RGB evidence, no GT.
Random cohort seed20261008: [9,1,14,7]; 1/7 remain unsupported in this frozen ablation (midclip scale mask failure),
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

## External evaluation state

Four frozen, unmatched FORM DEV clips; all first96 contiguous original frames.
FORM reconstructed multiview pseudo-GT is quarantined on VM02. All4×A/B and
external sampling must seal before any truth read; reserved4 remain unopened.
Exact-ID and all30 Track1 RGB alias guards pass, but near/reencoded aliases and
pretrained-checkpoint independence are NOT proven.

DEV acquisition225097082dad29ebbbacb82daa09c71a3ea83bdb completed167.933s;
7322B SHA90694c8d032f4b3b8eedcd422304cb51510820087d6add4be10e4e20671055dd.
Public RGB/text only transported directly Azure02→01; no laptop media.
All4 localizations806300da64a238c889f76bfeb4d0992d85c8a654 complete;
9645B SHA2f04c671fb0623f01bbb11f432323f3e4215683548835cc71762a73cfafdadbf.

First FORM clip125aab2: SAM42.528s, body/depth94.543s, Objects45.973s sealed;
prepare stopped with verified owned-container absence because full743576-face
raster costs were excessive. Byte-exact successful-prefix reuse implemented;
failed preparation never reused. Eight final4D predictions remain incomplete:
**no external reference metrics yet, no verified victory**.

CPU02 isolated Numba runtimef4d379bd8c7cf0ef23364b7f2e3b1bce215ceea6:
actual55.228sPASS, untouched public metric operator matched NumPy, reversed/
degenerate topology, analytic geometry, and all96 frames of shared Sim3 CD/ACC/PEN
qualification. This is numerical evaluator parity, NOT reconstructed-data accuracy.
Childsha256:d24051da178c12ce3f3f3193a5e0c8ca90e1e7a1c765077ec3e22fdfc3752004;
report10889B SHAdba8aace26df84ff250c67a083ed631241803bfbd27764164d96a74fc595a68e.

Full RGB/052/B previews published privately: EP9=415frames151200B;
EP14=442frames91777B. Both QA failures visible, 1/7 scale-mask failures retained.
Publication07944fe633f1b523529f24231b5347bec951e884 receipt3124B
SHA865c5b4e92f3cb8adfaacb828a6afc8c33e1ed981e4481145c241fde314a58d3.
On-demand memory-only viewer; no autoplay/preloading/download to laptop.

## Contact feasibility: completed alternatives

Translation-only nearest/original multibranch candidates841094d/4e24cf2 failed
on full EP9/frame99 and EP14/frame231: a common relative translation cannot
repair altered bimanual separation. No candidate geometry was fabricated.

Native articulation continuationc25f0df completed **96.371s**, all frozen records
retained: EP9=27.968s, EP14=64.127s. Nine whole-clip dyadic proposals each rejected
by placement/contact checks; both returned **full moving A, no improvement**.
Replayed native A reference uses the exact original IDs/activations/parameters;
saved A and original comparison gates unchanged. Native-vs-saved witness gap
roundoff max.463µm/.119µm was sealed before proposals, no tolerance relaxation.
A scalar direction can be infeasible at one contact at every positive step;
next distinct hypothesis is native articulated feasible-start SQP (separate doc),
not another weight/interpolation sweep. No actual SQP gain yet.

Private RGB/052/B videos remain available as diagnostics. The C viewer adapter
now separates136/shape/scale controls from the strict seven-field display ABI;
no geometry repair or checker relaxation. C rendering80eef93 completed, full415/442 original frames, with
fallback explicitly labeled no gain. Publicationf03dcf24 completed: receipt3366B
SHAe70c7bc3669772fa5924e3da12a5d53760b296bc4a93601f0c1d64bfeb6ba150.
EP9/14 MP4=151736/91818B. Playback only on demand; no laptop media.
The historical unsupported-reason text in this sealed receipt is stale; the
actual cause is the scale-mask failure documented below, not MHR absence.

### Exact prefix raster runtime — actual PASS

Native fine-kernel empty-prefix termination qualifiede27604c:96.477s including
isolated four-unit compilation. No global package/binary overwrite, no Pulsar
registration, no change in valid-face processing, resolution, geometry or models.
Procedural and actual743576-face object B1/B4: exact masks AND depth bits.
Actual B1 AABB1.409663→.545676s (2.58×); B4 5.817943→2.153679s (2.70×).
Against original full-capacity8.536546s, actual B4~3.96×; single-trial implementation
speed only, no reconstruction/leaderboard gain. Full4800-raster pose search is
still substantial; native contact cost is measured separately below, not inferred from complexity.

Runtime report4810B SHAacf983bc2fcc545542dcd06446fa3e5a3f856b017af0591bf4ad43a7b9be77da;
qualification5420B SHA5a2e84f576756f9e79c85b030b475089c7dc768cd0a591a99d648f847aa18bd0.
CPU-only export822B SHA4d1a7e0f60cd7811e227dfdb957820d5cedace5d01563bf0163b25b8bda5b01a;
binary878448B SHAf80eca190ea59532876cd4bea416e505d2600ba7eca3c72fbf89d5e3eb7329f7.
Explicit activation keeps original native base image/backward/other operators.

### Actual native contact backend qualification

Six actual full2318-hand ×743576-face CUDA probes0319c8c finished7.327s;
14195B SHA660803625e7edf9fd66dadca7f981b7f3762ccb022fbef202ffffb8917896b8a.
All original faces below native.005area threshold. All six selected-pair
values and Adam point updates are byte-identical; five of six human gradients
are byte-identical (six satisfy the declared numerical threshold),
but native face IDs differ and CPU witnesses tie: **replacement rejected**.
Native exhaustive reference costs about.0233s/hand; CPU BVH about.023–.192s/query.
The previous complexity-based assumption that CUDA contact must dominate was
wrong for these probes. Keep the unchanged native CUDA operator; no more backend
engineering until actual fit timing identifies a bottleneck.

FORM resumed producer80eef93 uses exact successful-stage reuse, qualified raster
prefix/capacity and frame-local byte-identical mesh dedup. Full25hypotheses and
full96 timeline preserved; actual duplicate gain remains unmeasured. Native fit
now observes three complete real updates and stops before update4 if projected
inclusive2400s/VRAM80% gates fail; native run/loss/objective/update count unchanged.
Scheduled PEN timing is rechecked after its first three actual updates.
The actual all-four producer80eef93 was dispatched and remains active in first
clip preparation at2026-10-10T15:04Z; no sealed prepare/fit result yet.

Sparse SQP core on two independent authored seeds reduces its known synthetic
objective95.22%/94.69% with unchanged witness bounds and root-motion increments.
This is NOT native MHR or real HOI accuracy. Native callback/launcher integration
and actual all-frame EP9/14 test remain required before adoption.

### Coverage correction

Original052 EP1/7 failures were **scale_smoke midpoint object mask empty** (244/
128 empty masks), not MHR inference failure. Human-visible-only scale anchoring
already handles the cause. Further full-T occlusion initializer/consumer routing
must preserve missing evidence and rerun native pose fitting; no copied/static
edge predictions or replacement clips. Actual recovery is not completed yet.

Official challenge reread2026-10-10: monocular RGB only, full metric shared-frame
HOI; website describes accuracy/physics axes50/50, while official FAQ says
aggregate sum-of-metric-ranks (lowest wins). Keep all five raw metrics and flag
this scoring ambiguity before a ranking claim.
The screenshot is sorted CD-H, not proof of aggregate first place. Smaller
self-motion acceleration alone does not establish acceleration error improvement.
