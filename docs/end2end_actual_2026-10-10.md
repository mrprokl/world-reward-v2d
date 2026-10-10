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
reference score. Full three-column Azure previews completed exit0, EP9=415frames151200B;
EP14=442frames91777B, both QA failures shown. Publication dispatched; playback
awaits sealed tiny receipt. DEVv2 failed due to absent host ffprobe; decoder
installed remotely, fresh v3 producer2250970 dispatched. No truth values decoded.
CPU02 existing image imports NumPy/Torch/Trimesh/SciPy; optional Numba absent,
not yet a qualified full evaluator execution.

### Acquired actual four-DEV PASS

Producer225097082dad29ebbbacb82daa09c71a3ea83bdb,167.933s completeexit0:
all4 native30fps1536x1152, lengths764/616/396/409, eligiblefirst96exact
aliasguardPASS (notproofnearaliases). No GTarrays/masks/calibration decoded.
Report7322B SHA25690694c8d032f4b3b8eedcd422304cb51510820087d6add4be10e4e20671055dd.
Rawotherarchivesremoved; firstretainedauthenticatedarchive reuseno1GBrefetch.
DirectpublicRGB/textAzure02→01 transport nowdispatched, no laptop media.
Publishedfulltimelinepreviewreceipt3124B
SHA256865c5b4e92f3cb8adfaacb828a6afc8c33e1ed981e4481145c241fde314a58d3.
Userloopbackondemandvieweropened49933 TTL3600, noautoplay/preloadmedia.

### Direct Azure public transport

VM02 publish1c9ffee completeexit0, publicmanifest4465B
SHA2563151519bc11e602060c92bfad3750c68c824f155d8934ad8bb98a4ba8d479459.
VM01fetch0077f73 dispatched. Remote source references never transferred.
Externalpredictionproducer806300d (493frozenruntimefiles), own per-sequence
one-shotVertexauth preparedinRAM pipeline; pendingactualcohortresults.

## Next structural hypothesis (not tested/adopted)

Nativecontact gradient reachesnewhumantranslation; it is notmissing. The
originalfactor trades 200×squared nearest distance (all2318handvertices,
initialnetworkgate) against normalized RGB/pose priors. TrainCOCO ignores
wrists, hands/rotationsfixed; rootRGB canbuyseparation. Samewitness gap is not
wholehandminimum: recordboth beforeclaiming allcontactworse. Native diagnostic
contact_distance_mean includes inactivehands; only activehandstatisticcomparable.

C, after FORM A/B: image optimization subject to contact-feasibility rather
than strongerarbitraryweight. Automaticcurrentanatomicalpatch normalconstraints
with tangent sliding/releasefree; minimumchange proximalprojection, not
sharedcoordinate reparameterization (equivalentobjective), nofreezes/scalechanges.
SameA/Bhands+rotations remainfixed toisolatethecause. Infeasible/ill-conditioned
constraints mustabstain explicitly. This is ouruntestedproposal, inspiredby
https://arxiv.org/html/2605.20992v4#S4.SS3 and https://arxiv.org/abs/2012.09856 ,
not a reproduction or evidence of generalization. Gates: trueactive wholehand
continuoussurface and unchangedwitness, held-outRGBproxy, PEN, actualreference
CD/ACC and unchangedfulltimelines. Complete existing externalpredictions first.

### Localization technical retry

First806300d hostdriver failedbeforecohort/models/API: hostNumPyabsent.
Installed host control-only numerical packages (NumPy1.21.5/SciPy1.8.0),
no host model fallback. Same immutable source/input/credential namespace reused
in a new unitname since no prediction output/credential consumption existed.
Wrapperfuturepreflight fixed. Originalfailedunit/log preserved; not response
shopping, hyperparameter selection, or scientific rerun.
