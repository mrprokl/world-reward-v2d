# Fresh FIT32/CAL16 identity freeze — prospective, source-only

The independent saved census audit qualified a **count-only** receipt, not a
selected corpus or learned model: TRAIN195, VAL191,386 private identity rows,
448 historical/pilot photo exclusions. Original source is
`b9270d55a3ed10c28e080e019107cedb18e63fd1` (308files/313entries, closure
`eb0dfc966a2a8dba1b249358d5152a23e16b42ff58eab1d66112551b74aebf76`).
Report57524/SHA256
`703e0e341d3ceb8c44d40f5a90823299719ade1559c36945bc6f8d92120ffee0`;
inventory24991/SHA256
`ea7503e6e2e09cc6b4536ce12bfed9ad73698488ebc50b22db0fcfe8276fbd70`.
No real inventory values were read locally to implement this pure helper.

`freeze_fit_cal_cohort(report_raw, inventory_raw, *, excluded_photo_ids,
expected_input_proof)` verifies the exact saved byte pins **before JSON decode**,
then validates the complete source-bound count receipt, all386 identity-only
rows,195/191 counts, unique native IDs/photo IDs and all448 exclusions. The caller
must authenticate original source/config/23inputs/18assets and reconstruct the
432historical+16pilot photo set before/after. The supplied proof is compared via
canonical JSON round trip; this pure function cannot certify caller authenticity
or filesystem seals. Its output explicitly says `source_authenticated=False`.
Historical/pilot references remain hash-only and TEST role values stay unopened.

Freeze one new namespace, `world_reward.vcoco_fit_cal_v1/`. Separately sort TRAIN
and VAL by `(SHA256(namespace + native_image_id padded to12decimal digits),
native_image_id)` and take exactly first32TRAIN as FIT and first16VAL as CAL.
Slots0..31 are FIT; slots32..47 are CAL. No availability, annotation/action,
prediction, wrist support, HOI proposal, score or quality filter is consulted.
No seed sweep, alternate namespace, resampling, substitution or retry exists.
The private output records only slot, study/official split, native image/photo
IDs and deterministic rank digest, plus frozen provenance/recipe identities.
It contains no RGB, URL, boxes, labels, role endpoints or reference files.
This metadata is **never** mounted into any model/predictor: a later caller
projects only opaque six-key RGB inputs, with split/GT/photo IDs hidden.

**All48 acquired original photos are required** to qualify a full FIT/CAL corpus.
If even one slot fails, preserve48 denominators and each miss and close without
smaller training, availability-selected subsets or new seed. Acquisition is a
separate future source-frozen stage; no HTTP, image bytes or job exists here.
All complete blind endpoint/DWPose/HOI banks must freeze before FIT references
are read. Retain all people, both sides, all3600 generic objects and native HOI
pairs; zero-HOI banks do not erase base proposals. Missing anatomy stays missing.

The scientific recipe is a separate future single prospective configuration,
not numbers adopted from cost fixtures. FIT alone supplies scales/parameters;
CAL is evaluated once after the trained artifacts/config are frozen. CAL is not
hyperparameter tuning or an independent final test, and the earlier pilot's
reserved reference values remain unopened. This48-photo freeze neither proves
17-parameter identifiability nor effective sample size, convergence or quality.

V-COCO positives are published general person–object roles, not anatomical hand
ownership, manipulated task targets, contact, complete negatives/OFF, temporal
identity or shared3D truth. Publisher CC-BY2 metadata is not independently
verified creator provenance; creator identities and checkpoint COCO/V-COCO
exposure remain unknown. Photo-disjointness is not author-/training-disjointness.
No official AP, challenge eligibility, adoption or victory over CARI4D is claimed.
Manufactured tiny tests qualify only deterministic projection and failure gates.

Source qualification: author86 related controls PASS0.33s, root127 PASS0.46s
and independent39 dedicated PASS0.06s. All owned fixtures were removed. Final
7780-byte driver SHA is
`a1300cabfa410c2d1b74723cb073a59664f6f0462d68260db890ce91cb55ab1e`.
No actual selected identity corpus, acquisition or learned model follows from
this release; a separately source-bound Azure caller must execute the freeze.
