# Positive V-COCO role-pair retrieval — pure numerical contract

`evaluate_vcoco_pair_retrieval` consumes caller-frozen original **pixel** Nx4
person/Mx4 object boxes, full NxM scores/support, unique source IDs, a genuine
`VcocoRoleReference`, **all** original same-image COCO instances and the
authenticated image ID/grid. `reference.rows` is not exhaustive context; the
separate original instance list is mandatory. This API authenticates structure
and numerical consistency, not dataset bytes/rights/freshness or bank completeness.

Continuous IoU threshold is fixed at0.5, without+1, clipping, coordinate repair,
per-image alignment or GT-driven candidate selection. Every category/crowd/
unlabelled context instance participates in matching before censoring; an
automatic endpoint must match uniquely and then be a noncrowd positive-area
person/nonperson instance. Multiple matches remain unknown, even coincident
category aliases. Zero-area/off-grid automatic boxes stay in the bank.

The primary ceiling is retrieved **unique localized positive agent/object ID
pairs** divided by all such pairs for that image; repeated actions/roles do not
multiply weight. Separate endpoint ceilings concern those positive pair endpoints.
Supported-pair ceiling additionally requires at least one supported automatic
pair. Missing role0, unscorable/person/self/crowd roles remain diagnostic counts,
not invented localized positives or verified negative labels. Images with no
localized pairs return an explicit unscorable status/None recalls, not a success.

Supported scores must be finite; unsupported scores must be NaN with an explicit
false mask. Scoring uses all supported unique coordinate pairs. Exact automatic
box aliases require identical full scores AND support; otherwise fail instead
of averaging/max repair. Numerical aliases count once, not physical identities.
Maximum-score exact ties receive fractional known-positive credit. Zero proposals
or no supported pairs returns0 positive-not-retrieved, never OFF/no-interaction.

Inputs are copied and fingerprint-checked, output arrays/scopes are immutable.
No model, threshold fitting, selector, reference file access or actual dataset is
performed here. Manufactured tests establish implementation behavior only.
This is not official V-COCO AP, exhaustive precision, contact, anatomical hand
ownership, task-target truth or a leaderboard gain. Actual DEV/RES opening still
requires the parent's separately frozen provenance/availability/bank/DEV gates.

Parent review and196 combined tiny tests passed in1.45s, including38 dedicated
controls, parser and existing retrieval/input contracts. Source12116 bytes/SHA
`13aa086cd2a7a9ac68d3b1485fd5c5945a0dc33d6180802a0f344e4bc362f026`.
No actual reference semantics, bank or metric was evaluated by these checks.

Independent read-only review atad6f491 found no concrete blocker;79 relevant
manufactured tests passed. Complete original instance context and pre-frozen
support remain caller obligations. This still does not validate an actor,
contact, anatomical owner or challenge target on real data.
