# Fresh role-reference seams — source/tiny controls only

2026-10-06. V-COCO metadata acquisition is qualified; **no semantic census,
photo acquisition, FIT, selector adoption or measured accuracy** follows yet.

V-COCO is action-major. Its flat role IDs use `role*N+row`, unlike COCO's
image-major annotation rows. The new two-pass `vcoco_role_stream` consults
only action/role names and structural agent/image IDs on pass one. On pass two
it decodes labels/role IDs for a frozen eligible-image set only, even when
semantic fields precede structural fields. Original action/row indices remain
explicit. Full lexical UTF8/syntax/duplicate-key/EOF checks and whole-byte hashes
cover both passes; excluded semantic numbers are not interpreted.

The caller must authenticate the **same immutable source before each open**.
Final post-hash rejection is not permission to read an unauthenticated changed
source: field order could otherwise consult a changed row before rejection.
Byte-limit arguments fail before source open; eligible IDs are snapshotted.
The filtered COCO annotation iterator decodes only fresh eligible images and
must be exhausted for full EOF/ZIP CRC, including excluded rows.

`vcoco_role_reference` accepts already projected fresh records only. It joins
original COCO2014 annotation/image IDs, retains every action/role row and raw
XYWH without clipping, and counts a native person/object pair once. Positive
dangling/cross-image endpoints fail integrity; zero is UNKNOWN, never OFF.
Self/person/crowd/invalid geometry remains explicitly unscorable for the
nonperson-pair view. Label0/unlisted pairs are not exhaustive relation negatives.
The parser content fingerprint is not provenance, rights or leakage proof.

Root149 combined tiny tests PASS0.12s (stream/reference/lexer); a larger
pre-filter suite256 PASS0.33s. Independent parser review finds no concrete
semantic blocker; four independent RAM poison/order/change/truncation controls
pass for the action projector under its immutable-source contract. No real
dataset values or local models were read. These qualify metadata mechanics,
not anatomical ownership, manipulation, temporal identity or official role AP.

Next census, separately frozen before values: exclude all432 historical slots,
join original per-photo grant metadata, count crowded localized role positives
in official val/test. Minimum8 distinct photos per split is a capacity gate,
not a selected cohort or statistically established quality threshold. General
roles include supports/instruments; they do not identify the challenge task.
