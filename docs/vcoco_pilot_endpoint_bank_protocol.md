# V-COCO16 native endpoint banks — prospective

Code-only caller for the separately frozen/acquired pilot. Root must authenticate
actual sealed acquisition and availability ≥6DEV/≥6RES before dispatch; this
caller receives only public manifest SHA/bytes and acquired count1…16. It cannot
consult private selection, roles, reference files, split or attribution.

Public source is `/srv/world-reward-data/vcoco_role_pilot_v1/inputs`: original
six-key acquired-only records with holes retained. `rgb_bank_inputs` validates
the exact directory, original JPEG bytes/grid and slots0…15. Native decoding
does not crop, resize, rotate EXIF or convert an unsupported color format.
Acquired ordinal, source slot and opaque image identity remain separate; every
static photo has original frame index0. Missing slots are not inferred/resampled.

Unchanged `rgb_endpoint_bank` helpers authenticate original GDI/OWL runtime,
installed RECORD/source/assets and full strict checkpoint state, load two models
once, generate every900 GDI native query and every3600 OWL patch, and losslessly
publish all17 arrays. The original `person.` query/thresholds/NMS are unchanged;
all resulting persons remain, including zero-person outputs. OWL performs no
objectness threshold, topK, NMS or prompt. No DWPose, HOI, ranking or selector is
loaded in this pass. Model math, source configs and closed callers are unchanged.

Exact image `sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252`
is required, not B47/7eb. Original OWL procedural source/model/runtime receipts
remain qualification evidence only, never independent real transfer accuracy.
Native process is offline `env-i`, Python`-I -B`, four CPUs/8GiB/one cooperative
H100 lock, readonly individually pinned source/assets/JPEGs; no private data or
whole repository mount. Budget600 seconds inclusive,20 seconds native cleanup
reserve,615 outer cleanup. Inputs/banks≤16MiB each,16M original pixels,
256MiB total; same-FD final report demotes any late PASS.

Host authenticates full sources/modes/input/runtime/model before and after even
failure, validates complete native counters and original slot mappings, proves
owned CID/name removal and seals artifacts500/400. Native arrays retain raw
masked `-inf` GDI padding exactly; all other arrays have their original finite
shape/dtype and full bank IDs. Empty/missing are not OFF or contact evidence.

Bank generation alone is neither fit, ownership evaluation nor a verified win.
All banks must be sealed and independently audited before any DEV reference
semantics are opened; RESERVED remains unopened until a separately frozen DEV
gate. Pretraining includes COCO/OpenImages/web scope; exact photo/challenge
overlap remains unknown, not leakage-free or unseen. Publisher-image grant is
the previously accepted private-pilot CC-BY2 metadata interpretation, not creator
authentication, checkpoint submission clearance or competition legal certainty.

## Root-owned configuration

`configs/vcoco_pilot_endpoint_bank_v1.json`: exact schema
`world_reward.vcoco_pilot_endpoint_bank.v1`, fixed paths/capacity16/image fd268,
600/615/20s,4 CPUs/8GiB host+container, original person/model policy,256MiB
input/output totals,16MiB per image/bank and16M pixels. `original_recipe` is the
unchanged `configs/rgb_endpoint_bank_v2.json` identity; `reused_helper_pins`
contains every original `rgb_endpoint_bank.HELPERS` plus
`src/world_reward/rgb_bank_inputs.py`. No acquisition receipt/actual count is
invented in config: explicit manifest CLI pins/count come from separate audit.

## Pre-execution source qualification

Independent read-only review identified a cleanup gap: checking only the
container name after removal could miss the saved CID surviving under another
name. Both exact CID and name must now be absent; unexpected ownership is never
removed. Final driver28413 bytes/SHA
`0fa39ee74e8cc8021d13256a4e8fb918389a0be97f701d07f0567fda6595b74d`
has no remaining concrete blocking issue from that review. Root368 combined tiny
tests passed in23.40s, including44 dedicated controls, AST and shell syntax.
No actual pilot RGB, model, CUDA inference or positive-pair accuracy was tested.
