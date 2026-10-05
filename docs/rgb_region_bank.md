# RGB region proposals, frozen before either new cohort

The closed generic-text Open Images study stays closed. The new reusable
`rgb_region_bank` entry is a region-only SAM2.1 producer, not an interaction
selector. It uses the already measured frontend image/checkpoint and original
AMG32×32/.8/.95/NMS.7/no-crop defaults. Every returned mask, bbox, quality,
native position and duplicate is retained. Empty banks remain empty; resource
caps abort the experiment rather than truncate or rescue an image. Raw predicted
IoU is a model output, not a calibrated probability; the adapter does not clip
finite values into[0,1]. No new checkpoint or prompt vocabulary is introduced.

Host checks the full immutable Git dispatch closure, original runtime ancestry,
and independently pinned public manifest/RGB before inference and afterward.
Native mounts **individual whitelisted files**, not the current code directory
or all configs: renderer, scene recipe, reserved split and reference annotations
are absent. Native rehashes those files and the installed SAM2 source/extension.
It sees only opaque image IDs, files, byte hashes and original dimensions. One
model load, one AMG call per frozen RGB; all data and banks remain on Azure.

Two prospectively separate validation namespaces are allowed: new author-owned
procedural stress, and new legally acquired external RGB. Neither reopens the
closed OI/HO-Cap studies. Private engine masks can measure procedural mask recall;
published external boxes can measure **bbox proposal recall**, not mask IoU or
ownership. Freeze all banks before evaluator reference access. Missed or
unavailable fixed slots count, not disappear. Exact metric/adoption gates belong
to each protocol, never the producer. Synthetic success does not establish real
transfer; new OI photo/author exclusion does not establish unseen pretraining.

Initial local verification:79 tiny public-contract/native-callback tests PASS.
These contain generated toy RGB/masks only: no actual native inference, proposal
quality, actor/object identity or leaderboard gain is claimed by this note.
