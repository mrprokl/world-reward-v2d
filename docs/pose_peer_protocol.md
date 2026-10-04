# Azure-private full-video pose inputs

This is transport and execution isolation, not a new prediction algorithm.
VM01 produces automatic Track1 RGB/masks/body/depth/scale/object initializers;
VM02 may run the exact unchanged full-video object tracker on those inputs.
Neither a complete transport nor a native execution PASS proves pose accuracy.

Before inventory, independently freeze original producer revision/entrypoint,
six producer helper hashes, nine tracker helpers and six actual report hashes
in `pose_peer_EP_source_pins.json`. Require full official frame coverage,
automatic prompts, identical video/metric gauge and all native no-oracle flags.
Missing scale or any original depth/mask frame fails before transport; EP7's
closed empty-mask failure is not rescued by moving it to another VM.

Inventory is CPU-only on VM01. Its exact allowlist contains one Track1 RGB,
public episode metadata/input manifest, all person/object masks, every full-depth
NPZ, body **report only**, object GLB/transform/intrinsics and scale report, plus
the genuine2KB official mesh-budget helper. No Body predictions, checkpoints,
SAM3D Objects image/source/weights, other videos, Track2/3, GT or calibration.
Volume mode adds its three proposal files, two control/build receipts and
unchanged committed volume pins. Full manifest≤2MB and archive remain Azure;
only measured source/manifest/archive/receipt pins≤4096B return locally.

New per-episode SSH keys use root0700/0600, pinned Ed25519 host key and one
forced command from10.0.0.4 to10.0.0.9:2222. Never copy/print private keys or
reuse deleted credentials. The receiver checks complete archive size/hash,
then the independently hashed FIRST bounded manifest before parsing/extraction.
Fresh root0700 namespaces never merge originals; only new public copied leaf
files become0444 for individual read-only UID1000 container mounts.

VM02 requires original immutable image7eb/44layers and exact tracker helpers.
Acquire/recheck the existing FD9 GPU lock and empty compute apps; use explicit
Python entrypoint, UID1000, network none, read-only/cap-drop/no-new-privileges.
The sole writable isolated episode work parent is owned by UID1000; every
input leaf is mounted read-only. Budget10800s,16GB RAM,4CPU, no second GPU job.
All original indices, native receipt fields and three output hashes must pass
before atomic Linux RENAME_NOREPLACE promotes the owned pose directory into
the still-absent canonical target. Source/input/image/lock checks run afterward;
preserve any failure without overwriting, merging or guessing poses.

Tiny control tests and read-only independent audit pass. No real archive,
private SSH transfer or VM02 full-pose execution is claimed until each actual
sealed Azure receipt independently verifies. Image entrypoint/parent traversal
are runtime concerns; mocked tests are not successful native execution.
