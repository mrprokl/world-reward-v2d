# Proposal stress v1 — frozen, new authored cohort

## Question and boundary

Does an unchanged automatic region/proposal producer retain visible human and
object instances in a deliberately crowded synthetic scene, and at what cost?
This measures **proposal/mask IoU recall and runtime only**. It cannot establish
person→hand→object ownership, task-target selection, real-image fidelity,
challenge accuracy, or an improvement over CARI4D.

No OI, HO-Cap, challenge media, old synthetic recipes, predictions, calibration
or reference records are inputs. Source mechanics alone are reused from the
qualified MHR/PyTorch3D camera renderer. The new recipe is fixed before any RGB,
model observation or QA. All 16 scenes are retained; **zero resampling**. A failed
manufacture closes this producer rather than prompting per-scene repairs.

## Fixed cohort and truth

Sixteen independently authored scenes: 8 DEV and 8 reserved, two original-grid
640×480 images each. DEV/reserved membership and scene/frame correspondence
remain private. Each scene contains two or three distinct MHR identities with
clip-constant shape, zero native scale controls and a fixed authored world scale.
The latter is a declared authored mesh transform, not a native 68-scale identity
estimate or a reconstruction of human metric scale.
Four fixed object meshes include cuboid, cylinder, ellipsoid and a duplicate;
identical material/shape distractors, depth overlap, background people and small
temporal motions vary mechanically. Geometry/shape/material/identity do not
change between the two frames. These are posed mannequins and primitive props,
not a claim of realistic interactions or a learned crowd distribution.

One pinned MHR checkpoint is loaded once, followed by one original batched MHR
forward with corrective shapes enabled. Named arm controls are checked against
the actual checkpoint limits. Native cm axes are converted once to an authored
OpenCV metre frame. PyTorch3D 0.7.9 uses the unchanged audited OpenCV projection,
hard first-face raster, perspective correction and Phong shading mechanics.
All original triangles are retained; no invisible-object deletion, alignment,
mask filling, resizing or proxy-driven fitting occurs.

The 300s manufacture budget, near-plane checks, full native ABI, finite RGB/Z,
≥1024 foreground pixels and ≥2 humans with ≥32 visible pixels are engineering
gates, **not proposal-quality gates**. Objects with zero visible area remain in
private truth and are explicitly ineligible for visible-mask IoU, not silently
deleted or counted as recovered. No claim of amodal-mask recovery is possible:
the private reference masks are scene-occlusion-aware visible rasters only.

## Public/private separation

`validation/proposal_stress_v1/inputs` contains exactly 32 PNGs and one manifest:

```json
{"schema":"world_reward.rgb_proposal_inputs.v1","images":[
  {"image_id":"opaque32hex","file":"image_000000.png","bytes":1,
   "sha256":"64hex","width":640,"height":480}
]}
```

Opaque IDs are deterministically ordered; neither scene/frame nor split/role is
exposed. `eval_private` retains the recipe, mapping, native controls/rig names,
camera, complete meshes, face-instance IDs, depth, visible raster masks and
manufacture receipts. RGB and all receipts are hashed; outputs are immutable.
The producer creates no inferred mask, target prompt or selector output.

**Future inference must mount only public RGB plus independently frozen public
pins and a clean inference-source whitelist. Never mount this renderer,
`proposal_stress_v1.json`, any scene recipe, complete renderer source checkout,
or `eval_private`, even though source is publicly versioned.** A broad configs
or validation mount violates the blind observation boundary. Reserved truth
must remain unread until predictions and method parameters are frozen.

## Planned evaluation (separate, not implemented here)

For each visible reference instance, best IoU over every retained native region
is a proposal-recall statistic, not ownership. Evaluate fixed IoU thresholds
0.25/0.50/0.75, per-human/per-object instance recall and macro scene recall;
retain duplicates and report mask count/runtime/peak memory. Report visibility
bins from private truth as diagnostics, never use them to select proposals.
DEV may guide a general method; the eight reserved scenes are evaluated once
after a frozen prediction artifact. This small authored study cannot validate
real-world or challenge generalization.

## Azure-only prerequisites and execution

- VM02 owns the GPU scheduler; take the existing H100 lock and verify GPU idle.
- Exact classic image `sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3`,
  Torch 2.5.1+cu124, PyTorch3D 0.7.9, audited source revision
  `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`. Native execution verifies installed
  `direct_url.json` against the exact Git URL/commit and `v0.7.9` revision; the
  version or a source constant alone is insufficient. No new runtime/checkpoint.
- Selected archive `75fdea08fb3b43f62f1c4b4f5e646674595cdd0b` manifest and original
  extraction receipt, independently pinned by `frontend_asset_archive_pins.json`.
- Its original MHR asset: 696110248 bytes,
  SHA256 `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc`.
  Only that leaf and the tiny selected receipts are mounted, not the 20GB tree.
- Source/recipe freeze, public commit accessibility and immutable dispatch
  markers must precede manufacture. The wrapper is one-use, network-offline,
  root with dropped capabilities, 16GB RAM/4 CPUs, own bounded container cleanup.

All media/models remain on Azure. Asset/source ancestry is checked, but external
model licensing/overlap eligibility is **not newly established** by this renderer.
No Azure run has occurred as part of implementation/testing this protocol.
