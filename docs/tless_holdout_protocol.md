# T-LESS frame holdout v1 — acquisition only

**Purpose:** independently captured real object-camera validation after the
positive TUD-L depth result. This is not full human-object/temporal validation,
not a leakage audit, and not proof of victory over CARI4D.

## Frozen before private labels

- Protocol: `configs/tless_frame_holdout_protocol.json`.
- Dataset revision `5fd309a04476a842d93abfb584fba9ee7caecdf1` (HF, 2025-02-24).
- Primesense scenes **1, 10, 20**, four sorted RGB ranks **floor(k N / 5)**,
  k=1,2,3,4 independently per scene. N≥5; when N=200 this is 40/80/120/160.
  No scene/frame replacement based on annotation values or image inspection.
- Twelve original **720×540 RGB** PNGs only, plus a public attributed manifest.
  No private camera, depth, instance labels/masks, mesh or source GT is public.
- Private split retains selected depth PNGs, original camera/depthscale records,
  and visibility masks for **every** annotated instance, including occluded ones.
  No pose alignment, instance filtering, sensor resampling or prediction fitting.

## Sources and licence

[BOP T-LESS](https://bop.felk.cvut.cz/datasets/#T-LESS) and the
[pinned HF card](https://huggingface.co/datasets/bop-benchmark/tless/raw/5fd309a04476a842d93abfb584fba9ee7caecdf1/README.md)
concordantly declare **CC BY 4.0**, permitting commercial reuse with attribution.
The 958-byte BOP section and 30-byte HF card have frozen SHA/byte identities in
the protocol. Original dataset-info/embedded licence text must also identify
T-LESS and explicitly agree; missing, ambiguous or contrary terms cause STOP.
The author's older project site was inaccessible during the metadata audit;
no licence waiver is inferred. Licence evidence and original base metadata stay
private; the public manifest supplies authors, citation, source/revision,
licence link and a description of modifications.

Exactly two immutable downloads, **825,326,589 compressed bytes total**:

| Archive | Bytes | SHA256 |
|---|---:|---|
| `tless_base.zip` | 49,597 | `dd70ca884b7c471a530a952f70c5ab2c212f3d2c2f371be86397442b97d70a7e` |
| `tless_test_primesense_bop19.zip` | 825,276,992 | `1a18f6bbfb5ac4ced8529f7a35225adfed88c0f62ef38067933e2b541ef1d00b` |

No training/full-test archive or object-model archive is acquired. Archive
layout and embedded licence have not been checked by downloading assets locally;
the driver deliberately refuses unknown layouts rather than guessing paths.
Format metadata is pinned to BOP toolkit commit
`b72b3015c87a96fa6398c2ef4c196e85f798d3e6`, native test grid 720×540.

## Azure execution and fail-fast contract

Prepared driver/wrapper only; root owns deployment and actual dispatch. All data
remain on Azure under `validation/tless_frame_holdout_v1`. A fresh absent target
is mandatory; never overwrite, resume or clean an old failure implicitly.
CPU only, stdlib, **600 s** Python alarm / 603 s TERM wrapper / 10 s kill grace,
16 GiB address-space cap. No model imports, GPU, Docker or installation.

Whole archive SHA/bytes pass before ZIP parsing. All ZIP members are audited
before any extraction (safe paths, no symlinks/encryption/duplicates, fixed
8 GB declared uncompressed size, 500,000 members, 128 MB/member). Complete
filename selection is sealed before private JSON interpretation; extracted
selected data are capped at 256 MB. Bounds are preregistered, not enlarged after
a failure. Unknown layout, missing RGB/depth/all-instance masks, nonfinite JSON,
licence conflict, changed source or pin mismatch means STOP.

Readonly sources and protocol are hashed before/after. Public files are0444;
private directories0700 and files0400. One immutable private acquisition report
records actual selection, source/archive/licence identities and exact retained
file inventories. ZIPs are removed even on failure; useful failed receipts and
retained evidence are preserved. No heavy data transit locally.

## Future comparison, not implemented here

Freeze original native MoGe/DA3 methods and all12 prediction bytes before private
evaluation. RGB-only K = focal800, centre(360,270), +0.5 pixel centres. The fixed
10% border is x<72 or x≥648 or y<54 or y≥486: **background proxy only**, without
semantic human/object exclusion. At least1024 paired pixels and95% paired border
coverage per image; one median of four per-image ratio medians per fixed scene.
No per-frame anchor, offset, hyperparameter tuning or validity drop/expansion.

Private native depthscale/K and union of every visible instance mask provide
sensor truth only. Original predicted camera XYZ remains untouched. Score four
paired frame camera Chamfer means per scene, then scene-relative gains:
median≥5%, no scene worse than−5%, both modes≥95% object coverage every frame,
exact candidate/baseline validity. No evaluation alignment; zero/insufficient
baseline support rejects. Preserve prior negative DA3 results and limited scope
of the TUD-L positive result. Cross-dataset object-geometry identity and model
training/challenge overlap are **unverified**, not asserted disjoint.

## Actual outcome — STOP, 2026-10-03

The original acquisition failed in **0.993362 s** before the 825 MB test ZIP,
private annotation values, RGB selection or inference. The byte-pinned base
contains an extra `test_targets_bop18.json`; a separate Azure-only metadata audit
also finds no embedded licence declaration in `dataset_info.md`. Neither meets
this frozen protocol. Preserve the original failure, remove disposable ZIPs,
and do **not** widen the layout or infer a licence waiver to rerun it. No T-LESS
generalization result or twelve-image input set exists.
