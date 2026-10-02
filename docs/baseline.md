# Audit CARI4D exécutable — Track 1, 2 octobre 2026

## Révisions et sources primaires
- V2D code inspecté: `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` (GitHub API, HEAD lors de l'audit).
  https://github.com/nvidia-isaac/video_to_data/tree/7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80/reconstruction/modules/v2d_cari4d
- CARI4D original inspecté: `71fa7cbe46081467edadd11ab534b0c14aa9d913`.
  https://github.com/NVlabs/CARI4D/tree/71fa7cbe46081467edadd11ab534b0c14aa9d913
- Papier primaire, CVPR2026: https://arxiv.org/html/2512.11988 ; https://nvlabs.github.io/CARI4D/
- Kit officiel téléchargé le 2 octobre, archive datée du 1 octobre2026.
  https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip
  Archive SHA256 `b1f8f703c2e772684a57d048cf28f40c7b9b54cf7a61aff6917b46d826915e1f`. Extraction limitée aux modules publics de métriques Track1 et outils de packing; archive multi-tracks supprimée après extraction. Aucun GT, contenu de sample/data Track2/3 ou FORM-HOI source n'a été lu.

## Conclusion opérationnelle
La release V2D CARI4D v0.3 est monoculaire **avec mesh objet requis**, pas vidéo seule: elle attend RGB, masks, mesh. Pour Track1 il faut ajouter reconstruction et échelle objet depuis les seules vidéos Track1. Ni mesh fourni Track2 ni scanner/stereo n'est une entrée licite.

Baseline minimale fidèle: SAM2 human/objet -> sélection frames visibles -> SAM3D Objects mesh -> métrification du mesh -> CARI4D v0.3 -> conversion MHR kit -> packing -> preflight. Deux risques bloquants immédiats: gated weights et autoscale absent.

## Poids et environnement
- `nvidia/cari4d_commercial`, revision `1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9`, checkpoint `2026-08-25-09-35-57/step200000.pth`, SHA256 `78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3`. Gating automatique, NVIDIA Open Model Agreement.
- `facebook/sam-3d-body-dinov3`, revision `11aaa346c7204874a1cbafe3d39a979080b2c55a`. Gating **manuel**, SAM License.
- `facebook/sam-3d-objects`, revision publique actuelle `2e73555018d2741ccd486e56c24fac41155a1dc6`, gating **manuel**. Wrapper de download non pinned: fixer cette révision et hash des artifacts.
- MoGe2 `Ruicheng/moge-2-vitl-normal` revision `b135031bae30b5ac2ae141a0e68717795ce38340`, MIT, non gated.
- Source MoGe2 `925b8ed835a7a9cdb7578ba15c658a0afc969030`; DINOv3 `6876159a11b4df116f30f667f8c9888617df0751`, DINOv2 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- Docker CARI4D: torch2.5.1/cu12.4, PyTorch3D0.7.9 compilé avec SM9.0, Kaolin0.18.0, transformers5.3.0, numpy1.26.3, nvdiffrast, FoundationPose NVlabs PyTorch.
- Docker SAM3D Objects inclut `flash-attn==2.7.4.post1`: **indispensable H100**, car le code choisit automatiquement ce backend sur A100/H100/H200. SM9.0 explicitement compilé. Vérifier `nvidia-smi`, forward torch, rasterisation CUDA et EGL après build.
- V2D source Apache2.0, docs CC-BY4.0; NVlabs CARI4D original source research noncommercial. Ne pas mêler silencieusement la licence du repository original dans une release Apache. Les modèles SAM gardent leur licence, ne pas les redistribuer comme Apache. MHR modèle de référence doit provenir de Meta MHR Apache release1.0.1, pas SAM bundled MHR.

## Commandes baseline officielles (dans reconstruction, rtk partout)
```sh
rtk proxy ./scripts/install_packages.sh
rtk proxy python3 -m v2d.sam2.docker.build
rtk proxy python3 -m v2d.sam3d.docker.build
rtk proxy python3 -m v2d.cari4d.docker.build
rtk proxy python3 -m v2d.sam2.docker.run_download_weights --output_dir /data/weights/sam2
rtk proxy python3 -m v2d.sam3d.docker.run_download_weights --output_dir /data/weights/sam3d
rtk proxy python3 -m v2d.cari4d.docker.run_download_weights --output_dir /data/weights/cari4d
rtk proxy python3 -m v2d.sam2.docker.run_video_to_masks --video_path /data/input/SEQ.0.color.mp4 --prompts_path /data/input/prompts.json --masks_dir /data/output/masks --weights_dir /data/weights/sam2
rtk proxy python3 -m v2d.cari4d.docker.run_pack_masks --video_path /data/input/SEQ.0.color.mp4 --human_masks_path /data/output/masks/0 --object_masks_path /data/output/masks/1 --output_path /data/output/SEQ_masks_k0.h5
rtk proxy python3 -m v2d.sam3d.docker.run_image_to_mesh --image_path /data/input/selected.png --mask_path /data/output/masks/1/000100.png --mesh_path /data/output/object.glb --transform_path /data/output/object-transform.json --intrinsics_path /data/output/object-intrinsics.json --weights_dir /data/weights/sam3d
# AVANT inference: métrifier le mesh! Le CLI ne consomme pas object-transform.json et ne redimensionne pas automatiquement.
rtk proxy python3 -m v2d.cari4d.docker.run_inference --video_path /data/input/SEQ.0.color.mp4 --mask_h5_path /data/output/SEQ_masks_k0.h5 --object_mesh_path /data/output/object-metric.glb --weights_path /data/weights/cari4d --output_dir /data/results --expected_frames N
rtk proxy python3 -m v2d.cari4d.docker.run_tests
```
Tokens doivent être dans environnement secret non affiché. `run_download_weights` est pinned pour CARI4D/Body/MoGe mais SAM3D Objects séparé ne l'est pas. Les images ont encore git/pip mutable pour certaines dépendances: enregistrer digest image et lock versions, pas seulement source SHA.

## Échelle objet depuis la vidéo (implémentation nécessaire)
Le module officiel ne contient aucun outil autoscale autonome malgré son avertissement README. Trois options, en ordre coût:
1. SAM3D Objects accepts `--pointmap_path`, `--pointmap_intrinsics_path`: utiliser XYZ MoGe2 aligné à l'humain depuis Track1 pour grounding; pose/scale retournés restent séparés. Recentrer mesh et **appliquer scale uniquement** dans sa géométrie avant CARI4D, sans double transform. Contrôler rendered projection.
2. Reproduire méthode **conceptuelle** du papier: registration FoundationPose d'un mesh sous ~30 échelles coarse0.03..3 et ~10 échelles refinement autour top3, classées par Chamfer unidirectionnelle observed object depth -> posed mesh + silhouette. **Ne pas copier original NC sans gestion licence.** Adaptation importante: utiliser plusieurs frames hautes confiances et shared scale, depth métrifiée à MHR; score robust trimmed distances pour occlusion. Des rotations candidates mal choisies confondent scale avec pose; rejet IoU/depth obligatoire.
3. Joint video refinement du mesh/scale avec intrinsics fixed + poses initiales, objective silhouette/photométrie/features/estimated depth. Holdout temporel d'observations pour ranking, jamais GT caché.
Ne pas importer `scale_mesh_srt.py` tel quel: son contrat CuSFM + FoundationStereo et objet stationnaire/two_stage est faux pour interactions mobiles.

## Conversion vers soumission
- Artifact final: `/data/results/SEQ/inference/refined.pth`, bloc `pr`, `frames`, metadata; initial CoCoNet dans `coconet.pth`.
- Clés **NON directement compatibles kit**: globalrot6d6, trans3 mètres, bodycont260, hand108, shape45, scale PCA28, face72. Kit exige pose136, scale68 et shape45 constants.
- Méthode sûre: `MHRLayer.from_mhr_assets(...).mhr_forward_vertices(pr)` -> vertices `[T,18439,3]` caméra mètres, puis converter kit avec modèle MHR public Apache.
- Converter input déjà `[1,-1,-1]` transform et mètres via CARI layer; ne pas reflip ni /100 encore.
- Model Meta v1.0.1: https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip , `assets/mhr_model.pt` sha256 `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc`.
```sh
rtk proxy python3 tools/track1/mesh_to_mhr_params.py --model /data/mhr_model.pt --input /data/verts.npy --output /data/episode_000003_params.npz --precision float32 --model-batch 256
```
- Converter fits identity once across whole episode and zero facial expression. CARI shape may vary frame-wise; least squares projection is not lossless then. Record report actual residual; fit/choose shared identity before final refine improves structure.
- Faster direct option native head `return_model_params=True` produces204 params. Official SAM body export stores native translated camera root with Y/Z unflip and `*10` (model transform translation gain), **pas naïvement100**. Must roundtrip against reference MHR to verify vertices/joints; converter safer initial baseline.
- Mesh partenaire: `export/SEQ/object_mesh/output_aligned.glb`; CARI recenters/orients preserving scale. `metadata.object_mesh_to_training_transform` must be applied to submitted local mesh paired with `pr.pose_abs`. Wild export is normally identity transform. `object_scale=1` if metric scale baked into mesh.
- Episode NPZ naming `episode_000003.npz`, arrays indexed by original VIDEO FRAME, not only scored compact indices: pose[T136],scales68,shape45,object_rotation[T33],object_translation[T3],object_scale scalar. Mesh `episode_000003_object.glb`. Read sample IDs, packer does exact subset automatically.
- Packer decimates to4096 faces/4096 vertices then pads with degenerate faces. Geometry QA must test packed geometry too, not only original.

## Failure points et expériences rentables
1. **Object tracking drift**: default registration once then tracking sets reliable=True for all later tracked frames despite silhouette measurements. It continues even low-support masks. Source run_foundationpose_mhr_export.py613-637. Alternative public available: `--register-every-frame --no-first-usable-frame-gt-rotation-oracle`, which uses score-ordered retries + silhouette/temporal filtering. **Mandatory negative oracle flag:** this mode otherwise enables GT oracle by default! Assert no poses.npy, metadata uses_gt_pose=False. Test sparse confidence-based re-registration plus bidirectional track rather than costly every-frame.
2. **CoCoNet96 window seams**: overlap policy first_occurrence; stride96. Smaller stride alone does NOT fuse predictions. Implement overlapping confidence/tapered fusion with SO3 rotation averaging and shared identity. Measure seam error with rendered keypoints and silhouette heldout observations, accel and visual smoothness.
3. **Hands frozen**: commercial config uses `body12_freeze_hand_face_all_losses`, w_mhr_hand0. Parity refinement fixes hands, root, trans, shape, scale and faces; optimizes body rotation + object translation, optional object rotation. Therefore add independent hand pose estimation/crop cues and hand-only anti-penetration/contact refinement. Avoid forcing all uncertain hands onto object.
4. **Contact wrong/lost**: a contact predicted by network only activates if raw CoCoNet min hand surface distance <5cm, fixed once. Correct contacts farther than5cm never recover. Better visibility/confidence-gated continuation, with explicitly visual evidence, no mere attraction across occlusion.
5. **PEN mismatch**: refinement penalizes object sample points inside human, scorer hand512 points inside own object. Add correctly oriented closed mesh hand-to-object SDF loss matching physical quantity, and preserve cavity orientation. Don't exploit open/inverted meshes or move object away to lower PEN.
6. **Temporal vs accuracy Pareto**: paper rowe feedforward CDH7.01, CDO11.59, AccH1.75,AccO3.78; rowf full CDH8.41,CDO11.57,AccH1.06,AccO0.38. Smoothness strongly helps object acceleration but can hurt body accuracy. Evaluate init/raw/refined and regularization strength separately on lawful external validation, not presume default300steps wins.
7. **Shape**: multi candidate seeds and frames visible whole object -> select candidate with heldout frame render/reprojection agreement. Include analytic geometry for visibly regular object classes only if derived from input, not hidden templates.

## Table3 vs challenge scorer
Same FIRST human-only Sim3 alignment principle, no per-frame alignment in paper either. Differences: paper original SMPLH+UniDepth+Hunyuan vs challenge MHR and new commercial weights; BEHAVE62 and InterCap22 chosen early object visible (86%+) vs unknown Track1 episodes; kit body64 fixed vertices, hands512, joint22. First scored frame may be >video0; same transform applies object too (initial object error counts). CD is sum of both directed means, not half and not squared, cm. Accelerations use 2nd frame differences **no fps² multiplier**, contiguous scored stretches only. PEN averages all512 points with outside=0 against submitted own mesh and reference enters through alignment scale only. Scores episode-means and dataset mean. Thus paper metrics cannot establish challenge victory; lawful local validation + actual submissions necessary.

## Clean research gates
- Access proof before big builds; smoke 8-32 frames par initialiseur et **96+ frames end-to-end** (CARI4D exige clip_len96) avec reference model roundtrip before processing30 episodes.
- Use licensed external data split by sequence/object/person; block any FORM-HOI source of challenge and any Track2/3 GT.
- Never feed challenge labels, test meshes, clean trajectories or accidental GT files into inference. Native prep source has training-oracle branches, so top-level wrapper and assertions essential.
- Log only pinned revisions, artifact hashes, run config, metrics, decision and failure reason. Keep predictions/models outside code; prune debug files after visual QA but keep minimal evidence images and final metadata.
