# 4DAnyone : audit primaire et pertinence Track 1

Audit du **4 octobre 2026**, limité aux publications/releases disponibles au
**30 septembre 2026**. Sources/API initialement bornées à300KB ; aucune vidéo,
image, dataset, poids ou PDF téléchargé. Le texte HTML complet v1 (342894B,
SHA256 `22f99ee14c55a3f09d7d81140713a2effb7543889ea955f75da376c8ff9dc130`)
a ensuite été vérifié intégralement en bytes, sous une borne texte de500KB ;
méthode, ablations/protocole et limitations ont été lus. Pas d'inspection des
figures ou preuve d'exécution. Performances rapportées par les auteurs,
pas des mesures World Reward.

## Verdict

[4DAnyone](https://4danyone.github.io/) est une proposition pertinente de
**reconstruction humaine par génération de vues cohérentes**, pas un système
publié de reconstruction complète humain–objet directement substituable à
CARI4D. Code et poids sont effectivement disponibles avant le cutoff. En
revanche, la reconstruction 4DGS utilisée dans le papier n'est pas publiée :
le guide livré exporte **un timestamp vers une reconstruction 3DGS statique**
et indique explicitement qu'il ne reproduit pas les résultats 4DGS du papier.

L'intérêt principal pour notre framework est le **contexte séquentiel structuré**
et la séparation entre observations, géométrie estimée et prior génératif.
Adopter une marque ou empiler son pipeline ne résout ni le mesh objet rigide,
ni les mains/contact, ni la jauge métrique commune, ni l'export Track 1.

## Sources et disponibilité vérifiables

| Source primaire | Révision/date | Identité SHA256 du texte audité |
|---|---|---|
| [Papier, arXiv 2608.20335](https://arxiv.org/abs/2608.20335), *Create Anyone in 4D from a Casual Monocular Video*, SIGGRAPH Asia 2026 | v1 : 20 août 2026 | abstract HTML `58a23af2dd824f70e2997fa2549545d272212e33e5726237ea56c328af0def8b` |
| [Code ant-research/4DAnyone](https://github.com/ant-research/4DAnyone/tree/9cc2aa230fe5d364da2c5634ec3dabadadcb0d60) | `9cc2aa230fe5d364da2c5634ec3dabadadcb0d60`, 23 septembre | README `80f24f0e8e7fa8d6957e191499580a7432e58c4993328f5f8848006175de4527` |
| [Guide de reconstruction publié](https://github.com/ant-research/4DAnyone/blob/9cc2aa230fe5d364da2c5634ec3dabadadcb0d60/docs/nerfstudio.md) | même pin, lignes 72–75 | `bc61f8112e2a3c3ef4e46e48b18bf458a9a65d2d06a4ae0990aa0d2bca519524` |
| [Poids et notices AntResearch/4DAnyone](https://huggingface.co/AntResearch/4DAnyone/tree/1a644be15442ca5c0b4ae628bc2c861bc6b209ef) | `1a644be15442ca5c0b4ae628bc2c861bc6b209ef`, dernière modification 19 septembre | LICENSE.md `44d01721d4e45d913609fb52f3210a8267eeddaca6410fcd2c9f03f5633b07e1` |
| [Licence du code](https://github.com/ant-research/4DAnyone/blob/9cc2aa230fe5d364da2c5634ec3dabadadcb0d60/LICENSE) | Apache-2.0 | `1c706f0f86f34b3245bdd3f29cadc63aa81616c849bfc197fd4d116a9d90a218` |
| [CARI4D original NVlabs](https://github.com/NVlabs/CARI4D/tree/71fa7cbe46081467edadd11ab534b0c14aa9d913) | pin déjà audité dans `docs/baseline.md` | README `dc756c907bc4d50122cb5b05af52689f94c72051fae91f6ee46314024f8b1be0` |

Les révisions complètes font foi ; les pages courantes ne constituent pas une
preuve de disponibilité historique de modifications ultérieures.

## Mécanisme et limites de la release

1. **RCP — Reference Context Packing** : références à résolutions mixtes dans
   un budget de contexte borné, plutôt qu'un contexte croissant avec le nombre
   de vues cibles. **TCR — Target Context Routing** : groupes de vues différents
   au cours du débruitage, échange global à bruit élevé puis stabilité des
   détails. Ce sont des mécanismes de cohérence de génération, pas une mesure
   supplémentaire de la scène réelle.
2. La géométrie conditionnante vient de **GVHMR/SMPL-X**, puis d'un régresseur
   de squelette **MHR70/Goliath**. `skeleton/pipeline.py` emploie SMPL-X neutral
   à 10 475 sommets et une conversion SMPL. Un squelette MHR70 **n'est pas** le
   modèle MHR natif à 18 439 sommets, ses 204 contrôles ou un estimateur HOI.
3. Les caméras cibles sont construites autour de l'humain estimé, dans un repère
   canonique dont l'origine utilise la racine initiale/projection au sol.
   `cameras.json` décrit ces **caméras de synthèse**, pas une calibration mesurée
   de la caméra d'entrée. La reconstruction monoculaire ne démontre pas une
   échelle métrique absolue, un sol vrai ou une position fiable des surfaces
   invisibles. Une apparence plausible peut halluciner une géométrie erronée.
4. Écart important avec le headline « unknown camera / mild motion » :
   [`motion/gvhmr.py`](https://github.com/ant-research/4DAnyone/blob/9cc2aa230fe5d364da2c5634ec3dabadadcb0d60/fdanyone/motion/gvhmr.py)
   **rejette `static_cam=False`**, désactive SimpleVO et appelle
   `predict(..., static_cam=True)` ; les intrinsics source sont estimées.
   SHA `13ef1744bff6b5046ca11d38e686e271b7a9311e18783520ddec07d6d7cdbd3a`.
   La tolérance visuelle à de petits mouvements n'est pas une estimation
   publiée de caméra mobile.
5. La release impose **121 frames** et recommande un humain unique, visible,
   des vidéos portrait nettes et peu de mouvement caméra. Il faudrait un
   nouveau contrat séquentiel pour nos vidéos complètes : jamais tronquer,
   resampler ou perdre les indices pour satisfaire ce défaut.

## Ce que la méthode/les ablations établissent réellement

Le papier v1 emploie40keypoints :17corps,6pieds,10knuckles et7landmarks
auxiliaires ;30joints de doigts exclus. Les détails de doigts sont donc
**générés depuis l'apparence**, pas explicitement reconstruits/validés pour HOI.
Le z-buffer de squelette résout un ordre d'occlusion estimé, pas un contact.
RCP conserve le source et3références compressées ×2 plus4 ×4, budget fixe ;
les références proviennent de **vues générées**, pas d'observations réelles.
TCR regroupe4vues avec circular sliding à bruit élevé puis groupes adjacents
fixes à faible bruit. Le gain n'est pas obtenu par regroupement arbitraire :
l'ablation DNA rapporte PSNR22.63/SSIM.796/LPIPS.191 pour Sliding,
contre22.20/.788/.197 Random et22.06/.786/.198 Strided. Ces métriques
sont une cohérence visuelle multivue, pas précision absolue humaine/objet.

L'évaluation principale utilise10scènes DNA et3DyMVHumans,16caméras/98frames ;
génération121 tronquée à98 pour ce protocole **auteur**, pas permis pour nos
trajectoires complètes. ReCamMaster est fine-tuné avec mêmes données etRCP/TCR,
les deux autres modèles comparés zero-shot : ne pas présenter un contrôle
strictement identique entre tous les systèmes. L'annexeH constate que les
vues cohérentes peuvent hériter d'une **mauvaise pose HMR**, et que les vêtements
loin du squelette divergent. Pour World Reward : prior génératif distinct
avec son incertitude ; cohérence inter-vues ne certifie jamais exactitude3D.

## Comparaison avec la vraie baseline et faisabilité

Le [papier CARI4D](https://arxiv.org/abs/2512.11988) et son code NVlabs original
visent **la reconstruction conjointe humain–objet** : humain SMPL-H/NLF,
profondeur UniDepth, objet Hunyuan/FoundationPose, puis CoCoNet/raffinement.
La variante V2D/MHR au pin `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` est
distincte ; ses entrées légitimes, son raffinement final et son export sont
documentés dans `docs/baseline.md`. Ne pas confondre ces trois body models ou
interpréter un squelette MHR70 comme une compatibilité de soumission.

4DAnyone évalue notamment **DNA-Rendering/DyMVHumans** et la fidélité des vues
générées/4DGS. PSNR, SSIM et LPIPS ne prouvent pas un gain de CD humain/objet,
accélération ou pénétration Track 1. Le contenu objet dans une vue générée ne
constitue pas une trajectoire rigide clip-constante ni une preuve de contact.

Coût primaire détaillé :
[`docs/inference_performance.md`](https://github.com/ant-research/4DAnyone/blob/9cc2aa230fe5d364da2c5634ec3dabadadcb0d60/docs/inference_performance.md),
SHA `7e177e920ee7e118cc2e001694bcd4bf6415577f75e6a2ea8d99e92d9ca3d02b`.
Turbo 4 étapes/6 vues : **0,97 min H200 SDPA**, plus **1,20 min GVHMR +
1,25 min squelette**, pic 24,06 GB pour la génération ; 24 vues : 5,11 min
H200 SDPA. Le README annonce aussi 27 s sur 4090, sans concordance avec cette
table détaillée. Pas d'ETA H100/full-4DGS extrapolée, ni de mesure locale.

Code et checkpoint principal Apache-2.0 ne couvrent pas tout le pipeline :
**Turbo LoRA par défaut CC-BY-NC-SA-4.0**, GVHMR research/non-profit,
SMPL-X séparé, schéma MHR70 sous Sapiens2, YOLOv8 AGPL. L'original NVlabs
CARI4D reste research non-commercial (LICENSE SHA
`57bddada60db4635298fb4976cf17a26b9f5c978f3af0fa1aa6a2eee52bdc10c`).
Le papier mentionne MVGameHuman interne, lightstage et vidéos in-the-wild :
**recouvrement challenge non vérifié** ; aucune affirmation leakage-free.
Les scripts d'installation/téléchargement ne doivent pas être exécutés sans
audit de chaque source et licence.

## Deux idées testables, pas deux nouveaux empilements de modèles

**1. État séquentiel partagé + contexte borné.** Un framework unique porte
l'identité humaine, le mesh objet, K/jauge commune et les trajectoires ; ses
fenêtres utilisent un petit contexte de références automatiques et un échange
global explicite, inspiré de RCP/TCR. Hypothèse : réduire les incohérences de
fenêtre/occlusion sans sacrifier les positions ou l'interaction. Ce n'est ni
une moyenne de poses non couplées, ni le re-test d'un fit d'identité déjà rejeté.
Gate sur une **nouvelle validation externe** de longues séquences : couverture
totale, gain 3D apparié et accélération, sans régression objet/pénétration ;
gates et budget gelés avant données. Un gain de seam/IoU seul ne suffit pas.

**2. Géométrie sparse fiable, profondeur/appearance incertaines.** Conditionner
une représentation temporelle HOI commune sur les observations réelles de
squelette/masques/points avec leur confiance ; ne pas donner aux profondeurs
bruitées ni aux vues hallucinées le statut de mesure métrique. Hypothèse :
meilleure articulation et réacquisition après occlusion, sans dérive du repère
humain–objet. Une éventuelle génération reste un **prior/proposition**, vérifié
sur la vidéo réellement observée. Gate indépendant sur erreurs 3D/interaction
et identité après occlusion, pas sur PSNR ; pas de réouverture des anciens
fits racine/profondeur/photométrie rejetés sur leurs mêmes cohortes.

**Priorité de livraison :** une soumission Track 1 complète et gelée demeure
le baseline opérationnel ; ces idées ne bloquent pas sa production. Un bon
framework sépare composants réutilisables, contrats de données et configuration
scientifique globale. Pins, budgets et hyperparamètres préenregistrés sont
nécessaires ; prompts/focal/seuils choisis par épisode ou après ses résultats
sont du bricolage interdit. Aucun accès FORM-HOI apparié, calibration source,
vraies autres vues challenge ou GT ne découle de cet audit.
