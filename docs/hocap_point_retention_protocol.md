# HO-Cap : persistance de points dans des masques sémantiques

**5 octobre 2026 — gel prospectif avant AMG/Boots ; pas exécuté ni adopté.**
Le protocole numérique est `configs/hocap_point_retention_protocol_v1.json`.
Une future implémentation/pins authentiques restent nécessaires. Ce diagnostic
n'est ni identité physique, association acteur–objet, contact, correspondance
matérielle, reconstruction3D, calibration, ni victoire sur CARI4D.

## Cohorte et séparation

Deux clips HO-Cap subject_5/caméra105322251564, test-only HPE/OPE :
20231027_112303 (702 frames) et20231027_113202 (791), toutes1493 frames.
Le manifeste réel358318B/SHA58f888… et report5940B/SHA17b5ee… sont entièrement
pinnés dans le JSON ; les anciens échecs d'acquisition restent FAIL.
Original `arange(T)` et source IDs inchangés ; aucun nouveau clip/remplacement.
Ce n'est pas un heldout garanti hors entraînement des checkpoints : overlap inconnu.
CC-BY-4.0 couvre les données ; toolkit GPL3 distinct, aucun MANO/mesh chargé.

Inputs predictor : RGB sélectionnés et métadonnées publiques clip/camera/T.
Tous les JPEGs sont décodés/validés avant le modèle. Pas de rawmeta, label,
pose, K capteur, autre caméra, CAD, main ou cible annotée sur GPU.
Les deux prédictions Boots complètes doivent être scellées/authentifiées,
avec banque/runtime/source/pré-posthash, **avant toute lecture de label privé**.

## Banque et tracking communs

SAM2 AMG class-agnostic original, mêmes filtres natifs fixés par son protocole :
aucun prompt `object.`, liste de catégories ou adaptation sur les deux clips.
Ancres `floor(k*(T-1)/4)`, k=0..4 : [0,175,350,525,701] et[0,197,395,592,790].
Conserver **tous** les retours natifs : hiérarchie/parties, overlaps, fond,
raw scores et Q0. Limite ressource ⇒ STOP whole bank, jamais truncation/top1.
La grille4×4 inside-mask originale produit les queries `(ancre,y+.5,x+.5)`
sans refill. Aucun masque/point n'est une identité acceptée.

Un seul load BootsTAPIR existant ; une forward full-T/all-query par clip.
Même source/checkpoint, FP32, pyramid1, native256, chunk32, seed0, noTF32/AMP,
`is_training=False`. Préprocessing/conversion XY natifs, pas second demi-pixel.
Garder queries/IDs/births, XY256/original, occlusion/expected_dist/visible,
y compris avant birth ; visible natif n'est pas une vérité/confidence calibrée.
Le diagnostic A=XY initial statique n'est **pas** une trajectoire admissible
d'interaction. B=Boots conserve mouvement ou stationnarité native sans filtre.

## Référence privée : contrat primaire, pas meilleur matching

URLs/revisions/bytes/SHA complets des sources sont dans le JSON :
[loader@576c63e](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/loaders/sequence_loader.py#L246-L261),
[factory](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/hocap_toolkit/factory/dataset_factory.py#L204-L218),
[catalogue](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/config/hocap_info.yaml),
[viewer](https://github.com/IRVLUTD/HO-Cap/blob/576c63ebf3b84dfec8744ba0f021234213bf0dab/examples/image_label_viewer.py#L28-L59).
`SequenceLoader.masks` est une validité/crop RGB-D, **pas** GTseg.

CPU lit exactement `subject_5/<clip>/<cam>/label_<source_id:06d>.npz`,
1493 membres du labels.zip authentifié. `allow_pickle=False`, accès aux seules
clés `seg_mask`, `obj_class_inds`, `obj_class_names`; pas même lecture des arrays
cam_K/poses/joints. Pas de constructeur SequenceLoader/full YAML ou CAD.
Conserver types entiers natifs non-bool SEG/classes, grilleH/W originale ;
pas cast/resize/relabel. Strings natives fixes, sans dtypeobject/pickle.

La factory énumère SEG positives triées après0=fond : la k-ième correspond à
classes[k]/names[k]. **SEG n'est pas class_index+1** ; +1 est un export COCO/BOP.
Vérifier cardinalités, nom/index dans catalogue original et cohérence full-T.
Schema/doublons sémantiques contradictoires/labels manquants ⇒ INCONCLUSIVE job,
pas conversion en fond ni suppression du dénominateur. Cela doit être qualifié
sur les annotations réelles avant interprétation ; aucune valeur lue ici.
Si deux objets partagent une classe, aucune identité physique n'est résolue.

Pour chaque query, une seule référence est attribuée **à son pixel initial**
(floor du XY continu) dans le label de son ancre ; class/index restent fixes.
Pas Hungarian, pose matching, meilleure seed/masque, regroupement GT predictor,
ou changement de référence quand le point dérive. Fond/HAND initiaux restent
contamination séparée, jamais acteurs/cibles. Tous les variants/queries sont
conservés ; histogramme/pureté/overlap rapportés sans sélection.

## Métriques et décision

Évaluer seulement t>birth ; le point forcé initial et prébirth restent stockés.
Chaque query initialement référencée objet contribue T−1−birth slots possibles,
même occluse, out-of-frame ou quand sa classe n'a aucun pixel au label futur.
Q0=NO_QUERIES ; dernière ancre=NO_FUTURE, jamais score1 ou réussite artificielle.
Correct-supported = visible natif ET XY fini/in-grid ET SEG à floor(XY)
appartient à la classe objet initiale. Hidden/outgrid/fond/HAND/mauvaise classe
ne sont pas corrects ; aucune missing query/frame/classe n'est retirée.

Primaire par clip : fraction micro correct-supported sur **tous** les slots
objet/query/futureframe. Wrong-object-class utilise le même dénominateur ;
fond/HAND/hidden restent rapports séparés. Afficher aussi macro seed/ancre,
Q0/NO_FUTURE, tous les per-query counts, contamination et couverture par classe.
Rapporter chaque frame : possible/support/correct/wrong/fond/HAND, classe absente,
fraction native-visible et gaps/minima. **Pas de floor visibilité par frame** :
l'occlusion naturelle est valide ; elle est déjà pénalisée dans le primaire.
Queries/ancres/hiérarchie corrélées ne sont pas des réplications indépendantes.

QUALIFIED_SEMANTIC_MASK_POINT_RETENTION uniquement si B>A correct-supported
**sur chacun** des deux clips, et wrong-object-class B≤A sur chacun.
Zéro slot objet possible ou GT inconnue/malformée ⇒ INCONCLUSIVE ; sinon REJECT.
Pooled gain seul insuffisant ; aucune suppression de clips immobiles/fond,
aucun nouveau prompt/seuil/refill après échec. Pas de seuil3D ou gainHOI caché.

Budgets prospectifs natifs : AMG900s, Boots2clips1800s, CPUévaluation300s ;
grace cleanup60s/stage, hôte distinct. Runtime/receipts réels doivent être
gelés avant évaluation. Ce protocole ne prétend ni disponibilité de ces nouveaux
drivers ni PASS de modèle, qualité ou référentiel physique.
