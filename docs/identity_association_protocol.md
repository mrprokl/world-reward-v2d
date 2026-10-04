# Association temporelle multi-instance — proposition scientifique

**4 octobre 2026 : primitive mathématique implémentée ; protocole externe
PROPOSÉ, ni préenregistré, ni autorisé, ni exécuté.** Les seuils, modèles,
coûts, budgets, licences complètes et pins d'acquisition restent à geler.
YCB48–50 et les pilotes clos ne sont pas recyclés pour ajuster cette méthode.

## 1. Ce qui existe, ce qui manque

`world_reward.temporal_identity` fournit `IdentityNode`, `IdentityGraph`,
`IdentityPath`, `IdentityRanking` et `rank_identity_paths`. Un graph représente
un couple persistant Z=(acteur, objet), avec états/observations possibles par
frame. Les états explicitement absents appartiennent au même Z ; aucune identité
n'est réinitialisée et aucun point/pose/masque observé n'est inventé.

Le solveur reçoit coûts unaires u, transitions v, arêtes admissibles booléennes
et références d'evidence sur toute la timeline int64 `arange(T)`. Il minimise
`E_Z(s)=sum_t u_Z,t(s_t)+sum_t v_Z,t(s_t,s_(t+1))` par programmation dynamique.
Tous les slots de coûts fournis sont finis, même interdits ; valeurs négatives
permises. Les couches sont triées par clés stables et leurs coûts réordonnés.
Copies readonly sans alias ; ties déterministes par clés, pas par ordre d'entrée.

Il retourne tous les couples classés et leur écart brut, les min-marginales
`min_{s:s_t=j} E_Z(s)`, regrets locaux et coût alternatif d'evidence différente.
Deux états nuisance portant la même evidence ne sont pas deux identités.
Une min-marginale infinie signifie état inatteignable ; sans alternative,
le coût alternatif est infini. Aucun écart n'est une probabilité ou une confiance
calibrée ; son échelle dépend des coûts et de T. Comparabilité inter-graphs,
pénalité d'absence et règles d'acceptation restent la responsabilité du protocole.

Une banque malformée/infeasible échoue entière : erreur d'intégration, distincte
d'une abstention scientifique. Un concurrent all-missing/peu soutenu reste classé
(`NO_EVIDENCE`/support explicite) ; le premier résultat n'est **jamais accepté**
automatiquement. `source_ref` référence une preuve, mais ne l'authentifie pas.
Le module n'apprend rien, n'exécute aucun modèle et ne produit ni scène ni contact.

Manquent : candidats automatiques, coûts d'apparence/points/reprojection,
association à une géométrie canonique, calibration et décision d'abstention.
Forme, échelle, K et jauge sont hors graph, clip-constants. Indices de mains
conditionnels, pas attraction universelle ni rigidité main–objet imposée.

## 2. Tests discriminants avant acquisition

Tests data-free : optimum/min-marginales contre énumération exhaustive,
permutations/ties, influence de futures observations, absence conservée,
concurrents all-missing/distracteurs et overflow/malformed/infeasible.
Ces **coûts synthétiques fournis** sont un oracle de test du solveur, pas des
observations apprises ni une mesure de qualité ou un entraînement ML.
Avant modèles, ajouter contrôles croisement, distracteur proche de la main,
relâchement/glissement et symétrie indiscernable ; tester l'évaluateur pour
qu'une abstention ou mauvaise identité ne gagne pas par suppression de frames.

## 3. Cohorte externe proposée : DexYCB

[DexYCB primaire](https://dex-ycb.github.io/), CVPR2021 : scènes multi-objets,
poses rigides et main ; **pas corps entier**, ni objets nouveaux face à YCBV.
Données CC-BY-NC4.0, toolkit GPL3 au
[pin64551b001d360ad83bc383157a559ec248fb9100](https://github.com/NVlabs/dex-ycb-toolkit/tree/64551b001d360ad83bc383157a559ec248fb9100).
MANO/meshes ont leurs droits distincts à vérifier ; modèle/challenge overlap
inconnu. Environ12GB/sujet selon le publisher, aucun asset acquis ici.

Design proposé à geler avant acquisition : sujet `20200709-subject-01`, caméra
`836212060125`, six premières séquences lexicographiques, toutes les frames
natives. Calibration séparée proposée : trois premières séquences du sujet
`20200813-subject-02`, même caméra. Pas de remplacement selon visibilité/labels.
Ce sont des choix de design, **pas des pins mesurés ou un split déjà disponible**.

Le toolkit `dex_ycb.py` lit `meta.yml` : les vrais champs sont `ycb_ids` et
`ycb_grasp_ind` (pas `object_ids`/`grasp_ind`). La règle privée figée est : vérifier
IDs uniques et index valide, puis cible=`ycb_ids[ycb_grasp_ind]` et ligne de pose
correspondante. `grasp_eval.py` confirme cette association. Le selector privé
s'exécute après gel complet des deux prédictions, **avant calcul des erreurs**,
sans recherche du meilleur matching. Ces champs, MANO/labels, K/depth capteurs,
objets scannés et autres caméras ne sont jamais fournis au predictor RGB-only.

## 4. Deux bras et décision à préenregistrer

A : notre initialiseur unique/abstention et tracking dynamique natifs, pas un
« single-box CARI4D » intrinsèque. B : association temporelle multi-instance.
Même observations automatiques gelées, modèles aval et règles de reconstruction ;
seule l'association change. Chaque identité proposée garde sa géométrie/scale.
K et jauge commune sont inférés des RGB/modèles, potentiellement biaisés : aucune
calibration ou échelle GT initiale, ni alignement par frame. Gain relatif seul
insuffisant ; erreur **absolue** objet/main et relative main–objet obligatoire.

Primaire : taux de succès conjoint par séquence sur toutes les frames, identité
correcte ET erreurs3D sous seuils gelés ; mauvaise identité/abstention/gap compte
échec, sans score de pose favorable absent. Rapport séparé : couverture, wrong-ID,
switches, gaps, erreurs conditionnelles et dénominateurs de GT disponible.
Main/contact non annotés ne deviennent pas une vérité inventée.

Seuils d'erreur, pénalités/acceptation, gain minimal, non-régressions par séquence,
budget et traitement des labels absents doivent être fixés **avant évaluation**,
sur le split externe de calibration distinct. Pas de nombres empruntés après
résultats ni tuning sur ces six séquences. Gel/preuves avant privé ; PASS technique,
soutien scientifique et adoption restent séparés. Même succès ne certifierait
ni full-HOI Track1 ni supériorité SOTA/CARI4D. Aucun nouveau transport requis ici.
