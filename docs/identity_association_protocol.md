# Association temporelle multi-instance — proposition scientifique

**4 octobre 2026 : solveur de coûts et features de mouvement implémentés ; protocole externe
PROPOSÉ, pas encore gelé ni exécuté.** Les seuils, modèles,
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

Manquent : exécution des candidats automatiques, coûts d'apparence/points/reprojection,
association à une géométrie canonique, calibration et décision d'abstention.
Forme, échelle, K et jauge sont hors graph, clip-constants. Indices de mains
conditionnels, pas attraction universelle ni rigidité main–objet imposée.

### Features relationnelles implémentées, pas encore mesurées sur vidéo

`relational_motion.relational_motion_features` extrait des caractéristiques 2D
à partir de tracks automatiques **fournis**, par instance main/objet. Un affine
OLS unique est estimé sur les correspondances de fond, partagé par tous les
couples ; ce nuisance image n'est pas une caméra physique. Moins de trois points
ou rang dégénéré ⇒ support absent, jamais mouvement nul crédible de secours.

Résidus médians par instance, contraste de vitesse main–objet et énergies sont
normalisés par diagonale image et Δt réel. Comptages, dispersion, résidus et
condition du fond restent bruts. Toutes les instances/frames et absences sont
conservées ; aucune proximité ne choisit l'identité. OLS est sensible au fond
mobile, aux outliers/parallaxe ; une rotation symétrique peut annuler la médiane.
Les tests exposent ces limites : **189 tests intégrés PASS0.44s**, pas gain vidéo.

L'orchestrateur à callbacks `automatic_candidate_bank` est implémenté :
DINO `hand.`/`object.` fixes, propositions SAM2 et queries Boots par grille de
masque automatique, plus fond hors de l'union de toutes les instances. Pas de
requête cible privée, point oracle RoboTAP ou sélection top1. Les anciens outputs
`automatic_masks` ne contiennent que deux entités sélectionnées : cette banque
n'existe pas dans ces artifacts et ne peut être fabriquée depuis leurs diagnostics
incomplets. Un seul encodeur SAM2 et batch de toutes les boxes ; masques Q=0
conservés, pas de point inventé/refill. Root215 tests combinés PASS0.20s avec
callbacks factices. L'exécution réelle sur RGB et le tracking full-T restent
nécessaires, sans revendication de gain à partir de ces tests.

Choix scientifique proposé : contraste discriminatif de faible dimension appris
sur sujet02 externe **après gel des features automatiques**, puis modèle/règle
d'abstention gelés avant sujet01. La cible privée sert uniquement à entraîner
l'association dans ce split distinct, jamais aux inputs GPU. Sans labels contact,
les régimes de co-mouvement ne doivent pas être nommés contact/prise/relâchement.
Stationnarité ou mouvement caméra commun ne permet pas toujours l'identité.
Un score calibré n'est pas encore un gain de pose absolue.

Ne pas appeler le minimum de chemins `temporal_identity` une vraisemblance HMM
marginalisée : MAP/profilage et log-sum-exp des chemins diffèrent, et des banques
de nuisance de tailles différentes peuvent biaiser le minimum. Comparabilité,
modèle d'absence et calibration doivent être explicitement résolus, pas cachés
par des poids ad hoc. Aucun nouveau checkpoint relationnel n'est encore adopté.

## 2. Tests discriminants avant acquisition

Tests data-free : optimum/min-marginales contre énumération exhaustive,
permutations/ties, influence de futures observations, absence conservée,
concurrents all-missing/distracteurs et overflow/malformed/infeasible.
Ces **coûts synthétiques fournis** sont un oracle de test du solveur, pas des
observations apprises ni une mesure de qualité ou un entraînement ML.
Avant modèle appris, compléter les contrôles croisement, distracteur proche de la main,
relâchement/glissement et symétrie indiscernable ; tester l'évaluateur pour
qu'une abstention ou mauvaise identité ne gagne pas par suppression de frames.

## 3. Cohorte externe proposée : DexYCB

[DexYCB primaire](https://dex-ycb.github.io/), CVPR2021 : scènes multi-objets,
poses rigides et main ; **pas corps entier**, ni objets nouveaux face à YCBV.
Données CC-BY-NC4.0, toolkit GPL3 au
[pin64551b001d360ad83bc383157a559ec248fb9100](https://github.com/NVlabs/dex-ycb-toolkit/tree/64551b001d360ad83bc383157a559ec248fb9100).
MANO/meshes ont leurs droits distincts à vérifier ; modèle/challenge overlap
inconnu. Aucun asset acquis ici. Archives sujet01/02 seulement, tailles Drive
publiques12,412,314,463/12,004,145,048B ; aucun checksum publisher vérifié.
Un SHA acquis sur Azure sera une identité mesurée, pas un hash publisher indépendant.
Pas d'endpoint par séquence identifié ; ne pas télécharger24GB sans nécessité.
`joint_3d[21,3]` natif est en mètres et directement disponible (sentinel−1) :
évaluer les joints ne nécessite pas MANO, contrairement à une surface main.

Plan d'acquisition fixé avant toute annotation : sujets `20200709-subject-01`
(évaluation) et `20200813-subject-02` (calibration), caméra `836212060125`,
indices lexicographiques `[0,16,32,48,64,80]` sur les100 séquences, toutes les
frames natives. Sujet02 : fit `[0,32,64]`, décision `[16,48,80]` ; sujet01 : les
six pour évaluation. Le toolkit impose100 séquences/sujet et groupe objets par
`i//5`, répétitions par `i%5` : premières3–6 séquences étaient un mauvais choix
avant acquisition car quasi un même groupe. Cette règle espacée est décidée
**avant données/labels**, pas après un résultat. Aucun remplacement selon
visibilité, cible ou qualité. Les noms/hashes restent à mesurer ; ce n'est pas
un split officiel unseen ni une preuve statistique avec seulement six clips.

Le toolkit `dex_ycb.py` lit `meta.yml` : les vrais champs sont `ycb_ids` et
`ycb_grasp_ind` (pas `object_ids`/`grasp_ind`). La règle privée figée est : vérifier
IDs uniques et index valide, puis cible=`ycb_ids[ycb_grasp_ind]` et ligne de pose
correspondante. `grasp_eval.py` confirme cette association. Le selector privé
s'exécute après gel complet des deux prédictions, **avant calcul des erreurs**,
sans recherche du meilleur matching. Ces champs, MANO/labels, K/depth capteurs,
objets scannés et autres caméras ne sont jamais fournis au predictor RGB-only.

Avant un pilote couplé, qualifier un pilote **identité/pose objet seulement** :
RGB original et requête automatique `hand.` fixe, pas `person.` ni corps entier
absent du cadrage. Identité cible, abstentions, translation/rotation absolues sans
meshes scannés sont suffisants pour un premier rejet discriminant. Main3D relative
seulement après mapping de prédicteur qualifié. Sujet02 n'est nécessaire que si
calibration de coûts/acceptation doit être apprise ; pas une dépense automatique.
Do-as-I-Do fournit des **mains GT** dans son benchmarkDexYCB/HOI4D pour isoler
l'objet (§4.1), donc ses résultats n'établissent pas le gain RGB-only couplé.

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
sur le split externe de calibration distinct. Si fit manque positifs/négatifs
ou si décision ne discrimine pas, ABSTAIN sans nouveau clip/prompt/feature.
GTseg→candidat en calibration exige une règle préfixée de dominance/pureté ;
cible absente ou masque mélangé reste non-étiquetable et dans le bilan.
Pas de nombres empruntés après résultats ni tuning sur ces six séquences. Gel/preuves avant privé ; PASS technique,
soutien scientifique et adoption restent séparés. Même succès ne certifierait
ni full-HOI Track1 ni supériorité SOTA/CARI4D. Aucun nouveau transport requis ici.

Ablations discriminantes proposées : même banque/visibilité avec nuisance de
fond supprimé ; permutation temporelle des mouvements objet conservant support
et couverture. Une amélioration venant seulement du candidat le plus visible
ne prouve pas une relation main–objet. Les seuils de ces essais ne sont pas gelés.
