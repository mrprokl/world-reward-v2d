# Framework World Reward : recherche, scène partagée, preuves

**Statut : SharedScene, temporal_identity et features relational_motion implémentés ;
extensions de scène et framework appris unifié non encore opérationnels.**
Objectif : améliorer la reconstruction Track 1, pas multiplier les wrappers ou
confondre un pipeline exécutable avec une victoire sur CARI4D. Code et décisions
restent locaux ; médias, modèles, caches et expérimentations restent sur Azure.

## 1. Ce que le dépôt implémente réellement

| Brique | Interface existante | Limite actuelle |
|---|---|---|
| Contrat final | `contracts.Reconstruction.validate`, `submission.Track1Episode` | Pas de scène avec observations et incertitudes. |
| Scène partagée | `shared_scene.SharedScene`, `SceneCamera`, `ObservationRef`, adapters Track1/Reconstruction | API readonly sur reconstruction déjà convertie ; K/gauge fournis, support déclaré, pas de caméra/contacts/confiance appris. |
| Temps original | `timeline.finite_chunks`, `first_occurrence_ownership` | Politique native96 explicite ; pas une fusion temporelle. |
| Identité humaine | `shared_identity.share_first_frame_identity`, `cari_shared_prepare.prepare_geometry` | Politique frame0 fixe, pas identité optimisée multi-vues. |
| Hypothèses objet | `point_candidate_pool.CandidatePool`, `pose_selection.PosePath` | Pool natif25 et sélection ; manque un contrat commun aux autres expériences. |
| Forme partagée | `shape_model.apply_fixed_shape`, `shape_fit.fit_shared_shape`, `shape_selection.select_shape_candidate` | Opérateurs expérimentaux ; pas adoption automatique. |
| Proposition canonique brute | `raw_shape_proposal.RawShapeProposal`, `AutomaticShapeAnchor` | Frontière générique modèle→géométrie : octets natifs F32/F64/int64 et tous les triangles conservés avant certification, références/ancre/gauge déclarées non authentifiées ; aucune inférence ou adoption. |
| Association pose/image | `point_pose_comparison.compare`, `rgb_pose_tracking.track_rgb_pose` | Coût ou suivi image, pas preuve de précision3D. |
| Objectif joint à points persistants | `joint_point_objective.run_joint_point_refinement` | Extension explicite de la classe native : perte originale + reprojection robuste, géométrie/rotation et attachments fixes, poids/échelle externes obligatoires. Dix contrôles Torch et boucle native301 sur état fabriqué PASS ; vrai MHR/contact et validation HOI encore requis. |
| Composition pose commune | `fixed_shape_point_pose.compare_fixed_shape_sequence` | Extraction générique du raccord pool/query/tracks A/B ; callbacks natifs, aucune nouvelle inférence ou amélioration mesurée. |
| Identité temporelle | `temporal_identity.IdentityGraph`, `rank_identity_paths` | Ranking global et min-marginales de coûts fournis ; aucun générateur de candidats/costs appris, identité acceptée ou probabilité. |
| Evidence relationnelle | `relational_motion.relational_motion_features` | Résidus 2D après nuisance affine du fond, comptages/support ; tracks fournis, pas génération/identité/contact/confiance ni caméra physique. |
| Vraisemblance conditionnelle | `relational_likelihood.conditional_relational_likelihood` | Gaussian HMM forward sur innovations fournies, marginales communes, null et poids relatifs ; hypothèses d'indépendance non validées, pas modèle appris/calibration/identité/contact. |
| Diagnostic de bruit partagé | `tracker_noise_null`, lifecycle `tracker_noise_experiment` | Trois inférences natives full-T et évaluation externe gelée exécutées ; transfert Gaussian A/B rejeté sur un des deux clips, aucune adoption. |
| Banque automatique | `automatic_candidate_bank.build_automatic_candidate_bank` | Callbacks DINO/SAM2 : toutes les propositions retenues, un encodeur et batch de masques, queries + fond ; pas encore une exécution de modèles ou une identité acceptée. |
| Mains spécialisées | `hand_observations.HandObservations`, `HandInstances`, `LandmarkEvidence` | Contrat readonly full-T/ragged/21 points, coordonnées et handedness séparés ; pas de modèle exécuté, identité ou visibilité certifiée. |
| Scan de mains | `hand_scan.scan_hands`, `NativeHandResult`, `HandScanResult` | Callback streaming une fois par frame originale, sorties natives et saturation conservées ; aucun modèle importé, paramètre natif ou précision vérifié par la primitive. |
| Pilote natif de mains | `infra/mediapipe_hand_scan.py`, `hand_evaluation.evaluate_hand_clip` | Exécution CPU full-T et diagnostic séparé vérifiés sur3 clips externes/216 frames ;171/177 positives associées, EPE conditionnelle17 joints9.65px. Ni identité/contact/3D ni adoption challenge. |
| Propositions de masques mains | `hand_mask_proposals.propose_hand_masks` | Callback apparié A=box/B=mêmebox+17points, slotsfullT explicites ; pas de tracking ou d'identités physiques. |
| Ablation native de masques mains | `infra/hand_mask_infer.py`, `infra/mediapipe_hand_evaluate.py` | SAM2 apparié full218 frames Dex04 exécuté et gelé avant labels ; Dice0.367→0.376 mais régression d'un clip et contamination objet accrue : gate rejeté, aucune adoption. |
| Mémoire temporelle des mains | `hand_temporal_masks.stream_temporal_hand_masks`, opt-in v3 des mêmes exécuteurs | Full220frames/440 masques Dex05 qualifiés, Dice0.547→0.642 et lacunes50→0 mais contamination objet accrue : gate rejeté, aucun transfert3D/adoption. |
| Géométrie sérialisée | `mesh_serialization.serialization_preflight` | Diagnostic readonly F32/weld8 digits/ties-even et triangles exacts ; pas de compilateur, réparation, fidélité ou qualification runtime. |
| Correspondance de solides | `solid_forest_fidelity.compare_solid_forest_fidelity` | Bijection de composantes par births I/J et forest complet conservé ; prédicats géométriques, Euler, volumes et légalité des collapses restent des gates séparés, pas un compilateur qualifié. |
| Chart numérique | `mesh_conditioning.prepare_conditioning`, opt-in `--conditioned` du même compilateur | Chart source-derived readonly avec inverse physique ; nouvelle sémantique QSlim, quatre contrôles géométriques frais qualifiés, pas de mesh de production ni adoption. |
| Implémentation et transfert mesh | `--conditioned-cache`, `object_budget_conditioned`, `conditioned_geometry_loader`, opt-in du tracker existant | Huit bras procéduraux byte-exacts ; ancien transfert fermé sur source non qualifiée. Ne pas lui attribuer le succès du nouveau whole-solid. |
| Solide entier | `oriented_solid_compiler`, `object_budget_solid`, `solid_geometry_loader`, opt-in strict du tracker existant | Chart-v2 qualifié sur quatre contrôles et première proposition réelle EP9/neuf composantes. Suivi full415 et nouvelle préparation native PASS ; roundtrip indépendant du mesh/poses effectivement F32 qualifié. Échec historique conservé ; aucun gain HOI/adoption. Pins v2 bornés raccordent cette source aux mêmes algorithmes CARI modernes sans alias ou changement de coûts. |
| Lacunes | `occlusion_bridge.initialize_occluded_gaps` | Initialiseur conditionnel ; pas contact observé ni validation RGB complète. |
| Baseline complète | `cari_shared_prepare`, `cari_full_forward`, `cari_full_refine`, `cari_full_export` | Adaptateurs natifs séparés, preuve et exécution souvent imbriquées. |
| Évaluation externe | `point_motion_evaluation`, `point_bop_evaluation`, `ycbv_point_evaluate` | Portée rigide relative ; pas validation HOI globale. |
| Soumission | `submission.assemble_submission`, `verify_roundtrip`, `write_submission` + kit officiel | Schéma/fidélité ne prouvent ni qualité ni licences. |
| Plan/exécution | `pipeline.plan_episode`, `azure_job` | Planner historique forward-only ; transport immuable, pas moteur scientifique. |

Les contrats historiques et leurs preuves restent consultables à leur révision.
La route proposée ne présente ni le planner historique ni les opérateurs purement
numériques comme un framework complet déjà opérationnel.

La route d'estimation reste [un seul objectif joint à points](joint_point_objective.md),
pas un empilement de corrections par vidéo. Sa qualification réelle a échoué à
la disponibilité des attaches ; une [allocation masque générique](mask_query_quantile_protocol.md)
est étudiée séparément avant tout nouveau fit. L'audit du kit a aussi identifié
une surcontrainte propre : la fermeture solide n'est **pas** une exigence
universelle Track1. Une [branche surfacique fidèle](surface_lod_protocol.md)
séparée doit qualifier le domaine ouvert sans modifier les anciens résultats.
La priorité immédiate est donc l'admissibilité/représentation et les observations,
non l'achat d'un nouvel empilement de modèles. Le [rééquilibrage des sommes exactes](balanced_solid_sum.md)
traite séparément la profondeur numérique du compilateur ; il ne change ni la
géométrie ni les critères d'acceptation. Les qualifications anciennes ne qualifient
pas ce nouveau source : quinze contrôles puis quatre compilations géométriques
fraîches sont nécessaires avant une nouvelle proposition de production.

La frontière `RawShapeProposal` ne modifie ni compacte les tableaux : faces
répétées/dégénérées et vertices inutilisés restent disponibles pour le validateur
séparé. Son hash porte les octets et déclarations, pas une preuve d'origine ou de
solidité. Elle n'impose pas le budget final au générateur et ne présente pas les
unités natives comme des mètres. La certification, le grounding métrique, le
tracking et l'ajustement joint restent des étapes explicites ; ce contrat ne les
remplace pas. Aucun backend alternatif réel n'est encore exécuté par cette API.

Deux primitives génériques supplémentaires existent maintenant :
`surface_identity.prepare_surface_identity` valide uniquement le domaine
vertex-manifold ouvert/fermé sous-budget, avec births identité et tableaux
natifs byte-immuables, sans volume/weld/repair. La readiness F32 est un diagnostic
séparé ; aucun loader/packer réel ni embedding n'est qualifié par ce contrat.
`point_surface_queries.canonical_mask_quantile_queries` partage le noyau de raycast
historique, alloue32 quantiles du masque frame0 avant profondeur et conserve
les diagnostics de tous les slots. Moins de8 témoins ou ambiguïté ⇒ FAIL, sans
refill. La route legacy et les jobs historiques ne sont pas changés/adoptés.
Root289tests PASS ; la mesure est de disponibilité/représentation, pas de qualité3D.

### Next integration gate, not a new model stack

The useful seam is `joint_point_native_qualify.numerical_control`: allocate
mask quantiles on the actual native-loaded F32 mesh using the original PNG,
inferred K/depth and constructor's frame0 object state. Freeze face IDs and
barycentrics before any future tracker result; bind unchanged full-T Boots
observations through `joint_point_evidence`. The legacy `fixed_shape_point_pose`
selector must not be implicitly substituted, and the failed EP21 cohort stays
closed. No per-video rescue or point-weight choice is justified by availability.

Before a positive-weight experiment, a fresh preregistered native-bundle control
must exercise real MHR/contact/render kernels and the original versus weight0
subclass pair: identical seeds/config, loss/gradient probes0/181, all301 updates,
byte-exact state/history/result. Supplied manufactured bundles qualify runtime,
not inferred RGB geometry or held-out HOI accuracy. A subsequent real tracker
control remains separate; previous authored/analytic cohorts are not fresh
validation. Do not manufacture more wrappers while this underlying seam is
unqualified. Full-HOI calibration/held-out rights remain unresolved; see the
October5 targeted licence audit. No positive weight or backend adoption yet.

### Première migration réalisée

`world_reward.shared_scene` ajoute des adaptateurs explicites
`from_track1_episode`, `from_reconstruction`, `to_track1_episode`. K fourni,
convention pixel et gauge déclarées ; arrays copiés readonly sans alias,
scale/mesh non rebakés, provenance tiny deep-freeze, support d'observation
manquant conservé. Layout MHR officiel, toute la timeline et SE3 restent
validés par les contrats existants. Root129PASS1optionalKITskip0.51s.
Cela n'est ni un estimateur de caméra, ni un fit, ni un système intelligent
complet ; roundtrip logique/byte-exact seulement, sans nouveau gain3D revendiqué.

## 2. Un flux scientifique, trois frontières

`Research → Evidence → Hypotheses → SharedScene → Fit → Validate → Submit`

- **Research** : question falsifiable, sources primaires, licence/chevauchement,
  budget, baseline, ablations et critère d'arrêt avant acquisition/inférence.
- **Evidence** : observations automatiques brutes, provenance et unités ; séparer
  exécution vérifiée, validité géométrique, gain mesuré et éligibilité juridique.
- **Hypotheses** : propositions génériques de forme, identité, caméra, pose,
  visibilité/contact ; pas de correction manuelle d'un épisode du challenge.
- **SharedScene/Fit** : même scène et observations pour chaque branche ; optimiser
  des variables autorisées, en conservant le temps original et un seul repère.
- **Validate** : gel des prédictions et preuves avant accès privé ; évaluateur
  distinct sans capacité d'injecter ses labels/caméras/masques dans l'inférence.
- **Submit** : branche adoptée seulement, packing fidèle et règles/quota vérifiés.

Frontières : modèle→observations, hypothèse→scène ajustée, prédiction gelée→vérité.
Le contrôleur transporte du code et ordonnance des ressources ; il ne sélectionne
pas une hypothèse parce qu'un job ou une vérification d'intégrité a passé.

## 3. API `shared_scene.v1` existante et extensions proposées

L'API actuelle porte `SceneCamera`, une `Reconstruction` MHR convertie, le mesh
objet, l'expression, la provenance et des `ObservationRef` avec support full-T.
Elle ne contient pas tous les champs riches ci-dessous : timestamps, contrôles
natifs typés, scores/confiances calibrés, contacts et banque multi-instance sont
des **extensions proposées**, pas des interfaces déjà exécutables.
`temporal_identity` reste distinct : il classe une banque de couples persistants
acteur/objet sur des observations référencées, sans fabriquer de géométrie.
Voir `docs/identity_association_protocol.md` pour la portée et la validation
encore proposée de cette primitive.

`automatic_candidate_bank` réalise désormais l'orchestration frame0 via callbacks
de modèles : requêtes fixes `hand.`/`object.`, NMS classwise existante (.3/.7),
tous les candidats retenus sans top1/cap/nearest, un `SAM2.set_image` puis un
`predict` batch. Queries 4×4 par masque et fond 8×8 hors union de tous les masques,
sans refill ; les masques vides restent Q=0. Coordonnées originales continues
`[0,row+.5,col+.5]`, copies readonly ; scores bruts non calibrés. Root215 tests
combinés PASS0.20s. Ce sont des tests à callbacks factices, pas une exécution GPU,
une licence/absence d'overlap certifiée ou un gain d'identité/3D.

`hand_observations` sépare les 21 coordonnées image de leurs unités/support
numérique, du Z relatif au poignet et du XYZ centré-main. Toute frame originale
possède un record, même vide ; tous les slots sont conservés sans les nommer
identités persistantes. Handedness n'est jamais confiance de détection/visibilité.
L'adapter natif normalisé applique seulement xW/yH, aucun mirror/offset/crop,
conserve les valeurs hors grille/non supportées et ne crée pas de jauge globale.
Root152 tests de contrats PASS0.11s ; aucun run MediaPipe ni gain appris.

`hand_scan` fournit l'orchestration streaming commune à une future exécution :
une seule requête par frame RGB originale, aucun retry/stride/flip/crop, erreurs
propagées, slots natifs sans identités, conservation des XYZ normalisés et du
world centré-main. Le budget de la primitive ne couvre pas les hashes ou exports
du caller ; le budget du pilote doit les inclure séparément. Les valeurs du
protocole IMAGE/4/.5 sont déclarées, pas vérifiées dans un callback opaque.
Root138 tests combinés PASS0.12s ; pas de scan MediaPipe réel ou d'adoption.

| Champ | Invariant |
|---|---|
| `clip`, `frame_index[T]`, `timestamps[T]` | Identité source ; `arange(T)` original, horodatages dérivés sans re-numérotation. |
| `camera` | K clip-constant RGB-inféré, dimensions et convention pixel explicites ; transformations proprement orientées, jamais calibration source interdite. |
| `gauge` | Unités/convention d'axes, scalaire positif constant et application unique déclarés ; mètres déclarés ne prouvent pas une gauge exacte. |
| `body.identity` | Forme et échelles clip-constantes avec `layout_id` ; PCA28 natif et scales68 officiels ne sont pas interchangeables. |
| `body.motion` | Paramètres/SE3 sur T complet, modèle/décodeur identifiés ; joints nommés avec mapping et repère explicites. |
| `object.geometry` | Une géométrie canonique, faces/orientation, échelle appliquée une fois et qualification topologique ; pas de maillage par frame. |
| `object.motion` | SE3 propre sur T complet dans le même repère que l'humain ; pas de scale/shear dans les poses. |
| `observations` | RGB-références, masques, points, profondeur, mains, tracks ; chaque array déclare indices, grille, validité et producteur. |
| `uncertainty` | Scores bruts, support et visibilité séparés ; confiance calibrée seulement si protocole externe gelé l'établit. |
| `contacts` | Hypothèses main/objet et evidence associée ; stabilité relative ou recouvrement ne sont pas une preuve de contact. |
| `provenance` | Références hashées sources/modèles/protocoles/artifacts ; preuves volumineuses restent sur Azure. |

Une observation absente reste absente : pas de NaN remplacé par zéro crédible,
pas de point interpolé marqué observé. Un état intermédiaire peut être incomplet ;
une reconstruction finale exige toute la trajectoire finie et ses incertitudes.
Les lacunes initiales/finales non identifiables ne déclenchent pas un objet statique.
Raw et packed sont deux artifacts distincts ; le second exige un gate de fidélité
de géométrie et des poses, pas une substitution silencieuse du premier.

## 4. Hypothèses, scoring et ajustement

Une proposition contient `parent_scene`, `method_id`, `config_id`, variables
modifiées, coût/budget et diagnostics. Les candidats sont comparables parce qu'ils
partagent observations, indices, géométrie de référence et protocole de mesure.
Le fit conserve l'identité/forme/K/gauge comme variables de clip, et les poses
comme variables temporelles ; tout changement de K impose une chaîne géométrique
cohérente, jamais une simple correction de translations sauvegardées.

Objectifs admissibles : reprojection, silhouette visible, profondeur inférée,
photométrie/features, tracks, indices de mains et régularisation temporelle.
Contact/anti-pénétration s'ajoutent avec evidence et géométrie qualifiée ; éviter
l'attraction systématique main→objet, le rétrécissement ou la dérive hors interaction.
Conserver aussi les résidus non régularisés : un coût plus faible peut cacher une
géométrie fausse. Changer de solveur/pondération constitue une ablation scientifique.

Apprentissage des poids/confiances et choix des hyperparamètres **hors challenge**,
sur splits distincts scène/objet/personne, puis calibration/validation séparées.
La sélection sur frames tenues à l'écart de la même vidéo est un diagnostic
self-supervisé, pas une précision held-out démontrée. Rien ne retourne du privé.

L’export immuable et son consommateur ont des provenances distinctes : une
évolution des wrappers ne justifie ni de régénérer les prédictions ni de
réécrire l’historique. Le consommateur authentifie la closure Git réelle du
producteur, puis vérifie séparément la compatibilité de ses opérateurs
numériques actuels. Les receipts d’échec restent immuables ; lorsqu'un consommateur
natif impose un chemin canonique, un échec partiel peut être archivé par renommage
atomique sans écrasement, après arrêt prouvé et inventaire hashé avant/après.
Il ne devient jamais une entrée PASS. Tout replay technique explicitement motivé
dispose d'une sortie fraîche et d'une révision identifiée, sans reprendre le partiel.

La géométrie sérialisée et celle effectivement lue sont distinctes. Trimesh5.1.0
projette implicitement certaines matrices de rotation F32 par SVD, même avec
`process=False`. Pour la route solide, authentifier le code installé, démontrer
exactement POSITION/indices → matrice brute A puis graphe natif B=`fix_rigid(A)`
→ triangles chargés → vertices natifs F32. Conserver metadata A et poses natives,
jamais inverser B pour améliorer le résultat. Vérifier toute la timeline dans le
repère caméra avec les vertices et poses réellement F32 et la borne héritée1e-5m.
Déclarer explicitement cette projection native ; aucun arrondi supplémentaire,
modification de surfaces/connectivité, relaxation de seuil ou gain3D n'est déduit.

## 5. Gates et expérimentation parallèle

Un `ExperimentSpec` proposé déclare cohorte, entrées autorisées, baseline,
config/version, hypothèses, ressources, métriques et arrêt. Un `GateResult`
référence evidence + décision + portée ; pas un unique `PASS` universel.
Séparer gates **intégrité**, **numérique/couverture**, **géométrie**, **qualité**,
**licence/règles** et **packing/soumission**. Seuils globaux/protocolaires, jamais
lookups par épisode. Les pins historiques restent spécifiques pour l'authenticité,
mais ne deviennent pas des constantes d'algorithme ni des scores de candidats.

Paralléliser d'abord audits et ablations disjointes sur scène/observations gelées.
Un stage appris coûteux produit une fois des artifacts immuables ; cache identifié
par sources/modèles/config/entrées/indices/conventions, pas simplement épisode.
Ordonnancer GPU exclusif par VM et CPU borné séparément ; libérer GPU avant
encodage/évaluation CPU. Aucun job dupliqué, retry implicite ou fusion partielle.

## 6. Matrice d'ablations externe, à geler avant valeurs privées

| Axe | Comparaison minimale | Cas/gate incontournable |
|---|---|---|
| Forme | Mesh natif fixe vs proposition multi-anchor choisie automatiquement | Géométrie irrégulière, cavités/composants ; précision surface + silhouettes, pas baisse de PEN par shrink. |
| Pose | Même pool natif + coût image vs ajout tracks automatiques | Mouvement accéléré/rotation/ambiguïtés ; erreur3D relative sur T entier, no-regression par strate. |
| Mains | Body natif vs observation indépendante + fit confidence-aware | Masque humain IoU élevé mais doigts erronés ; repère/joint mapping/support vérifiés avant contact. |
| Profondeur/gauge | Gauge humaine fixe vs estimation commune autorisée | Biais focal/Z et support perdu ; erreur absolue + couverture, aucun alignement par frame. |
| Occlusion | Baseline vs initializer+fit commun avec evidence bilatérale | Carry/free/slip/regrasp, gaps et extrémités ; aucune deletion/static rescue, uncertainty et erreur sur frames cachées. |
| Contact | Fit sans contact vs evidence-gated contact/SDF | Non-contact, saisie, glissement ; gain de précision et interactions conservées, pas PEN seul. |

Fixtures idéales testent les opérateurs, non les observations apprises. EP4 masque
absent, pilot mains `57d109c` rejeté, échecs forme/topologie et ambiguïtés gauge
délimitent les strates nécessaires : ne pas recycler leurs labels pour sauver
une branche. YCB rigide et TUM profondeur ne suffisent pas à certifier les mains
ou HOI. Toute nouvelle cohorte doit être autorisée et screened licence/overlap.
Critères de non-régression, couverture et budget sont gelés par protocole ; un
FAIL reste FAIL. Une reprise technique conserve les artifacts et sa justification.

## 7. Migration incrémentale, pas réécriture générale

1. Documenter la route complète actuelle et ses vrais adapters ; ne pas exécuter
   le vieux planner comme s'il contenait déjà shared-prepare/refinement/export.
2. Ajouter un adapter en lecture seule vers SharedScene, commencer par baseline
   gelée et tests de roundtrip exacts sur arrays/indices/conventions.
3. Regrouper nouveaux orchestrateurs dans un seul catalogue de stages typés :
   validate-inputs / execute / validate-outputs / seal ; garder les wrappers natifs.
4. Une preuve réutilisable par artifact remplace les validateurs copiés ; conserver
   l'authentification indépendante du producteur historique, distincte du consumer.
5. Migrer une ablation à la fois ; test numérique + un benchmark réel borné avant
   adoption. Aucun budget gagné par gate supprimé ou math modifiées sans mesure.
6. Marquer anciens launchers dépréciés sans toucher aux sources ayant produit des
   preuves ; nettoyer seulement bruit jetable, conserver décisions et références.

Priorité : une ablation externally validée de pose sur observations réellement
inférées, puis mains/occlusion si cohorte recevable ; pas un nouveau sous-framework
de transport. Une seule scène/ledger minimal doit rendre les échecs comparables.
Le résultat final reste un Parquet gelé pour les cinq compétitions **World Reward**,
avec commit accessible, règles acceptées et quota vérifié ; aucune victoire
annoncée à partir d'un proxy, d'un gate CPU ou d'un packing PASS.

### Audit complémentaire : prior adaptatif dans les propositions, pas guidance3D

Le [code Do-as-I-Do pré-cutoff](https://github.com/malik-group/do-as-i-do/tree/824591b808c342b20079c3b4198a8c2bdf88c74e)
isole un cœur Fast-SAM3D sans mains, mais ses points règlent la force du prior via
une rotation **2D image**, pas une perte géométrique de reprojection. Son CLI
interpole des vitesses mal supportées, saute des masques vides et peut lire des
extrinsèques HDF5 : il ne respecte pas notre full-T et ne sera pas importé tel
quel. Le [papier](https://arxiv.org/abs/2606.19333) emploie des mains GT dans
son évaluation HOI, donc ne prouve pas le gain RGB-only recherché.

Hypothèse distincte, **différée** : prior précédent constant contre prior modulé
par mouvement image observé, à forme/échelle/caméra/observations identiques.
Avant tout run, prouver la compatibilité exacte des poids/configurations et
l'injection explicite de nos pointmaps/K, sans nouveau modèle caché. Aucun
modèle forké non audité, oracle, frame supprimée ou interpolation présentée
comme observation. Le code racine et le fork sont MIT ; droits des checkpoints
et overlap restent séparés et non vérifiés. Source principale Fast-SAM3D
`track_object.py` au pin823d4784dbf73a9a58ca8b2384dee1eaf8fc1c6b :83898B,
SHA136531d4248fd2f57c1f812d9a770839bf2de5d91eb48d14efe81f3185001cca.
Pas d'expérience ou gain revendiqué ; le diagnostic ne justifie pas une nouvelle
stack de wrappers. Réutiliser d'abord les primitives actuelles de scène/pool/points.

Le raccord déjà présent dans l'expérience YCB est maintenant extrait dans
`fixed_shape_point_pose`, sans modifier cet ancien producteur ou rouvrir sa
cohorte : géométrie/caméra/gauge unique, temps original et sourceIDs explicites,
queries automatiques figées, pool natif25 complet avant un unique callback de
tracks. A/B partagent le même pool et les opérateurs/poids existants ; pas de
feedback tracks→seeds, nouveau score, PnP, Gaussian ou fallback statique.
Copies readonly et preuves de mutation encadrent les callbacks. Les pointmaps
streamées ne sont pas retenues : weakrefs vérifient les arrays encore vivants ;
seuls RGB restent présents jusqu'au callback natif. Le test générateur/GC vérifie
ce contrat mémoire. Root223 tests combinés PASS0.94s, dont40 nouveaux tests ;
aucun modèle/Azure/cohorte/mesure3D. `<40` observations par frame reste un échec :
la composition ne résout pas seule l'occlusion, l'identité ou la jauge absolue.


## 8. Next integrated experiment decision — October4

The fixed-chart compiler has passed four procedural geometry controls, not a
production/adoption gate. Before large predicted meshes, evaluate an exact
transactional physical-coordinate cache against the qualified whole-decode
implementation: native collapse changes only its two endpoints, but source
numeric roundtrip is not a byte proof. Initialize cache by the same inverse,
keep native queue/AABB/volume/serialization callback order unchanged, recompute
full final physical geometry, and require frozen source/prediction/counter/birth
parity. This is a technical scaling hypothesis, not another mesh-quality gain.
That implementation regression has now passed eight real-collapse arms on the
four previously successful procedural sources; the immutable qualified cached
binary is retained on Azure. See mesh_conditioned_cache_protocol.md. This
establishes measured byte parity for those controls, not production mesh or HOI
accuracy. The sole marker-only technical replay authenticated its inputs but
rejected the production source component arrangement before QEM. That scientific
failure is closed. A readonly diagnostic found nine manifold components (six
outward, three inward), not a certificate that these are valid cavities. The
new general oriented-forest adjudicator preserves every component and requires
independently certified embedding/containment; it cannot repair an invalid seed.

The next substantive reconstruction hypothesis is **joint persistent actor/object
association plus explicit visible ownership** over one shared automatic evidence
bank, followed by unchanged CARI initialization/refinement. Appearance, point
correspondences and rigid reprojection supply global costs; relational co-motion
is secondary evidence, not contact or object truth. Ambiguous hand/object overlap
stays unknown instead of subtracting intersections. Missing observations stay
missing and all original frames remain in the scene. A real cost producer is
required: another graph contract alone is not an experiment.

Current hand-memory results motivate this intervention but do not validate it:
better availability also increased object contamination. Use three fresh full
single-camera HOI recordings selected by a naming rule before references,
freeze both automatic predictions, then a separate evaluator measures coverage,
wrong-ID, human/object error and object-to-wrist relative error. Native topology
or 2D tracking PASS must not be called measured HOI gain. All source/body/object
mappings, private reference provenance and lawful competition-related evaluation
rights must be verified before acquisition; no qualified cohort is currently
available. InterCap requires account access and explicit clarification of its
noncommercial/research terms for this context. No registration or request has
been sent. No model, rendering stack or per-episode threshold is added merely
because validation access is missing.

### Statistical boundary for the association experiment

The existing temporal_identity primitive ranks paths by MAP costs. It is not a
marginal likelihood or a calibrated posterior. A proposed learned relation model
must use the same complete observation bank, background nuisance, K and geometry
for every candidate pair. Its binary temporal latent represents statistical
dependence, not contact. The new conditional Gaussian primitive marginalizes
regimes with forward log-sum-exp and an explicit null. Every hypothesis shares
the whole bank and the same per-entity means/covariances; only the paired joint
cross-covariance differs. Block-diagonal finite-df Student-t is not an independence
null because its radial scale is shared, so that proposed shortcut is not used. Missing
dimensions contribute their marginal emission, not an invented zero displacement;
absence/clutter laws and normalization must be common across candidate pairs.
Duplicate nuisance representations cannot increase probability mass.

Calibration/abstention and final evaluation need distinct external splits with
persisting instance IDs and automatic proposal-to-instance associations. Current
hand slots do not identify full-body actors: without an automatic measured
body-wrist association, the proposed experiment is conditional hand-object only.
Keep stationary/common-motion/symmetric ambiguity explicit, with no nearest
fallback. The implementation is mathematically tested, not empirically qualified:
OLS camera error creates shared residual covariance and Boots/adjacent differences
remain autocorrelated. Conditioning on the same camera estimate does not prove
independence. A causal innovation/noise model, external negatives, lawful training
data and separate calibration/evaluation are still required. Direct relative
log weights cancel common bank factors algebraically, not by subtracting enormous
absolute likelihoods; they are neither normalized nor calibrated probabilities.

### Two source-level routes, not a generator shopping list

1. Raw TRELLIS is an independent canonical-mesh proposal, not a repair of SAM.
   The pre-cutoff source is MIT; use raw `outputs['mesh'][0]`, not its default
   simplify/GLB postprocessor. FlexiCubes does not itself certify a closed,
   consistently oriented embedded solid. Checkpoint terms and challenge overlap
   remain unverified. No checkpoint downloaded or native run performed.
2. On an already valid seed, jointly fit a single bijectively deformed canonical
   shape and full-T proper SE3 using the existing automatic observations and
   renderer. Keep the human/shared gauge and K fixed; visible silhouette,
   robust inferred depth, safe free-space and reserved point evidence are
   complementary signals. A positive-Jacobian deformation cannot fix a bad
   seed topology. BundleSDF is methodological inspiration, not a drop-in:
   its experiment uses real RGB-D and its code has noncommercial restrictions.

Hunyuan3D-2.1 is not an eligible shortcut for this France-based user: its primary
licence excludes the EU and also restricts outputs outside its territory.
4DAnyone remains a human-only sequence-reference inspiration, not an object
geometry/interaction solution. Never enable bundled mesh cleanup as if it were
source-generation novelty. Fresh procedural controls test operators first;
lawfully qualified real full-HOI evaluation is still required for an accuracy
claim and adoption.

For the cheaper relation hypothesis, specialist wrist/palm point queries plus
one full-video Boots inference avoid the rejected SAM2 hand-mask contamination.
Use SAM2 for objects only. Fresh DexYCB subjects06/07 were identified before
acquisition, but rights are unresolved and those references cannot establish
full-body actor identity. This is a conditional target-selection experiment,
not persistent person association or full-HOI validation. The modern SAM2 API
can add automatic births between propagation slices on the same state; it does
not decide whether a birth is a new physical instance or a duplicate/reappearance.
Neither this audit nor another contract substitutes for an actual paired run.

Two attributable comparisons: same-bank independence versus learned dependence;
then identical relation model without/with rigid reprojection on reserved queries
not used for pose fitting. First require lower wrong-ID without reduced full-T
coverage, then paired object and object-to-wrist 3D gains under the same downstream
CARI fit, without human regression or per-frame alignment. A 2D residual alone
cannot establish a reconstruction-quality gain.

The first actual homogeneous tracker-noise pilot is now **closed REJECT**:
the frozen Gaussian4 composite model gains density on one heldout clip but loses
on the second. It is not an adopted correction or learned association prior.
Do not tune that cohort, pool away its regression or use its descriptive shift
contrast as a new acceptance gate. Dependence-aware association remains a
research hypothesis requiring a substantively new model, fresh validation and
actual paired wrong-ID/3D/coverage evidence before integration.
