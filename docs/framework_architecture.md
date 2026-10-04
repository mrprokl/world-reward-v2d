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
| Association pose/image | `point_pose_comparison.compare`, `rgb_pose_tracking.track_rgb_pose` | Coût ou suivi image, pas preuve de précision3D. |
| Identité temporelle | `temporal_identity.IdentityGraph`, `rank_identity_paths` | Ranking global et min-marginales de coûts fournis ; aucun générateur de candidats/costs appris, identité acceptée ou probabilité. |
| Evidence relationnelle | `relational_motion.relational_motion_features` | Résidus 2D après nuisance affine du fond, comptages/support ; tracks fournis, pas génération/identité/contact/confiance ni caméra physique. |
| Banque automatique | `automatic_candidate_bank.build_automatic_candidate_bank` | Callbacks DINO/SAM2 : toutes les propositions retenues, un encodeur et batch de masques, queries + fond ; pas encore une exécution de modèles ou une identité acceptée. |
| Mains spécialisées | `hand_observations.HandObservations`, `HandInstances`, `LandmarkEvidence` | Contrat readonly full-T/ragged/21 points, coordonnées et handedness séparés ; pas de modèle exécuté, identité ou visibilité certifiée. |
| Scan de mains | `hand_scan.scan_hands`, `NativeHandResult`, `HandScanResult` | Callback streaming une fois par frame originale, sorties natives et saturation conservées ; aucun modèle importé, paramètre natif ou précision vérifié par la primitive. |
| Pilote natif de mains | `infra/mediapipe_hand_scan.py`, `hand_evaluation.evaluate_hand_clip` | Exécution CPU full-T et diagnostic séparé vérifiés sur3 clips externes/216 frames ;171/177 positives associées, EPE conditionnelle17 joints9.65px. Ni identité/contact/3D ni adoption challenge. |
| Propositions de masques mains | `hand_mask_proposals.propose_hand_masks` | Callback apparié A=box/B=mêmebox+17points, slotsfullT explicites ; pas de tracking ou d'identités physiques. |
| Ablation native de masques mains | `infra/hand_mask_infer.py`, `infra/mediapipe_hand_evaluate.py` | SAM2 apparié full218 frames Dex04 exécuté et gelé avant labels ; Dice0.367→0.376 mais régression d'un clip et contamination objet accrue : gate rejeté, aucune adoption. |
| Mémoire temporelle des mains | `hand_temporal_masks.stream_temporal_hand_masks`, opt-in v3 des mêmes exécuteurs | Full220frames/440 masques Dex05 qualifiés, Dice0.547→0.642 et lacunes50→0 mais contamination objet accrue : gate rejeté, aucun transfert3D/adoption. |
| Géométrie sérialisée | `mesh_serialization.serialization_preflight` | Diagnostic readonly F32/weld8 digits/ties-even et triangles exacts ; pas de compilateur, réparation, fidélité ou qualification runtime. |
| Chart numérique | `mesh_conditioning.prepare_conditioning`, opt-in `--conditioned` du même compilateur | Chart source-derived readonly avec inverse physique ; nouvelle sémantique QSlim, quatre contrôles géométriques frais qualifiés, pas de mesh de production ni adoption. |
| Lacunes | `occlusion_bridge.initialize_occluded_gaps` | Initialiseur conditionnel ; pas contact observé ni validation RGB complète. |
| Baseline complète | `cari_shared_prepare`, `cari_full_forward`, `cari_full_refine`, `cari_full_export` | Adaptateurs natifs séparés, preuve et exécution souvent imbriquées. |
| Évaluation externe | `point_motion_evaluation`, `point_bop_evaluation`, `ycbv_point_evaluate` | Portée rigide relative ; pas validation HOI globale. |
| Soumission | `submission.assemble_submission`, `verify_roundtrip`, `write_submission` + kit officiel | Schéma/fidélité ne prouvent ni qualité ni licences. |
| Plan/exécution | `pipeline.plan_episode`, `azure_job` | Planner historique forward-only ; transport immuable, pas moteur scientifique. |

Les contrats historiques et leurs preuves restent consultables à leur révision.
La route proposée ne présente ni le planner historique ni les opérateurs purement
numériques comme un framework complet déjà opérationnel.

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


## 6. Next integrated experiment decision — October4

The fixed-chart compiler has passed four procedural geometry controls, not a
production/adoption gate. Before large predicted meshes, evaluate an exact
transactional physical-coordinate cache against the qualified whole-decode
implementation: native collapse changes only its two endpoints, but source
numeric roundtrip is not a byte proof. Initialize cache by the same inverse,
keep native queue/AABB/volume/serialization callback order unchanged, recompute
full final physical geometry, and require frozen source/prediction/counter/birth
parity. This is a technical scaling hypothesis, not another mesh-quality gain.
No such cache or persistent native install exists at this revision.

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
