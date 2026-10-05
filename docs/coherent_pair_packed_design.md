# Marginale cohérente compacte : architecture proposée

2026-10-05 — aucun code GPU, coût, FIT ou hyperparamètre qualifié ici.
La recette hard-max512 sous720s reste **fermée, coût non qualifié**. Cette
architecture vise la nouvelle marginale, pas un sauvetage du contrôle.

## Préparation unique, indépendante des coefficients et références

Entrée : `PairCache` authentifié, toutes les propositions P/O et routes K.
Partager les facteurs FP64 base[P,2,O,6], local[P,2,K,4], bridge[K,O,2],
leurs supports, cinq indicateurs et les marges natives. Les NaN bruts restent
dans la preuve ; contribution masquée zéro ne signifie pas anatomie observée.

La clé complète de l'oracle est une concaténation, pas une somme :12 valeurs
**brutes masquées**,12 supports,5 indicateurs et présence de route. Donner des
IDs canoniques aux blocs base/local/bridge, puis dédupliquer les composites
`(base_id, local_id, bridge_id, route_bit)`. Égalité des octets complets,
zéros signés canonisés ; jamais hash seul ni fusion liée aux échelles FIT,
theta ou alpha. Des valeurs brutes distinctes restent distinctes même si leur
coefficient normalisé est désactivé. Classes/marges ne définissent pas la
géométrie. Sans route utilisable : état base, route_bit0 ; pas route fictive.

CSR : paire de groupes de coordonnées → côtés supportés → géométries
distinctes → marges natives distinctes conditionnelles. Conserver chaque slot,
ID natif et correspondance route→état. Les vues natives et groupes peuvent
partager les tables mais ont leurs propres incidences : ne pas répliquer le
score groupé sur un alias dont les observations diffèrent. Alias géométriques
sans masse supplémentaire, **sans prétendre une identité physique**.
Sans ancres supportées : score NaN/support faux, jamais OFF.

## Distribution, score et dérivées

Pour une paire, prior
`pi(s,g,m)=1/n_sides * 1/n_geom(s) * 1/n_margin(s,g)` sur ses seuls états
supportés. La réduction hiérarchique normalisée est
`S=tau*log(sum pi*exp((theta.x + alpha*m)/tau))`, tau>0, alpha>=0.
Assembler local et bridge de **la même route** avant réduction. Aucun topK,
max de marges, poids de confiance inventé ou LSE non normalisé.

Le posterior permet les VJP `dS/dtheta=E[x]`, `dS/dalpha=E[m]` : agréger les
poids vers les trois tables de facteurs, sans matérialiser chaque vecteur17.
À alpha0, A/B partagent score et VJP géométrique. La dérivée droite vaut la
moyenne posterior des moyennes des marges distinctes par géométrie. Un simple
`if alpha==0: return A` déconnecte alpha : prévoir VJP analytique ou autograd
personnalisé correct, testé, pas un gradient substitutif.

Réductions segmentées stables et ordre/chunks fixes ; aucune accumulation
atomique CUDA non déterministe supposée acceptable. FP64 sans AMP/TF32.
Les clés, supports, IDs et masse sont exacts. Les produits/réductions Torch
peuvent différer de NumPy `x@theta` : **aucune tolérance CPU/GPU choisie ici**
ni promesse bit-exacte inter-runtime. Choisir le contrat numérique avant tout
contrôle ; A/B alpha0 utilisent un seul chemin dans le runtime candidat.

## Coût, API et qualification avant FIT

P4/O3600/K64 donne1,843,200 routes. Les17 doubles seuls représentent239MiB,
mais millions d'objets Python et duplication native/groupes aggravent l'oracle.
Facteurs et indices CSR donnent un stockage O(P2O+P2K+KO+routes), avec mémoire
de travail par bloc ; pas tenseur global G×K×17. H10095GB rend ce stockage
plausible, **pas son coût démontré**. Trier les composites/incidences une fois.

API minimale proposée : `prepare_marginal_packed(cache)` puis
`score_and_vjp(packed, theta, temperature, alpha)` avec mêmes sorties natives,
groupées, supports et provenance que `coherent_pair_marginal`. Constructeur
authentifié contre la source, pas facteurs auto-certifiés. Réutiliser cet
oracle et le lifecycle existant, pas nouvelle stack/transport.

Avant toute référence FIT : énumération minuscule indépendante, FD17/droite0,
côtés absents, marges inégales, alias/permutations, supports/échelles désactivées,
tables forgées ; puis contrôle neuf H100 full-native/groupes, préparation,
objectif/VJP, mémoire et répétabilité. Objectif complet image-équilibré :
LSE tous groupes moins LSE positifs +L2 ; inconnus pas négatifs certifiés.

Future optimisation non convexe : A full-batch L-BFGS strong-Wolfe, B scalaire
borné avec KKT projeté, géométrie gelée. Budget, tolérances, température,
régularisation et arrêt **non gelés ici** ; à déclarer avant FIT. Nonfinite,
ligne échouée ou budget épuisé ferme la recette sans restart. Stationnarité,
PASS mécanique et capacité ne prouvent ni ownership, contact ni qualité 3D.

## Préparation compacte réalisée, qualification minuscule seulement

`coherent_pair_packed` implémente la préparation coefficient-free : trois
tables brutes/supports/normalisées, composites et CSR, refs natifs complets,
constructeur uniquement sur PairCache authentifié. Aucun scorer/Torch/FIT.
16 contrôles dédiés passent ; root162 combinés PASS1.89s. Audit indépendant
sur5548B fabriqués reconstruit les clés depuis l'evidence originale, pas le
helper de préparation :18 refs, marges inégales, alias, zéros signés, supports,
échelles désactivées, absences et rejet de tables forgées PASS. Cela n'établit
ni débit/mémoire full-bank, ni performance GPU ou précision de sélection.
