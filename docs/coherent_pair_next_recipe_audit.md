# Audit : prochaine recette cohérente, pas sauvetage d'un essai fermé

2026-10-05 — proposition non gelée, aucun FIT exécuté.

## Décision

Six comparaisons cache/original bit-exactes, projection au pire passage36389s :
**recette512 sous720s non qualifiée**, pas hypothèse relationnelle réfutée.
La mesure n'est pas une borne de durée/convergence. Garder anciens résultats et
gates fermés ; ne pas supprimer des3600 objets.

Priorité : nouvelle recette différentiable sur le même graphe. Défaut précis :
hard-max non lisse et milliers de gradients de routes ex aequo assemblés en
Python. Hard-max strictGPU reste possible ; marginalisation ci-dessous =
**autre hypothèse**, pas optimisation bit-exacte.

## Marginale latente et masse déclarée

Pour chaque groupe automatique de coordonnées P/O, conserver tous les slots,
deux côtés latents et chaque route complète. Avec `g=theta.x` (17 composantes
masquées/indicateurs), utiliser
`S=tau*log(sum_z pi_z*exp((g_z+alpha*m_z)/tau))`, `tau>0`.
Route entière avant réduction, jamais minima incompatibles. NaN dans la preuve,
contribution masquée+indicateur dans le modèle ; support structurel commun.
Sans route utilisable : un état base, pas OFF/contact négatif. Sans ancres :
support inconnu, score NaN, échec de couverture conservé.

`pi` fixé sans référence/coefficient : masse égale entre côtés supportés, puis
géométries complètes distinctes. Alias sans masse supplémentaire ; à géométrie
identique, moyenne sur marges natives distinctes, sans max B.
À `alpha=0`, même distribution A/B ; chemin calcul partagé requis pour parité
bit-exacte, pas deux réductions Torch réordonnées. La clé géométrique inclut
valeurs/supports de la route entière ; elle ne désigne aucune identité physique.
Préparer ces groupes une fois, pas de déduplication dépendant de theta/alpha.
Contrôle énumératif avant gel, notamment côtés absents : logsumexp non normalisé
favorise les banques nombreuses, pas une marginale justifiée.

Perte positive-set image-équilibrée
`LSE(S_all)-LSE(S_positifs)+lambda*||params||²/2` : plusieurs positifs autorisés,
concurrents non annotés **inconnus**, pas négatifs fiables. Elle favorise un positif, pas toutes les interactions. FIT seul fournit
échelles sans centrage et A ; géométrie gelée pour B, qui apprend seulement
alpha>=0. Même support/prior/tau dans les deux bras.

## Exécution et arrêt proposés

PyTorchFP64/H100, facteurs immuables par image/blocs, logsumexp stabilisé,
VJP analytique sur CSR fixes ; pas AMP/TF32 ni nouveaux modèles. Le premier
plan autograd est remplacé par l'audit des réductions et le prototype explicite
`coherent_pair_packed_torch`, pas par une modification d'un FIT historique.
Objectif complet déterministe,
pas minibatches différents dans une recherche linéaire.

L-BFGS strong-Wolfe pour A lisse. TorchLBFGS sans bornes : pas alpha softplus
(zéro impossible) ou carré
(gradient nul au départ). Pour B, préférer L-BFGS-B borné/scalarline-search avec
KKT projeté, via callback du même objectifGPU. Déclarer limites d'évaluations,
tolérance, zéro initial et échec de ligne avant FIT. Un arrêt numérique n'est
ni optimum global : différence de logsumexp et latents rendent la perte non
convexe. Nonfinite/ligne échouée/budget épuisé ferme la recette, sans restart.

Avant références : contrôle nouveau sur mêmes fixtures procédurales, FDFP64,
alpha0, aliases/permutations, absences, mémoire et objectif/gradientCPU-GPU.
Budget1–2h proposé, pas gelé ; mesurer l'objectif complet, pas extrapolation-victoire.
Tau/lambda/arrêt DEV externe seul, gelés avant RESERVED ;96photos fermé exclu.
Validation future : retrieval personne-objet puis vidéo stationnaire/bystanders,
occlusion/release ; jamais propriété/contact/3D déduits d'un PASS mécanique.

[Ilse2018](https://proceedings.mlr.press/v80/ilse18a.html), MIL invariant
aux permutations, pas preuve d'ownership ;
[Torch2.5.1 LBFGS](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/lbfgs.py),
source18037B/SHAf181e9224a72596f0f9027cfc42c0bf3819fd136828aef0fdc002a4e94495367.

## Référence mécanique implémentée, pas recette apprise

`coherent_pair_marginal` prépare une référence énumérative NumPyFP64 sur
PairCache authentifié ; aucun FIT, perte/optimiseur, température adoptée ou GPU.
Les clés géométriques comprennent12 valeurs **brutes masquées**, leurs12
supports,5 indicateurs et présence de route : désactiver un coefficient via
une échelle FIT ne fusionne pas deux géométries brutes distinctes. Classes/
marges ne définissent pas la géométrie. IDs/supports natifs restent complets.

26 contrôles dédiés passent. Audit indépendant en RAM sur4180B fabriqués :
énumération hiérarchique,17 dérivées FD, dérivée droite alpha0, chemin A/B
bit-exact, absence HOI, immutabilité et rejet de facteurs forgés PASS. Root274
contrôles combinés PASS1.93s incluant ce module et parser/census/cache existants.
Ce résultat qualifie uniquement les mathématiques et contrats : coût full-bank,
objectif/optimiseurGPU, convergence, ownership et qualité restent à tester.
