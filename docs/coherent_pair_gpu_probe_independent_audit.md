# Audit indépendant du tiny GPU probe

2026-10-06 — **READY, portée source/lifecycle uniquement**. Aucun import Torch,
GPU/Azure, donnée ancienne, modèle, référence ou FIT exécuté par cet audit.
L'ABI CUDA `segment_reduce`, les tolérances et la répétabilité restent à
vérifier lors du seul nouveau contrôle natif, pas déduites des tests AST.

## Sources relues, figées

- Driver :31277B, SHA256
  `1e54b9aad3186abd31c511e79927aecd9f46dc44dbd061d9d12127a079c185f9`.
- Wrapper : SHA256
  `b1747221aea35a9400b65fad5073a11d4f3a2b339409c302237008dfd392051d`.
- Tests : SHA256
  `3732fb0653f2f85c1a91ef58cbc37e14e46f09f6dd2a5340ed839acd3f530911`.
- Protocole relu : SHA256
  `588ca73b174e533672a61bf6169f91fa43cac5c43f0da15ef79f270b9f07b681`.

## Défauts concrets trouvés puis corrigés par l'owner

Le premier validateur acceptait runtime absent,20 enregistrements de segments
vides et FD omis sur un cas supporté. Il exige maintenant runtime
Torch2.5.1+cu124/CUDA12.4/H100,20 contrôles ordonnés width/empty/op/SHA,
indices/comptages/empreintes et applicabilité FD exacte pour les huit fixtures.
Un contrôle stdlib en RAM confirme rejet runtime absent, segments vides,
FD supporté sauté, index incorrect et empreinte de résultat invalide.

Le cleanup prouvait seulement l'absence du nom. Il interroge maintenant aussi
le CID sauvegardé : un conteneur renommé encore vivant provoque FAIL sans
mutation étrangère. Régression indépendante mockée confirmée.

## Contrat désormais cohérent

Huit nouvelles fixtures complètes ; deux sorties par alpha comparées en
dtype/shape/octets, NaN inclus. Comparaison contre les deux oracles NumPy
1e-12 ; FD17 h1e-6/tol1e-7 et alpha0 par différence droite d'ordre2. Alpha0
partage les buffers natifs/groupés et VJP géométrique ; sa dérivée reste testée.
Contrôles CUDA widths1/9/6/2/17 avec trous et segments vides ; overflow doit
échouer sans clipping. Toutes les tables device mutables et tables/IDs host
sont rehashées avant/après, y compris inverse CSR et contrôle d'échec.

VM02 explicite, B47 immuable, FD9 nonblocking, réseau none, source leaves RO,
aucun checkpoint/RGB. Host stdlib, Torch seulement enfant. CUBLAS configuré
avant import, FP64 sans TF32, déterminisme strict.6GiB Docker concerne RAM
host ; fraction/peaks6GiB Torch ne prétendent pas borner contexte/driver CUDA.
Namespace neuf, umask explicite, CID/image/name/owner authentifiés, deadline
1200s incluant scellement/cleanup ; publication sur même FD démote PASS tardif.

AST/bash-n et pins relus ; aucun blocker source restant identifié. Le PASS
natif futur qualifiera uniquement cette arithmétique minuscule, pas coût
full-bank, optimisation, ownership physique, performance ou adoption. Ancienne
recette512/720 et études fermées restent inchangées.
