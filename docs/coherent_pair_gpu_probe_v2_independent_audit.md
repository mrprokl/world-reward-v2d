# Audit indépendant — tiny GPU v2

2026-10-06 : **READY source/lifecycle uniquement**. Aucun Azure, import
Torch, GPU, modèle, référence, FIT ou calcul natif exécuté par cet audit v2.

## Pourquoi cette correction est technique

Selon le diagnostic saved-only de la racine, le v1 `a48aaec…` a échoué sur
`proof → image → control` : B47 absent du Docker privé VM02. Aucun dossier
résultat, reçu, conteneur ou calcul Torch n'a été produit. `docs/compute.md`
distingue B47 VM01 du runtime classique `7ebfff…` VM02. Le v1 reste FAILED ;
on ne le requalifie pas et ses sources ne sont pas modifiées rétroactivement.
Les deux tentatives d'observateur incomplètes restent INCONCLUSIVE, pas une
preuve d'échec du scoreur.

## Vérifications indépendantes

Comparaison AST exacte avec le driver exécuté v1 : `fixtures`,
`segment_controls`, `arithmetic`, `native`, `validate_native`, `cleanup`,
`publish`, `image`, `proof` et fonctions d'empreintes sont inchangées.
Seuls le schéma du manifeste, la propagation CLI et la sélection explicite du
profil/host/namespace changent. Tous les champs numériques du manifeste sont
identiques hors schéma ; retour v2→v1 vérifié. Le wrapper v1 est identique en
octets. Les deux wrappers passent `bash -n`.

V2 impose VM01 `scenesmith-ncc-h100-01` / `SCENESMITH-H100`, entry
`run_coherent_pair_gpu_probe_v2`, résultat `coherent-pair-gpu-probe-v2`,
`--profile v2` transmis au fils, même B47 strict. Aucun fallback, téléchargement,
installation, alias d'image ou changement de tolérance. Le schéma v2 et les
leaves authentifiées empêchent une confusion avec les reçus v1.

FD9, Docker privé explicite, réseau none, source RO, quatre CPUs, 1200 s
inclusifs, nettoyage CID/name/image/owner et publication sur même FD restent
inchangés. RAM host 6 GiB et limite allocateur Torch 6 GiB ne sont pas une
borne de tout le contexte CUDA.

## Sources relues

- Driver 32097B : `989b531802797e0fc5eb90a1ef6a4a19e4c7c68d797e8b0451934f68e5a1571b`.
- Wrapper v2 1735B : `56e769626442d15c626070d0d8792a4ad8822eaed53f8f7d7cbfedaf5df80402`.
- Tests 16008B : `0ffe3f9e89227942eaac8c96a56cc0198be176e3888c53d1e7b9e4b77c3a353c`.
- Protocole 2702B : `69ad98a2b89ebaec39df924231d048536084e22c357eef4412bc8930cf8a92bb`.

Aucun blocker source concret restant identifié. Image réelle, ABI CUDA,
parité, FD et répétabilité restent **non qualifiées** jusqu'au seul contrôle
v2 figé. Un éventuel PASS qualifiera l'arithmétique minuscule, pas le coût
full-bank, une optimisation, l'ownership physique, la qualité ou l'adoption.
