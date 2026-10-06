# V-COCO16 — copie aveugle des observations HOI natives

**Prospectif, non exécuté.** Deux unités CPU distinctes, export VM02 puis import
VM01. Limite inclusive de 180 s par unité (195 s externe), stdlib seule, mémoire
256 MiB. Aucun modèle, GPU, décodage JPEG/NPZ, référence, apprentissage ou sélection.

## Population et provenance

On conserve les **16 NPZ originaux**, tous les 1500 queries par image et 165
paires main–objet natives, sans filtrage. S'y ajoutent `host.json` (copie du
`report.json` original), `model.json` et `model_proof.json` : **19 charges utiles**.
Producteur `fc3c91bd9d156c501ed42abd9e342d1158883047`, source 344 fichiers / 349
entrées, closure `cc35bb23acd75b89e4ce9ac7f6580604464a76f25ba3138795069b456c8b98d7`.
Les trois reçus sont épinglés par octets/SHA256 dans le code. L'export vérifie
source complète, permissions, empreintes, runtime enregistré et vivant, absence
des overlays/CID et provenance des 16 endpoints d'origine avant et après.

L'import valide les métadonnées exactes des **19 tableaux/image** et les SHA256
des NPZ opaques. Les identités d'image, grille, frame 0, slot/ordinal, personnes
et empreintes endpoint sont comparées à la completion `21ac084…` authentifiée
sur VM01. Les preuves runtime/source copiées restent des **déclarations du
producteur**, jamais une vérification du runtime/source sender vivant sur VM01.
Aucune copie de RGB, poids, référence privée, CID ou archive de source.

## Transport et publication

Blob privé MI existant : clé `articulated-runtime-<export_revision>.tar`, URL
fixe sans query/secrets. PUT exclusif `If-None-Match:*`, HEAD de l'export puis GET
unique de l'import. Le contrôleur doit fournir indépendamment pins archive,
manifest et reçu export (base64 canonique borné) avant import. Aucun retry.

USTAR limité à 32 MiB, manifest-first ≤256 KiB, **20 membres** exacts. Avant
installation : SHA complet, noms/ordre/tailles, headers/checksums, SHA par membre,
padding/tail zéro, EOF complet. Pas de PAX/GNU, chemins/prefixes, liens, membres
supplémentaires, `extractall` ou lecture non bornée. Destination neuve uniquement
`/srv/world-reward-data/vcoco_hoi_saved_replica_v1` : root 700, `banks/` 500,
19 feuilles 400. Le parent DATA 700 root-owned doit déjà exister ; pas de bootstrap
ni chmod d'une namespace existante. Commit Linux NOREPLACE après fsync.

Publication via le helper sealed-callback, pas l'ancien publisher de replica.
Le DELETE unique `If-Match` de l'ETag exact a lieu **après scellement** et seulement
si la copie/source/endpoints sont intacts. PASS final requiert DELETE 202 et
contrôles après callback/deadline. Une erreur conserve copie et reçu FAIL, ne
réécrit pas l'ancien producteur et ne déclenche aucune relance automatique.

## Interface pour jointure ultérieure

`authenticate_receiver(code, replica_revision, receipt_pin)` renvoie
`images`, `banks`, `hoi_rows`, `files` (paths absolus/pins), `states`,
`import_source`, `import_identity` et déclaration de source originale ; aucun
tableau n'est décodé. La jointure doit qualifier séparément ses données/algorithmes.
Les tests sont des octets/JSON **fabriqués**, pas une qualification native réelle.
Les licences/overlaps restent ceux des modèles/données upstream ; cette copie
ne prouve ni droits des photos, absence de fuite, contact, ownership, qualité,
adoption ou victoire leaderboard.
Root151 related controls PASS0.40s; AST and shell syntax PASS. Owned fixtures
removed. Root wrapper supplies exact Azure host/source/environment,1GiB address
space and195s outer failure grace around180s inclusive technical work. This is
source qualification only; no export/import, actual archive or new join exists
yet, and no prediction-quality claim follows.
