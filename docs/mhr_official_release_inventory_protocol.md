# Independent MHR release inventory diagnostic

2026-10-05, prospective. One CPU-only Azure acquisition of the exact public
MHR v1.0.1 archive:198943157B/SHAe4f4f205…6b94. The original acquisition
`4ed17ad…` failed at inventory after whole-archive identity passed; its sealed
four files, original full source closure and failure receipt must authenticate
before and after this diagnostic. No source/asset/FAIL replacement.

The explicit `acquire_weights.sh --mhr-release-inventory-only` mode reuses the
same acquisition primitives and unchanged protocol, but has an exclusive
`results/mhr-official-release-inventory-v1` output. Before opening ZIP metadata,
verify the complete published SHA. Record bounded central-directory rows,
compression/type/flags/expanded sizes and the first original v1 guard rejection.
Do not infer that `phase=inventory` means malicious geometry or archive content.

Read notice payloads only if every structural inventory guard passes; each is
bounded64KiB/total256KiB and its exact primary-Apache equality is descriptive.
Never open, extract, stream-hash or load a model member. Remove only the owned
download after source/model/original-failure postchecks.300s inclusive, outer320s
and cleanup grace20s. All retained text/scalar receipts stay sealed on Azure.

A diagnostic PASS means authentic bounded metadata inspection completed. It
does **not** pass the original acquisition, certify asset licence scope, legal
eligibility, training overlap or model membership. Any packaging correction
would need its own explicit future protocol/producer; no v1 gate relaxation.
Root integration212tiny tests PASS,23.49s; no archive was downloaded locally.
