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

## Actual bounded diagnostic (2026-10-05)

Producer `bd58681a04fa7e5f82807997ab5d92a7bdc9723c`: PASS in2.966602s.
Receipt17451B/SHA256`5e3188aa1fefa41ffe231575fbe445f3b476a683c9067b076f12e5fde623b193`.
The authenticated archive has19 entries and4,767,087,611 expanded bytes. Its
first v1 rejection is entry1, `assets/corrective_blendshapes_lod0.npz`,
2,651,004,394B, exceeding the1GiB per-member cap. No payload was read. The actual
notice member is `assets/LICENSE.txt`, not the v1 assumed `assets/LICENSE`.
These are packaging findings, not malicious-content findings or licence proof.
An independent read-only metadata/receipt/source/model/old-failure posthash
audit also passed. Original v1 FAIL remains unchanged; no generic cap is raised.
