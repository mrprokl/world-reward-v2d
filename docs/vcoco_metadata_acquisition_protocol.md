# V-COCO metadata acquisition v1 — prospective, private research only

This first CPU-only job acquires **opaque bytes**, not a cohort or labels used by
a model. No JSON/ID parsing, RGB, census, prediction, historical reference,
checkpoint, package installation or GPU is allowed. Code/tests are not an actual
acquisition result. A later separately reviewed census must retain all 432 old
slots' photo/known-byte exclusions; creators and model overlap remain UNKNOWN.

## Frozen publisher identities

V-COCO [commit](https://github.com/s-gupta/v-coco/tree/489cc4db74f2f10ab4b134f67da3874afbf245ab)
is dated 2017-06-16, root tree `517cee7d005cf57fa4e6e5f5bc1eaeb3b3118ac7`.
The source contains exact Git blob SHA1/byte counts for all four original role
JSONs and all five original `.ids` splits. Acquisition computes standard
`SHA1("blob "+decimal_length+NUL+bytes)` and a descriptive full SHA256 by streaming,
without decoding values. The raw README, MIT license, Fast-RCNN BSD notice and
three original helper files retain software-header notices; they are never run.
Repository-wide MIT is a documented **private research/metadata qualification
interpretation**, not a definitive annotation-specific grant or legal guarantee.
No public image redistribution or competition eligibility is asserted.

COCO's pinned [download page](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/aaa6a5a0cc24bf1350247169cc512edd7ddf28b9/dataset/download.htm)
lists `annotations_trainval2014.zip`; its original publisher S3 object is used
via [strict TLS](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip).
On 2026-10-05 22:02 UTC HEAD returned 200, 252872794 bytes, application/zip,
Last-Modified 2018-07-10; ETag `0a379cfc70b0e71301e0f377548639bd` is **not SHA256**.
The branded HTTPS endpoint failed TLS locally; no insecure fallback is used.
No existing COCO2014 metadata receipt was assumed.

Exactly 4096 tail bytes were read in RAM with HTTP206, Content-Range
`bytes 252868698-252872793/252872794`, SHA256
`38c634460cccfad846182e4b24d721978afcfcec8c3675c34857a9417849139f`.
Only the ZIP central catalogue was inspected: six regular DEFLATE members,
original instances/keypoints/captions train2014 and val2014, 844813048 expanded
bytes. No payload or JSON values were read locally. Names, sizes, CRC, flags and
external attributes are frozen in the driver; the same tail is authenticated
after the full Azure download. The original archive has no publisher SHA256:
first full-read SHA256 is a descriptive identity, **not publisher certification**.
All six members are streamed for decompressed SHA/CRC verification without
extracting files or consulting values, including unused keypoints/captions.

The pinned [COCO terms](https://raw.githubusercontent.com/cocodataset/cocodataset.github.io/aaa6a5a0cc24bf1350247169cc512edd7ddf28b9/dataset/termsofuse.htm)
are retained: annotation grant CC-BY-4.0 is separate from underlying photograph
copyright/Flickr licenses. This job does not verify creator/photo rights or
pretraining/challenge overlap, and has no quality, ownership or adoption claim.

## Execution and failure boundary

Entrypoint `run_vcoco_metadata_acquire`: Azure VM02 root CPU, clean environment,
Python `-I -B`, GPU hidden, 8 GiB address-space cap. One sequential HTTP attempt
per frozen URI, no redirect/proxy/retry/replacement; 15 s per blocking request,
1 MiB streams, 300 MiB total compressed, 512 MiB per expanded member and 1 GiB
total expanded. Source authentication, transfer, CRC, full post-hashes and final
same-FD receipt publication are inside **600 s**; shell615 s allows cleanup only.

Fresh `/srv/world-reward-data/vcoco_metadata_v1` is private700 until publication,
then500; original artifact and report files400. Exact immutable source/helper
hashes and modes are checked before/after. Failed partial writes remove only the
opened regular inode; completed original artifacts remain sealed and inventoried
even if later CRC fails. Foreign leaves are never removed. Deadline/seal/fsync
failure demotes the same opened report FD to FAIL. Sanitized fixed exception
classes only; no raw URLs/errors/annotation values in stdout. A technical PASS is
`METADATA_QUALIFIED_PENDING_SEPARATE_CENSUS`, never permission to train or submit.

## Actual metadata-only result — 2026-10-06

Producer `9976f0f0d9caccdf72105d6cb7b7d8a3cc1d689d` sealed a PASS receipt in
15.304575789 s: 272739841 acquired bytes, 18 original assets plus report. The
report is 8529 bytes/SHA256
`aaf173055797b021fb6e1ecbc578b5676a3de5ea333534aa29860d0508b667d1`;
COCO2014 archive SHA256 is
`031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009`
(first-read descriptive identity, not publisher certification).

Saved-only independent audit v2 authenticated all 292 Git source files/297
entries, original markers/modes, protocol, 18 artifact hashes, 15 publisher Git
blobs, six central-directory entries/tail and source/output post-hashes. CRC and
expanded SHA evidence is authenticated **producer evidence**; the observer did
not decompress or parse any annotation. Exact native driver was absent. The
systemd unit was already collected/missing: unit exit status is **unavailable**,
and its default success/0 fields are not native execution proof. Acquisition
qualification rests on the sealed, source-bound receipt with this limitation.
The first observer's unit-not-found PARTIAL remains unchanged; v2 did not rerun
acquisition. References: `results/audits/vcoco_metadata_v1_actual_v2.json`.
No capacity, photograph rights, ownership quality, FIT or model adoption is
qualified. A separately frozen metadata census is still required.

## Source qualification, October6, before execution

Root159 combined tiny tests PASS0.41s, with no real HTTP/annotation values.
Independent four stdlib fault controls pass: header rejection before body,
incorrect Git blob cleans only its partial, DEFLATE CRC corruption preserves
the original, and late receipt publication closes FAIL. Fresh directory
device/inode/UID/GID and700 mode are now checked before inventory/publication;
this was corrected before the single frozen acquisition. These checks preceded
execution; the actual result above supersedes the prospective dispatch status.
