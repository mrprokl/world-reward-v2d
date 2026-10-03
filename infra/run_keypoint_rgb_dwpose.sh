#!/usr/bin/env bash
# One CPU session on15 public RGBs; no object PNGs, body output or private truth.
set -euo pipefail
(( $# == 0 )) || { echo 'Keypoint DWPose observations accept no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
test -f "$CODE/infra/dwpose_smoke.py"
test -f "$CODE/infra/run_dwpose_smoke.sh" # source_identity audits the unchanged helper wrapper.
test -f "$CODE/infra/dwpose_acquire.py"
test -f "$CODE/infra/dwpose_wheel_audit.py"
BASE="$ROOT/validation/keypoint_rgb_v1"; ASSETS="$ROOT/weights/dwpose_native_v1"
AUDIT="$ROOT/results/dwpose-wheel-audit-v3"; FAILED="$ROOT/results/dwpose-acquisition-v1.json"
PREVIOUS="$ROOT/results/dwpose-wheel-audit-v2/report.json"
SMOKE_V1="$ROOT/validation/dwpose_smoke_v1/report.json"; SMOKE_V2="$ROOT/validation/dwpose_smoke_v2/report.json"
OUT="$BASE/dwpose_v1"
python3 - "$ROOT" "$ASSETS" "$AUDIT" "$FAILED" "$PREVIOUS" "$SMOKE_V1" "$SMOKE_V2" "$BASE/inputs" "$BASE/automatic_masks/report.json" "$OUT" <<'PYSAFE'
from pathlib import Path
import sys
root,assets,audit,failed,previous,v1,v2,inputs,masks,out=map(Path,sys.argv[1:])
if not root.is_dir() or root.resolve()!=root.absolute(): raise ValueError('Canonical task root required')
paths=[assets,audit,failed,previous,v1,v2,inputs,masks,out]
for c in range(3):
    for f in range(5):
        stem=f'clip_{c:02d}_frame_{f:03d}'
        paths.extend([inputs/(stem+'.png'), masks.parent/(stem+'_human.png')])
for path in paths:
    if any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in path.parents) or path.is_symlink():
        raise ValueError('Symlink/nonregular input ancestor rejected')
if not assets.is_dir() or not audit.is_dir() or not inputs.is_dir() or out.exists(): raise FileExistsError('Fresh output/regular inputs required')
for path in (failed,previous,v1,v2,masks,*paths[9:]):
    if not path.is_file(): raise FileNotFoundError('Every frozen public input must exist')
PYSAFE
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
mkdir "$OUT"; chmod 755 "$OUT"; chown scenesmith:scenesmith "$OUT"
MOUNTS=()
for c in 00 01 02; do
 for f in 000 001 002 003 004; do
  MASK="$BASE/automatic_masks/clip_${c}_frame_${f}_human.png"
  MOUNTS+=(--mount "type=bind,src=$MASK,dst=$MASK,readonly")
 done
done
timeout --signal=TERM --kill-after=10s 183s docker run --rm --network none --memory 8g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" \
 --env CUDA_VISIBLE_DEVICES= --env PYTHONPATH="$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ASSETS,dst=$ASSETS,readonly" \
 --mount "type=bind,src=$AUDIT,dst=$AUDIT,readonly" \
 --mount "type=bind,src=$FAILED,dst=$FAILED,readonly" \
 --mount "type=bind,src=$PREVIOUS,dst=$PREVIOUS,readonly" \
 --mount "type=bind,src=$SMOKE_V1,dst=$SMOKE_V1,readonly" \
 --mount "type=bind,src=$SMOKE_V2,dst=$SMOKE_V2,readonly" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$BASE/automatic_masks/report.json,dst=$BASE/automatic_masks/report.json,readonly" \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/keypoint_rgb_dwpose.py"
