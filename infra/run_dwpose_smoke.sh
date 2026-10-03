#!/usr/bin/env bash
# CPU native-source ABI/replay; only two public RGBs and automatic human masks.
set -euo pipefail
(( $# == 0 )) || { echo 'DWPose smoke accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
test -f "$CODE/infra/dwpose_acquire.py"
test -f "$CODE/infra/dwpose_wheel_audit.py"
BASE="$ROOT/validation/identity_rgb_v2"; ASSETS="$ROOT/weights/dwpose_native_v1"
AUDIT="$ROOT/results/dwpose-wheel-audit-v3"; FAILED="$ROOT/results/dwpose-acquisition-v1.json"
PREVIOUS="$ROOT/results/dwpose-wheel-audit-v2/report.json"; OUT="$ROOT/validation/dwpose_smoke_v1"
python3 - "$ROOT" "$ASSETS" "$AUDIT" "$OUT" "$FAILED" "$PREVIOUS" "$BASE/inputs/manifest.json" "$BASE/automatic_masks/report.json" \
 "$BASE/inputs/clip_00_frame_000.png" "$BASE/inputs/clip_01_frame_000.png" \
 "$BASE/automatic_masks/clip_00_frame_000_human.png" "$BASE/automatic_masks/clip_01_frame_000_human.png" <<'PY'
from pathlib import Path
import sys
root,assets,audit,out,*files=map(Path,sys.argv[1:])
if not root.is_dir() or root.resolve()!=root.absolute(): raise ValueError('Canonical root required')
for path in (assets,audit,out,*files):
    if any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in path.parents) or path.is_symlink(): raise ValueError('Symlink/nonregular ancestor rejected')
if not assets.is_dir() or not audit.is_dir() or any(not p.is_file() for p in files) or out.exists(): raise FileExistsError('Fresh output and regular frozen inputs required')
PY
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
mkdir "$OUT"; chmod 755 "$OUT"; chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 183s docker run --rm --network none --memory 8g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" --env WR_DWPOSE_SMOKE_RESERVED=1 \
 --env CUDA_VISIBLE_DEVICES= --env PYTHONPATH="$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ASSETS,dst=$ASSETS,readonly" \
 --mount "type=bind,src=$AUDIT,dst=$AUDIT,readonly" \
 --mount "type=bind,src=$FAILED,dst=$FAILED,readonly" \
 --mount "type=bind,src=$PREVIOUS,dst=$PREVIOUS,readonly" \
 --mount "type=bind,src=$BASE/inputs/manifest.json,dst=$BASE/inputs/manifest.json,readonly" \
 --mount "type=bind,src=$BASE/automatic_masks/report.json,dst=$BASE/automatic_masks/report.json,readonly" \
 --mount "type=bind,src=$BASE/inputs/clip_00_frame_000.png,dst=$BASE/inputs/clip_00_frame_000.png,readonly" \
 --mount "type=bind,src=$BASE/inputs/clip_01_frame_000.png,dst=$BASE/inputs/clip_01_frame_000.png,readonly" \
 --mount "type=bind,src=$BASE/automatic_masks/clip_00_frame_000_human.png,dst=$BASE/automatic_masks/clip_00_frame_000_human.png,readonly" \
 --mount "type=bind,src=$BASE/automatic_masks/clip_01_frame_000_human.png,dst=$BASE/automatic_masks/clip_01_frame_000_human.png,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/dwpose_smoke.py"
