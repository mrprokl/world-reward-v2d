#!/usr/bin/env bash
# New offline CPU audit only; original acquisition and assets are mounted RO.
set -euo pipefail
(( $# == 0 )) || { echo 'DWPose wheel audit accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:?Require Azure root}"; CODE="${WR_CODE:?Require immutable source}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$(uname -s)" == Linux && "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ ]]
test -f "$CODE/infra/dwpose_acquire.py" # Explicit immutable dependency for code-only launcher closure.
ASSETS="$ROOT/weights/dwpose_native_v1"; RECEIPT="$ROOT/results/dwpose-acquisition-v1.json"
OUT="$ROOT/results/dwpose-wheel-audit-v2"
python3 - "$ROOT" "$ASSETS" "$RECEIPT" "$OUT" <<'PY'
from pathlib import Path
import sys
root,assets,receipt,out=map(Path,sys.argv[1:])
if not root.is_dir() or root.resolve()!=root.absolute(): raise ValueError('Canonical task root required')
for path in (assets,receipt,out):
    if any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in path.parents) or path.is_symlink():
        raise ValueError('Symlink/nonregular audit ancestor rejected')
if not assets.is_dir() or not receipt.is_file() or out.exists(): raise FileExistsError('Fresh audit target/regular inputs required')
PY
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
mkdir "$OUT"; chmod 755 "$OUT"; chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
timeout --signal=TERM --kill-after=10s 123s docker run --rm --network none --memory 8g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$WR_CODE_REVISION" --env WR_IMAGE_ID="$IMAGE" \
 --env WR_DWPOSE_AUDIT_RESERVED=1 --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ASSETS,dst=$ASSETS,readonly" \
 --mount "type=bind,src=$RECEIPT,dst=$RECEIPT,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/dwpose_wheel_audit.py"
