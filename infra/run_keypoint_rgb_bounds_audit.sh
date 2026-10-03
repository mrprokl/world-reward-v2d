#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || { echo 'Bounds audit accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
BASE="$ROOT/validation/keypoint_rgb_v1"
OUT="$ROOT/results/keypoint-rgb-bounds-audit-v2"
python3 - "$ROOT" "$OUT" <<'PYSAFE'
from pathlib import Path
import sys
root,out=map(Path,sys.argv[1:])
if root.resolve()!=root.absolute() or not root.is_dir():raise ValueError('Canonical root required')
if out.exists() or any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in (out,*out.parents)):raise ValueError('Fresh nonsymlink audit output required')
PYSAFE
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
UID_SCENE="$(id -u scenesmith)"; GID_SCENE="$(id -g scenesmith)"
mkdir "$OUT"; chown "$UID_SCENE:$GID_SCENE" "$OUT"
# Historical source is hash-only input, not an imported runtime dependency.
NAME=keypoint_rgb_fit.py
OLD="$ROOT/jobs/52f35c0b0df9a1448adf16c1cff796007fb41198/run_keypoint_rgb_fit/code/infra/$NAME"
timeout --signal=TERM --kill-after=10s 63s docker run --rm --network none --cpus 4 --memory 8g \
 --user "$UID_SCENE:$GID_SCENE" --entrypoint python --env "WR_ROOT=$ROOT" --env "WR_CODE_REVISION=$WR_CODE_REVISION" \
 --env "WR_IMAGE_ID=$IMAGE" --env CUDA_VISIBLE_DEVICES= --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/weights/mhr/mhr_model.pt,dst=$ROOT/weights/mhr/mhr_model.pt,readonly" \
 --mount "type=bind,src=$ROOT/results/keypoint-rgb-bounds-audit-v1/report.json,dst=$ROOT/results/keypoint-rgb-bounds-audit-v1/report.json,readonly" \
 --mount "type=bind,src=$BASE/inputs/manifest.json,dst=$BASE/inputs/manifest.json,readonly" \
 --mount "type=bind,src=$BASE/baseline_v1,dst=$BASE/baseline_v1,readonly" \
 --mount "type=bind,src=$BASE/root_fit_v1,dst=$BASE/root_fit_v1,readonly" \
 --mount "type=bind,src=$OLD,dst=$OLD,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/keypoint_rgb_bounds_audit.py"
