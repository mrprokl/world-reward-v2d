#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || { echo 'Soft silhouette capability accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
OUT="$ROOT/results/soft-silhouette-probe-v1"
[[ -d "$ROOT/results" && ! -L "$ROOT" && ! -L "$ROOT/results" && ! -e "$OUT" && ! -L "$OUT" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
mkdir "$OUT"; chmod 755 "$OUT"; chown 1000:1000 "$OUT"
timeout --signal=TERM --kill-after=10s 183s docker run --rm --gpus all --network none --memory 16g --cpus 4 \
 --user 1000:1000 --entrypoint python --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" \
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/soft_silhouette_probe.py"
