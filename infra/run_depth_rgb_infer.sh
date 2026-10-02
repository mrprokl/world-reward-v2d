#!/usr/bin/env bash
# Each backend has only original RGB input; no private or other-backend mount.
set -euo pipefail
[[ $# == 2 && "$1" == --backend && ( "$2" == moge || "$2" == da3 ) ]] || exit 2
BACKEND="$2"; ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/depth_rgb_v1"; OUT="$BASE/${BACKEND}_predictions_v1"
[[ ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -L "$BASE" && -d "$BASE/inputs" && ! -e "$OUT" && ! -L "$OUT" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
MOUNTS=(); EXTRA_PATH=''
if [[ "$BACKEND" == da3 ]]; then
 WHEEL="$ROOT/vendor/research/da3_dependencies_v1/addict-2.4.0-py3-none-any.whl"
 [[ -f "$WHEEL" && ! -L "$WHEEL" && "$(stat -c %s "$WHEEL")" == 3832 ]]
 [[ "$(sha256sum "$WHEEL" | cut -d ' ' -f1)" == 249bb56bbfd3cdc2a004ea0ff4c2b6ddc84d53bc2194761636eb314d5cfa5dfc ]]
 EXTRA_PATH=":$WHEEL"
 MOUNTS=(--mount "type=bind,src=$WHEEL,dst=$WHEEL,readonly"
  --mount "type=bind,src=$ROOT/vendor/research/da3_metric_v1,dst=$ROOT/vendor/research/da3_metric_v1,readonly"
  --mount "type=bind,src=$ROOT/weights/research/da3_metric_v1,dst=$ROOT/weights/research/da3_metric_v1,readonly"
  --mount "type=bind,src=$ROOT/results/da3-metric-acquisition-v1.json,dst=$ROOT/results/da3-metric-acquisition-v1.json,readonly")
else
 MOUNTS=(--mount "type=bind,src=$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal,dst=$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal,readonly"
  --mount "type=bind,src=$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c,dst=$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c,readonly"
  --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly")
fi
mkdir "$OUT"; chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 183s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" --env PYTHONPATH="$CODE/src:$CODE/infra$EXTRA_PATH" \
 --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env WANDB_MODE=disabled \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "${MOUNTS[@]}" "$IMAGE" "$CODE/infra/depth_rgb_infer.py" --backend "$BACKEND"
