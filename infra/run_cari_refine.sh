#!/usr/bin/env bash
# Exact public parity refinement; GPU serialization is an external queue's job.
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
source "$CODE/infra/cari_wrapper_common.sh"
wr_parse_cari_arguments refine "$@"
wr_cari_dependency refine
printf -v PADDED '%06d' "$WR_EPISODE"
BASE="$ROOT/outputs/episode_$PADDED"
OUT="$BASE/cari_refined"
[[ ! -L "$ROOT" && ! -L "$ROOT/outputs" && ! -L "$BASE" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
timeout --signal=TERM --kill-after=10s 7203s docker run --rm --gpus all --network none --memory 64g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 --env WANDB_MODE=disabled \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache --env MPLCONFIGDIR=/tmp/world-reward-matplotlib \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/vendor/video_to_data,dst=$ROOT/vendor/video_to_data,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/sam3d_body,dst=$ROOT/weights/cari4d/sam3d_body,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/refinement,dst=$ROOT/weights/cari4d/refinement,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --mount "type=bind,src=$ROOT/results/cari-refinement-assets.json,dst=$ROOT/results/cari-refinement-assets.json,readonly" \
 --mount "type=bind,src=$BASE/cari_forward,dst=$BASE/cari_forward,readonly" \
 --mount "type=bind,src=$BASE/cari_inputs,dst=$BASE/cari_inputs,readonly" \
 --mount "type=bind,src=$BASE/body_full,dst=$BASE/body_full,readonly" \
 --mount "type=bind,src=$BASE/object_pose_full,dst=$BASE/object_pose_full,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/cari_refine.py" --episode "$WR_EPISODE"
