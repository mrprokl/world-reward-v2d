#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
OUT="$ROOT/outputs/episode_000015/identity_xy_fit_v1"
[[ -d "$ROOT" && ! -L "$ROOT" && ! -L "$ROOT/outputs" && ! -L "$ROOT/outputs/episode_000015" && ! -e "$OUT" && ! -L "$OUT" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
mkdir "$OUT";chmod 755 "$OUT";chown 1000:1000 "$OUT"
timeout --signal=TERM --kill-after=10s 603s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user 1000:1000 --entrypoint python --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" \
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/vendor/video_to_data,dst=$ROOT/vendor/video_to_data,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3,dst=$ROOT/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3,readonly" \
 --mount "type=bind,src=$ROOT/results/input-manifest.json,dst=$ROOT/results/input-manifest.json,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --mount "type=bind,src=$ROOT/data/track_1/meta/episodes.jsonl,dst=$ROOT/data/track_1/meta/episodes.jsonl,readonly" \
 --mount "type=bind,src=$ROOT/outputs/episode_000015/body_full/report.json,dst=$ROOT/outputs/episode_000015/body_full/report.json,readonly" \
 --mount "type=bind,src=$ROOT/outputs/episode_000015/body_full/predictions.npz,dst=$ROOT/outputs/episode_000015/body_full/predictions.npz,readonly" \
 --mount "type=bind,src=$ROOT/outputs/episode_000015/automatic_masks/report.json,dst=$ROOT/outputs/episode_000015/automatic_masks/report.json,readonly" \
 --mount "type=bind,src=$ROOT/outputs/episode_000015/automatic_masks/prompts.json,dst=$ROOT/outputs/episode_000015/automatic_masks/prompts.json,readonly" \
 --mount "type=bind,src=$ROOT/outputs/episode_000015/automatic_masks/masks/0,dst=$ROOT/outputs/episode_000015/automatic_masks/masks/0,readonly" \
 --mount "type=bind,src=$ROOT/outputs/episode_000015/automatic_masks/masks/1,dst=$ROOT/outputs/episode_000015/automatic_masks/masks/1,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/identity_xy_fit.py"
