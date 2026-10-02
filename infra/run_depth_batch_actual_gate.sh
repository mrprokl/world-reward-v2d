#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
ATTEMPT=v1
if (( $# > 0 )); then
 [[ $# == 2 && "$1" == --attempt && ( "$2" == v1 || "$2" == v2 ) ]] || { echo 'Require --attempt v1|v2 only' >&2; exit 2; }
 ATTEMPT="$2"
fi
BASE="$ROOT/outputs/episode_000015"
OUT="$ROOT/results/depth-batch-actual-$ATTEMPT"
[[ ! -L "$ROOT" && ! -L "$ROOT/results" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
MOUNTS=()
for FILE in "$BASE/depth_full/report.json" "$BASE/scale_smoke/report.json" "$BASE/cari_inputs/report.json" "$BASE/cari_inputs/aligned_depth.h5"; do
 [[ -f "$FILE" && ! -L "$FILE" ]]
 MOUNTS+=(--mount "type=bind,src=$FILE,dst=$FILE,readonly")
done
for INDEX in {0..8}; do
 printf -v FRAME '%06d' "$INDEX"
 FILE="$BASE/depth_full/$FRAME.npz"
 [[ -f "$FILE" && ! -L "$FILE" ]]
 MOUNTS+=(--mount "type=bind,src=$FILE,dst=$FILE,readonly")
done
timeout --signal=TERM --kill-after=5s 243s docker run --rm --network none --memory 16g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/vendor/video_to_data,dst=$ROOT/vendor/video_to_data,readonly" \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/depth_batch_actual_gate.py" --attempt "$ATTEMPT"
