#!/usr/bin/env bash
# Source closure: /infra/end2end_preview.py /infra/full4d_video.py /infra/full4d_publish.py
set +x
set -euo pipefail
[[ ( $# == 2 || ( $# == 4 && "$3" == --candidate-kind && "$4" =~ ^(joint|continuation)$ ) ) \
 && "$2" =~ ^[0-9a-f]{40}$ && "$1" =~ ^(render|publish)$ ]] || exit 2
KIND="${4:-joint}"; PREFIX=native-joint-real
[[ "$KIND" != continuation ]] || PREFIX=native-contact-continuation-real
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_end2end_preview/code" && "$(hostname)" == scenesmith-ncc-h100-01 ]] || exit 2
if [[ "$1" == publish ]]; then
 exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 PYTHONPATH="$CODE/src:$CODE/infra" PYTHONDONTWRITEBYTECODE=1 python3 -B "$CODE/infra/end2end_preview.py" "$@"
fi
exec 8<"$ROOT/jobs/.world-reward-h100.lock"; flock -n 8
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
OUT="$ROOT/results/end2end-preview-$REV"; [[ ! -e "$OUT" ]] || exit 2; mkdir -m 755 "$OUT"
NAME="wr-end2end-preview-$REV"
cleanup() {
 local status=$? cid=''; trap - EXIT
 if [[ -f "$OUT/.container.cid" ]]; then
  cid="$(cat "$OUT/.container.cid")"; [[ "$cid" =~ ^[0-9a-f]{64}$ ]] || exit 1
  if [[ -n "$(docker ps -aq --no-trunc --filter "id=$cid")" ]]; then
   [[ "$(docker inspect "$cid" --format '{{.Image}}|{{index .Config.Labels "world_reward.preview.owner"}}')" == "$IMAGE|$REV" ]] || exit 1
   docker rm -f "$cid" >/dev/null || status=1
  fi
  [[ -z "$(docker ps -aq --no-trunc --filter "id=$cid")" ]] || status=1
 fi
 printf '{"process_exit_code":%d,"owned_container_absent":true}\n' "$status" > "$OUT/host-exit.json"
 chmod 444 "$OUT/host-exit.json"; exit "$status"
}
trap cleanup EXIT; trap 'exit 143' TERM; trap 'exit 130' INT
SOURCE_MOUNTS=()
if [[ "$KIND" == continuation ]]; then
 SOURCE_MOUNTS+=(--mount "type=bind,src=$ROOT/jobs/$2/run_native_contact_continuation_real,dst=$ROOT/jobs/$2/run_native_contact_continuation_real,readonly")
fi
timeout --signal=TERM --kill-after=10s 1200s docker run --rm --cidfile "$OUT/.container.cid" \
 --name "$NAME" --label "world_reward.preview.owner=$REV" --gpus all --network none --read-only \
 --user 0:0 --cap-drop ALL --security-opt no-new-privileges --cpus 4 --memory 16g --tmpfs /tmp:rw,nosuid,size=256m \
 --mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly" \
 --mount "type=bind,src=$ROOT/experiments/full4d-v1-052ba1554e9a573d566713a99a61d89a5f27681c,dst=$ROOT/experiments/full4d-v1-052ba1554e9a573d566713a99a61d89a5f27681c,readonly" \
 --mount "type=bind,src=$ROOT/results/$PREFIX-$2,dst=$ROOT/results/$PREFIX-$2,readonly" "${SOURCE_MOUNTS[@]}" \
 --mount "type=bind,src=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera,dst=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" \
 --entrypoint /usr/bin/env "$IMAGE" -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp \
 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONPATH="$CODE/src:$CODE/infra" \
 /opt/conda/bin/python -B "$CODE/infra/end2end_preview.py" "$@"
