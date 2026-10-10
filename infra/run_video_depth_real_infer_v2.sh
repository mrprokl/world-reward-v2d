#!/usr/bin/env bash
# Source closure: /infra/video_depth_real_run.py /infra/video_depth_assets.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ $# == 0 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_video_depth_real_infer_v2/code" \
 && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
BASE="$ROOT/validation/video_depth_real_v1"
OUTPUT="$ROOT/validation/video_depth_real_output_v2"
[[ ! -e "$OUTPUT" ]] || exit 2
mkdir "$OUTPUT"
[[ -f "$ROOT/vendor/research/vda_dependencies_v1/easydict/__init__.py" && -f "$ROOT/results/video-depth-dependencies-v1.json" ]] || exit 2
[[ -f "$ROOT/validation/video_depth_real_output_v1/predictions/report.json" ]] || exit 2
exec 9<"$ROOT/jobs/.world-reward-h100.lock"
flock -n 9
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 2
NAME="wr-video-depth-real-v2-$REV"
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]] || exit 2
cleanup() {
 local status=$? id=''
 trap - EXIT INT TERM
 id="$(docker ps -aq --no-trunc --filter "name=^/$NAME$")"
 if [[ -n "$id" ]]; then
  [[ "$(docker inspect "$id" --format '{{.Image}}|{{index .Config.Labels "world_reward.video_depth.owner"}}')" == "$IMAGE|$REV" ]] || exit 1
  docker rm -f "$id" >/dev/null || status=1
 fi
 [[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]] || status=1
 printf '{"producer_revision":"%s","process_exit_code":%d,"owned_container_absent":true}\n' "$REV" "$status" > "$BASE/infer-v2-host-exit.json"
 exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
timeout --signal=TERM --kill-after=10s 1800s docker run --rm --name "$NAME" \
 --label "world_reward.video_depth.owner=$REV" --gpus all --network none --read-only \
 --user 0:0 --cap-drop ALL --security-opt no-new-privileges --cpus 8 --memory 32g \
 --tmpfs /tmp:rw,nosuid,size=1g \
 --mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$OUTPUT,dst=$OUTPUT" \
 --mount "type=bind,src=$ROOT/vendor/research/vda_metric_small_v1,dst=$ROOT/vendor/research/vda_metric_small_v1,readonly" \
 --mount "type=bind,src=$ROOT/weights/research/vda_metric_small_v1,dst=$ROOT/weights/research/vda_metric_small_v1,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/hf_home,dst=$ROOT/weights/cari4d/hf_home,readonly" \
 --mount "type=bind,src=$ROOT/vendor/research/vda_dependencies_v1,dst=$ROOT/vendor/research/vda_dependencies_v1,readonly" \
 --mount "type=bind,src=$ROOT/results/video-depth-dependencies-v1.json,dst=$ROOT/results/video-depth-dependencies-v1.json,readonly" \
 --mount "type=bind,src=$ROOT/results/video-depth-assets-v2.json,dst=$ROOT/results/video-depth-assets-v2.json,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --entrypoint /usr/bin/env "$IMAGE" -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp \
 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONPATH="$ROOT/vendor/research/vda_dependencies_v1:$CODE/src:$CODE/infra" \
 /opt/conda/bin/python -B "$CODE/infra/video_depth_real_run.py" infer_v2
