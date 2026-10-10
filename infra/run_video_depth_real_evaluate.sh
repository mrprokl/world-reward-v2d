#!/usr/bin/env bash
# Source closure: /infra/video_depth_real_run.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ $# == 0 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_video_depth_real_evaluate/code" \
 && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
BASE="$ROOT/validation/video_depth_real_v1"; OUTPUT="$ROOT/validation/video_depth_real_output_v1"
[[ -f "$OUTPUT/predictions/report.json" && ! -e "$OUTPUT/evaluation" ]] || exit 2
timeout --signal=TERM --kill-after=10s 600s docker run --rm --name "wr-video-depth-eval-$REV" \
 --network none --read-only --user 0:0 --cap-drop ALL --security-opt no-new-privileges \
 --cpus 4 --memory 8g --tmpfs /tmp:rw,nosuid,size=256m \
 --mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly" \
 --mount "type=bind,src=$BASE,dst=$BASE,readonly" \
 --mount "type=bind,src=$OUTPUT,dst=$OUTPUT" \
 --entrypoint /usr/bin/env "$IMAGE" -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp \
 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONPATH="$CODE/src:$CODE/infra" \
 /opt/conda/bin/python -B "$CODE/infra/video_depth_real_run.py" evaluate
