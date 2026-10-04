#!/usr/bin/env bash
set -euo pipefail
[[ ( $# == 2 || $# == 4 ) && $1 == --episode && $2 =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
EPISODE="$2"
BACKEND=volume
if [[ $# == 4 ]]; then
 [[ $3 == --backend && ( $4 == volume || $4 == conditioned ) ]] || exit 2
 BACKEND="$4"
fi
ROOT="${WR_ROOT:?}"
CODE="${WR_CODE:?}"
# Source closure: /infra/mesh_volume_qem.cpp and /infra/mesh_guarded_qem.cpp.
export DOCKER_HOST="unix://$ROOT/docker.sock"
printf -v PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$PADDED"
OUT="$BASE/object_budget_$BACKEND"
[[ ! -L "$ROOT/outputs" && ! -L "$BASE" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
IMAGE="$(python3 - "$ROOT/results/image-volume-qem.json" "$BACKEND" <<'PY'
import json,re,sys
r=json.load(open(sys.argv[1]));assert r['status']=='pass'
assert re.fullmatch('sha256:[0-9a-f]{64}',r['image_id']);print(r['image_id'])
if sys.argv[2]=='conditioned':
 assert r['image_id']=='sha256:c8fb1632a6908a82aeeeb73c36a00f17a26b81f95f4d2d498c53f75894e21137'
PY
)"
RUN=(timeout --signal=TERM --kill-after=5s 903s docker run)
EXTRA=()
if [[ $BACKEND == conditioned ]]; then
 RUN[3]=1803s
 HELPER="$ROOT/vendor/v2d_submission_kit/v2dlb/mesh_budget.py"
 [[ -f "$HELPER" && ! -L "$HELPER" ]]
 EXTRA+=(--read-only --cap-drop ALL --security-opt no-new-privileges
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=8g --env WR_CODE="$CODE"
  --mount "type=bind,src=$HELPER,dst=$HELPER,readonly")
else
 EXTRA+=(--mount "type=bind,src=$ROOT/vendor/v2d_submission_kit,dst=$ROOT/vendor/v2d_submission_kit,readonly"
  --mount "type=bind,src=$ROOT/validation/volume_qem_v1,dst=$ROOT/validation/volume_qem_v1,readonly")
fi
"${RUN[@]}" --rm --network none --memory 16g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
 --mount "type=bind,src=$ROOT/data,dst=$ROOT/data,readonly" \
 --mount "type=bind,src=$BASE,dst=$BASE,readonly" \
 "${EXTRA[@]}" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/object_budget_volume.py" "$@"
