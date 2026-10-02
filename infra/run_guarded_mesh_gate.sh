#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"
CODE="${WR_CODE:?}"
# Actual binary's own source hash is checked; freeze /infra/mesh_guarded_qem.cpp.
export DOCKER_HOST="unix://$ROOT/docker.sock"
OUT="$ROOT/validation/guarded_qem_v1"
[[ ! -L "$ROOT/validation" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
IMAGE="$(python3 - "$ROOT/results/image-guarded-qem.json" <<'PY'
import json,re,sys
r=json.load(open(sys.argv[1]))
assert r['stage']=='world_reward_guarded_qem_build' and r['status']=='pass'
assert re.fullmatch(r'sha256:[0-9a-f]{64}',r['image_id'])
print(r['image_id'])
PY
)"
timeout --signal=TERM --kill-after=5s 183s docker run --rm --network none --memory 16g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/results/image-guarded-qem.json,dst=$ROOT/results/image-guarded-qem.json,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" python "$CODE/infra/guarded_mesh_gate.py"
