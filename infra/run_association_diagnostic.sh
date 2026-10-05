#!/usr/bin/env bash
# All complete frozen exports; CPU-only diagnostics cannot alter predictions.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_association_diagnostic/code" ]]
OUT="$ROOT/results/association-diagnostic-$REV"
[[ ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"; chmod 755 "$OUT"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]]
NAME="world-reward-association-diagnostic-${REV:0:12}"
timeout --signal=TERM --kill-after=10s 903s docker run --rm --name "$NAME" \
 --network none --read-only --memory 3g --cpus 2 --tmpfs /tmp:rw,noexec,nosuid,size=64m \
 --entrypoint python --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_DIAGNOSTIC_OUTPUT=$OUT" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp \
 --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 --env MKL_NUM_THREADS=2 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/association_diagnostic.py"
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]]
