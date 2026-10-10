#!/usr/bin/env bash
# Source closure: /infra/form_track1_alias_bank.py /infra/form_hoi_external_dev.py
set +x
set -euo pipefail
[[ $# == 1 && "$1" =~ ^(build|publish)$ ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_form_track1_alias_bank/code" && "$(hostname)" == scenesmith-ncc-h100-01 ]] || exit 2
BASE=/srv/world-reward-data/track1_rgb_alias_bank_v1
if [[ "$1" == publish ]]; then
 exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 PYTHONPATH="$CODE/src:$CODE/infra" PYTHONDONTWRITEBYTECODE=1 python3 -B "$CODE/infra/form_track1_alias_bank.py" publish
fi
[[ ! -e "$BASE" ]] || exit 2; mkdir -p -m 755 "$BASE"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
NAME="wr-track1-alias-bank-$REV"
# No GPU lease is needed: exact RGB decoding only, in parallel with native fit.
timeout --signal=TERM --kill-after=10s 610s docker run --rm --name "$NAME" \
 --label "world_reward.alias_bank.owner=$REV" --network none --read-only --cpus 4 --memory 2g \
 --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,size=64m \
 --mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly" \
 --mount "type=bind,src=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera,dst=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera,readonly" \
 --mount "type=bind,src=$ROOT/results/input-manifest.json,dst=$ROOT/results/input-manifest.json,readonly" \
 --mount "type=bind,src=$BASE,dst=$BASE" --entrypoint /usr/bin/env "$IMAGE" -i \
 PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
 WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONPATH="$CODE/src:$CODE/infra" \
 /opt/conda/bin/python -B "$CODE/infra/form_track1_alias_bank.py" build
