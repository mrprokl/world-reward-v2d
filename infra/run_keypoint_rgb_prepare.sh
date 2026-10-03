#!/usr/bin/env bash
# Independent render120s then automatic RGB-only masks180s; no truth mask access.
set -euo pipefail
(( $# == 0 )) || { echo 'Keypoint RGB prepare accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/keypoint_rgb_v1"
[[ -d "$ROOT" && ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -e "$BASE" && ! -L "$BASE" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
python3 - "$ROOT" "$BASE" <<'PYSAFE'
from pathlib import Path
import sys
root,base=map(Path,sys.argv[1:])
if not root.is_dir() or root.resolve()!=root.absolute():raise ValueError('Canonical root required')
if base.exists() or any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in (base,*base.parents)):raise ValueError('Fresh nonsymlink preparation root required')
PYSAFE
mkdir "$BASE";chmod 700 "$BASE";chown scenesmith:scenesmith "$BASE"
timeout --signal=TERM --kill-after=10s 123s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" --env PYTHONPATH="$CODE/src:$CODE/infra" \
 --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/weights/mhr,dst=$ROOT/weights/mhr,readonly" \
 --mount "type=bind,src=$ROOT/results/mhr-finger-semantics-v4.json,dst=$ROOT/results/mhr-finger-semantics-v4.json,readonly" \
 --mount "type=bind,src=$BASE,dst=$BASE" "$IMAGE" "$CODE/infra/keypoint_rgb_render.py"
MASK_IMAGE="$(docker image inspect world-reward/grounding:0.1 --format '{{.Id}}')"
[[ "$MASK_IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
OUT="$BASE/automatic_masks"
[[ ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT";chmod 755 "$OUT";chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 183s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$MASK_IMAGE" --env PYTHONPATH="$CODE/src:$CODE/infra" \
 --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env WANDB_MODE=disabled \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$ROOT/weights/grounding_dino,dst=$ROOT/weights/grounding_dino,readonly" \
 --mount "type=bind,src=$ROOT/weights/sam2,dst=$ROOT/weights/sam2,readonly" \
 --mount "type=bind,src=$ROOT/results/image-grounding.json,dst=$ROOT/results/image-grounding.json,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$MASK_IMAGE" "$CODE/infra/keypoint_rgb_masks.py"
