#!/usr/bin/env bash
# H101 render-only stage. Inference receives public inputs in a later container.
set -euo pipefail
(( $# == 0 )) || { echo 'Human photometric prepare accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/human_photometric_v1"; export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
python3 - "$ROOT" "$CODE" "$BASE" <<'PYSAFE'
from pathlib import Path
import sys
root,code,base=map(Path,sys.argv[1:])
if not root.is_dir()or root.resolve()!=root.absolute()or not code.is_dir()or code.resolve()!=code.absolute():raise ValueError('Canonical root/code required')
if base.exists()or any(p.is_symlink()or(p.exists()and not p.is_dir())for p in(base,*base.parents)):raise ValueError('Fresh nonsymlink manufacturing root required')
PYSAFE
for path in "$ROOT/weights/mhr/mhr_model.pt" \
 "$ROOT/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt" \
 "$ROOT/results/weights-acquisition.json" "$ROOT/results/mhr-finger-semantics-v4.json"; do [[ -f "$path" && ! -L "$path" ]]; done
mkdir "$BASE"; chmod 700 "$BASE"; chown scenesmith:scenesmith "$BASE"
timeout --signal=TERM --kill-after=10s 123s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/weights/mhr/mhr_model.pt,dst=$ROOT/weights/mhr/mhr_model.pt,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt,dst=$ROOT/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --mount "type=bind,src=$ROOT/results/mhr-finger-semantics-v4.json,dst=$ROOT/results/mhr-finger-semantics-v4.json,readonly" \
 --mount "type=bind,src=$BASE,dst=$BASE" "$IMAGE" "$CODE/infra/human_photometric_render.py"
