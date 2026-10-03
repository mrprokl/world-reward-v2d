#!/usr/bin/env bash
# One public-only H101 observer stage; never expose the manufacturing namespace.
set -euo pipefail
(( $# == 2 )) && [[ "$1" == --stage ]] || { echo 'Require --stage masks|body' >&2;exit 2; }
STAGE="$2";case "$STAGE" in masks|body) ;; *) exit 2;; esac
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/human_photometric_v1";export DOCKER_HOST="unix://$ROOT/docker.sock"
if [[ "$STAGE" == masks ]];then OUT="$BASE/automatic_masks";TAG=world-reward/grounding:0.1;LIMIT=123;else OUT="$BASE/predictions_v1";TAG=world-reward/cari4d-source:0.1;LIMIT=603;fi
python3 - "$ROOT" "$CODE" "$OUT" <<'PYSAFE'
from pathlib import Path
import sys
root,code,out=map(Path,sys.argv[1:])
if not root.is_dir()or not code.is_dir()or any(p.resolve()!=p.absolute()or any(a.is_symlink()for a in(p,*p.parents))for p in(root,code,out)):
 raise ValueError('Canonical source/root/output required')
if out.exists()or any(p.exists()and not p.is_dir()for p in(out,*out.parents)):raise ValueError('Fresh stage output required')
PYSAFE
IMAGE="$(docker image inspect "$TAG" --format '{{.Id}}')"
if [[ "$STAGE" == masks ]];then
 [[ "$IMAGE" == sha256:53b33bc4b60e0e3e8f83b401775b4701b18eef54408fd585fbe3a5d376c042e1 ]]
 PATHS=("$ROOT/weights/grounding_dino" "$ROOT/weights/sam2" "$ROOT/results/image-grounding.json" "$ROOT/results/weights-acquisition.json")
else
 [[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
 PATHS=("$BASE/automatic_masks" "$ROOT/vendor/video_to_data" "$ROOT/weights/cari4d/sam3d_body" "$ROOT/results/weights-acquisition.json" "$ROOT/results/mhr-finger-semantics-v4.json")
fi
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly" --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly")
for p in "${PATHS[@]}";do [[ -e "$p" && ! -L "$p" ]];MOUNTS+=(--mount "type=bind,src=$p,dst=$p,readonly");done
mkdir "$OUT";chmod 755 "$OUT";chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s "${LIMIT}s" docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 --env WANDB_MODE=disabled \
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/human_photometric_observe.py" --stage "$STAGE"
