#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || { echo 'Keypoint RGB baseline accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
BASE="$ROOT/validation/keypoint_rgb_v1"
OUT="$BASE/baseline_v1"
python3 - "$ROOT" "$BASE" "$OUT" <<'PYSAFE'
from pathlib import Path
import sys
root,base,out=map(Path,sys.argv[1:])
if not root.is_dir() or root.resolve()!=root.absolute():raise ValueError('Canonical root required')
if out.exists() or any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in (out,*out.parents)):raise ValueError('Fresh nonsymlink baseline output required')
PYSAFE
[[ ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -L "$BASE" && -d "$BASE/inputs" && -d "$BASE/automatic_masks" ]]
[[ ! -e "$OUT" && ! -L "$OUT" ]]
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
UID_SCENE="$(id -u scenesmith)"; GID_SCENE="$(id -g scenesmith)"
mkdir "$OUT"; chown "$UID_SCENE:$GID_SCENE" "$OUT"
COMMON=(--rm --network none --cpus 4 --user "$UID_SCENE:$GID_SCENE" --entrypoint python
 --env "WR_ROOT=$ROOT" --env "WR_CODE_REVISION=$WR_CODE_REVISION" --env "WR_IMAGE_ID=$IMAGE"
 --env "PYTHONPATH=$CODE/src:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 --env WANDB_MODE=disabled
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp
 --mount "type=bind,src=$CODE,dst=$CODE,readonly"
 --mount "type=bind,src=$ROOT/vendor/video_to_data,dst=$ROOT/vendor/video_to_data,readonly"
 --mount "type=bind,src=$ROOT/vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py,dst=$ROOT/vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py,readonly"
 --mount "type=bind,src=$ROOT/weights/cari4d/sam3d_body,dst=$ROOT/weights/cari4d/sam3d_body,readonly"
 --mount "type=bind,src=$ROOT/weights/mhr/mhr_model.pt,dst=$ROOT/weights/mhr/mhr_model.pt,readonly"
 --mount "type=bind,src=$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal,dst=$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal,readonly"
 --mount "type=bind,src=$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c,dst=$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c,readonly"
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly"
 --mount "type=bind,src=$ROOT/results/mhr-finger-semantics-v4.json,dst=$ROOT/results/mhr-finger-semantics-v4.json,readonly"
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly"
 --mount "type=bind,src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly")
timeout --signal=TERM --kill-after=10s 603s docker run "${COMMON[@]}" --gpus all --memory 32g \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/keypoint_rgb_baseline.py"
