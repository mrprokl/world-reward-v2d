#!/usr/bin/env bash
set -euo pipefail
(( $# == 1 )) || { echo 'Require factorial stage masks|body|dwpose' >&2;exit 2; }
STAGE="$1";case "$STAGE" in masks|body|dwpose) ;; *) exit 2;; esac
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/factorial_rgb_v1";export DOCKER_HOST="unix://$ROOT/docker.sock"
case "$STAGE" in
 masks) OUT="$BASE/automatic_masks";PIN="$CODE/configs/factorial_rgb_manifest_pins_v1.json";TAG=world-reward/grounding:0.1;MEMORY=32g;SECONDS_LIMIT=183;;
 body) OUT="$BASE/body_v1";PIN="$CODE/configs/factorial_rgb_public_pins_v1.json";TAG=world-reward/cari4d-source:0.1;MEMORY=32g;SECONDS_LIMIT=303;;
 dwpose) OUT="$BASE/dwpose_v1";PIN="$CODE/configs/factorial_rgb_public_pins_v1.json";TAG=world-reward/cari4d-source:0.1;MEMORY=8g;SECONDS_LIMIT=183;;
esac
python3 - "$ROOT" "$CODE" "$OUT" "$PIN" <<'PYSAFE'
from pathlib import Path
import sys
root,code,out,pin=map(Path,sys.argv[1:])
if root.resolve()!=root.absolute()or not root.is_dir()or code.resolve()!=code.absolute():raise ValueError('Canonical root/code required')
if out.exists()or any(p.is_symlink()or(p.exists()and not p.is_dir())for p in(out,*out.parents)):raise ValueError('Fresh nonsymlink observer output required')
if pin.resolve()!=pin.absolute()or pin.parent!=code/'configs'or not pin.is_file()or pin.stat().st_mode&0o222:raise ValueError('Immutable source-bundle pins required')
PYSAFE
IMAGE="$(docker image inspect "$TAG" --format '{{.Id}}')"
if [[ "$STAGE" != masks ]];then [[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]];fi
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly" --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly")
GPUTAGS=();CPU_ENV=();PATHS=()
if [[ "$STAGE" == masks ]];then
 GPUTAGS=(--gpus all);PATHS=("$ROOT/weights/grounding_dino" "$ROOT/weights/sam2" "$ROOT/results/image-grounding.json" "$ROOT/results/weights-acquisition.json")
else
 MASK_REV="$(python3 - "$PIN" <<'PYPIN'
import json,re,sys
row=json.load(open(sys.argv[1]))['automatic_masks'];rev=row['producer_revision']
if type(rev)is not str or not re.fullmatch('[0-9a-f]{40}',rev):raise ValueError('Explicit historical mask producer required')
print(rev)
PYPIN
 )"
 PATHS=("$ROOT/jobs/$MASK_REV/run_factorial_rgb_observe/code/infra/factorial_rgb_observe.py" "$BASE/automatic_masks/report.json")
 if [[ "$STAGE" == body ]];then
  GPUTAGS=(--gpus all);PATHS+=("$ROOT/vendor/video_to_data" "$ROOT/weights/cari4d/sam3d_body" "$ROOT/results/weights-acquisition.json" "$ROOT/results/mhr-finger-semantics-v4.json" "$BASE/automatic_masks")
 else
  CPU_ENV=(--env CUDA_VISIBLE_DEVICES=);PATHS+=("$ROOT/weights/dwpose_native_v1" "$ROOT/results/dwpose-wheel-audit-v3" "$ROOT/results/dwpose-acquisition-v1.json" "$ROOT/results/dwpose-wheel-audit-v2/report.json" "$ROOT/validation/dwpose_smoke_v1/report.json" "$ROOT/validation/dwpose_smoke_v2/report.json")
  for g in 00 01 02 03 04 05 06 07;do for f in 000 001 002;do PATHS+=("$BASE/automatic_masks/group_${g}_frame_${f}_human.png");done;done
 fi
fi
for path in "${PATHS[@]}";do [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly");done
mkdir "$OUT";chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
timeout --signal=TERM --kill-after=10s "${SECONDS_LIMIT}s" docker run --rm --network none --cpus 4 --memory "$MEMORY" \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python "${GPUTAGS[@]}" "${CPU_ENV[@]}" \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 --env WANDB_MODE=disabled --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/factorial_rgb_observe.py" "$STAGE" --public-pins "$PIN"
