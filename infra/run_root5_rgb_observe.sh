#!/usr/bin/env bash
set -euo pipefail
(( $# == 1 )) || { echo 'Require stage masks|baseline|dwpose' >&2; exit 2; }
STAGE="$1"; case "$STAGE" in masks|baseline|dwpose) ;; *) exit 2 ;; esac
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Runtime source inventories require these unchanged helper wrappers as code.
test -f "$CODE/infra/run_dwpose_smoke.sh"
test -f "$CODE/infra/dwpose_acquire.py"
test -f "$CODE/infra/dwpose_wheel_audit.py"
test -f "$CODE/infra/run_keypoint_rgb_dwpose.sh"
BASE="$ROOT/validation/root5_rgb_v1"
case "$STAGE" in
 masks) FOLDER=automatic_masks; BUDGET=183; MEMORY=32g; TAG=world-reward/grounding:0.1 ;;
 baseline) FOLDER=baseline_v1; BUDGET=603; MEMORY=32g; TAG=world-reward/cari4d-source:0.1 ;;
 dwpose) FOLDER=dwpose_v1; BUDGET=183; MEMORY=8g; TAG=world-reward/cari4d-source:0.1 ;;
esac
OUT="$BASE/$FOLDER"
python3 - "$ROOT" "$BASE" "$OUT" <<'PYSAFE'
from pathlib import Path
import sys
root,base,out=map(Path,sys.argv[1:])
if not root.is_dir()or root.resolve()!=root.absolute():raise ValueError('Canonical root required')
if out.exists()or any(p.is_symlink()or(p.exists()and not p.is_dir())for p in(out,*out.parents)):
 raise ValueError('Fresh non-symlink output required')
if not(base/'inputs').is_dir()or(base/'inputs').is_symlink():raise ValueError('Public RGB directory required')
PYSAFE
IMAGE="$(docker image inspect "$TAG" --format '{{.Id}}')"
if [[ "$STAGE" == masks ]];then [[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]];else
 [[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]];fi
mkdir "$OUT";chmod 755 "$OUT";chown scenesmith:scenesmith "$OUT"
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly" --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly")
GPUTAGS=()
case "$STAGE" in
 masks)
  GPUTAGS=(--gpus all)
  for path in "$ROOT/weights/grounding_dino" "$ROOT/weights/sam2" "$ROOT/results/image-grounding.json" "$ROOT/results/weights-acquisition.json";do
   [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly");done ;;
 baseline)
  GPUTAGS=(--gpus all)
  for path in "$ROOT/vendor/video_to_data" "$ROOT/vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py" \
   "$ROOT/weights/cari4d/sam3d_body" "$ROOT/weights/mhr/mhr_model.pt" \
   "$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal" \
   "$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c" \
   "$ROOT/results/weights-acquisition.json" "$ROOT/results/mhr-finger-semantics-v4.json" "$BASE/automatic_masks";do
   [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly");done ;;
 dwpose)
  for path in "$ROOT/weights/dwpose_native_v1" "$ROOT/results/dwpose-wheel-audit-v3" \
   "$ROOT/results/dwpose-acquisition-v1.json" "$ROOT/results/dwpose-wheel-audit-v2/report.json" \
   "$ROOT/validation/dwpose_smoke_v1/report.json" "$ROOT/validation/dwpose_smoke_v2/report.json" \
   "$BASE/automatic_masks/report.json";do
   [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly");done
  for c in 00 01 02;do for f in 000 001 002 003 004;do
   path="$BASE/automatic_masks/clip_${c}_frame_${f}_human.png"
   [[ -f "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
  done;done ;;
esac
CPU_ENV=();if [[ "$STAGE" == dwpose ]];then CPU_ENV=(--env CUDA_VISIBLE_DEVICES=);fi
timeout --signal=TERM --kill-after=10s "${BUDGET}s" docker run --rm --network none --cpus 4 --memory "$MEMORY" \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python "${GPUTAGS[@]}" "${CPU_ENV[@]}" \
 --env "WR_ROOT=$ROOT" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 --env WANDB_MODE=disabled \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/root5_rgb_observe.py" "$STAGE"
