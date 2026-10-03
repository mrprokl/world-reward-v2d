#!/usr/bin/env bash
set -euo pipefail
(( $# == 1 )) || { echo 'Require root5 operation fit|replay' >&2; exit 2; }
OPERATION="$1";case "$OPERATION" in fit|replay) ;; *) exit 2;; esac
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
BASE="$ROOT/validation/root5_rgb_v1"
if [[ "$OPERATION" == fit ]];then OUT="$BASE/root_fit_v1";TIMEOUT=303;else OUT="$BASE/root_replay_v1";TIMEOUT=123;fi
PINARGS=(--public-pins "$CODE/configs/root5_rgb_public_pins_v1.json")
if [[ "$OPERATION" == replay ]];then PINARGS+=(--fit-pins "$CODE/configs/root5_rgb_fit_pins_v1.json");fi
python3 - "$ROOT" "$OUT" <<'PYSAFE'
from pathlib import Path
import sys
root,out=map(Path,sys.argv[1:])
if root.resolve()!=root.absolute()or not root.is_dir():raise ValueError('Canonical root required')
if out.exists()or any(p.is_symlink()or(p.exists()and not p.is_dir())for p in(out,*out.parents)):raise ValueError('Fresh nonsymlink output required')
PYSAFE
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
mkdir "$OUT";chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
MOUNTS=()
for path in "$ROOT/vendor/video_to_data" "$ROOT/vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py" \
 "$ROOT/weights/cari4d/sam3d_body" "$ROOT/weights/mhr/mhr_model.pt" \
 "$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal" \
 "$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c" \
 "$ROOT/weights/dwpose_native_v1" "$ROOT/results/weights-acquisition.json" "$ROOT/results/mhr-finger-semantics-v4.json" \
 "$ROOT/weights/grounding_dino" "$ROOT/weights/sam2" "$ROOT/results/image-grounding.json" \
 "$ROOT/results/dwpose-wheel-audit-v3" "$ROOT/results/dwpose-acquisition-v1.json" \
 "$ROOT/results/dwpose-wheel-audit-v2/report.json" "$ROOT/validation/dwpose_smoke_v1/report.json" \
 "$ROOT/validation/dwpose_smoke_v2/report.json" "$BASE/inputs" "$BASE/automatic_masks" "$BASE/baseline_v1/report.json" "$BASE/baseline_v2" "$BASE/dwpose_v1" \
 "$ROOT/jobs/8084688a4d84bbad9ba8c0580e4d1b2803745511/run_root5_rgb_observe/code/infra/root5_rgb_observe.py" \
 "$ROOT/jobs/feae71ea16a1d942f08e95ccafc131b6467dffb9/run_root5_rgb_observe/code/infra";do
 [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
if [[ "$OPERATION" == replay ]];then MOUNTS+=(--mount "type=bind,src=$BASE/root_fit_v1,dst=$BASE/root_fit_v1,readonly");fi
timeout --signal=TERM --kill-after=10s "${TIMEOUT}s" docker run --rm --network none --cpus 4 --memory 32g --gpus all \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" --env "WR_CODE=$CODE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env MOMENTUM_ENABLED=0 --env WANDB_MODE=disabled --env HOME=/tmp --env XDG_CACHE_HOME=/tmp \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" "${MOUNTS[@]}" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/root5_rgb_fit.py" "$OPERATION" "${PINARGS[@]}"
