#!/usr/bin/env bash
# Manufacture public RGB, then automatic masks and Body in isolated containers.
set -euo pipefail
(( $# == 0 )) || { echo 'Photometric capability accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/photometric_native_v1"; export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
MASK_IMAGE="$(docker image inspect world-reward/grounding:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
[[ "$MASK_IMAGE" == sha256:53b33bc4b60e0e3e8f83b401775b4701b18eef54408fd585fbe3a5d376c042e1 ]]
python3 - "$ROOT" "$CODE" "$BASE" "$MASK_IMAGE" <<'PYSAFE'
import json,sys
from pathlib import Path
root,code,base=map(Path,sys.argv[1:4])
if not root.is_dir()or root.resolve()!=root.absolute()or not code.is_dir()or code.resolve()!=code.absolute():raise ValueError('Canonical root/code required')
for p in (base,base/'inputs',base/'automatic_masks',base/'capability_v1'):
 if p.exists()or any(a.is_symlink()or(a.exists()and not a.is_dir())for a in (p,*p.parents)):raise ValueError('Fresh nonsymlink entire route required')
for name in ('photometric_render.py','photometric_capability.py'):
 p=code/'infra'/name
 if not p.is_file()or p.is_symlink()or p.stat().st_mode&0o222:raise ValueError('Immutable actual stage source required')
if json.loads((root/'results/image-grounding.json').read_text()).get('Id')!=sys.argv[4]:raise ValueError('Exact original grounding build required')
PYSAFE
for path in "$ROOT/weights/mhr" "$ROOT/results/mhr-finger-semantics-v4.json" \
 "$ROOT/weights/grounding_dino" "$ROOT/weights/sam2" "$ROOT/results/image-grounding.json" "$ROOT/results/weights-acquisition.json" \
 "$ROOT/vendor/video_to_data" "$ROOT/weights/cari4d/sam3d_body"; do [[ -e "$path" && ! -L "$path" ]]; done
mkdir "$BASE"; chmod 700 "$BASE"; chown scenesmith:scenesmith "$BASE"
COMMON=(--rm --gpus all --network none --memory 32g --cpus 4 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV"
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 --env WANDB_MODE=disabled
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4
 --mount "type=bind,src=$CODE,dst=$CODE,readonly")
printf 'phase=manufacture utc=%s\n' "$(date -u +%FT%TZ)"
timeout --signal=TERM --kill-after=10s 123s docker run "${COMMON[@]}" --env "WR_IMAGE_ID=$IMAGE" \
 --mount "type=bind,src=$ROOT/weights/mhr,dst=$ROOT/weights/mhr,readonly" \
 --mount "type=bind,src=$ROOT/results/mhr-finger-semantics-v4.json,dst=$ROOT/results/mhr-finger-semantics-v4.json,readonly" \
 --mount "type=bind,src=$BASE,dst=$BASE" "$IMAGE" "$CODE/infra/photometric_render.py"
OUT="$BASE/automatic_masks"; [[ ! -e "$OUT" && ! -L "$OUT" ]]; mkdir "$OUT"; chmod 755 "$OUT"; chown scenesmith:scenesmith "$OUT"
printf 'phase=masks utc=%s\n' "$(date -u +%FT%TZ)"
timeout --signal=TERM --kill-after=10s 123s docker run "${COMMON[@]}" --env "WR_IMAGE_ID=$MASK_IMAGE" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$ROOT/weights/grounding_dino,dst=$ROOT/weights/grounding_dino,readonly" \
 --mount "type=bind,src=$ROOT/weights/sam2,dst=$ROOT/weights/sam2,readonly" \
 --mount "type=bind,src=$ROOT/results/image-grounding.json,dst=$ROOT/results/image-grounding.json,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$MASK_IMAGE" "$CODE/infra/photometric_capability.py" --stage masks
OUT="$BASE/capability_v1"; [[ ! -e "$OUT" && ! -L "$OUT" ]]; mkdir "$OUT"; chmod 755 "$OUT"; chown scenesmith:scenesmith "$OUT"
printf 'phase=body utc=%s\n' "$(date -u +%FT%TZ)"
timeout --signal=TERM --kill-after=10s 183s docker run "${COMMON[@]}" --env "WR_IMAGE_ID=$IMAGE" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly" \
 --mount "type=bind,src=$ROOT/vendor/video_to_data,dst=$ROOT/vendor/video_to_data,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/sam3d_body,dst=$ROOT/weights/cari4d/sam3d_body,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --mount "type=bind,src=$ROOT/results/mhr-finger-semantics-v4.json,dst=$ROOT/results/mhr-finger-semantics-v4.json,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/photometric_capability.py" --stage body
printf 'phase=complete utc=%s\n' "$(date -u +%FT%TZ)"
