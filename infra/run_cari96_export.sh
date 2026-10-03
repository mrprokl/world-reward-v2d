#!/usr/bin/env bash
# Direct refined96 native parameters, never LM, image inference or quality.
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
OUT="$ROOT/validation/cari96_export_v1"
python3 - "$ROOT" "$CODE" "$OUT" <<'PYSAFE'
from pathlib import Path
import sys
root,code,out=map(Path,sys.argv[1:])
if not root.is_dir() or not code.is_dir() or any(p.resolve()!=p.absolute() or any(x.is_symlink() for x in (p,*p.parents)) for p in (root,code,out)):
 raise ValueError('Canonical root/code/output required')
if out.exists() or any(p.exists() and not p.is_dir() for p in (out,*out.parents)):
 raise ValueError('Never overwrite complete or incomplete export')
PYSAFE
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly")
for path in "$ROOT/validation/cari96_public_v1" "$ROOT/validation/cari96_forward_v4" "$ROOT/validation/cari96_refined_v1" \
 "$ROOT/vendor/video_to_data" "$ROOT/vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py" \
 "$ROOT/weights/cari4d/sam3d_body" "$ROOT/weights/mhr/mhr_model.pt" "$ROOT/results/weights-acquisition.json" \
 "$ROOT/weights/cari4d/refinement" "$ROOT/results/cari-refinement-assets.json"; do
 [[ -e "$path" && ! -L "$path" ]]; MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
for name in input prepare forward refined; do [[ -f "$CODE/configs/cari96_${name}_pins.json" ]]; done
mkdir "$OUT"; chmod 755 "$OUT"; chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 183s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 \
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/cari96_export.py"
