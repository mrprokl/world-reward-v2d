#!/usr/bin/env bash
# Original full native export only: exact public files and producer namespaces RO.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
EPISODE="$2"
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
printf -v PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$PADDED"; OUT="$BASE/cari_shared_export_v1"
PIN="$CODE/configs/cari_clip_${PADDED}_input_pins.json"
python3 - "$ROOT" "$CODE" "$OUT" "$PIN" <<'PYSAFE'
from pathlib import Path
import sys
root,code,out,pin=map(Path,sys.argv[1:])
if (not root.is_dir() or not code.is_dir() or not pin.is_file()
    or any(p.resolve()!=p.absolute() or any(a.is_symlink() for a in (p,*p.parents)) for p in (root,code,out,pin))):
 raise ValueError('Canonical root/code/pin/output required')
if out.exists() or not out.parent.is_dir():raise ValueError('Exclusive full export under existing episode required')
PYSAFE
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
SOURCES="$(PYTHONPATH="$CODE/src:$CODE/infra" python3 - "$PIN" "$EPISODE" <<'PYPATHS'
import json,sys
from pathlib import Path
from cari_clip_inputs import PublicClipSpec,source_paths,validate_pins
pins=json.loads(Path(sys.argv[1]).read_text());spec=PublicClipSpec(**pins['clip_spec'])
if spec.episode_index!=int(sys.argv[2]):raise ValueError('Explicit episode differs from pinned clip')
validate_pins(spec,pins)
print('\n'.join(sorted(source_paths(spec))))
PYPATHS
)"
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly")
while IFS= read -r relative; do
 path="$ROOT/$relative"; [[ -f "$path" && ! -L "$path" ]]
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done <<< "$SOURCES"
# Explicit authenticated historical source is provenance only, never imported.
HISTORICAL_PIN="$CODE/configs/cari_clip_${PADDED}_historical_source_pins.json"
CONTAINER_USER="$(id -u scenesmith):$(id -g scenesmith)"; HISTORICAL_ROOT=0
if [[ -e "$HISTORICAL_PIN" || -L "$HISTORICAL_PIN" ]];then
 [[ -f "$HISTORICAL_PIN" && ! -L "$HISTORICAL_PIN" ]] || exit 1
 # The container authenticates this readonly provenance before substantive work.
 CONTAINER_USER=0:0; HISTORICAL_ROOT=1
 HISTORICAL="$ROOT/jobs/672b10ee5d8b8532686cf39ccd44134adc26178b/run_cari_full_refine_queued/code"
 for path in "$HISTORICAL" "${HISTORICAL%/code}/revision" "${HISTORICAL%/code}/source-sha256" \
  "$ROOT/results/episode3-queued-source-cache-audit.json";do
  [[ -e "$path" && ! -L "$path" ]] || exit 1
  MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
 done
fi
for stage in prepare forward refined; do
 [[ -f "$CODE/configs/cari_clip_${PADDED}_shared_${stage}_pins.json" ]]
 path="$BASE/cari_shared_${stage}_v1"; [[ -d "$path" && ! -L "$path" ]]
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
for path in "$ROOT/vendor/video_to_data" "$ROOT/vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py" \
 "$ROOT/weights/cari4d/sam3d_body" "$ROOT/weights/mhr/mhr_model.pt" "$ROOT/results/weights-acquisition.json" \
 "$ROOT/weights/cari4d/refinement" "$ROOT/results/cari-refinement-assets.json"; do
 [[ -e "$path" && ! -L "$path" ]]; MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
mkdir "$OUT"; chmod 755 "$OUT"; chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 603s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$CONTAINER_USER" --entrypoint python \
 --env "WR_HISTORICAL_READONLY_ROOT=$HISTORICAL_ROOT" \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 \
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/cari_full_export.py" --episode "$EPISODE"
