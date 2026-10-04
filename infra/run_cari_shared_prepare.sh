#!/usr/bin/env bash
# Full original native decode/replay only. Public sources RO, new output alone RW.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
EPISODE="$2"
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
printf -v PADDED '%06d' "$EPISODE"
OUT="$ROOT/outputs/episode_$PADDED/cari_shared_prepare_v1"
PIN="$CODE/configs/cari_clip_${PADDED}_input_pins.json"
python3 - "$ROOT" "$CODE" "$OUT" "$PIN" <<'PYSAFE'
from pathlib import Path
import sys
root,code,out,pin=map(Path,sys.argv[1:])
if (not root.is_dir() or not code.is_dir() or not pin.is_file()
    or any(p.resolve()!=p.absolute() or any(a.is_symlink() for a in (p,*p.parents)) for p in (root,code,out,pin))):
 raise ValueError('Canonical root/code/pin/output required')
if out.exists() or not out.parent.is_dir():raise ValueError('Exclusive full shared output under existing episode required')
PYSAFE
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
# Enumerate only the exact public fifteen from the committed pure input helper.
SOURCES="$(PYTHONPATH="$CODE/src:$CODE/infra" python3 - "$PIN" "$EPISODE" <<'PYPATHS'
import json,sys
from pathlib import Path
from cari_clip_inputs import PublicClipSpec,source_paths,source_profile,validate_pins
pins=json.loads(Path(sys.argv[1]).read_text());spec=PublicClipSpec(**pins['clip_spec'])
if spec.episode_index!=int(sys.argv[2]):raise ValueError('Explicit episode differs from pinned clip')
validate_pins(spec,pins)
print('\n'.join(sorted(source_paths(spec,object_source=source_profile(pins)))))
PYPATHS
)"
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly")
while IFS= read -r relative;do
 path="$ROOT/$relative";[[ -f "$path" && ! -L "$path" ]]
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done <<< "$SOURCES"
for path in "$ROOT/vendor/video_to_data" "$ROOT/vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py" \
 "$ROOT/weights/cari4d/sam3d_body" "$ROOT/weights/mhr/mhr_model.pt" "$ROOT/results/weights-acquisition.json";do
 [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
mkdir "$OUT";chmod 755 "$OUT";chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 603s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 \
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/cari_shared_prepare.py" --episode "$EPISODE"
