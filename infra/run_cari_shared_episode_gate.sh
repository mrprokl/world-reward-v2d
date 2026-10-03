#!/usr/bin/env bash
# CPU-only real frozen full-native loader; sole new output is its JSON receipt.
set -euo pipefail
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
EPISODE="$2"
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_cari_shared_episode_gate/code" ]]
printf -v PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$PADDED";OUT="$BASE/cari_shared_episode_v1"
PIN="$CODE/configs/cari_clip_${PADDED}_input_pins.json"
EXPORT_PIN="$CODE/configs/cari_clip_${PADDED}_shared_export_pins.json"
python3 -I -B - "$ROOT" "$CODE" "$OUT" "$PIN" "$EXPORT_PIN" <<'PYSAFE'
from pathlib import Path
import stat,sys
root,code,out,pin,exported=map(Path,sys.argv[1:])
for path in (root,code,out,pin,exported):
 if path.absolute()!=path or path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
  raise ValueError('Canonical source/pins/exclusive consumer output required')
if not root.is_dir() or not code.is_dir() or out.exists() or not out.parent.is_dir():
 raise ValueError('Absent consumer output under existing full episode required')
for path in (pin,exported,code/'infra/cari_shared_episode_gate.py',code/'infra/run_cari_shared_episode_gate.sh'):
 if not path.is_file() or not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_mode&0o222:
  raise ValueError('Committed actual readonly consumer/pins required')
PYSAFE
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
# Host source-list bootstrap stays stdlib-only: no NumPy/consumer/model import.
SOURCES="$(PYTHONPATH="$CODE/src:$CODE/infra" PYTHONDONTWRITEBYTECODE=1 python3 - "$PIN" "$EPISODE" <<'PYPATHS'
import json,sys
from pathlib import Path
from cari_clip_inputs import PublicClipSpec,source_paths,validate_pins
pins=json.loads(Path(sys.argv[1]).read_text());spec=PublicClipSpec(**pins['clip_spec'])
if spec.episode_index!=int(sys.argv[2]):raise ValueError('Explicit episode differs from pinned full clip')
validate_pins(spec,pins)
print('\n'.join(sorted(source_paths(spec))))
PYPATHS
)"
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly")
while IFS= read -r relative;do
 path="$ROOT/$relative";[[ -f "$path" && ! -L "$path" ]]
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done <<< "$SOURCES"
if [[ -f "$CODE/configs/cari_clip_${PADDED}_historical_source_pins.json" ]];then
 HISTORICAL="$ROOT/jobs/672b10ee5d8b8532686cf39ccd44134adc26178b/run_cari_full_refine_queued/code"
 for path in "$HISTORICAL" "${HISTORICAL%/code}/revision" "${HISTORICAL%/code}/source-sha256" \
  "$ROOT/results/episode3-queued-source-cache-audit.json";do
  [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
 done
fi
for stage in prepare forward refined export;do
 [[ -f "$CODE/configs/cari_clip_${PADDED}_shared_${stage}_pins.json" ]]
 path="$BASE/cari_shared_${stage}_v1";[[ -d "$path" && ! -L "$path" ]]
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
for path in "$ROOT/vendor/video_to_data" "$ROOT/vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py" \
 "$ROOT/weights/mhr/mhr_model.pt" "$ROOT/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3" \
 "$ROOT/weights/cari4d/refinement" "$ROOT/results/weights-acquisition.json" "$ROOT/results/cari-refinement-assets.json";do
 [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
mkdir "$OUT";chmod 755 "$OUT";chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 303s docker run --rm --network none --memory 4g --cpus 2 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 \
 --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 --env MKL_NUM_THREADS=2 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/cari_shared_episode_gate.py" --episode "$EPISODE"
