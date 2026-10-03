#!/usr/bin/env bash
# CPU-only one-episode unmodified official packer smoke; scratch payloads deleted.
# Original sample row_id-only projection happens inside our source-bound gate.
# The official reader receives a separate row_id-only CSV, never sample XYZ.
set -euo pipefail
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
EPISODE="$2"
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_official_track1_pack_gate/code" ]]
printf -v PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$PADDED";OUT="$BASE/official_track1_pack_smoke_v2"
PIN="$CODE/configs/cari_clip_${PADDED}_input_pins.json"
EXPORT_PIN="$CODE/configs/cari_clip_${PADDED}_shared_export_pins.json"
RUNTIME_PIN="$CODE/configs/official_pack_runtime_pins.json"
[[ -f "$RUNTIME_PIN" && ! -L "$RUNTIME_PIN" ]]
python3 -I -B - "$ROOT" "$CODE" "$OUT" "$PIN" "$EXPORT_PIN" "$REV" "${BASH_SOURCE[0]}" <<'PYSAFE'
from pathlib import Path
import re,stat,sys
root,code,out,pin,exported=map(Path,sys.argv[1:6])
revision=sys.argv[6];executing=Path(sys.argv[7])
if executing!=code/"infra/run_official_track1_pack_gate.sh" or executing.resolve()!=executing:
 raise ValueError("Actual immutable launcher entrypoint required")
if (code.parent/"revision").read_text().strip()!=revision or not re.fullmatch("[0-9a-f]{64}",(code.parent/"source-sha256").read_text().strip()):
 raise ValueError("Exact immutable source revision/archive markers required")
for path in code.rglob("*"):
 if path.is_symlink() or (not path.is_dir() and not stat.S_ISREG(path.lstat().st_mode)) or path.stat().st_mode&0o222:
  raise ValueError("Complete actual source closure must remain readonly and regular")
for path in (root,code,out,pin,exported):
 if path.absolute()!=path or path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
  raise ValueError('Canonical source/pins/exclusive consumer output required')
if not root.is_dir() or not code.is_dir() or out.exists() or not out.parent.is_dir():
 raise ValueError('Absent consumer output under existing full episode required')
for path in (pin,exported,code/'infra/official_track1_pack_gate.py',code/'infra/run_official_track1_pack_gate.sh'):
 if not path.is_file() or not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_mode&0o222:
  raise ValueError('Committed actual readonly consumer/pins required')
PYSAFE
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Entire runtime JSON/receipt/helper binding is stdlib-only, before Docker or
# output reservation. The builder is never imported or executed here.
PINNED_IMAGE="$(python3 -I -B - "$ROOT" "$CODE" <<'PYRUNTIME'
import sys
from pathlib import Path
root,code=map(Path,sys.argv[1:])
sys.path[:0]=[str(code/'infra'),str(code/'src')]
from official_track1_pack_gate import load_runtime
print(load_runtime(root,code)['pins']['image_id'])
PYRUNTIME
)"
IMAGE="$(docker image inspect world-reward/official-pack-cpu:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == "$PINNED_IMAGE" ]]
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
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly"
 --mount "type=bind,src=$ROOT/results/official-pack-image-build.json,dst=$ROOT/results/official-pack-image-build.json,readonly")
while IFS= read -r relative;do
 path="$ROOT/$relative";[[ -f "$path" && ! -L "$path" ]]
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done <<< "$SOURCES"
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
# Source-only official namespace closure, not the full kit/scorers/assets tree.
for relative in tools/pack_reconstruction.py v2dlb/report_schema.py v2dlb/mesh_budget.py \
 v2dlb/mhr_submission.py v2dlb/mesh_common.py v2dlb/mhr_metrics.py data/track_1_sample_submission.parquet;do
 path="$ROOT/vendor/v2d_submission_kit/$relative";[[ -f "$path" && ! -L "$path" ]]
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
# Existing CPU image must have the official packer's transitive dependencies.
# Source gate checks actual source bytes and no-Torch/model execution itself.
mkdir "$OUT";chmod 755 "$OUT";chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 303s docker run --rm --network none --memory 4g --cpus 2 \
 --user "$CONTAINER_USER" --entrypoint python \
 --env "WR_HISTORICAL_READONLY_ROOT=$HISTORICAL_ROOT" \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 \
 --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 --env MKL_NUM_THREADS=2 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/official_track1_pack_gate.py" --episode "$EPISODE"
