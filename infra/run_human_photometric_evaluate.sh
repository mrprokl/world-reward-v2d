#!/usr/bin/env bash
# One private evaluation after complete immutable public native predictions.
set -euo pipefail
(( $# == 0 )) || { echo 'Human photometric evaluation accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/human_photometric_v1"; OUT="$BASE/quality_v1"; PIN="$CODE/configs/human_photometric_quality_pins.json"
SOURCE_LIST="$(python3 - "$ROOT" "$CODE" "$OUT" "$PIN" <<'PYSAFE'
import json,re,sys
from pathlib import Path
root,code,out,pin=map(Path,sys.argv[1:])
if not root.is_dir()or not code.is_dir()or any(p.resolve()!=p.absolute()or any(a.is_symlink()for a in(p,*p.parents))for p in(root,code,out,pin)):
 raise ValueError('Canonical root/code/output/config required')
if out.exists()or any(p.exists()and not p.is_dir()for p in(out,*out.parents)):raise ValueError('Exclusive evaluation output required')
if not pin.is_file()or pin.stat().st_mode&0o222 or pin.parent!=code/'configs':raise ValueError('Actual immutable code/configs pins required')
data=json.loads(pin.read_text())
if type(data)is not dict or set(data)!={'schema','render','masks','body'}or data['schema']!='world-reward-human-photometric-quality-pins-v1':raise ValueError('Complete explicit producer pins required')
paths=set()
for role in('render','masks','body'):
 row=data[role]
 if type(row)is not dict or set(row)!={'producer_revision','report_sha256','report_bytes','script_sha256'}:raise ValueError('Exact producer pin fields required')
 for k,n in(('producer_revision',40),('report_sha256',64),('script_sha256',64)):
  if type(row[k])is not str or not re.fullmatch('[0-9a-f]{'+str(n)+'}',row[k]):raise ValueError('Actual source/receipt hashes required')
 if type(row['report_bytes'])is not int or row['report_bytes']<=0:raise ValueError('Positive receipt size required')
 stem,name=('run_human_photometric_prepare','human_photometric_render.py')if role=='render'else('run_human_photometric_observe','human_photometric_observe.py')
 path=root/'jobs'/row['producer_revision']/stem/'code'/'infra'/name
 if not path.is_file()or path.stat().st_mode&0o222 or path.resolve()!=path.absolute()or any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('Actual canonical immutable producer source required')
 paths.add(path)
if data['masks']['producer_revision']!=data['body']['producer_revision']or data['masks']['script_sha256']!=data['body']['script_sha256']:raise ValueError('Same immutable observer required')
for path in sorted(paths):print(path)
PYSAFE
)"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly")
while IFS= read -r path; do MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly"); done <<< "$SOURCE_LIST"
for path in "$BASE/inputs" "$BASE/automatic_masks" "$BASE/predictions_v1" "$BASE/eval_private" \
 "$ROOT/vendor/video_to_data" "$ROOT/weights/cari4d/sam3d_body" \
 "$ROOT/weights/grounding_dino" "$ROOT/weights/sam2" "$ROOT/results/image-grounding.json" \
 "$ROOT/results/weights-acquisition.json" "$ROOT/results/mhr-finger-semantics-v4.json"; do
 [[ -e "$path" && ! -L "$path" ]]; MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
mkdir "$OUT"; chmod 755 "$OUT"; chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 123s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 \
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/human_photometric_evaluate.py" --quality-pins "$PIN"
