#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || { echo 'Diagnostic accepts no arguments; explicit completed pins required' >&2;exit 2; }
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/factorial_rgb_v1";OUT="$BASE/quality_v1"
PUBLIC_PINS="$CODE/configs/factorial_rgb_public_pins_v1.json"
QUALITY_PINS="$CODE/configs/factorial_rgb_quality_pins_v1.json"
SOURCE_LIST="$(python3 - "$ROOT" "$CODE" "$OUT" "$PUBLIC_PINS" "$QUALITY_PINS" <<'PYSAFE'
import json,re,sys
from pathlib import Path
root,code,out,pp,qp=map(Path,sys.argv[1:])
if root.resolve()!=root.absolute()or not root.is_dir()or code.resolve()!=code.absolute():raise ValueError('Canonical root/code required')
if out.exists()or any(p.is_symlink()or(p.exists()and not p.is_dir())for p in(out,*out.parents)):raise ValueError('Exclusive diagnostic output required')
for p in(pp,qp):
 if p.parent!=code/'configs'or p.resolve()!=p.absolute()or not p.is_file()or p.stat().st_mode&0o222:raise ValueError('Explicit immutable code/configs pins required')
public=json.loads(pp.read_text());quality=json.loads(qp.read_text());paths=set()
for role in('render','body','dwpose','masks'):
 pin=public['automatic_masks']if role=='masks'else quality[role];rev=pin['producer_revision']
 if type(rev)is not str or not re.fullmatch('[0-9a-f]{40}',rev):raise ValueError('Canonical completed producer required')
 stem,name=('run_factorial_rgb_prepare','factorial_rgb_render.py')if role=='render'else('run_factorial_rgb_observe','factorial_rgb_observe.py')
 paths.add(root/'jobs'/rev/stem/'code'/'infra'/name)
for p in sorted(paths):
 if p.resolve()!=p.absolute()or not p.is_file()or p.stat().st_mode&0o222 or any(q.is_symlink()for q in(p,*p.parents)):
  raise ValueError('Immutable canonical source-only producer required')
 print(p)
PYSAFE
)"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly")
while IFS= read -r SOURCE;do MOUNTS+=(--mount "type=bind,src=$SOURCE,dst=$SOURCE,readonly");done <<< "$SOURCE_LIST"
for path in "$ROOT/vendor/video_to_data" "$ROOT/weights/cari4d/sam3d_body" \
 "$ROOT/weights/grounding_dino" "$ROOT/weights/sam2" "$ROOT/results/image-grounding.json" \
 "$ROOT/results/weights-acquisition.json" "$ROOT/results/mhr-finger-semantics-v4.json" \
 "$ROOT/weights/dwpose_native_v1" "$ROOT/results/dwpose-wheel-audit-v3" "$ROOT/results/dwpose-acquisition-v1.json" \
 "$ROOT/results/dwpose-wheel-audit-v2/report.json" "$ROOT/validation/dwpose_smoke_v1/report.json" "$ROOT/validation/dwpose_smoke_v2/report.json" \
 "$BASE/inputs" "$BASE/automatic_masks" "$BASE/body_v1" "$BASE/dwpose_v1" "$BASE/eval_private";do
 [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
mkdir "$OUT";chmod 755 "$OUT";chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 123s docker run --rm --network none --cpus 4 --memory 8g \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python --env CUDA_VISIBLE_DEVICES= \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/factorial_rgb_evaluate.py" --public-pins "$PUBLIC_PINS" --quality-pins "$QUALITY_PINS"
