#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || { echo 'Quality takes no arguments; explicit immutable configurations required' >&2;exit 2; }
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
PUBLIC_PINS="$CODE/configs/root5_rgb_public_pins_v1.json"
QUALITY_PINS="$CODE/configs/root5_rgb_quality_pins_v1.json"
BASE="$ROOT/validation/root5_rgb_v1";OUT="$BASE/quality_v1"
# Command substitution propagates failed Python validation under set -e.
SOURCE_LIST="$(python3 - "$ROOT" "$CODE" "$OUT" "$PUBLIC_PINS" "$QUALITY_PINS" <<'PYSAFE'
import json,re,sys
from pathlib import Path
root,code,out,public_path,quality_path=map(Path,sys.argv[1:])
if root.resolve()!=root.absolute()or not root.is_dir()or code.resolve()!=code.absolute():raise ValueError('Canonical root/code required')
if out.exists()or any(p.is_symlink()or(p.exists()and not p.is_dir())for p in(out,*out.parents)):raise ValueError('Exclusive quality output required')
for p in(public_path,quality_path):
 if p.parent!=code/'configs'or p.resolve()!=p.absolute()or not p.is_file()or p.stat().st_mode&0o222:raise ValueError('Explicit immutable code/configs pins required')
public=json.loads(public_path.read_text());quality=json.loads(quality_path.read_text());paths=set()
for role in('fit','replay'):
 rev=quality[role+'_revision']
 if type(rev)is not str or not re.fullmatch('[0-9a-f]{40}',rev):raise ValueError('Explicit immutable source revision required')
 paths.add(root/'jobs'/rev/'run_root5_rgb_fit'/'code'/'infra'/'root5_rgb_fit.py')
for name in('root5_rgb_render.py','identity_rgb_render.py','joint_rgb_render.py'):
 paths.add(root/'jobs'/'077a03910d3568634c3b0d0e0b8ae0978303ce28'/'run_root5_rgb_prepare'/'code'/'infra'/name)
revisions=public['producer_revision']
if revisions!={'masks':'8084688a4d84bbad9ba8c0580e4d1b2803745511','baseline':'feae71ea16a1d942f08e95ccafc131b6467dffb9','dwpose':'feae71ea16a1d942f08e95ccafc131b6467dffb9'}:
 raise ValueError('Exact completed public producer revisions required')
paths.add(root/'jobs'/revisions['masks']/'run_root5_rgb_observe'/'code'/'infra'/'root5_rgb_observe.py')
paths.add(root/'jobs'/revisions['baseline']/'run_root5_rgb_observe'/'code'/'infra')
for p in sorted(paths):
 if p.resolve()!=p.absolute()or not p.exists()or any(q.is_symlink()for q in(p,*p.parents))or(p.is_file()and p.stat().st_mode&0o222):
  raise ValueError('Immutable canonical source-only producer path required')
 print(p)
PYSAFE
)"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
SOURCES=();while IFS= read -r SOURCE;do SOURCES+=(--mount "type=bind,src=$SOURCE,dst=$SOURCE,readonly");done <<< "$SOURCE_LIST"
MOUNTS=()
for path in "$ROOT/vendor/video_to_data" "$ROOT/vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py" \
 "$ROOT/weights/cari4d/sam3d_body" "$ROOT/weights/mhr/mhr_model.pt" \
 "$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal" \
 "$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c" \
 "$ROOT/weights/grounding_dino" "$ROOT/weights/sam2" "$ROOT/results/image-grounding.json" \
 "$ROOT/results/weights-acquisition.json" "$ROOT/results/mhr-finger-semantics-v4.json" \
 "$ROOT/weights/dwpose_native_v1" "$ROOT/results/dwpose-wheel-audit-v3" "$ROOT/results/dwpose-acquisition-v1.json" \
 "$ROOT/results/dwpose-wheel-audit-v2/report.json" "$ROOT/validation/dwpose_smoke_v1/report.json" "$ROOT/validation/dwpose_smoke_v2/report.json" \
 "$BASE/inputs" "$BASE/automatic_masks" "$BASE/baseline_v1/report.json" "$BASE/baseline_v2" "$BASE/dwpose_v1" \
 "$BASE/root_fit_v2" "$BASE/root_fit_v1/report.json" "$BASE/root_replay_v1" "$BASE/eval_private";do
 [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
mkdir "$OUT";chmod 755 "$OUT";chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 123s docker run --rm --network none --cpus 4 --memory 8g \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" --env CUDA_VISIBLE_DEVICES= \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" "${SOURCES[@]}" "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" \
 "$IMAGE" "$CODE/infra/root5_rgb_evaluate.py" --public-pins "$PUBLIC_PINS" --quality-pins "$QUALITY_PINS"
