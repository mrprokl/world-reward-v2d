#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || { echo 'Paired keypoint quality accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/keypoint_rgb_v1"; OUT="$BASE/quality_v1"
python3 - "$ROOT" "$BASE" "$OUT" <<'PYSAFE'
from pathlib import Path
import sys
root,base,out=map(Path,sys.argv[1:])
if root.resolve()!=root.absolute() or not root.is_dir():raise ValueError('Canonical task root required')
for path in (base,out):
 if path.is_symlink() or any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in path.parents):raise ValueError('No symlink traversal')
if out.exists():raise FileExistsError('Exclusive quality output required')
for name in ('inputs','automatic_masks','baseline_v1','dwpose_v1','root_fit_v1','eval_private'):
 path=base/name
 if not path.is_dir() or path.is_symlink():raise ValueError('Every frozen stage must exist')
PYSAFE
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
mkdir "$OUT"; chmod 755 "$OUT"; chown scenesmith:scenesmith "$OUT"
MOUNTS=()
# Hash-only producer text, not imported.
for NAME in keypoint_rgb_fit.py keypoint_rgb_render.py identity_rgb_render.py joint_rgb_render.py; do
 if [[ "$NAME" == keypoint_rgb_fit.py ]]; then
  SOURCE="$ROOT/jobs/$REV/run_keypoint_rgb_fit/code/infra/$NAME"
 else
  SOURCE="$ROOT/jobs/42f457fc46fbeb814befb48b8104403b06922f63/run_keypoint_rgb_prepare/code/infra/$NAME"
 fi
 python3 - "$SOURCE" <<'PYSOURCE'
from pathlib import Path
import sys
p=Path(sys.argv[1])
if not p.is_file() or p.resolve()!=p.absolute() or p.stat().st_mode & 0o222:raise ValueError('Immutable source-only producer file required')
PYSOURCE
 MOUNTS+=(--mount "type=bind,src=$SOURCE,dst=$SOURCE,readonly")
done
for path in "$ROOT/vendor/video_to_data" "$ROOT/vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py" \
 "$ROOT/weights/cari4d/sam3d_body" "$ROOT/weights/mhr/mhr_model.pt" \
 "$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal" \
 "$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c" \
 "$ROOT/weights/dwpose_native_v1" "$ROOT/results/weights-acquisition.json" "$ROOT/results/mhr-finger-semantics-v4.json" \
 "$ROOT/results/dwpose-wheel-audit-v3" "$ROOT/results/dwpose-acquisition-v1.json" \
 "$ROOT/results/dwpose-wheel-audit-v2/report.json" "$ROOT/validation/dwpose_smoke_v1/report.json" \
 "$ROOT/validation/dwpose_smoke_v2/report.json" \
 "$ROOT/jobs/776e479c0a3492078313b93b8bf92e7443b274cf/run_keypoint_rgb_dwpose/code/infra" \
 "$BASE/inputs" "$BASE/automatic_masks" "$BASE/baseline_v1" "$BASE/dwpose_v1" "$BASE/root_fit_v1" "$BASE/eval_private"; do
 [[ -e "$path" && ! -L "$path" ]]
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
timeout --signal=TERM --kill-after=10s 123s docker run --rm --network none --cpus 4 --memory 8g \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" --env CUDA_VISIBLE_DEVICES= \
 --env "PYTHONPATH=$CODE/src:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" "${MOUNTS[@]}" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/keypoint_rgb_evaluate.py"
