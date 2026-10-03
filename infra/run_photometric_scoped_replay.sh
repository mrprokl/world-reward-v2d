#!/usr/bin/env bash
# H100c: preserve original immutable failure, reuse only pinned public inputs.
set -euo pipefail
(( $# == 0 )) || { echo 'Native replay accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/photometric_native_v1"; OUT="$BASE/capability_scoped_v1"
OLD="$ROOT/jobs/8c68fba8d4f9034c2e19ed2fa9eac394c9f64758/run_photometric_capability/code/infra/photometric_capability.py"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
python3 - "$ROOT" "$CODE" "$OUT" "$OLD" <<'PYSAFE'
from pathlib import Path
import sys
root,code,out,old=map(Path,sys.argv[1:])
if any(p.resolve()!=p.absolute()or any(a.is_symlink()for a in(p,*p.parents))for p in(root,code,out,old)):raise ValueError('Canonical routes required')
if out.exists()or not old.is_file()or old.stat().st_mode&0o222:raise ValueError('Fresh output and original immutable source required')
if not code.is_dir()or not root.is_dir():raise ValueError('Actual source/root required')
PYSAFE
for p in "$BASE/inputs" "$BASE/automatic_masks" "$BASE/render-report.json" "$BASE/capability_v1/report.json" "$BASE/capability_replay_v1/report.json" \
 "$ROOT/vendor/video_to_data" "$ROOT/weights/cari4d/sam3d_body" "$ROOT/results/weights-acquisition.json" "$ROOT/results/mhr-finger-semantics-v4.json"; do [[ -e "$p" && ! -L "$p" ]]; done
mkdir "$OUT"; chmod 755 "$OUT"; chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 183s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 --env WANDB_MODE=disabled \
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" --mount "type=bind,src=$OLD,dst=$OLD,readonly" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly" \
 --mount "type=bind,src=$BASE/render-report.json,dst=$BASE/render-report.json,readonly" \
 --mount "type=bind,src=$BASE/capability_v1/report.json,dst=$BASE/capability_v1/report.json,readonly" \
 --mount "type=bind,src=$BASE/capability_replay_v1/report.json,dst=$BASE/capability_replay_v1/report.json,readonly" \
 --mount "type=bind,src=$ROOT/vendor/video_to_data,dst=$ROOT/vendor/video_to_data,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/sam3d_body,dst=$ROOT/weights/cari4d/sam3d_body,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --mount "type=bind,src=$ROOT/results/mhr-finger-semantics-v4.json,dst=$ROOT/results/mhr-finger-semantics-v4.json,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/photometric_scoped_replay.py"
