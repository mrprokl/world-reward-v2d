#!/usr/bin/env bash
# Only committed source, official single RGB and independently pinned exports.
set +x
set -euo pipefail
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
EPISODE="$2"; ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
    && "$CODE" == "$ROOT/jobs/$REV/run_reconstruction_preview/code" ]]
printf -v PADDED '%06d' "$EPISODE"
OUT="$ROOT/results/reconstruction-preview-$REV-ep$PADDED"
VIDEO="$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_$PADDED.mp4"
EXPORT="$ROOT/outputs/episode_$PADDED/cari_shared_export_v1"
MANIFEST="$ROOT/results/input-manifest.json"
python3 -I -B - "$CODE" "$OUT" "$VIDEO" "$EXPORT" "$MANIFEST" <<'PY'
from pathlib import Path
import sys
for x in sys.argv[1:]:
 p=Path(x)
 if p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents)):raise ValueError('Canonical preview paths required')
code,out,video,export,manifest=map(Path,sys.argv[1:])
if out.exists()or not video.is_file()or not export.is_dir()or not manifest.is_file():raise ValueError('Exclusive preview and original input required')
out.mkdir(mode=0o755)
PY
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]]
NAME="world-reward-preview-${REV:0:12}-ep$PADDED"
timeout --signal=TERM --kill-after=10s 180s docker run --rm --name "$NAME" \
 --network none --read-only --memory 2g --cpus 2 --tmpfs /tmp:rw,noexec,nosuid,size=64m \
 --entrypoint python --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_PREVIEW_OUTPUT=$OUT" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 --env MKL_NUM_THREADS=2 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" --mount "type=bind,src=$VIDEO,dst=$VIDEO,readonly" \
 --mount "type=bind,src=$EXPORT,dst=$EXPORT,readonly" --mount "type=bind,src=$MANIFEST,dst=$MANIFEST,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/reconstruction_preview.py" --episode "$EPISODE"
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]]
