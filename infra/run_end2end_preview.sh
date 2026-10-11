#!/usr/bin/env bash
# Source closure: /infra/end2end_preview.py /infra/full4d_video.py /infra/full4d_publish.py /infra/terminal_success.py
set +x
set -euo pipefail
[[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ && "$1" =~ ^(render|publish|render-and-publish)$ ]] || exit 2
MODE="$1"; TARGET="$2"; shift 2
KIND=joint; KIND_SEEN=0; WAIT_FOR=''
while (( $# )); do
 case "$1" in
  --candidate-kind)
   [[ $# -ge 2 && "$KIND_SEEN" == 0 && "$2" =~ ^(joint|continuation|sqp|smoothing)$ ]] || exit 2
   KIND="$2"; KIND_SEEN=1; shift 2 ;;
  --after-terminal)
   [[ $# -ge 2 && -z "$WAIT_FOR" && "$2" =~ ^world-reward-[a-z0-9][a-z0-9-]{0,80}(\.service)?$ ]] || exit 2
   WAIT_FOR="${2%.service}.service"; shift 2 ;;
  *) exit 2 ;;
 esac
done
PREFIX=native-joint-real
[[ "$KIND" != continuation ]] || PREFIX=native-contact-continuation-real
[[ "$KIND" != sqp ]] || PREFIX=native-pose-sqp-real
[[ "$KIND" != smoothing ]] || PREFIX=pose-smoothing-real
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_end2end_preview/code" && "$(hostname)" == scenesmith-ncc-h100-01 ]] || exit 2
source_identity() {
 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/infra" python3 -B - "$ROOT" "$CODE" "$REV" <<'PYSOURCE'
from pathlib import Path
import sys
from mediapipe_cpu_runtime_verify import source
value=source(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],'run_end2end_preview',
 ('infra/end2end_preview.py','infra/run_end2end_preview.sh','infra/terminal_success.py'))
print(value['closure_sha256'])
PYSOURCE
}
if [[ -n "$WAIT_FOR" ]]; then
 BEFORE="$(source_identity)"
 env PYTHONDONTWRITEBYTECODE=1 python3 -B "$CODE/infra/terminal_success.py" "$WAIT_FOR"
 [[ "$(source_identity)" == "$BEFORE" ]] || exit 2
fi
if [[ "$MODE" == render-and-publish ]]; then
 # Original render wrapper releases/verifies its OWN GPU container before the
 # CPU publication. Same immutable ENTRY/source, no refit or new media method.
 bash "$CODE/infra/run_end2end_preview.sh" render "$TARGET" --candidate-kind "$KIND"
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT/results/end2end-preview-$REV/host-exit.json" <<'PYEXIT'
from pathlib import Path
import json,stat,sys
p=Path(sys.argv[1]);s=p.lstat()
if p.resolve()!=p or p.is_symlink()or not stat.S_ISREG(s.st_mode)or s.st_nlink!=1 or s.st_mode&0o222 or not 0<s.st_size<=4096:raise ValueError('Sealed original render host cleanup required')
r=json.loads(p.read_bytes())
if r!={'process_exit_code':0,'owned_container_absent':True}or type(r['process_exit_code'])is not int or r['owned_container_absent']is not True:raise ValueError('Successful owned render cleanup required before CPU publication')
PYEXIT
 exec bash "$CODE/infra/run_end2end_preview.sh" publish "$REV" --candidate-kind "$KIND"
fi
ARGS=("$MODE" "$TARGET" --candidate-kind "$KIND")
if [[ "$MODE" == publish ]]; then
 exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 PYTHONPATH="$CODE/src:$CODE/infra" PYTHONDONTWRITEBYTECODE=1 python3 -B "$CODE/infra/end2end_preview.py" "${ARGS[@]}"
fi
exec 8<"$ROOT/jobs/.world-reward-h100.lock"; flock -n 8
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
OUT="$ROOT/results/end2end-preview-$REV"; [[ ! -e "$OUT" ]] || exit 2; mkdir -m 755 "$OUT"
NAME="wr-end2end-preview-$REV"
cleanup() {
 local status=$? cid=''; trap - EXIT
 if [[ -f "$OUT/.container.cid" ]]; then
  cid="$(cat "$OUT/.container.cid")"; [[ "$cid" =~ ^[0-9a-f]{64}$ ]] || exit 1
  if [[ -n "$(docker ps -aq --no-trunc --filter "id=$cid")" ]]; then
   [[ "$(docker inspect "$cid" --format '{{.Image}}|{{index .Config.Labels "world_reward.preview.owner"}}')" == "$IMAGE|$REV" ]] || exit 1
   docker rm -f "$cid" >/dev/null || status=1
  fi
  [[ -z "$(docker ps -aq --no-trunc --filter "id=$cid")" ]] || status=1
 fi
 printf '{"process_exit_code":%d,"owned_container_absent":true}\n' "$status" > "$OUT/host-exit.json"
 chmod 444 "$OUT/host-exit.json"; exit "$status"
}
trap cleanup EXIT; trap 'exit 143' TERM; trap 'exit 130' INT
SOURCE_MOUNTS=()
if [[ "$KIND" == continuation ]]; then
 SOURCE_MOUNTS+=(--mount "type=bind,src=$ROOT/jobs/$TARGET/run_native_contact_continuation_real,dst=$ROOT/jobs/$TARGET/run_native_contact_continuation_real,readonly")
elif [[ "$KIND" == smoothing ]]; then
 SOURCE_MOUNTS+=(--mount "type=bind,src=$ROOT/jobs/$TARGET/run_pose_smoothing_real,dst=$ROOT/jobs/$TARGET/run_pose_smoothing_real,readonly")
elif [[ "$KIND" == sqp ]]; then
 SOURCE_MOUNTS+=(--mount "type=bind,src=$ROOT/jobs/$TARGET/run_native_pose_sqp_real,dst=$ROOT/jobs/$TARGET/run_native_pose_sqp_real,readonly")
fi
timeout --signal=TERM --kill-after=10s 1200s docker run --rm --cidfile "$OUT/.container.cid" \
 --name "$NAME" --label "world_reward.preview.owner=$REV" --gpus all --network none --read-only \
 --user 0:0 --cap-drop ALL --security-opt no-new-privileges --cpus 4 --memory 16g --tmpfs /tmp:rw,nosuid,size=256m \
 --mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly" \
 --mount "type=bind,src=$ROOT/experiments/full4d-v1-052ba1554e9a573d566713a99a61d89a5f27681c,dst=$ROOT/experiments/full4d-v1-052ba1554e9a573d566713a99a61d89a5f27681c,readonly" \
 --mount "type=bind,src=$ROOT/results/$PREFIX-$TARGET,dst=$ROOT/results/$PREFIX-$TARGET,readonly" "${SOURCE_MOUNTS[@]}" \
 --mount "type=bind,src=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera,dst=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" \
 --entrypoint /usr/bin/env "$IMAGE" -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp \
 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONPATH="$CODE/src:$CODE/infra" \
 /opt/conda/bin/python -B "$CODE/infra/end2end_preview.py" "${ARGS[@]}"
