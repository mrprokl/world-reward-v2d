#!/usr/bin/env bash
# Source closure: /infra/native_joint_real.py
# Durable offline exact-native pilot; original frozen sources remain RO.
set +x
set -euo pipefail
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(9|1|14)$ ]] || exit 2
EPISODE="$2"; printf -v PADDED '%06d' "$EPISODE"
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
SOURCE=052ba1554e9a573d566713a99a61d89a5f27681c
FRONTEND=b658881079871508c6b3ec14d001dc1299956996
TRACK=09f516d9085f0d64d725b46b0ad6aa52fe7c0d84
CONTACT=40183b3a83ba59080c192c3cdf2db9e1d76ef021
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_native_joint_real/code" \
 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
BASE="$ROOT/results/native-joint-real-$REV"
OUT="$BASE/episode_$PADDED"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Reuse existing shared GPU lease read-only; never truncate lock contents.
exec 8<"$ROOT/jobs/.world-reward-h100.lock"
flock -n 8
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 2
MOUNTS=(--mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly")
for path in \
 "$ROOT/experiments/full4d-v1-$SOURCE" \
 "$ROOT/results/gemini-sam31-$FRONTEND" \
 "$ROOT/results/sequence-pose-probe-$TRACK-v2" \
 "$ROOT/results/sequence-pose-contact-$CONTACT" \
 "$ROOT/vendor/video_to_data" "$ROOT/weights/cari4d/sam3d_body" \
 "$ROOT/weights/cari4d/refinement" "$ROOT/results/cari-refinement-assets.json" \
 "$ROOT/results/weights-acquisition.json" \
 "$ROOT/weights/dwpose_native_v1" "$ROOT/results/dwpose-acquisition-v1.json" \
 "$ROOT/results/dwpose-wheel-audit-v2" "$ROOT/results/dwpose-wheel-audit-v3" \
 "$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_$PADDED.mp4"; do
 [[ -e "$path" && ! -L "$path" ]] || exit 2
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
# Base may hold a different sealed episode from THIS producer, never arbitrary files.
/usr/bin/python3 -I -B - "$BASE" "$OUT" <<'PYOWN'
import os,pathlib,stat,sys
base,out=map(pathlib.Path,sys.argv[1:])
if any(p.resolve()!=p or any(q.is_symlink() for q in(p,*p.parents)) for p in(base,out)):raise SystemExit(2)
if base.exists():
 s=base.stat()
 if not stat.S_ISDIR(s.st_mode) or s.st_uid!=0 or stat.S_IMODE(s.st_mode)!=0o755:raise SystemExit(2)
 for p in base.iterdir():
  if p.name not in {'episode_000001','episode_000009','episode_000014'} or p.is_symlink() or not p.is_dir():raise SystemExit(2)
  done=p/'host-exit.json'
  if not done.is_file() or done.is_symlink() or done.stat().st_mode&0o222:raise SystemExit(2)
else:base.mkdir(mode=0o755)
out.mkdir(mode=0o755)
PYOWN
NAME="wr-native-joint-$REV-$PADDED"; CIDFILE="$OUT/.container.cid"
cleanup() {
 local status=$? cid='' ids='' found='' absent=false
 trap - EXIT TERM INT
 if [[ -f "$CIDFILE" && ! -L "$CIDFILE" ]]; then
  cid="$(cat "$CIDFILE")"
  if [[ "$cid" =~ ^[0-9a-f]{64}$ ]]; then
   ids="$(timeout 20s docker ps -aq --no-trunc --filter "id=$cid")" || status=1
   if [[ "$ids" == "$cid" ]]; then
    found="$(timeout 20s docker inspect "$cid" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.native_joint.owner"}}')" || status=1
    if [[ "$found" == "$IMAGE|/$NAME|$REV" ]]; then timeout 20s docker rm -f "$cid" >/dev/null || status=1
    else status=1; fi
   elif [[ -n "$ids" ]]; then status=1; fi
   ids="$(timeout 20s docker ps -aq --no-trunc --filter "id=$cid")" || status=1
   [[ -z "$ids" ]] && absent=true
  else status=1; fi
  chmod 444 "$CIDFILE"
 else
  ids="$(timeout 20s docker ps -aq --no-trunc --filter "name=^/$NAME$")" || status=1
  [[ -z "$ids" ]] && absent=true
 fi
 printf '{"producer_revision":"%s","container_absence_verified":%s,"process_exit_code":%d}\n' \
  "$REV" "$absent" "$status" > "$OUT/host-exit.json"
 chmod 444 "$OUT/host-exit.json"
 exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[[ -z "$(docker ps -aq --no-trunc --filter "name=^/$NAME$")" ]] || exit 2
timeout --signal=TERM --kill-after=10s 1803s docker run --rm --cidfile "$CIDFILE" \
 --name "$NAME" --label "world_reward.native_joint.owner=$REV" --gpus all \
 --network none --read-only --user 0:0 --cap-drop ALL --security-opt no-new-privileges \
 --memory 64g --cpus 4 --tmpfs /tmp:rw,nosuid,size=1g \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" \
 --entrypoint /usr/bin/env "$IMAGE" -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp \
 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MOMENTUM_ENABLED=0 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" \
 PYTHONPATH="$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" \
 /opt/conda/bin/python -B "$CODE/infra/native_joint_real.py" --episode "$EPISODE"
