#!/usr/bin/env bash
# Source closure: /infra/camera_render.py /src/world_reward/raster_capacity.py
# Allocation-only gate, exact procedural + sealed external inferred geometry.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_raster_capacity_gate/code" \
 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
[[ $# == 12 ]] || exit 2
OP=''; BP=''
for ((i=1; i<=12; i+=2)); do
 j=$((i+1)); key="${!i}"; value="${!j}"
 case "$key" in
  --external-object-report) [[ -z "$OP" ]] || exit 2; OP="$value";;
  --external-body-report) [[ -z "$BP" ]] || exit 2; BP="$value";;
  --external-object-report-bytes|--external-body-report-bytes) [[ "$value" =~ ^[1-9][0-9]*$ ]] || exit 2;;
  --external-object-report-sha256|--external-body-report-sha256) [[ "$value" =~ ^[0-9a-f]{64}$ ]] || exit 2;;
  *) exit 2;;
 esac
done
[[ "$OP" == "$ROOT/results/"*/object/report.json && "$BP" == "$ROOT/results/"*/body_depth/report.json ]] || exit 2
OUT="$ROOT/results/raster-capacity-gate-$REV"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
exec 8<"$ROOT/jobs/.world-reward-h100.lock"
flock -n 8
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 2
MOUNTS=(--mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly")
for path in "$OP" "${OP%/*}/object.glb" "${OP%/*}/transform.json" "$BP" "${BP%/*}/gauge.json"; do
 [[ -f "$path" && ! -L "$path" ]] || exit 2
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
mkdir -m 755 "$OUT"
NAME="wr-raster-capacity-$REV"; CID="$OUT/.container.cid"
cleanup() {
 local status=$? id='' absent=false found=''
 trap - EXIT TERM INT
 if [[ -f "$CID" && ! -L "$CID" ]]; then
  id="$(cat "$CID")"
  if [[ "$id" =~ ^[0-9a-f]{64}$ ]]; then
   found="$(timeout 20s docker ps -aq --no-trunc --filter "id=$id")" || status=1
   if [[ "$found" == "$id" ]]; then
    found="$(timeout 20s docker inspect "$id" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.raster_capacity.owner"}}')" || status=1
    if [[ "$found" == "$IMAGE|/$NAME|$REV" ]]; then timeout 20s docker rm -f "$id" >/dev/null || status=1
    else status=1; fi
   elif [[ -n "$found" ]]; then status=1; fi
   found="$(timeout 20s docker ps -aq --no-trunc --filter "id=$id")" || status=1
   [[ -z "$found" ]] && absent=true
  else status=1; fi
  chmod 444 "$CID"
 fi
 printf '{"producer_revision":"%s","container_absence_verified":%s,"process_exit_code":%d}\n' \
  "$REV" "$absent" "$status" > "$OUT/host-exit.json"
 chmod 444 "$OUT/host-exit.json"
 exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]] || exit 2
timeout --signal=TERM --kill-after=10s 1203s docker run --rm --cidfile "$CID" \
 --name "$NAME" --label "world_reward.raster_capacity.owner=$REV" --gpus all \
 --network none --read-only --user 0:0 --cap-drop ALL --security-opt no-new-privileges \
 --memory 64g --cpus 4 --shm-size 2g --tmpfs /tmp:rw,nosuid,exec,size=1g \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" \
 --entrypoint /usr/bin/env "$IMAGE" -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp \
 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" \
 PYTHONPATH="$CODE/src:$CODE/infra" /opt/conda/bin/python -B "$CODE/infra/camera_render.py" \
 --capacity-gate --root "$ROOT" --report "$OUT/report.json" "$@"
