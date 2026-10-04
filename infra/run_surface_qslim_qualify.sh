#!/usr/bin/env bash
# Source closure: /infra/surface_qslim_qualify.py /infra/official_track1_pack_gate.py
# Source closure: /infra/official_pack_geometry.py /infra/mediapipe_cpu_runtime_verify.py
# Source closure: /src/world_reward/surface_identity.py /src/world_reward/raw_shape_proposal.py
# Source closure: /infra/surface_identity_qualify.py /infra/surface_qslim_build.py /infra/surface_qslim.cpp
# Source closure: /configs/surface_qslim_build_pins.json /configs/surface_identity_qualification_pins.json
set +x
set -euo pipefail
[[ $# == 0 && "$(uname -s)" == Linux && "$(id -u)" == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_surface_qslim_qualify/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_surface_qslim_qualify.sh" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock" PYTHONDONTWRITEBYTECODE=1
DRIVER="$CODE/infra/surface_qslim_qualify.py"
BEFORE="$(timeout 45s python3 -I -B "$DRIVER" proof)"
PATHS="$(timeout 10s python3 -I -B "$DRIVER" mounts)"
IMAGE=sha256:1a04b1930f713ef9ffb411489e80ddebbce59a5ce26e713add4095cd9b5303f0
[[ "$(timeout 5s docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]]
OUT="$ROOT/results/surface-qslim-qualify-$REV";CID="$OUT.container.cid"
NAME="world-reward-surface-qslim-$REV"
[[ ! -e "$OUT" && ! -L "$OUT" && ! -e "$CID" && ! -L "$CID" ]]
[[ -d "$ROOT/results" && ! -L "$ROOT/results" ]]
MOUNTS=()
while IFS= read -r path;do
 [[ -e "$path" && ! -L "$path" ]]
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done <<< "$PATHS"
mkdir -m 755 "$OUT";chown 1000:1000 "$OUT"
inspect_owned() {
 local cid="$1" value rc
 if value="$(timeout 5s docker inspect "$cid" --format '{{.Name}} {{.Image}} {{index .Config.Labels "world-reward.revision"}}' 2>&1)";then
  [[ "$value" == "/$NAME $IMAGE $REV" ]] || return 2
  return 0
 else
  rc=$?
  if (( rc == 1 )) && [[ "$value" == "Error: No such object: $cid" \
   || "$value" == "error: no such object: $cid" \
   || "$value" == "Error: No such container: $cid" \
   || "$value" == "Error response from daemon: No such container: $cid" ]];then return 1;fi
  return 2
 fi
}
cleanup() {
 local cid rc
 [[ -f "$CID" && ! -L "$CID" && "$(stat -c '%u:%h:%s' "$CID")" =~ ^0:1:6[45]$ ]] || return 1
 cid="$(cat "$CID")";[[ "$cid" =~ ^[0-9a-f]{64}$ ]] || return 1
 if inspect_owned "$cid";then
  timeout 10s docker rm -f "$cid" >/dev/null || return 1
  if inspect_owned "$cid";then return 1;else rc=$?;(( rc == 1 )) || return 1;fi
 else rc=$?;(( rc == 1 )) || return 1;fi
 chmod 400 "$CID"
}
trap 'status=$?;if cleanup;then exit "$status";else (( status != 0 )) && exit "$status";exit 1;fi' EXIT
STATUS=0
timeout --signal=TERM --kill-after=5s 302s docker run --rm --cidfile "$CID" --name "$NAME" \
 --label "world-reward.revision=$REV" --network none --read-only --cap-drop ALL \
 --security-opt no-new-privileges --memory 16g --cpus 4 --user 1000:1000 \
 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp \
 --env CUDA_VISIBLE_DEVICES=-1 --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$DRIVER" native || STATUS=$?
CLEANUP=1
if ! cleanup;then CLEANUP=0;(( STATUS != 0 )) || STATUS=1;fi
trap - EXIT
timeout 45s python3 -I -B "$DRIVER" seal --before "$BEFORE" --status "$STATUS" --cleanup-verified "$CLEANUP"
