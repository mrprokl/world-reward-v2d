#!/usr/bin/env bash
set +x
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_precision_volume_gate/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_precision_volume_gate.sh" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
OUT="$ROOT/validation/precision_volume_v1"; JOB="${CODE%/code}"
[[ ! -L "$ROOT/validation" && ! -e "$OUT" && ! -L "$OUT" ]]
IMAGE="sha256:c8fb1632a6908a82aeeeb73c36a00f17a26b81f95f4d2d498c53f75894e21137"
HELPER="$ROOT/vendor/v2d_submission_kit/v2dlb/mesh_budget.py"
python3 - "$CODE/infra/precision_volume_gate.py" "$HELPER" <<'PY'
import ast,hashlib,re,sys
from pathlib import Path
p=Path(sys.argv[1]);tree=ast.parse(p.read_text())
pin=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BINARY_SHA' for t in n.targets))
assert isinstance(pin,ast.Constant) and isinstance(pin.value,str) and re.fullmatch('[0-9a-f]{64}',pin.value)
helper=Path(sys.argv[2]);assert helper.is_file() and not any(p.is_symlink()for p in(helper,*helper.parents))
assert helper.stat().st_size==2031 and hashlib.sha256(helper.read_bytes()).hexdigest()=='42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0'
PY
[[ "$(docker image inspect --format '{{.Id}}' "$IMAGE")" == "$IMAGE" ]]
NAME="world-reward-precision-volume-${REV:0:12}"
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]]
mkdir -m 700 "$OUT"; chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
CIDDIR="$(mktemp -d "$ROOT/validation/precision-volume-cid.XXXXXXXX")"; CIDFILE="$CIDDIR/container.cid"
finish() {
 STATUS=$?; trap - EXIT INT TERM; set +e; CLEANUP_OK=1
 if [[ -f "$CIDFILE" && ! -L "$CIDFILE" ]];then
  CID="$(cat "$CIDFILE")"
  if [[ "$CID" =~ ^[0-9a-f]{64}$ ]];then
   IDS="$(timeout 5s docker ps -aq --no-trunc --filter "id=$CID")"; [[ $? == 0 ]] || CLEANUP_OK=0
   if [[ -n "$IDS" ]];then
    OWNED="$(timeout 5s docker inspect "$CID" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}')"
    if [[ $? == 0 && "$IDS" == "$CID" && "$OWNED" == "$IMAGE|/$NAME|run_precision_volume_gate|$REV" ]];then
     timeout 10s docker stop --time 5 "$CID" >/dev/null || true
     LEFT="$(timeout 5s docker ps -aq --no-trunc --filter "id=$CID")"; [[ $? == 0 ]] || CLEANUP_OK=0
     if [[ "$LEFT" == "$CID" ]];then timeout 5s docker kill "$CID" >/dev/null; timeout 5s docker rm -f "$CID" >/dev/null || CLEANUP_OK=0;elif [[ -n "$LEFT" ]];then CLEANUP_OK=0;fi
    else CLEANUP_OK=0;fi
   fi
   LEFT="$(timeout 5s docker ps -aq --no-trunc --filter "id=$CID")"; [[ $? == 0 && -z "$LEFT" ]] || CLEANUP_OK=0
  else CLEANUP_OK=0;fi
 else CLEANUP_OK=0;fi
 if [[ "$CLEANUP_OK" == 1 ]];then rm -f "$CIDFILE"; rmdir "$CIDDIR" || STATUS=1;else STATUS=1;fi
 python3 -I -B - "$OUT" "$STATUS" "$CLEANUP_OK" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'host-cleanup.json'
with p.open('x')as h:json.dump({'status':'pass'if sys.argv[2]=='0'and sys.argv[3]=='1'else'fail','container_absent':sys.argv[3]=='1','job_exit_code':int(sys.argv[2])},h)
p.chmod(0o444)
PY
 [[ $? == 0 ]] || STATUS=1
 exit "$STATUS"
}
trap finish EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
# Native source closure: /infra/mesh_volume_qem.cpp /infra/mesh_guarded_qem.cpp.
# Only the actual audited mesh_budget.py is mounted; no v2dlb directory/data.
# Its verified 2031-byte source imports only future/numpy and lazy
# fast_simplification/trimesh from the immutable image; no sibling dependencies.
set +e
timeout --signal=TERM --kill-after=5s 3603s docker run --rm --name "$NAME" --cidfile "$CIDFILE" \
 --label world-reward.job=run_precision_volume_gate --label "world-reward.revision=$REV" \
 --network none --read-only --cap-drop ALL --security-opt no-new-privileges --memory 16g --cpus 4 \
 --tmpfs /tmp:rw,noexec,nosuid,nodev,size=512m --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE="$CODE" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly" --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly" \
 --mount "type=bind,src=$ROOT/results/image-volume-qem.json,dst=$ROOT/results/image-volume-qem.json,readonly" \
 --mount "type=bind,src=$HELPER,dst=$HELPER,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/precision_volume_gate.py"
STATUS=$?; set -e; exit "$STATUS"
