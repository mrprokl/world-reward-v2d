#!/usr/bin/env bash
# Source closure: /infra/joint_point_native_qualify.py
# Paired zero-weight qualification ONLY; caller owns the original FD9 GPU queue.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
[[ $# == 2 && "$1" == --episode && "$2" == 21 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_joint_point_native_qualify/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
[[ "$(timeout 10s docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]]
OUT="$ROOT/results/joint-point-native-qualify-$REV"
NAME="world-reward-joint-point-qualify-$REV"
BEFORE="$(PYTHONPATH="$CODE/infra" python3 - "$ROOT" "$CODE" "$REV" <<'PY'
import json,sys
from pathlib import Path
import joint_point_native_qualify as q
print(json.dumps(q.source_binding(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]),sort_keys=True))
PY
)"
PATHS="$(PYTHONPATH="$CODE/infra" python3 - "$ROOT" "$CODE" <<'PY'
import sys
from pathlib import Path
import joint_point_native_qualify as q
print('\n'.join(map(str,q.host_mounts(Path(sys.argv[1]),Path(sys.argv[2])))))
PY
)"
[[ ! -e "$OUT" && ! -L "$OUT" ]]
MOUNTS=()
while IFS= read -r path;do
 [[ -e "$path" && ! -L "$path" ]]
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done <<< "$PATHS"
# Parent scheduler must hold FD9, not an independent opportunistic GPU process.
[[ -e /proc/$$/fd/9 && "$(readlink /proc/$$/fd/9)" == "$ROOT/jobs/.world-reward-h100.lock" ]]
[[ -z "$(timeout 10s nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
mkdir "$OUT";chmod 755 "$OUT";chown scenesmith:scenesmith "$OUT"
CID="$OUT.container.cid";[[ ! -e "$CID" ]]
read_cleanup_cid() {
 python3 - "$CID" <<'PY'
from pathlib import Path
import os,re,stat,sys
p=Path(sys.argv[1]);before=p.lstat()
if (not p.is_absolute() or p.resolve()!=p or any(parent.is_symlink()for parent in(p,*p.parents))
    or not stat.S_ISREG(before.st_mode)or before.st_nlink!=1 or before.st_uid!=os.geteuid()
    or before.st_size not in(64,65)):raise ValueError('Owned regular bounded CID required')
raw=p.read_bytes();after=p.lstat()
if (not re.fullmatch(b'[0-9a-f]{64}\n?',raw)or any(getattr(before,k)!=getattr(after,k)
    for k in('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns'))):
 raise ValueError('CID bytes or identity changed')
print(raw.decode().rstrip('\n'))
PY
}
inspect_owned_container() {
 local cid="$1" inspected rc
 if inspected="$(timeout 10s docker inspect "$cid" --format '{{.Name}} {{.Config.Image}} {{index .Config.Labels "world-reward.revision"}}' 2>&1)";then
  [[ "$inspected" == "/$NAME $IMAGE $REV" ]] || return 2
  return 0
 else
  rc=$?
  if (( rc == 1 )) && [[ "$inspected" == "Error: No such object: $cid" \
    || "$inspected" == "Error: No such container: $cid" \
    || "$inspected" == "Error response from daemon: No such container: $cid" ]];then return 1;fi
  return 2
 fi
}
cleanup() {
 local status="${1:-$?}" cid rc failed
 failed=$((status == 0 ? 1 : status))
 [[ "$CID" == "$OUT.container.cid" ]] || return "$failed"
 if [[ ! -e "$CID" && ! -L "$CID" ]];then
  (( status != 0 )) && return "$status"
  return 1
 fi
 if ! cid="$(read_cleanup_cid)";then return "$failed";fi
 if inspect_owned_container "$cid";then
  if ! timeout 20s docker rm -f "$cid" >/dev/null;then return "$failed";fi
  # Removal's exit status is not absence proof. Requery the same exact CID.
  if inspect_owned_container "$cid";then rc=0;else rc=$?;fi
  (( rc == 1 )) || return "$failed"
 else
  rc=$?;(( rc == 1 )) || return "$failed"
 fi
 return "$status"
}
trap 'status=$?; if cleanup "$status";then exit "$status";else exit "$?";fi' EXIT
timeout --signal=TERM --kill-after=20s 7203s docker run --rm --cidfile "$CID" --name "$NAME" \
 --label "world-reward.revision=$REV" --gpus all --network none --read-only --cap-drop ALL \
 --security-opt no-new-privileges --memory 64g --cpus 4 --user "$(id -u scenesmith):$(id -g scenesmith)" \
 --tmpfs /tmp:rw,nosuid,nodev,size=2g --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env MOMENTUM_ENABLED=0 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/joint_point_native_qualify.py" --episode 21
PYTHONPATH="$CODE/infra" python3 - "$ROOT" "$CODE" "$REV" "$OUT" "$BEFORE" <<'PY'
import json,sys
from pathlib import Path
import joint_point_native_qualify as q
root,code,out=map(Path,(sys.argv[1],sys.argv[2],sys.argv[4]));rt=q.runtime()
if q.source_binding(root,code,sys.argv[3])!=json.loads(sys.argv[5]):raise ValueError('Full source changed')
report=rt.strict((out/'report.json').read_bytes())
expected={'status':'pass','phase':'complete','constructor_attempts':2,'constructor_returns':2,
 'probe_attempts':4,'probe_returns':4,'run_attempts':2,'run_returns':2,'actual_native_updates_total':602,
 'exact_initial_state_loss_gradient_parity':True,'exact_full_result_history_parity':True,
 'source_inputs_assets_rehashed_after':True,'quality_verified':False,'tracker_executed':False,
 'positive_weight_executed':False,'adoption':False}
if any(type(report.get(k))is not type(v)or report[k]!=v for k,v in expected.items()):raise ValueError('Complete pair required')
if {p.name for p in out.iterdir()}!={'report.json'}:raise ValueError('No new prediction bundles permitted')
if not 0<report['elapsed_seconds']<=q.BUDGET:raise ValueError('Inclusive native pair deadline')
PY
