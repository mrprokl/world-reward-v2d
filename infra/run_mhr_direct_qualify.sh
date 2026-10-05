#!/usr/bin/env bash
# Source closure: /infra/mhr_direct_qualify.py /infra/mhr_direct_bridge.py /infra/mhr_direct_gradients.py
set -euo pipefail
ARGS=();export WR_MHR_DIRECT_CONTROL=named
if (( $#==1 )) && [[ "$1" == --gradients ]];then ARGS=(--gradients);export WR_MHR_DIRECT_CONTROL=gradients
elif (( $#!=0 ));then exit 2;fi
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_mhr_direct_qualify/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock" PYTHONDONTWRITEBYTECODE=1
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
STEM=mhr-direct-;NAME="wr-mhr-direct-$REV"
if [[ "$WR_MHR_DIRECT_CONTROL" == gradients ]];then STEM=mhr-direct-gradients-;NAME="wr-mhr-direct-gradients-$REV";fi
OUT="$ROOT/results/${STEM}qualify-$REV";CONTROL="$ROOT/results/${STEM}control-$REV"
LOCK="$ROOT/jobs/.world-reward-h100.lock"
host() {
 timeout --signal=TERM --kill-after=1s 30s python3 -I -B - "$ROOT" "$CODE" "$REV" "$@" <<'PYHOST'
import json,os,sys
from pathlib import Path
root,code=map(Path,sys.argv[1:3]);revision,mode=sys.argv[3:5]
sys.path.insert(0,str(code/'infra'));import mhr_direct_qualify as q
q.runtime().require(Path(q.__file__)==code/'infra/mhr_direct_qualify.py','Actual committed host caller required')
out,control=q.namespaces(root,revision)
if mode=='before':
 proof=q.host_proof(root,code,revision)
 q.runtime().require(not out.exists()and not out.is_symlink()and not control.exists()and not control.is_symlink(),'Fresh direct control required')
 control.mkdir(mode=0o755)
 q.write_receipt(control/'proof.json',proof,lambda:None)
 print('\n'.join(map(str,q.mounts(proof,code))))
elif mode=='cleanup':q.cleanup(control,q.container_name(revision),revision)
elif mode=='seal':q.seal(root,code,revision,int(sys.argv[5]),sys.argv[6]=='1')
else:raise ValueError('Explicit host control mode required')
PYHOST
}
lock_identity() {
 timeout 5s python3 -I -B - "$LOCK" "${1:-path}" <<'PYLOCK'
import os,stat,sys
from pathlib import Path
p=Path(sys.argv[1]);a=p.lstat()
if p.resolve()!=p or any(x.is_symlink()for x in(p,*p.parents))or not stat.S_ISREG(a.st_mode)or a.st_nlink!=1:raise ValueError('Original cooperative lock required')
if sys.argv[2]=='fd'and(a.st_dev,a.st_ino)!=(os.fstat(9).st_dev,os.fstat(9).st_ino):raise ValueError('Cooperative inode changed')
print(str(a.st_dev)+':'+str(a.st_ino))
PYLOCK
}
[[ "$(timeout 5s docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]]
LOCK_BEFORE="$(lock_identity)";exec 9<"$LOCK"
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]]
flock --timeout 43200 9
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]]
APPS="$(timeout 5s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]]
EXISTING="$(timeout 5s docker ps -aq --filter "name=^/$NAME$")";[[ -z "$EXISTING" ]]
PATHS="$(host before)";MOUNTS=()
while IFS= read -r path;do [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly");done <<< "$PATHS"
mkdir "$OUT";chmod 755 "$OUT";chown scenesmith:scenesmith "$OUT"
finish() {
 local status=$? check cleaned=0
 trap - EXIT INT TERM;set +e
 host cleanup;check=$?;if (( check==0 ));then cleaned=1;elif (( status==0 ));then status=1;fi
 [[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]];check=$?;if (( check!=0 && status==0 ));then status=1;fi
 host seal "$status" "$cleaned";check=$?;if (( check!=0 && status==0 ));then status=1;fi
 exec 9<&-;exit "$status"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
timeout --signal=TERM --kill-after=5s 123s docker run --rm --name "$NAME" --cidfile "$CONTROL/.container.cid" \
 --label "world_reward.mhr_direct.owner=$REV" --gpus all --network none --read-only --cap-drop ALL \
 --security-opt no-new-privileges --cpus 4 --memory 16g --user "$(id -u scenesmith):$(id -g scenesmith)" \
 --tmpfs /tmp:rw,nosuid,nodev,size=128m --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env WR_MHR_DIRECT_LEASE=fd9 --env "WR_MHR_DIRECT_CONTROL=$WR_MHR_DIRECT_CONTROL" \
 --env "PYTHONPATH=$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env HOME=/tmp \
 "${MOUNTS[@]}" --mount "type=bind,src=$CONTROL/proof.json,dst=$CONTROL/proof.json,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/mhr_direct_qualify.py" "${ARGS[@]}"
