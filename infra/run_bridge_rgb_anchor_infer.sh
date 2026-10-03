#!/usr/bin/env bash
# Public four-anchor observations; no authored/private renderer paths mounted.
# Source closure: /infra/bridge_rgb_anchor_infer.py /infra/bridge_frontend_bindings.py
# Source closure: /infra/body_smoke.py /infra/hand_synthetic_infer.py /infra/object_synthetic_observations.py
# Source closure: /infra/frontend_selected_assets.py /infra/frontend_sam2_kernel_gate.py
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_bridge_rgb_anchor_infer/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_bridge_rgb_anchor_infer.sh" \
 && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
host() { /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -B "$CODE/infra/bridge_rgb_anchor_infer.py" "$1"; }
OUT="$ROOT/validation/bridge_rgb_anchor_v1/observations_anchor_v1"
NAME="world-reward-bridge-infer-${REV:0:12}";LOCK="$ROOT/jobs/.world-reward-h100.lock";OWNED=0;LOCK_OPEN=0
query() { timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "name=^/$NAME$"; }
gpu_idle() {
 local apps
 apps="$(timeout --signal=TERM --kill-after=2s 5s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
 [[ -z "${apps//[[:space:]]/}" ]]
}
lock_identity() {
 /usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B - "$LOCK" "${1:-path}" <<'PY'
import os,stat,sys
from pathlib import Path
p=Path(sys.argv[1]);s=p.lstat()
if p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents))or not stat.S_ISREG(s.st_mode)or s.st_nlink!=1:raise ValueError('Canonical cooperative lock required')
if sys.argv[2]=='fd'and(os.fstat(9).st_dev,os.fstat(9).st_ino)!=(s.st_dev,s.st_ino):raise ValueError('Cooperative lock inode changed')
print(str(s.st_dev)+':'+str(s.st_ino))
PY
}
BEFORE="$(host --preflight)";[[ -n "$BEFORE" ]] || exit 1
finish() {
 local status=$? ids after
 trap - EXIT INT TERM;set +e
 if [[ "$OWNED" == 1 ]];then
  ids="$(query)";[[ $? == 0 ]] || status=1
  if [[ -n "$ids" ]];then
   timeout --signal=TERM --kill-after=2s 8s docker stop --time 3 "$NAME" >/dev/null 2>&1 || true
   ids="$(query)";[[ $? == 0 ]] || status=1
   if [[ -n "$ids" ]];then
    timeout --signal=TERM --kill-after=2s 5s docker kill "$NAME" >/dev/null 2>&1 || true
    timeout --signal=TERM --kill-after=2s 5s docker rm --force "$NAME" >/dev/null 2>&1 || true
   fi
  fi
  ids="$(query)";[[ $? == 0 && -z "$ids" ]] || status=1
  gpu_idle || status=1
 fi
 after="$(host --verify)";[[ $? == 0 && "$after" == "$BEFORE" ]] || status=1
 if [[ "$LOCK_OPEN" == 1 ]];then
  after="$(lock_identity fd)";[[ $? == 0 && "$after" == "$LOCK_BEFORE" ]] || status=1;exec 9>&-
 fi
 exit "$status"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
IDS="$(query)";[[ -z "$IDS" ]] || exit 1
LOCK_BEFORE="$(lock_identity)";exec 9<"$LOCK";LOCK_OPEN=1
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]] || exit 1
flock --nonblock 9 || exit 1;gpu_idle || exit 1
[[ ! -e "$OUT" && ! -L "$OUT" && -d "${OUT%/*}" ]] || exit 1
CHAIN="$(/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B -c 'import json,sys;print(json.dumps(json.loads(sys.argv[1])["original_MoGe_link_graph"],sort_keys=True,separators=(",",":")))' "$BEFORE")"
MOUNT_TEXT="$(host --mounts)";[[ -n "$MOUNT_TEXT" && "$BEFORE" == "$(host --verify)" ]] || exit 1
MOUNTS=()
while IFS=$'\t' read -r source destination;do
 [[ "$source" == /* && "$destination" == /* && "$source" != *','* && "$destination" != *','* ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$source,dst=$destination,readonly")
done <<<"$MOUNT_TEXT"
umask 077;mkdir -m 700 "$OUT";OWNED=1
timeout --signal=TERM --kill-after=10s 190s docker run --rm --name "$NAME" --gpus all --network none \
 --user 0:0 --memory 12g --cpus 4 --read-only --cap-drop ALL --security-opt no-new-privileges \
 --tmpfs /tmp:rw,noexec,nosuid,size=512m --entrypoint /usr/bin/env \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" \
 sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252 \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp PYTHONPATH="$CODE/infra:$CODE/src:/workspace/v2d_sam3d_body/lib" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_MOGE_ORIGINAL_CHAIN="$CHAIN" \
 WR_IMAGE_ID=sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252 \
 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 /opt/conda/bin/python -B "$CODE/infra/bridge_rgb_anchor_infer.py" --run
