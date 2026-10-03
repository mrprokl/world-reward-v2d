#!/usr/bin/env bash
# NEW four RGB masks only; selected proof/models RO, never authored truth/code.
# Source closure: /infra/bridge_frontend_bindings.py /infra/hand_synthetic_masks.py
# Source closure: /infra/frontend_selected_assets.py /infra/frontend_sam2_kernel_gate.py
# Source closure: /configs/frontend_grounding_source_pins.json
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_bridge_rgb_anchor_masks/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_bridge_rgb_anchor_masks.sh" \
 && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
OUT="$ROOT/validation/bridge_rgb_anchor_v1/automatic_masks"
NAME="world-reward-bridge-masks-${REV:0:12}";LOCK="$ROOT/jobs/.world-reward-h100.lock"
host_check() {
 /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PY'
from pathlib import Path
import sys
root,code=map(Path,sys.argv[1:3]);revision=sys.argv[3];entry=Path(sys.argv[4])
sys.path[:0]=[str(code/'infra'),str(code/'src')]
import bridge_frontend_bindings as b
b.require(entry==code/'infra/run_bridge_rgb_anchor_masks.sh' and entry.resolve()==entry,'Actual immutable mask launcher required')
proof=b.authenticate(root,code,'run_bridge_rgb_anchor_masks',live=True)
records,frozen=b.public_inputs(root,code/b.INPUT_PINS);b.recheck(frozen)
print(proof['source_binding']['closure_sha256'])
PY
}
lock_identity() {
 /usr/bin/env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -B - "$LOCK" "${1:-path}" <<'PY'
from pathlib import Path
import os,stat,sys
p=Path(sys.argv[1]);s=p.lstat()
if p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents))or not stat.S_ISREG(s.st_mode)or s.st_nlink!=1:raise ValueError('Existing canonical cooperative lock required')
if sys.argv[2]=='fd'and(os.fstat(9).st_dev,os.fstat(9).st_ino)!=(s.st_dev,s.st_ino):raise ValueError('Cooperative lock inode changed')
print(str(s.st_dev)+':'+str(s.st_ino))
PY
}
gpu_idle() {
 local apps
 apps="$(timeout --signal=TERM --kill-after=2s 5s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
 [[ -z "${apps//[[:space:]]/}" ]]
}
query() { timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "name=^/$NAME$"; }
BEFORE="$(host_check)";OWNED=0;LOCK_OPEN=0;LOCK_BEFORE=''
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
 after="$(host_check)";[[ $? == 0 && "$after" == "$BEFORE" ]] || status=1
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
[[ ! -e "$OUT" && ! -L "$OUT" && -d "${OUT%/*}" && "$BEFORE" == "$(host_check)" ]] || exit 1
umask 077;mkdir -m 700 "$OUT"
DEST=/srv/world-reward-data/frontend-assets-extracted-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b
BUILD="$ROOT/jobs/d1fcb8ad158cf4ad4b1f7fe2bba7ea533ae565cc/run_frontend_grounding_build"
KERNEL="$ROOT/jobs/168a809369a294c1ab65154a585b83014c98d976/run_frontend_sam2_kernel_gate"
MOUNTS=()
for path in "${CODE%/code}" "$BUILD" "$KERNEL" \
 "$ROOT/results/frontend-grounding-build-v6/report.json" "$ROOT/results/frontend-grounding-build-v6/child-CPU-probe.log" \
 "$ROOT/results/frontend-sam2-kernel-gate-v1/report.json" \
 "$DEST/world-reward-frontend-assets-manifest.json" \
 "$ROOT/results/frontend-asset-extract-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b/report.json" \
 "$DEST/results/weights-acquisition.json" "$DEST/weights/grounding_dino" "$DEST/weights/sam2";do
 [[ -e "$path" && ! -L "$path" ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
for name in manifest.json clip_000000_frame_000000.png clip_000001_frame_000000.png clip_000002_frame_000000.png clip_000003_frame_000000.png;do
 path="$ROOT/validation/bridge_rgb_anchor_v1/inputs/$name";[[ -f "$path" && ! -L "$path" ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
OWNED=1
timeout --signal=TERM --kill-after=10s 123s docker run --rm --name "$NAME" --gpus all --network none \
 --user 0:0 --memory 16g --cpus 4 --read-only --cap-drop ALL --security-opt no-new-privileges \
 --tmpfs /tmp:rw,noexec,nosuid,size=512m --entrypoint /usr/bin/env \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" \
 sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252 \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp PYTHONPATH="$CODE/infra:$CODE/src" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID=sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252 \
 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 /opt/conda/bin/python -B "$CODE/infra/bridge_rgb_anchor_masks.py"
