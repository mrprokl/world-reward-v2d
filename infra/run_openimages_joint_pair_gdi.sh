#!/usr/bin/env bash
# New GDI only: no masks, actor selection, old datasets or reference mounts.
# Source closure: /infra/openimages_joint_pair_gdi.py /infra/bridge_frontend_bindings.py
# Source closure: /infra/frontend_selected_assets.py /infra/frontend_sam2_kernel_gate.py
# Source closure: /infra/mediapipe_cpu_runtime_verify.py /configs/frontend_grounding_source_pins.json
# Source closure: /configs/frontend_asset_archive_pins.json
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_openimages_joint_pair_gdi/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_openimages_joint_pair_gdi.sh" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
OUT="$ROOT/results/openimages-joint-pair-gdi-v1"
NAME="world-reward-oi-joint-gdi-${REV:0:12}";LOCK="$ROOT/jobs/.world-reward-h100.lock"
IMAGE=sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252
HOSTENV=(/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1)
control() { timeout --signal=TERM --kill-after=2s 8s "$@"; }
query() { control docker ps -aq --no-trunc --filter "name=^/$NAME$"; }
gpu_idle() { local apps;apps="$(control nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1;[[ -z "${apps//[[:space:]]/}" ]]; }
lock_identity() {
 "${HOSTENV[@]}" /usr/bin/python3 -I -B - "$LOCK" "${1:-path}" <<'PY'
from pathlib import Path
import os,stat,sys
p=Path(sys.argv[1]);s=p.lstat()
assert p.resolve()==p and not any(a.is_symlink()for a in(p,*p.parents)) and stat.S_ISREG(s.st_mode) and s.st_nlink==1
if sys.argv[2]=='fd':assert(os.fstat(9).st_dev,os.fstat(9).st_ino)==(s.st_dev,s.st_ino)
print(str(s.st_dev)+':'+str(s.st_ino))
PY
}
OWNED=0;LOCK_OPEN=0;LOCK_BEFORE=''
cleanup() {
 local ids
 [[ "$OWNED" == 1 ]] || return 0
 ids="$(query)" || return 1
 if [[ -n "$ids" ]];then
  control docker stop --time 3 "$NAME" >/dev/null 2>&1 || true
  ids="$(query)" || return 1
  if [[ -n "$ids" ]];then
   control docker kill "$NAME" >/dev/null 2>&1 || true
   control docker rm --force "$NAME" >/dev/null 2>&1 || true
  fi
 fi
 ids="$(query)" || return 1;[[ -z "$ids" ]] && gpu_idle
}
finish() {
 local status=$? after
 trap - EXIT INT TERM;set +e
 cleanup || status=1
 if [[ "$LOCK_OPEN" == 1 ]];then
  after="$(lock_identity fd)";[[ $? == 0 && "$after" == "$LOCK_BEFORE" ]] || status=1;exec 9>&-
 fi
 exit "$status"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
IDS="$(query)";[[ -z "$IDS" && ! -e "$OUT" && ! -L "$OUT" ]] || exit 1
LOCK_BEFORE="$(lock_identity)";exec 9<"$LOCK";LOCK_OPEN=1
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]] || exit 1
flock --nonblock 9;gpu_idle
PROJECTED="$("${HOSTENV[@]}" /usr/bin/python3 -I -B "$CODE/infra/openimages_joint_pair_gdi.py" --prepare)"
read -r INPUT_SHA INPUT_BYTES <<<"$PROJECTED"
[[ "$INPUT_SHA" =~ ^[0-9a-f]{64}$ && "$INPUT_BYTES" =~ ^[0-9]+$ ]] || exit 1
DEST=/srv/world-reward-data/frontend-assets-extracted-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b
BUILD="$ROOT/jobs/d1fcb8ad158cf4ad4b1f7fe2bba7ea533ae565cc/run_frontend_grounding_build"
KERNEL="$ROOT/jobs/168a809369a294c1ab65154a585b83014c98d976/run_frontend_sam2_kernel_gate"
MOUNTS=()
for path in "${CODE%/code}" "$BUILD" "$KERNEL" \
 "$ROOT/results/frontend-grounding-build-v6/report.json" "$ROOT/results/frontend-grounding-build-v6/child-CPU-probe.log" \
 "$ROOT/results/frontend-sam2-kernel-gate-v1/report.json" "$DEST/world-reward-frontend-assets-manifest.json" \
 "$ROOT/results/frontend-asset-extract-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b/report.json" \
 "$DEST/results/weights-acquisition.json" "$DEST/weights/grounding_dino" "$OUT/inputs.json";do
 [[ -e "$path" && ! -L "$path" ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
PATHS="$("${HOSTENV[@]}" /usr/bin/python3 -I -B - "$OUT/inputs.json" <<'PY'
import json,re,sys
from pathlib import Path
data=json.load(open(sys.argv[1]));assert set(data)=={'schema','images'}and len(data['images'])==50
for row in data['images']:
 assert set(row)=={'image_id','file','width','height','bytes','sha256'}and re.fullmatch('[0-9a-f]{16}',row['image_id'])
 p=Path(row['file']);assert p==Path('/srv/world-reward-data/openimages_joint_pair_acquisition_v1')/row['image_id']/'rgb.jpg'
 assert p.resolve()==p and not any(a.is_symlink()for a in(p,*p.parents));print(p)
PY
)"
while IFS= read -r path;do
 [[ -f "$path" && ! -L "$path" ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done <<<"$PATHS"
gpu_idle;OWNED=1
set +e
timeout --signal=TERM --kill-after=10s 615s docker run --rm --name "$NAME" --gpus all --network none \
 --user 0:0 --read-only --memory 16g --cpus 4 --cap-drop ALL --security-opt no-new-privileges \
 --tmpfs /tmp:rw,noexec,nosuid,size=512m --entrypoint /usr/bin/env \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT/native,dst=$OUT/native" "$IMAGE" \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" \
 /opt/conda/bin/python -I -B "$CODE/infra/openimages_joint_pair_gdi.py" --native \
 --inputs-sha256 "$INPUT_SHA" --inputs-bytes "$INPUT_BYTES"
RESULT=$?
set -e
cleanup
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]] || exit 1
"${HOSTENV[@]}" /usr/bin/python3 -I -B "$CODE/infra/openimages_joint_pair_gdi.py" --finalize "$RESULT"
