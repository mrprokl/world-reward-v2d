#!/usr/bin/env bash
# NEW synthetic proposal stress only. Renderer recipe/GT are never inference mounts.
# Source closure: /infra/proposal_stress_render.py /infra/camera_render.py
# Source closure: /infra/frontend_selected_assets.py /infra/mediapipe_cpu_runtime_verify.py
# Source closure: /configs/frontend_asset_archive_pins.json /configs/proposal_stress_v1.json
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_proposal_stress_render/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_proposal_stress_render.sh" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
OUT="$ROOT/validation/proposal_stress_v1";NAME="world-reward-proposal-stress-${REV:0:12}"
LOCK="$ROOT/jobs/.world-reward-h100.lock"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
DEST=/srv/world-reward-data/frontend-assets-extracted-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b
MODEL="$DEST/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
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
OWNED=0;LOCK_OPEN=0;LOCK_BEFORE='';BEFORE=''
cleanup() {
 local ids
 [[ "$OWNED" == 1 ]] || return 0
 ids="$(query)" || return 1
 if [[ -n "$ids" ]];then
  [[ "$ids" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ "$(control docker inspect "$ids" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.proposal_stress.owner"}}')" \
   == "$IMAGE|/$NAME|$REV-$BEFORE" ]] || return 1
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
 if [[ -n "$BEFORE" ]];then
  after="$("${HOSTENV[@]}" /usr/bin/python3 -I -B "$CODE/infra/proposal_stress_render.py" --verify)"
  [[ $? == 0 && "$after" == "$BEFORE" ]] || status=1
 fi
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
[[ "$(control docker image inspect "$IMAGE" --format '{{.Id}}|{{.Architecture}}|{{.Os}}')" == "$IMAGE|amd64|linux" ]] || exit 1
BEFORE="$("${HOSTENV[@]}" /usr/bin/python3 -I -B "$CODE/infra/proposal_stress_render.py" --verify)"
[[ "$BEFORE" =~ ^[0-9a-f]{64}$ ]] || exit 1
"${HOSTENV[@]}" /usr/bin/python3 -I -B - "$OUT" <<'PY'
from pathlib import Path
import os,sys
p=Path(sys.argv[1]);assert p.resolve()==p and not any(a.is_symlink()for a in(p,*p.parents)) and p.parent.is_dir() and not p.exists()
os.mkdir(p,0o700)
PY
MOUNTS=()
for path in "${CODE%/code}" "$MODEL" "$DEST/world-reward-frontend-assets-manifest.json" \
 "$ROOT/results/frontend-asset-extract-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b/report.json";do
 [[ -e "$path" && ! -L "$path" ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
gpu_idle;OWNED=1
set +e
timeout --signal=TERM --kill-after=10s 305s docker run --rm --name "$NAME" --gpus all --network none \
 --label "world_reward.proposal_stress.owner=$REV-$BEFORE" \
 --user 0:0 --read-only --memory 16g --cpus 4 --cap-drop ALL --security-opt no-new-privileges \
 --tmpfs /tmp:rw,noexec,nosuid,size=256m --entrypoint /usr/bin/env \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" \
 /opt/conda/bin/python -I -B "$CODE/infra/proposal_stress_render.py"
RESULT=$?
set -e
cleanup
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]] || exit 1
"${HOSTENV[@]}" /usr/bin/python3 -I -B "$CODE/infra/proposal_stress_render.py" --finalize "$RESULT" --expected-binding "$BEFORE"
