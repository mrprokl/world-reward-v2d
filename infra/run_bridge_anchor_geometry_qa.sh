#!/usr/bin/env bash
# Source closure: /infra/bridge_anchor_qa_run.py /infra/bridge_anchor_geometry_qa.py /infra/triangle_ray_gate.py
# Source closure: /infra/run_triangle_ray_gate.sh
# Source closure: /infra/bridge_frontend_bindings.py /infra/frontend_selected_assets.py /infra/frontend_sam2_kernel_gate.py
set +x
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_bridge_anchor_geometry_qa/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_bridge_anchor_geometry_qa.sh" \
 && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
host() {
 /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$CODE" "$1" <<'PY'
import sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'))
import bridge_anchor_qa_run as driver
if Path(driver.__file__).resolve()!=code/'infra/bridge_anchor_qa_run.py':raise ValueError('Actual CPU QA source required')
driver.main([sys.argv[2]])
PY
}
OUT="$ROOT/validation/bridge_rgb_anchor_v1/geometry_qa_v1"
NAME="world-reward-bridge-geometry-${REV:0:12}"; OWNED=0
query() { timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "name=^/$NAME$"; }
BEFORE="$(host --preflight)"; [[ -n "$BEFORE" ]] || exit 1
finish() {
 local status=$? ids after
 trap - EXIT INT TERM; set +e
 if [[ "$OWNED" == 1 ]]; then
  ids="$(query)"; [[ $? == 0 ]] || status=1
  if [[ -n "$ids" ]]; then
   timeout --signal=TERM --kill-after=2s 8s docker stop --time 3 "$NAME" >/dev/null 2>&1 || true
   timeout --signal=TERM --kill-after=2s 5s docker kill "$NAME" >/dev/null 2>&1 || true
   timeout --signal=TERM --kill-after=2s 5s docker rm --force "$NAME" >/dev/null 2>&1 || true
  fi
  ids="$(query)"; [[ $? == 0 && -z "$ids" ]] || status=1
 fi
 after="$(host --verify)"; [[ $? == 0 && "$after" == "$BEFORE" ]] || status=1
 exit "$status"
}
trap finish EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
IDS="$(query)"; [[ -z "$IDS" ]] || exit 1
/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B - "$OUT" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
if p.exists()or p.is_symlink()or p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents))or not p.parent.is_dir():raise ValueError('Exclusive canonical new QA output required')
PY
MOUNT_TEXT="$(host --mounts)"; [[ -n "$MOUNT_TEXT" && "$BEFORE" == "$(host --verify)" ]] || exit 1
MOUNTS=()
while IFS= read -r path; do
 [[ "$path" == /* && "$path" != *','* ]] || exit 1
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done <<<"$MOUNT_TEXT"
umask 077; mkdir -m 700 "$OUT"; OWNED=1
timeout --signal=TERM --kill-after=2s 183s docker run --rm --name "$NAME" --network none \
 --user 0:0 --memory 4g --cpus 4 --read-only --cap-drop ALL --security-opt no-new-privileges \
 --tmpfs /tmp:rw,noexec,nosuid,size=128m --entrypoint /usr/bin/env \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" \
 sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252 \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp WR_ROOT="$ROOT" WR_CODE="$CODE" \
 WR_IMAGE_ID=sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252 \
 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 /opt/conda/bin/python -I -B "$CODE/infra/bridge_anchor_qa_run.py" --run
