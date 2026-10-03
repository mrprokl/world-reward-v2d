#!/usr/bin/env bash
# Tiny procedural H100 operator only; no weights, data, source acquisition.
# Source closure: /infra/frontend_sam2_kernel_gate.py /configs/frontend_grounding_source_pins.json
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_frontend_sam2_kernel_gate/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_frontend_sam2_kernel_gate.sh" ]] || exit 2
[[ $# == 8 && "$1" == --build-revision && "$2" =~ ^[0-9a-f]{40}$ \
 && "$3" == --build-report-sha256 && "$4" =~ ^[0-9a-f]{64}$ \
 && "$5" == --build-report-bytes && "$6" =~ ^[1-9][0-9]*$ \
 && "$7" == --build-script-sha256 && "$8" =~ ^[0-9a-f]{64}$ ]] || exit 2
BUILD_REV="$2";ARGS=("$@")
export DOCKER_HOST="unix://$ROOT/docker.sock"
host_gate() { /usr/bin/env -i PATH=/usr/bin:/bin WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -B "$CODE/infra/frontend_sam2_kernel_gate.py" "$1" "${ARGS[@]}"; }
IMAGE="$(host_gate --preflight)";[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 1
LOCK="$ROOT/jobs/.world-reward-h100.lock"
[[ ! -L "$LOCK" && ( ! -e "$LOCK" || -f "$LOCK" ) ]] || exit 1
exec 9>>"$LOCK";flock --nonblock 9
APPS="$(timeout --signal=TERM --kill-after=2s 8s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || exit 1
OUT="$ROOT/results/frontend-sam2-kernel-gate-v1"
[[ ! -e "$OUT" && ! -L "$OUT" && -d "${OUT%/*}" ]] || exit 1
umask 077;mkdir -m 700 "$OUT"
NAME="world-reward-sam2-kernel-$REV";OWNED=0
query() { timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "name=^/$NAME$"; }
cleanup() {
 [[ "$OWNED" == 1 ]] || return 0
 local remaining;remaining="$(query)" || return 1
 if [[ -n "$remaining" ]];then
  timeout --signal=TERM --kill-after=2s 8s docker stop --time 3 "$NAME" >/dev/null 2>&1 || true
  remaining="$(query)" || return 1
  if [[ -n "$remaining" ]];then
   timeout --signal=TERM --kill-after=2s 5s docker kill "$NAME" >/dev/null 2>&1 || true
   timeout --signal=TERM --kill-after=2s 5s docker rm --force "$NAME" >/dev/null 2>&1 || true
  fi
 fi
 remaining="$(query)" || return 1
 [[ -z "$remaining" ]]
}
finish() { STATUS=$?;trap - EXIT INT TERM;set +e;cleanup || STATUS=1;host_gate --verify >/dev/null || STATUS=1;exit "$STATUS"; }
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
EXISTING="$(query)";[[ -z "$EXISTING" ]] || exit 1
OWNED=1
OLD="$ROOT/jobs/$BUILD_REV/run_frontend_grounding_build"
timeout --signal=TERM --kill-after=10s 120s docker run --rm --name "$NAME" --gpus all --network none \
 --user 0:0 --memory 4g --cpus 2 --read-only --cap-drop ALL --security-opt no-new-privileges \
 --tmpfs /tmp:rw,noexec,nosuid,size=128m --entrypoint /usr/bin/env \
 --mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly" \
 --mount "type=bind,src=$OLD,dst=$OLD,readonly" \
 --mount "type=bind,src=$ROOT/results/frontend-grounding-build-v6/report.json,dst=$ROOT/results/frontend-grounding-build-v6/report.json,readonly" \
 --mount "type=bind,src=$ROOT/results/frontend-grounding-build-v6/child-CPU-probe.log,dst=$ROOT/results/frontend-grounding-build-v6/child-CPU-probe.log,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 WR_KERNEL_IMAGE_ID="$IMAGE" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 /opt/conda/bin/python -I -B "$CODE/infra/frontend_sam2_kernel_gate.py" --run "${ARGS[@]}"
