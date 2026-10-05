#!/usr/bin/env bash
# Saved-only CPU; all native prediction hashes precede private reference decoding.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
OUT="$ROOT/results/proposal-stress-evaluation-v1"
NAME="world-reward-proposal-recall-${REV:0:12}"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
[[ "$(uname -s)" == Linux && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$CODE" == "$ROOT/jobs/$REV/run_proposal_stress_evaluate/code" \
 && ! -e "$OUT" && ! -L "$OUT" ]] || exit 2
[[ -z "$(docker ps -aq --no-trunc --filter "name=^/$NAME$")" ]] || exit 2
mkdir -m 700 "$OUT"
cleanup() {
 local status=$? cid
 trap - EXIT INT TERM;set +e
 cid="$(docker ps -aq --no-trunc --filter "name=^/$NAME$")"
 if [[ -n "$cid" ]];then
  if [[ "$cid" =~ ^[0-9a-f]{64}$ && "$(docker inspect "$cid" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.revision"}}')" == "$IMAGE|/$NAME|$REV" ]];then
   timeout --signal=TERM --kill-after=2s 10s docker rm --force "$cid" >/dev/null
  else status=1;fi
 fi
 [[ -z "$(docker ps -aq --no-trunc --filter "name=^/$NAME$")" ]] || status=1
 exit "$status"
}
trap cleanup EXIT;trap 'exit 143' TERM;trap 'exit 130' INT
timeout --signal=TERM --kill-after=10s 190s docker run --rm --network none \
 --name "$NAME" --label "world-reward.revision=$REV" --memory 8g --cpus 4 --pids-limit 128 \
 --read-only --user 0:0 --cap-drop ALL --security-opt no-new-privileges \
 --tmpfs /tmp:rw,noexec,nosuid,size=64m --entrypoint /usr/bin/env \
 --mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly" \
 --mount "type=bind,src=$ROOT/validation/proposal_stress_v1,dst=$ROOT/validation/proposal_stress_v1,readonly" \
 --mount "type=bind,src=$ROOT/results/proposal-stress-regions-v2,dst=$ROOT/results/proposal-stress-regions-v2,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" \
 "$IMAGE" \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 \
 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 /opt/conda/bin/python -I -B "$CODE/infra/proposal_stress_evaluate.py"
