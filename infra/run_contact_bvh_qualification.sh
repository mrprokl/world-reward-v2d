#!/usr/bin/env bash
# Source closure: /infra/contact_bvh_qualification.py /src/world_reward/contact_bvh.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_contact_bvh_qualification/code" \
 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/src:$CODE/infra"
exec 9<"$ROOT/jobs/.world-reward-h100.lock"; flock -n 9
PLAN="$(/usr/bin/python3 -B "$CODE/infra/contact_bvh_qualification.py" --plan "$@")"
OUT="$ROOT/results/contact-bvh-qualification-$REV"
[[ ! -e "$OUT" ]] || exit 2
mkdir -m 755 "$OUT"
NAME="wr-contact-bvh-qualification-$REV"; CIDFILE="$OUT/.container.cid"
cleanup() {
 local status=$? cid='' ids='' found='' verified=false
 trap - EXIT TERM INT
 if [[ -f "$CIDFILE" && ! -L "$CIDFILE" ]]; then
  cid="$(cat "$CIDFILE")"
  if [[ "$cid" =~ ^[0-9a-f]{64}$ ]]; then
   if ids="$(timeout 20s docker ps -aq --no-trunc --filter "id=$cid")" && [[ -z "$ids" || "$ids" == "$cid" ]]; then
    if [[ -n "$ids" ]]; then
     if found="$(timeout 20s docker inspect "$cid" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.contact_bvh.owner"}}')" \
      && [[ "$found" == "$IMAGE|/$NAME|$REV" ]]; then timeout 20s docker rm -f "$cid" >/dev/null || status=1; else status=1; fi
    fi
    if ids="$(timeout 20s docker ps -aq --no-trunc --filter "id=$cid")" && [[ -z "$ids" ]]; then verified=true; else status=1; fi
   else status=1; fi
  else status=1; fi
  chmod 444 "$CIDFILE"
 else
  if ids="$(timeout 20s docker ps -aq --no-trunc --filter "name=^/$NAME$")" && [[ -z "$ids" ]]; then verified=true; else status=1; fi
 fi
 printf '{"producer_revision":"%s","container_absence_verified":%s,"process_exit_code":%d,"GPU_requested":true}\n' \
  "$REV" "$verified" "$status" > "$OUT/host-exit.json"
 chmod 444 "$OUT/host-exit.json"
 exit "$status"
}
trap cleanup EXIT; trap 'exit 143' TERM; trap 'exit 130' INT
[[ -z "$(docker ps -aq --no-trunc --filter "name=^/$NAME$")" ]] || exit 2
mapfile -t MOUNTS < <(/usr/bin/python3 - "$PLAN" <<'PY'
import json,sys
p=json.loads(sys.argv[1])
for row in p['mounts']:
 print('--mount'); print('type=bind,src='+row['path']+',dst='+row['path']+(',readonly' if row['readonly'] else ''))
PY
)
timeout --signal=TERM --kill-after=15s 615s docker run --rm --cidfile "$CIDFILE" \
 --name "$NAME" --label "world_reward.contact_bvh.owner=$REV" --gpus all \
 --network none --read-only --user 0:0 --cap-drop ALL --security-opt no-new-privileges \
 --memory 16g --cpus 4 --tmpfs /tmp:rw,nosuid,exec,size=512m \
 "${MOUNTS[@]}" --entrypoint /usr/bin/env "$IMAGE" \
 -i PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" \
 PYTHONPATH="$CODE/src:$CODE/infra" PYTHONDONTWRITEBYTECODE=1 \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 /opt/conda/bin/python -B "$CODE/infra/contact_bvh_qualification.py" "$@"
