#!/usr/bin/env bash
# Source closure: /infra/contact_memory_real.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ $# -eq 0 ]] || exit 2
TRACK_SOURCE=09f516d9085f0d64d725b46b0ad6aa52fe7c0d84
SOURCE=052ba1554e9a573d566713a99a61d89a5f27681c
CONTACT_SOURCE=40183b3a83ba59080c192c3cdf2db9e1d76ef021
FRONTEND=b658881079871508c6b3ec14d001dc1299956996
POOL_SOURCE=4ef9a1b5290d656f6c20f9d4aa6a781b4bfba019
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_contact_memory_real/code" \
 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
OUT="$ROOT/results/contact-memory-real-$REV"
[[ ! -e "$OUT" ]] || exit 2
# CPU-only concurrency lease: do not truncate or mutate an existing GPU lock.
LOCK="$ROOT/jobs/.world-reward-contact-memory-real.lock"
LOCK_ID="$(/usr/bin/python3 -I -B -c '
import os,pathlib,stat,sys
p=pathlib.Path(sys.argv[1])
if p.resolve()!=p or any(q.is_symlink() for q in (p,*p.parents)):raise SystemExit(2)
try:fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o400);os.close(fd)
except FileExistsError:pass
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);s=os.fstat(fd);os.close(fd)
if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=0:raise SystemExit(2)
print(str(s.st_dev)+":"+str(s.st_ino))
' "$LOCK")"
exec 9<"$LOCK"
[[ "$(stat -Lc '%d:%i' /proc/$$/fd/9)" == "$LOCK_ID" \
 && "$(stat -c '%d:%i' "$LOCK")" == "$LOCK_ID" ]] || exit 2
flock -n 9
mkdir -m 755 "$OUT"
export DOCKER_HOST="unix://$ROOT/docker.sock"
NAME="wr-contact-memory-real-$REV"; CIDFILE="$OUT/.container.cid"
cleanup() {
 local status=$? cid='' ids='' found='' verified=false
 trap - EXIT TERM INT
 if [[ -f "$CIDFILE" && ! -L "$CIDFILE" ]]; then
  cid="$(cat "$CIDFILE")"
  if [[ "$cid" =~ ^[0-9a-f]{64}$ ]]; then
   if ids="$(timeout 20s docker ps -aq --no-trunc --filter "id=$cid")" \
    && [[ -z "$ids" || "$ids" == "$cid" ]]; then
    if [[ -n "$ids" ]]; then
     if found="$(timeout 20s docker inspect "$cid" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.contact_memory_real.owner"}}')" \
      && [[ "$found" == "$IMAGE|/$NAME|$REV" ]]; then
      timeout 20s docker rm -f "$cid" >/dev/null || status=1
     else status=1; fi
    fi
    if ids="$(timeout 20s docker ps -aq --no-trunc --filter "id=$cid")" \
     && [[ -z "$ids" ]]; then verified=true; else status=1; fi
   else status=1; fi
  else status=1; fi
  chmod 444 "$CIDFILE"
 else
  # Before a CID is written, do not stop an unproven name-matching container.
  if ids="$(timeout 20s docker ps -aq --no-trunc --filter "name=^/$NAME$")" \
   && [[ -z "$ids" ]]; then verified=true; else status=1; fi
 fi
 printf '{"producer_revision":"%s","container_absence_verified":%s,"process_exit_code":%d,"GPU_requested":false}\n' \
  "$REV" "$verified" "$status" > "$OUT/host-exit.json"
 chmod 444 "$OUT/host-exit.json"
 exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[[ -z "$(docker ps -aq --no-trunc --filter "name=^/$NAME$")" ]] || exit 2
timeout --signal=TERM --kill-after=10s 2400s docker run --rm --cidfile "$CIDFILE" \
 --name "$NAME" --label "world_reward.contact_memory_real.owner=$REV" \
 --network none --read-only --user 0:0 --cap-drop ALL --security-opt no-new-privileges \
 --memory 16g --cpus 4 --tmpfs /tmp:rw,nosuid,size=256m \
 --mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly" \
 --mount "type=bind,src=$ROOT/experiments/full4d-v1-$SOURCE,dst=$ROOT/experiments/full4d-v1-$SOURCE,readonly" \
 --mount "type=bind,src=$ROOT/results/gemini-sam31-$FRONTEND,dst=$ROOT/results/gemini-sam31-$FRONTEND,readonly" \
 --mount "type=bind,src=$ROOT/results/sequence-pose-probe-$TRACK_SOURCE-v2,dst=$ROOT/results/sequence-pose-probe-$TRACK_SOURCE-v2,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/refinement/mhr_hand_surface_spec.npz,dst=$ROOT/weights/cari4d/refinement/mhr_hand_surface_spec.npz,readonly" \
 --mount "type=bind,src=$ROOT/results/sequence-pose-contact-$CONTACT_SOURCE,dst=$ROOT/results/sequence-pose-contact-$CONTACT_SOURCE,readonly" \
 --mount "type=bind,src=$ROOT/results/sequence-contact-patch-real-$POOL_SOURCE,dst=$ROOT/results/sequence-contact-patch-real-$POOL_SOURCE,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" \
 --entrypoint /usr/bin/env "$IMAGE" \
 -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONPATH="$CODE/src:$CODE/infra" \
 /opt/conda/bin/python -B "$CODE/infra/contact_memory_real.py"
