#!/usr/bin/env bash
# Source closure: /infra/form_hoi_external_eval.py /infra/form_hoi_external_dev.py /infra/form_hoi_external_acquire.py
# CPU only; private unmatched FORM references mounted only AFTER ALL prediction seals.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
# Use each host's existing qualified image; never transfer/pull/retag multi-GB images.
case "$(hostname)" in
 scenesmith-ncc-h100-01) IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ;;
 world-reward-ncc-h100-02) IMAGE=sha256:d24051da178c12ce3f3f3193a5e0c8ca90e1e7a1c765077ec3e22fdfc3752004 ;;
 *) exit 2 ;;
esac
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_form_hoi_external_eval/code" && "$(id -u)" == 0 ]] || exit 2
RUNTIME="$ROOT/results/form-eval-numba-runtime-f4d379bd8c7cf0ef23364b7f2e3b1bce215ceea6"
[[ "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
/usr/bin/python3 -I -B - "$RUNTIME" "$IMAGE" <<'PYRUNTIME'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);image=sys.argv[2]
pins={'report.json':(10889,'dba8aace26df84ff250c67a083ed631241803bfbd27764164d96a74fc595a68e'),
 'qualification.json':(2352,'c2baab794ed676ae11eb531f9149e1af0a78a7b3d056664c38070374f86490e8')}
for name,(size,digest)in pins.items():
 p=root/name
 if p.resolve()!=p or any(q.is_symlink()for q in(p,*p.parents))or p.stat().st_mode&0o222:raise SystemExit(2)
 raw=p.read_bytes()
 if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=digest:raise SystemExit(2)
report=json.loads((root/'report.json').read_bytes());q=json.loads((root/'qualification.json').read_bytes())
if report['status']!='pass' or report['qualified_image_id']!=image or q['status']!='pass' or q['compiled_public_kernels_active']is not True:raise SystemExit(2)
PYRUNTIME
read -r DEVREV PREDROOT < <(/usr/bin/python3 -I -B - "$CODE" "$ROOT" <<'PYBOOT'
import json,re,sys
from pathlib import Path
code,root=map(Path,sys.argv[1:]);cfg=json.loads((code/'configs/form_hoi_external_eval_v1.json').read_bytes())
g=cfg['input_gate'];rev=g['dev_producer_revision'];p=Path(g['prediction_manifest_path'] or '/missing')
if not isinstance(rev,str) or not re.fullmatch('[0-9a-f]{40}',rev) or not p.is_absolute() or p.resolve()!=p or not p.is_relative_to(root) or any(x.is_symlink() for x in(p,*p.parents)) or any(c.isspace() for c in str(p)):raise SystemExit(2)
for k in('dev_report_identity','prediction_manifest_identity'):
 v=g[k]
 if not isinstance(v,dict) or set(v)!={'bytes','sha256'} or type(v['bytes']) is not int or not 0<v['bytes']<=128<<10 or not re.fullmatch('[0-9a-f]{64}',v['sha256']):raise SystemExit(2)
print(rev,p.parent)
PYBOOT
)
[[ "$DEVREV" =~ ^[0-9a-f]{40}$ && "$PREDROOT" == "$ROOT/"* ]] || exit 2
DEV="/srv/world-reward-data/form_hoi_external_dev_v1/$DEVREV"
[[ -d "$DEV" && -d "$PREDROOT" ]] || exit 2
OUT="$ROOT/results/form-hoi-external-eval-$REV";[[ ! -e "$OUT" ]] || exit 2
LOCK="$ROOT/jobs/.world-reward-form-external-eval.lock"
# A read-only flock lease never truncates another active job's lock.
/usr/bin/python3 -I -B - "$LOCK" <<'PYLOCK'
import os,pathlib,stat,sys
p=pathlib.Path(sys.argv[1])
if p.resolve()!=p or any(x.is_symlink() for x in(p,*p.parents)):raise SystemExit(2)
try:fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o400);os.close(fd)
except FileExistsError:pass
s=p.lstat()
if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=0:raise SystemExit(2)
PYLOCK
exec 9<"$LOCK";flock -n 9
mkdir -m 755 "$OUT"
export DOCKER_HOST="unix://$ROOT/docker.sock"
[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]] || exit 2
NAME="wr-form-external-eval-$REV";CIDFILE="$OUT/.container.cid"
cleanup() {
 local status=$? cid='' ids='' found='' verified=false
 trap - EXIT TERM INT
 if [[ -f "$CIDFILE" && ! -L "$CIDFILE" ]]; then
  cid="$(cat "$CIDFILE")"
  if [[ "$cid" =~ ^[0-9a-f]{64}$ ]] \
   && ids="$(timeout 20s docker ps -aq --no-trunc --filter "id=$cid")" \
   && [[ -z "$ids" || "$ids" == "$cid" ]]; then
   if [[ -n "$ids" ]]; then
    if found="$(timeout 20s docker inspect "$cid" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.form_external_eval.owner"}}')" \
     && [[ "$found" == "$IMAGE|/$NAME|$REV" ]]; then
     timeout 20s docker rm -f "$cid" >/dev/null || status=1
    else status=1;fi
   fi
   if ids="$(timeout 20s docker ps -aq --no-trunc --filter "id=$cid")" && [[ -z "$ids" ]];then verified=true;else status=1;fi
  else status=1;fi
  chmod 444 "$CIDFILE"
 else
  if ids="$(timeout 20s docker ps -aq --no-trunc --filter "name=^/$NAME$")" && [[ -z "$ids" ]];then verified=true;else status=1;fi
 fi
 printf '{"producer_revision":"%s","container_absence_verified":%s,"process_exit_code":%d,"GPU_requested":false}\n' "$REV" "$verified" "$status" > "$OUT/host-exit.json"
 chmod 444 "$OUT/host-exit.json";exit "$status"
}
trap cleanup EXIT;trap 'exit 143' TERM;trap 'exit 130' INT
[[ -z "$(docker ps -aq --no-trunc --filter "name=^/$NAME$")" ]] || exit 2
timeout --signal=TERM --kill-after=20s 1560s docker run --rm --cidfile "$CIDFILE" \
 --name "$NAME" --label "world_reward.form_external_eval.owner=$REV" \
 --network none --read-only --user 0:0 --cap-drop ALL --security-opt no-new-privileges \
 --memory 8g --cpus 4 --tmpfs /tmp:rw,nosuid,size=256m \
 --mount "type=bind,src=${CODE%/code},dst=${CODE%/code},readonly" \
 --mount "type=bind,src=$DEV,dst=$DEV,readonly" \
 --mount "type=bind,src=$PREDROOT,dst=$PREDROOT,readonly" \
 --mount "type=bind,src=$ROOT/vendor/v2d_submission_kit/v2dlb,dst=$ROOT/vendor/v2d_submission_kit/v2dlb,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/refinement/mhr_hand_surface_spec.npz,dst=$ROOT/weights/cari4d/refinement/mhr_hand_surface_spec.npz,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" \
 --mount "type=bind,src=$RUNTIME,dst=$RUNTIME,readonly" \
 --entrypoint /usr/bin/env "$IMAGE" -i PATH=/opt/conda/bin:/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 NUMBA_NUM_THREADS=4 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONPATH="/opt/world-reward/form-eval-numba:$CODE/src:$CODE/infra" \
 /opt/conda/bin/python -B "$CODE/infra/form_hoi_external_eval.py"
