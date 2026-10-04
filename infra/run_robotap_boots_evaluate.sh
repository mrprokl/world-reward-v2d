#!/usr/bin/env bash
# Azure VM02 CPU private adapter. Frozen native TAP first-query point metrics only; no GPU.
# Source closure: /infra/robotap_boots_evaluate.py /infra/robotap_boots_public.py /infra/robotap_boots_acquire.py.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_robotap_boots_evaluate/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_robotap_boots_evaluate.sh" ]] || exit 2
BASE="$ROOT/validation/robotap_boots_v1";OUT="$BASE/eval_v1";JOB="${CODE%/code}"
PINNED_IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
source_identity() {
 python3 -I -B - "$CODE" "$REV" <<'PYSOURCE'
from pathlib import Path
import hashlib,re,stat,sys
code=Path(sys.argv[1]);revision=sys.argv[2];digest=hashlib.sha256()
for path in (code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents))or(not stat.S_ISDIR(mode)and not stat.S_ISREG(mode))or mode&0o222:raise ValueError('Readonly canonical source required')
 if path.is_file():digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
for name in ('revision','source-sha256'):
 path=code.parent/name;before=path.lstat();data=path.read_bytes()
 if path.resolve()!=path or path.is_symlink()or not stat.S_ISREG(before.st_mode)or before!=path.lstat():raise ValueError('Original markers required')
 if(name=='revision'and data!=(revision+'\n').encode())or(name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',data)):raise ValueError('Exact original markers required')
 digest.update(name.encode()+b'\0'+data)
print(digest.hexdigest())
PYSOURCE
}
SOURCE_BEFORE="$(source_identity)"
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e
 if [[ -n "${CIDFILE:-}" && -f "$CIDFILE" && ! -L "$CIDFILE" ]];then
  CID="$(cat "$CIDFILE")"
  if [[ "$CID" =~ ^[0-9a-f]{64}$ ]];then
   OWNED="$(docker inspect "$CID" --format '{{.Image}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}' 2>/dev/null)"
   if [[ "$OWNED" == "$PINNED_IMAGE|run_robotap_boots_evaluate|$REV" ]];then docker rm -f "$CID" >/dev/null 2>&1 || STATUS=1;elif [[ -n "$OWNED" ]];then STATUS=1;fi
   chmod 400 "$CIDFILE"
  else STATUS=1;fi
 fi
 AFTER="$(source_identity)";CHECK=$?;[[ "$CHECK" == 0 && "$AFTER" == "$SOURCE_BEFORE" ]] || STATUS=1
 exit "$STATUS"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 -I -B - "$CODE" "$BASE" "$OUT" <<'PYPREFLIGHT'
import sys
from pathlib import Path
code,base,out=map(Path,sys.argv[1:]);sys.path.insert(0,str(code/'infra'))
import robotap_boots_acquire as source
protocol=source.read_protocol(code/'configs/robotap_boots_protocol.json');source.azure_vm02_identity(protocol)
for path in(base,out,base/'public_v2/inputs',base/'public_v2/report.json',base/'public_v2/selection.json',base/'infer_v1/report.json',base/'infer_v1/predictions',*[base/f'eval_private/pickles/robotap/robotap_split{i}.pkl'for i in range(5)],base/'report.json',base/'eval_private/retention-receipt.json'):
 source.canonical(path)
 if path!=out and not path.exists():raise ValueError('Complete original acquisition required')
 if path.is_file()and(path.stat().st_uid!=1000 or path.stat().st_mode&0o222):raise ValueError('Actual immutable private/public artifact owner required')
if out.exists():raise ValueError('Fresh output required; no retry or overwrite')
PYPREFLIGHT
export DOCKER_HOST="unix://$ROOT/docker.sock"
[[ "$(docker image inspect "$PINNED_IMAGE" --format '{{.Id}}')" == "$PINNED_IMAGE" ]] || exit 1
mkdir -m 700 "$OUT";chown 1000:1000 "$OUT"
CIDFILE="$OUT/.container.cid";[[ ! -e "$CIDFILE" ]] || exit 1
MOUNTS=(--mount "type=bind,src=$CODE,dst=$CODE,readonly" --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly" --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly" --mount "type=bind,src=$BASE/report.json,dst=$BASE/report.json,readonly" --mount "type=bind,src=$BASE/eval_private/retention-receipt.json,dst=$BASE/eval_private/retention-receipt.json,readonly" --mount "type=bind,src=$OUT,dst=$OUT")
for P in "$BASE/public_v2/report.json" "$BASE/public_v2/selection.json" "$BASE/public_v2/inputs" "$BASE/infer_v1/report.json" "$BASE/infer_v1/predictions";do MOUNTS+=(--mount "type=bind,src=$P,dst=$P,readonly");done
for i in 0 1 2 3 4;do P="$BASE/eval_private/pickles/robotap/robotap_split$i.pkl";MOUNTS+=(--mount "type=bind,src=$P,dst=$P,readonly");done
set +e
timeout --signal=TERM --kill-after=10s 250s docker run --rm --interactive --cidfile "$CIDFILE" --label world-reward.job=run_robotap_boots_evaluate --label "world-reward.revision=$REV" --network none --memory 16g --cpus 4 --user 1000:1000 --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,size=64m --entrypoint /usr/bin/env "${MOUNTS[@]}" "$PINNED_IMAGE" \
 -i PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$PINNED_IMAGE" WR_AZURE_VM02_VERIFIED=1 WR_ROBOTAP_EVALUATION_RESERVED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 /opt/conda/bin/python -I -B - "$CODE" <<'PYCONTROL'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));driver=code/'infra/robotap_boots_evaluate.py';sys.argv=[str(driver)]
runpy.run_path(str(driver),run_name='__main__')
PYCONTROL
STATUS=$?;set -e
[[ "$(docker image inspect "$PINNED_IMAGE" --format '{{.Id}}')" == "$PINNED_IMAGE" ]] || exit 1
exit "$STATUS"
