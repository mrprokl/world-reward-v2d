#!/usr/bin/env bash
# CPU private evaluator only; no model/GPU or acquisition recipe invocation.
# Source closure: /infra/ycbv_point_evaluate.py /infra/tudl_holdout_inputs.py
# /src/world_reward/point_bop_evaluation.py /src/world_reward/point_motion_evaluation.py
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_ycbv_point_evaluate/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_ycbv_point_evaluate.sh" \
 && "$(hostname -s)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 ]] || exit 2
BASE="$ROOT/validation/ycbv_point_pose_v1";OUT="$BASE/evaluation_v1";JOB="${CODE%/code}"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
NAME="world-reward-ycbv-point-evaluate-${REV:0:12}";CIDFILE=''
export DOCKER_HOST="unix://$ROOT/docker.sock"
host_identity() {
 /usr/bin/env -i PATH=/usr/bin:/bin WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$CODE" "$REV" <<'PY'
from pathlib import Path
import hashlib,re,runpy,stat,sys,json
code=Path(sys.argv[1]);revision=sys.argv[2];sys.path.insert(0,str(code/'infra'));digest=hashlib.sha256()
for p in(code,*sorted(code.rglob('*'))):
 mode=p.lstat().st_mode
 if p.resolve()!=p or any(x.is_symlink()for x in(p,*p.parents))or mode&0o222 or not(stat.S_ISREG(mode)or stat.S_ISDIR(mode)):raise ValueError('Complete immutable host dispatch required')
 if p.is_file():digest.update(str(p.relative_to(code)).encode()+b'\0'+hashlib.sha256(p.read_bytes()).digest())
for name in('revision','source-sha256'):
 p=code.parent/name;s=p.lstat();raw=p.read_bytes()
 if not stat.S_ISREG(s.st_mode)or s.st_nlink!=1 or any(getattr(s,k)!=getattr(p.lstat(),k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink')):raise ValueError('Actual immutable markers required')
 if name=='revision'and raw!=(revision+'\n').encode()or name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',raw):raise ValueError('Original marker values differ')
 digest.update(raw)
driver=runpy.run_path(str(code/'infra/ycbv_point_evaluate.py'),run_name='host_control')
pins=driver['validate_pins'](driver['files'].strict_json((code/driver['PINS']).read_bytes()));_,reports=driver['public_predictions'](driver['ROOT'],pins,decode=False);producer=driver['producer_sources'](driver['ROOT'],pins,reports);digest.update(json.dumps(producer,sort_keys=True).encode())
print(hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()+' '+digest.hexdigest())
PY
}
BEFORE="$(host_identity)";PROOF="${BEFORE%% *}";[[ "$PROOF" =~ ^[0-9a-f]{64}$ ]]
[[ ! -e "$OUT" && ! -L "$OUT" ]]
[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]]
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]]
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e
 if [[ -n "$CIDFILE" && -f "$CIDFILE" && ! -L "$CIDFILE" ]];then
  CID="$(cat "$CIDFILE")"
  if [[ "$CID" =~ ^[0-9a-f]{64}$ ]];then
   IDS="$(timeout 5s docker ps -aq --no-trunc --filter "id=$CID")";[[ $? == 0 ]] || STATUS=1
   if [[ -n "$IDS" ]];then
    OWNED="$(timeout 5s docker inspect "$CID" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}')"
    if [[ "$OWNED" == "$IMAGE|/$NAME|run_ycbv_point_evaluate|$REV" ]];then timeout 15s docker rm -f "$CID" >/dev/null || STATUS=1;else STATUS=1;fi
    IDS="$(timeout 5s docker ps -aq --no-trunc --filter "id=$CID")";[[ $? == 0 && -z "$IDS" ]] || STATUS=1
   fi
   chmod 400 "$CIDFILE"
  else STATUS=1;fi
 fi
 AFTER="$(host_identity)";[[ $? == 0 && "$AFTER" == "$BEFORE" ]] || STATUS=1
 [[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]] || STATUS=1
 exit "$STATUS"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
MOUNTS=()
for name in infra/ycbv_point_evaluate.py infra/run_ycbv_point_evaluate.sh infra/tudl_holdout_inputs.py src/world_reward/__init__.py src/world_reward/point_bop_evaluation.py src/world_reward/point_motion_evaluation.py configs/ycbv_point_evaluation_pins.json;do
 MOUNTS+=(--mount "type=bind,src=$CODE/$name,dst=$CODE/$name,readonly")
done
for path in "$JOB/revision" "$JOB/source-sha256" "$BASE/report.json" "$BASE/comparison_v1" "$BASE/automatic_masks_v1/report.json" "$BASE/eval_private";do
 [[ -e "$path" && ! -L "$path" ]];MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
for scene in 000048 000049 000050;do
 path="$BASE/automatic_masks_v1/scene_$scene/masks/1/000000.png";MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
mkdir -m 700 "$OUT";CIDFILE="$OUT/.container.cid"
set +e
timeout --signal=TERM --kill-after=10s 260s docker run --rm --interactive --name "$NAME" --cidfile "$CIDFILE" \
 --label world-reward.job=run_ycbv_point_evaluate --label "world-reward.revision=$REV" --network none --memory 8g --cpus 4 --user 0:0 \
 --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,size=64m \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" --entrypoint /usr/bin/env "$IMAGE" -i \
 PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" WR_YCBV_EVALUATION_HOST_PROOF_SHA256="$PROOF" \
 /opt/conda/bin/python -I -B - "$CODE" <<'PY'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]);sys.path[:0]=[str(code/'infra'),str(code/'src')];driver=code/'infra/ycbv_point_evaluate.py';sys.argv=[str(driver)]
runpy.run_path(str(driver),run_name='__main__')
PY
STATUS=$?;set -e
exit "$STATUS"
