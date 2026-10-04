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
import json,runpy,sys
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'))
driver=runpy.run_path(str(code/'infra/ycbv_point_evaluate.py'),run_name='host_control')
print(json.dumps(driver['host_snapshot'](driver['ROOT'],code,sys.argv[2]),sort_keys=True,separators=(',',':')))
PY
}
BEFORE="$(host_identity)"
PROOF="$(/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B -c 'import json,sys;print(json.loads(sys.argv[1])["pins_sha256"])' "$BEFORE")";[[ "$PROOF" =~ ^[0-9a-f]{64}$ ]]
[[ ! -e "$OUT" && ! -L "$OUT" ]]
[[ "$(docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]]
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]]
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e;CLEANUP_OK=0;CLEANUP_ERROR=0
 REPORT_BEFORE='null'
 if [[ -n "$CIDFILE" && -d "$OUT" && ! -L "$OUT" ]];then
  REPORT_BEFORE="$(/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B - "$CODE" "$OUT" <<'PYREPORT'
from pathlib import Path
import json,sys
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));import tudl_holdout_inputs as files
identity=files.identity(Path(sys.argv[2])/'report.json')
if identity['bytes']>262144:raise ValueError('Bounded aggregate-only report required')
print(json.dumps(identity,sort_keys=True,separators=(',',':')))
PYREPORT
  )";[[ $? == 0 ]] || { REPORT_BEFORE='null';[[ "$STATUS" != 0 ]] || STATUS=1; }
 fi
 if [[ -n "$CIDFILE" && -f "$CIDFILE" && ! -L "$CIDFILE" ]];then
  CID="$(cat "$CIDFILE")"
  if [[ "$CID" =~ ^[0-9a-f]{64}$ ]];then
   IDS="$(timeout 5s docker ps -aq --no-trunc --filter "id=$CID")";[[ $? == 0 ]] || CLEANUP_ERROR=1
   if [[ -n "$IDS" ]];then
    if [[ "$IDS" == "$CID" ]];then
     OWNED="$(timeout 5s docker inspect "$CID" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}')"
     if [[ $? == 0 && "$OWNED" == "$IMAGE|/$NAME|run_ycbv_point_evaluate|$REV" ]];then timeout 15s docker rm -f "$CID" >/dev/null || CLEANUP_ERROR=1;else CLEANUP_ERROR=1;fi
    else CLEANUP_ERROR=1;fi
    IDS="$(timeout 5s docker ps -aq --no-trunc --filter "id=$CID")";[[ $? == 0 && -z "$IDS" ]] || CLEANUP_ERROR=1
   fi
   chmod 400 "$CIDFILE";[[ $? == 0 ]] || CLEANUP_ERROR=1
   if [[ "$CLEANUP_ERROR" == 0 ]];then CLEANUP_OK=1;else STATUS=1;fi
  else STATUS=1;fi
 fi
 if [[ -n "$CIDFILE" && -d "$OUT" && ! -L "$OUT" ]];then
  /usr/bin/env -i PATH=/usr/bin:/bin WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/python3 -I -B - "$CODE" "$REV" "$BEFORE" "$REPORT_BEFORE" "$STATUS" "$CLEANUP_OK" <<'PYSEAL'
from pathlib import Path
import json,runpy,sys
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));driver=runpy.run_path(str(code/'infra/ycbv_point_evaluate.py'),run_name='host_post')
if not driver['write_host_seal'](driver['ROOT'],code,sys.argv[2],json.loads(sys.argv[3]),json.loads(sys.argv[4]),int(sys.argv[5]),sys.argv[6]=='1'):raise SystemExit(1)
PYSEAL
  [[ $? == 0 || "$STATUS" != 0 ]] || STATUS=1
 else STATUS=1;fi
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
