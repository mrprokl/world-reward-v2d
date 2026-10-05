#!/usr/bin/env bash
# Source closure: /src/world_reward/affine_span_gate.py /infra/mediapipe_cpu_runtime_verify.py
set -euo pipefail
(( $#==0 )) || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_mhr_scale_span/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock" PYTHONDONTWRITEBYTECODE=1
START=$SECONDS
bounded() { local limit="$1" remaining=$((75-(SECONDS-START)));shift;(( remaining>0 ))||return 1;(( remaining>=limit ))||limit=$remaining;timeout --signal=TERM --kill-after=1s "${limit}s" "$@"; }
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
OLD=8e7c14dbcc09e93638b19dbab71457ba4f8b1ec4
BASE="$ROOT/results/joint-point-component-preflight-$OLD"
auth() {
 bounded 20 python3 -I -B - "$ROOT" "$CODE" "$REV" "$OLD" "$BASE" <<'PYAUTH'
import hashlib,json,sys
from pathlib import Path
root,code=map(Path,sys.argv[1:3]);rev,old=sys.argv[3:5];base=Path(sys.argv[5])
sys.path.insert(0,str(code/'infra'));import mediapipe_cpu_runtime_verify as rt
rt.require(Path(rt.__file__).resolve()==code/'infra/mediapipe_cpu_runtime_verify.py','Actual current helper required')
source=rt.source(root,code,rev,'run_mhr_scale_span',('infra/run_mhr_scale_span.sh','src/world_reward/affine_span_gate.py','infra/mediapipe_cpu_runtime_verify.py'))
rows=(('native/report.json',3506,'d34c84b724ef1f4069dc410b94cbb9c9017ca551699e5cbeb0bda3f6e3de625a'),('native/mhr_metadata.npz',2976290,'c3df5caeffbd85935c4bd5a21f246694500b334fe521f8774ebc26aacaa094d0'),('native/mhr_metadata.json',6553,'9f0d8130d9ca2d56eb623799af002dc4c118aca185f39a570c47d2d00571ebfc'),('report.json',6503,'f441e26bfc29393436f6c2e83c0a7b9b7e101e76f7f41de358b4a5041c1129ed'),('proof.json',26258,'341208c1c614f4029be058ff0f4f86e213783a18939b38daa569205940b21af2'))
rt.require(base==root/'results'/('joint-point-component-preflight-'+old)and base.stat().st_mode&0o777==0o555 and(base/'native').stat().st_mode&0o777==0o555,'Original sealed failed scope required')
files={name:rt.identity(base/name,4<<20)for name,_,_ in rows}
rt.require(all(files[n]==dict(bytes=b,sha256=s)and(base/n).stat().st_mode&0o777==0o444 for n,b,s in rows),'Exact five original saved files required')
reports=[rt.strict((base/n).read_bytes())for n in('native/report.json','report.json','proof.json')]
rt.require(all(r.get('status')=='fail'for r in reports[:2])and reports[0]['control']=='component_preflight','Original failure remains FAIL')
historical=rt.source(root,root/'jobs'/old/'run_joint_point_authored_qualify/code',old,'run_joint_point_authored_qualify',('infra/joint_point_authored_qualify.py','infra/run_joint_point_authored_qualify.sh'))
rt.require(historical['entries']==245 and historical['closure_sha256']=='0d921499ed5ccd954dd1a60ba0e16fa1bd06a072fc42f211c84046b6674a9809'and all(all(historical[k]==r['source_binding'][k]for k in('producer_revision','markers','entries','closure_sha256'))for r in reports),'Complete original three source heads differ')
print(json.dumps(dict(source=source,old_source=historical,input_identities=files,original_failed_status_preserved=True),sort_keys=True))
PYAUTH
}
[[ "$(bounded 5 docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]]
BEFORE="$(auth)";CURRENT_PARENT="${CODE%/code}";MOUNTS=(--mount "type=bind,src=$CURRENT_PARENT,dst=$CURRENT_PARENT,readonly")
OLD_PARENT="$ROOT/jobs/$OLD/run_joint_point_authored_qualify"
MOUNTS+=(--mount "type=bind,src=$OLD_PARENT,dst=$OLD_PARENT,readonly")
for name in proof.json report.json native/report.json native/mhr_metadata.npz native/mhr_metadata.json;do MOUNTS+=(--mount "type=bind,src=$BASE/$name,dst=$BASE/$name,readonly");done
RESULT="$(bounded 31 docker run --rm -i --network none --read-only --cap-drop ALL \
 --security-opt no-new-privileges --cpus 2 --memory 4g --user 1000:1000 --tmpfs /tmp:rw,nosuid,nodev,size=16m \
 --entrypoint python --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 \
 --env "PYTHONPATH=$CODE/src" "${MOUNTS[@]}" "$IMAGE" -B - "$CODE" "$BASE" <<'PYDIAG'
from dataclasses import asdict
import hashlib,json,os,signal,sys,time
from pathlib import Path
code,base=map(Path,sys.argv[1:]);started=time.monotonic()
def expired(*_):raise TimeoutError('Saved algebraic diagnostic deadline')
signal.signal(signal.SIGALRM,expired);signal.alarm(30)
import numpy as np
import world_reward.affine_span_gate as gate
if Path(gate.__file__).resolve()!=code/'src/world_reward/affine_span_gate.py':raise ValueError('Actual current pure gate required')
if os.geteuid()!=1000 or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}or 'torch'in sys.modules:raise ValueError('Offline saved-only CPU required')
meta=json.loads((base/'native/mhr_metadata.json').read_bytes())
with np.load(base/'native/mhr_metadata.npz',allow_pickle=False)as archive:
 arrays={name:archive[name].copy()for name in archive.files}
expected={'bounds','parents','transform','lbs_indices','lbs_weights','faces','hand_pose_mean','hand_pose_comps','hand_left_indices','hand_right_indices','scale_mean','scale_comps'}
if set(arrays)!=expected or len(meta['parameter_names'])!=249 or len(meta['joint_names'])!=127 or meta['network_executed']is not False:raise ValueError('Exact original retained metadata required')
# Literal array-dict branch of the authenticated historical full.fingerprint.
h=hashlib.sha256();h.update(b'dict')
for name in sorted(arrays):
 value=arrays[name]
 if value.dtype.hasobject:raise ValueError('Object metadata forbidden')
 h.update(repr(name).encode());h.update(str((value.shape,value.dtype.str)).encode());h.update(value.tobytes(order='C'))
if h.hexdigest()!=meta['metadata_sha256']:raise ValueError('Original saved metadata fingerprint differs')
mean,comps=arrays['scale_mean'],arrays['scale_comps']
if mean.shape!=(68,)or comps.shape!=(28,68)or mean.dtype!=np.float32 or comps.dtype!=np.float32:raise ValueError('Actual F32 scale PCA ABI required')
result=gate.affine_span_gate(comps.T,mean,np.zeros(68,np.float32))
report=dict(status='diagnostic_complete',stage='mhr_saved_scale_affine_span_v1',result=asdict(result),nonmembership_certified=result.nonmembership_certified,
 metadata_fingerprint=h.hexdigest(),original_failed_status_preserved=True,models_loaded=False,network_executed=False,fit_performed=False,approximation_used=False,membership_certified=False,adoption=False,elapsed_seconds=time.monotonic()-started)
raw=json.dumps(report,sort_keys=True,allow_nan=False)
if len(raw)>8192 or time.monotonic()-started>30:raise TimeoutError('Bounded inclusive saved diagnostic')
print(raw);signal.alarm(0)
PYDIAG
)"
[[ "$(auth)" == "$BEFORE" ]]
bounded 3 python3 -I -B - "$BEFORE" "$RESULT" <<'PYFINAL'
import json,sys
proof,result=map(json.loads,sys.argv[1:])
if result['status']!='diagnostic_complete'or result['result']['status']not in('EXACT_NOT_IN_AFFINE_SPAN','INCONCLUSIVE'):raise ValueError('Certificate or inconclusive only')
result.update(input_identities=proof['input_identities'],source_binding=proof['source'],old_source=proof['old_source'],source_inputs_rehashed_after=True)
print(json.dumps(result,sort_keys=True,allow_nan=False))
PYFINAL
