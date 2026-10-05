#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
source "$CODE/infra/cari_wrapper_common.sh"
wr_parse_cari_arguments prepare "$@"
wr_cari_dependency prepare
# Surface provenance is hash-only: these exact old source parents are not on PYTHONPATH.
# Source closure: /infra/mediapipe_cpu_runtime_verify.py
HISTORICAL_MOUNTS=()
if [[ "$WR_MESH_SOURCE" == surface ]]; then
 [[ "$ROOT" == /srv/scenesmith/world-reward && "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$WR_CODE_REVISION/run_cari_prepare/code" && "$(uname -s)" == Linux && "$(id -u)" == 0 ]] || exit 2
 SURFACE_START=$SECONDS
 SURFACE_CLOCK="$(python3 -I -B -c 'import time;print(time.monotonic())')"
 HISTORICAL_SOURCES="$(timeout --signal=TERM --kill-after=5s 300s python3 -I -B - "$ROOT" "$CODE" "${WR_CODE_REVISION:?}" "$WR_EPISODE" <<'PYSURFACE'
import sys,re
from pathlib import Path
root,code=map(Path,sys.argv[1:3]);revision=sys.argv[3];episode=int(sys.argv[4])
sys.path.insert(0,str(code/'infra'));import mediapipe_cpu_runtime_verify as rt
rt.require(Path(rt.__file__).resolve()==code/'infra/mediapipe_cpu_runtime_verify.py','Actual current source helper required')
current=rt.source(root,code,revision,'run_cari_prepare',('infra/run_cari_prepare.sh','infra/cari_wrapper_common.sh'))
pinpath=code/f'configs/surface_mesh_{episode:06d}_pins.json';pin=rt.identity(pinpath,2_000_000);pins=rt.strict(pinpath.read_bytes())
rt.require(set(pins)=={'schema','episode_index','input_sha256','metric_scale_baked_once','report','files','source_helpers'} and pins['schema']=='world_reward.surface_mesh_pins.v1' and type(pins['episode_index'])is int and pins['episode_index']==episode,'Exact selected surface pins required')
q=rt.strict((code/'configs/surface_qslim_qualification_pins.json').read_bytes());first=rt.strict((code/'configs/surface_identity_qualification_pins.json').read_bytes())
rt.require(q['schema']=='world_reward.surface_qslim_qualification_pins.v1' and first['schema']=='world_reward.surface_identity_qualification_pins.v1','Original qualification profiles required')
producer=pins['report'];revisions=(producer['producer_revision'],q['producer_revision'],first['producer_revision'])
rt.require(all(re.fullmatch('[0-9a-f]{40}',r)for r in revisions),'Exact original revisions required')
records=((root/f'outputs/episode_{episode:06d}'/('object_budget_surface_'+revisions[0])/'report.json',{k:producer[k]for k in('bytes','sha256')},'source_binding'),(root/'results'/('surface-qslim-qualify-'+revisions[1])/'native.json',q['native'],'source_proof'),(root/'results'/('surface-identity-qualify-'+revisions[2])/'native.json',first['native'],'source_proof'))
parents=[];frozen={pinpath:pin}
for r,entry,(receipt,identity,key)in zip(revisions,('run_object_budget_solid','run_surface_qslim_qualify','run_surface_identity_qualify'),records):
 value=rt.pinned(receipt,identity,2_000_000);frozen[receipt]=identity;bound=value[key]if key=='source_binding'else value[key]['source_binding']
 parent=rt.canonical(root/'jobs'/r/entry);rt.require({p.name for p in parent.iterdir()}=={'code','revision','source-sha256'},'Exact source-only historical parent required')
 source=rt.source(root,parent/'code',r,entry,())
 rt.require(all(source[k]==bound[k]for k in('producer_revision','markers','entries','closure_sha256')),'Original complete historical source differs')
 for p in(parent/'code').rglob('*'):
  if p.is_file():rt.require((p.suffix in('.py','.sh','.cpp','.hpp','.h','.json','.toml')or p.name.startswith('Dockerfile'))and b'\0'not in p.read_bytes(),'No binary/asset in historical proof closure')
 parents.append(parent)
rt.require(current==rt.source(root,code,revision,'run_cari_prepare',('infra/run_cari_prepare.sh','infra/cari_wrapper_common.sh'))and all(rt.identity(p,2_000_000)==v for p,v in frozen.items()),'Historical proof inputs changed')
print('\n'.join(map(str,parents)))
PYSURFACE
)"
 while IFS= read -r path;do HISTORICAL_MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly");done <<< "$HISTORICAL_SOURCES"
fi
set -- --episode "$WR_EPISODE"
if [[ "$WR_MESH_SOURCE" != default ]]; then set -- "$@" --mesh-source "$WR_MESH_SOURCE"; fi
if (( WR_QUERY_REQUALIFICATION )); then set -- "$@" --query-requalification; fi
if [[ "$WR_MESH_SOURCE" == surface ]]; then
 REV="$WR_CODE_REVISION";printf -v PADDED '%06d' "$WR_EPISODE"
 OUT="$ROOT/outputs/episode_$PADDED/cari_inputs"
 CONTROL="$ROOT/results/cari-native-prepare-surface-$PADDED-$REV"
 IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
 NAME="wr-cari-prepare-surface-$PADDED-$REV";CID_FILE="$CONTROL/.container.cid"
 BEFORE='';CLEANUP_VERIFIED=0;POST_VERIFIED=0;NATIVE_VERIFIED=0;CREATED=0
 surface_host() {
  local remaining=$((7200+30-(SECONDS-SURFACE_START)))
  (( remaining>0 )) || return 1
  (( remaining<=300 )) || remaining=300
  timeout --signal=TERM --kill-after=1s "${remaining}s" python3 -I -B - "$ROOT" "$CODE" "$REV" "$WR_EPISODE" "$CONTROL" "$OUT" "$1" "$SURFACE_CLOCK" "${2:-1}" "$POST_VERIFIED" "$CLEANUP_VERIFIED" "$NATIVE_VERIFIED" <<'PYSURFACEHOST'
import hashlib,json,os,re,stat,sys,time
from pathlib import Path
root,code,revision,episode,control,out,mode,started,status,post,cleanup,native=sys.argv[1:]
root,code,control,out=map(Path,(root,code,control,out));episode=int(episode);started=float(started)
sys.path.insert(0,str(code/'infra'));import mediapipe_cpu_runtime_verify as rt
rt.require(os.geteuid()==0 and Path(rt.__file__).resolve()==code/'infra/mediapipe_cpu_runtime_verify.py','Actual root-owned host helper required')
rt.require(code==root/'jobs'/revision/'run_cari_prepare/code' and out==root/f'outputs/episode_{episode:06d}/cari_inputs' and control==root/'results'/f'cari-native-prepare-surface-{episode:06d}-{revision}','Exact surface control/output namespace required')
for p in(root,code,control,out):rt.canonical(p)
if mode=='seal':
    elapsed=time.monotonic()-started;good=status=='0' and post==cleanup==native=='1' and elapsed<=7200
    record=dict(stage='world_reward_cari_prepare_surface_host_v1',status='pass'if good else'fail',episode_index=episode,
        producer_revision=revision,elapsed_seconds=elapsed,budget_seconds=7200,cleanup_grace_seconds=30,
        source_inputs_rehashed_after=post=='1',owned_container_absence_verified=cleanup=='1',native_report_verified=native=='1',
        original_exit_status=int(status),gpu_used=False,ground_truth_used=False,adoption=False)
    proofpath=control/'proof.json';record['host_proof_identity']=rt.identity(proofpath,2<<20)
    if good:record['native_report_identity']=rt.identity(out/'report.json',32<<20,readonly=False)
    target=control/'report.json';fd=os.open(target,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
    with os.fdopen(fd,'w+b')as f:
        def write_record():
            a=target.lstat();b=os.fstat(f.fileno())
            rt.require(a.st_dev==b.st_dev and a.st_ino==b.st_ino and a.st_nlink==b.st_nlink==1,'Owned receipt inode changed')
            f.seek(0);f.truncate();f.write((json.dumps(record,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
        try:
            write_record()
            for p in control.iterdir():
                a=p.lstat();rt.require(p.name in('proof.json','report.json','.container.cid')and stat.S_ISREG(a.st_mode)and a.st_nlink==1 and a.st_uid==0 and not p.is_symlink(),'Only owned control artifacts permitted');p.chmod(0o444)
            control.chmod(0o555)
            directory=os.open(control,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:os.fsync(directory)
            finally:os.close(directory)
            elapsed=time.monotonic()-started
            if good and elapsed>7200:
                good=False;record.update(status='fail',elapsed_seconds=elapsed,failure_type='TimeoutError');write_record()
            rt.require(time.monotonic()-started<=7200+(0 if good else 30),'Inclusive surface completion budget exhausted')
            rt.require(not(status=='0'and not good),'A native PASS alone cannot pass host controls')
        except BaseException as error:
            record.update(status='fail',elapsed_seconds=time.monotonic()-started,failure_type=type(error).__name__)
            write_record();raise
    sys.exit(0)
source=rt.source(root,code,revision,'run_cari_prepare',('infra/run_cari_prepare.sh','infra/cari_wrapper_common.sh','infra/cari_prepare.py'))
base=out.parent;pinpath=code/f'configs/surface_mesh_{episode:06d}_pins.json';pins=rt.pinned(pinpath,rt.identity(pinpath,16384),16384)
rt.require(set(pins)=={'schema','episode_index','input_sha256','metric_scale_baked_once','report','files','source_helpers'}and pins['schema']=='world_reward.surface_mesh_pins.v1'and type(pins['episode_index'])is int and pins['episode_index']==episode,'Exact independent surface profile required')
q=rt.strict((code/'configs/surface_qslim_qualification_pins.json').read_bytes());first=rt.strict((code/'configs/surface_identity_qualification_pins.json').read_bytes())
revisions=(pins['report']['producer_revision'],q['producer_revision'],first['producer_revision']);rt.require(all(re.fullmatch('[0-9a-f]{40}',r)for r in revisions),'Exact historical revisions required')
proposal=base/('object_budget_surface_'+revisions[0]);qualification=root/'results'/('surface-qslim-qualify-'+revisions[1])
expected={proposal/n for n in('report.json','native.json','geometry.npz','object_fixed_canonical.glb','candidate_geometry.npz','mapping.json')}|{base/'object_grounded'/n for n in('report.json','object.glb','transform.json','intrinsics.json')}|{base/'scale_smoke/report.json',qualification/'native.json',qualification/'report.json',root/'results'/('surface-identity-qualify-'+revisions[2])/'native.json',root/'results/surface-qslim-independent-v2/report.json'}
rt.require(set(pins['files'])=={str(p.relative_to(root))for p in expected},'Exact fifteen geometry/source leaves required')
identities={str(p):rt.identity(p,256<<20 if p==proposal/'mapping.json'else 32<<20,readonly=False)for p in expected}
rt.require(all(identities[str(root/n)]==v for n,v in pins['files'].items()),'Original surface artifacts changed')
for name,pin in pins['source_helpers'].items():rt.require(rt.identity(code/name,empty=True)==pin,'Current pinned pure helper changed')
rt.require(pins['report']['sha256']==identities[str(proposal/'report.json')]['sha256']and pins['report']['bytes']==identities[str(proposal/'report.json')]['bytes'],'Independent producer report differs')
historical={}
for r,entry in zip(revisions,('run_object_budget_solid','run_surface_qslim_qualify','run_surface_identity_qualify')):
    historical[str(root/'jobs'/r/entry)]=rt.source(root,root/'jobs'/r/entry/'code',r,entry,())
manifest=root/'results/input-manifest.json';m=rt.strict(manifest.read_bytes());meta=root/'data/track_1/meta/episodes.jsonl'
video=root/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4'
rt.require((m['track'],m['repo_id'],m['revision'])==('track_1','nvidia/video_to_data_challenge','5f68335f3acc802033d1e80728c1633197521de8'),'Original Track1 manifest required')
for p in(meta,video):
    rows=[r for r in m['files']if r['path']==str(p.relative_to(root/'data'))];rt.require(len(rows)==1,'One original public input required');value=rt.identity(p,32<<30,readonly=False);rt.require(value=={k:rows[0][k]for k in('bytes','sha256')},'Original public input changed');identities[str(p)]=value
rows=[rt.strict(line)for line in meta.read_bytes().splitlines()if line.strip()];rows=[r for r in rows if r['episode_index']==episode];rt.require(len(rows)==1 and type(rows[0]['length'])is int and rows[0]['length']>=3,'Original full timeline required');total=rows[0]['length']
rt.require(identities[str(video)]['sha256']==pins['input_sha256'],'Surface and original video differ')
pose=base/'object_pose_full_surface';rt.require(stat.S_IMODE(pose.stat().st_mode)==0o555 and {p.name for p in pose.iterdir()}=={'report.json','geometry_and_poses.npz','object_fixed_canonical.glb'},'Complete sealed surface poses required')
for p in pose.iterdir():identities[str(p)]=rt.identity(p,1<<30)
pr=rt.strict((pose/'report.json').read_bytes());rt.require(all(type(pr.get(k))is type(v)and pr[k]==v for k,v in dict(stage='fixed_scale_full_object_pose_initializer',status='pass',mesh_source='surface',episode_index=episode,input_sha256=pins['input_sha256'],execution_verified=True,original_frame_coverage_verified=True,fixed_shape=True,ground_truth_used=False,hand_labeled_test=False,oracle_modes=[]).items()),'Qualified surface pose report required')
rt.require([r['frame_index']for r in pr['frames']]==list(range(total))and pr['geometry_and_poses_sha256']==identities[str(pose/'geometry_and_poses.npz')]['sha256']and pr['fixed_canonical_mesh_sha256']==identities[str(pose/'object_fixed_canonical.glb')]['sha256']==identities[str(proposal/'object_fixed_canonical.glb')]['sha256']and pr['topology_budget']['committed_pins_sha256']==rt.identity(pinpath)['sha256'],'Full surface pose byte lineage differs')
reports={}
for name,stage in(('body_full','sam3d_body_full_video_initializer'),('depth_full','monocular_moge2_full_video')):
    path=base/name/'report.json';record=rt.strict(path.read_bytes());rt.require(all(type(record.get(k))is type(v)and record[k]==v for k,v in dict(stage=stage,status='pass',episode_index=episode,total_video_frames=total,input_track='track_1',input_sha256=pins['input_sha256'],ground_truth_used=False,hand_labeled_test=False,oracle_modes=[]).items())and[r['frame_index']for r in record['frames']]==list(range(total)),'Full original body/depth report required');reports[name]=record;identities[str(path)]=rt.identity(path,32<<20,readonly=False)
adapter=base/'body_full/cari_adapter';ar=rt.strict((adapter/'report.json').read_bytes())
rt.require(all(type(ar.get(k))is type(v)and ar[k]==v for k,v in dict(stage='native_cari_body_adapter_full_video',status='pass',episode_index=episode,frames=total,input_track='track_1',ground_truth_used=False,hand_labeled_test=False,oracle_modes=[]).items())and ar['body_report_sha256']==identities[str(base/'body_full/report.json')]['sha256'],'Existing full body adapter must be reused unchanged')
for p in(adapter/'report.json',adapter/'canonical_initializer.pkl'):identities[str(p)]=rt.identity(p,1<<30,readonly=False)
rt.require(ar['canonical_initializer_sha256']==identities[str(adapter/'canonical_initializer.pkl')]['sha256'],'Original body adapter changed')
for directory,suffix in((base/'automatic_masks/masks/0','.png'),(base/'automatic_masks/masks/1','.png'),(base/'depth_full','.npz')):
    names={f'{i:06d}{suffix}'for i in range(total)}|({'report.json'}if suffix=='.npz'else set());rt.require({p.name for p in directory.iterdir()}==names,'Exact full original input directory required')
    for p in sorted(directory.iterdir()):identities[str(p)]=rt.identity(p,1<<30,readonly=False)
for r in reports['depth_full']['frames']:rt.require(identities[str(base/'depth_full'/f"{r['frame_index']:06d}.npz")]['sha256']==r['output_sha256'],'Full depth changed')
for r in reports['body_full']['frames']:rt.require(identities[str(base/'automatic_masks/masks/0'/f"{r['frame_index']:06d}.png")]['sha256']==r['mask_sha256'],'Full body mask changed')
for r in pr['frames']:rt.require(identities[str(base/'automatic_masks/masks/1'/f"{r['frame_index']:06d}.png")]['sha256']==r['object_mask_sha256'],'Full object mask changed')
for p in(manifest,base/'automatic_masks/report.json',base/'automatic_masks/prompts.json'):identities[str(p)]=rt.identity(p,32<<20,readonly=False)
proof=dict(source=source,inputs=identities,historical_sources=historical,total_frames=total,video_sha256=pins['input_sha256'])
digest=hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest()
if mode=='before':
    rt.require(not out.exists()and not out.is_symlink()and not control.exists()and not control.is_symlink(),'Fresh native preparation/control required')
    control.mkdir(mode=0o755);fd=os.open(control/'proof.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
    with os.fdopen(fd,'wb')as f:f.write((json.dumps(proof,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno())
else:
    rt.require(proof==rt.strict((control/'proof.json').read_bytes()),'Original source/inputs changed after preparation')
    if mode=='complete':
        result=rt.strict((out/'report.json').read_bytes());rt.require(all(type(result.get(k))is type(v)and result[k]==v for k,v in dict(stage='world_reward_native_cari_inputs',status='pass',frames=total,episode_index=episode,producer_revision=revision,input_sha256=pins['input_sha256'],original_frame_coverage_verified=True,object_source='surface',ground_truth_used=False,hand_labeled_test=False,oracle_modes=[]).items()),'Complete full native surface preparation required')
        rt.require(result['script_sha256']==source['helpers']['infra/cari_prepare.py']['sha256']and result['object_pose_source']==dict(report=str((pose/'report.json').relative_to(root)),geometry_and_poses=str((pose/'geometry_and_poses.npz').relative_to(root)),geometry_and_poses_sha256=identities[str(pose/'geometry_and_poses.npz')]['sha256'])and result['surface_geometry_validation']['source_rehashed_after']is True,'Native surface preparation lineage differs')
print(digest)
PYSURFACEHOST
 }
 surface_inspect() {
  timeout 6s python3 -I -B - "$1" "$NAME" "$IMAGE" "$REV" <<'PYSURFACEINSPECT'
import subprocess,sys
cid,name,image,revision=sys.argv[1:]
try:
    result=subprocess.run(['docker','inspect',cid,'--format','{{.Id}} {{.Name}} {{.Image}} {{index .Config.Labels "world_reward.cari_prepare.owner"}}'],capture_output=True,timeout=5)
except (OSError,subprocess.TimeoutExpired):sys.exit(2)
if result.returncode==0:
    sys.exit(0 if result.stdout==(f'{cid} /{name} {image} {revision}\n').encode()and result.stderr==b''else 2)
absent=tuple((prefix+cid).encode()for prefix in('Error: No such object: ','error: no such object: ','Error: No such container: ','Error response from daemon: No such container: '))
# Docker may emit an empty JSON list on stdout and leading empty stderr lines.
# Strip line separators only, never whitespace or daemon/error context.
sys.exit(1 if result.returncode==1 and result.stdout in(b'',b'[]\n')and result.stderr.strip(b'\r\n')in absent else 2)
PYSURFACEINSPECT
 }
 surface_cleanup() {
  local cid rc
  if [[ ! -e "$CID_FILE" && ! -L "$CID_FILE" ]];then (( CREATED==0 )) || return 1;return 0;fi
  cid="$(timeout 5s python3 -I -B - "$CID_FILE" <<'PYSURFACECID'
import os,re,stat,sys
from pathlib import Path
p=Path(sys.argv[1]);a=p.lstat()
if p.resolve()!=p or any(x.is_symlink()for x in(p,*p.parents))or not stat.S_ISREG(a.st_mode)or a.st_nlink!=1 or a.st_uid!=os.geteuid()or a.st_size not in(64,65):raise ValueError('Owned regular CID required')
b=p.read_bytes();z=p.lstat()
if not re.fullmatch(b'[0-9a-f]{64}\n?',b)or any(getattr(a,k)!=getattr(z,k)for k in('st_dev','st_ino','st_mode','st_nlink','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('CID changed')
print(b.decode().strip())
PYSURFACECID
)" || return 1
  if surface_inspect "$cid";then
   timeout 10s docker rm -f "$cid" >/dev/null 2>&1 || return 1
   if surface_inspect "$cid";then return 1;else rc=$?;[[ "$rc" == 1 ]] || return 1;fi
  else rc=$?;[[ "$rc" == 1 ]] || return 1;fi
  CLEANUP_VERIFIED=1
 }
 surface_finish() {
  local status=$? check
  trap - EXIT INT TERM;set +e
  surface_cleanup;check=$?;if (( check!=0 && status==0 ));then status=1;fi
  if [[ -n "$BEFORE" ]];then
   if (( status==0 ));then
    [[ "$(surface_host complete)" == "$BEFORE" ]];check=$?;if (( check==0 ));then POST_VERIFIED=1;NATIVE_VERIFIED=1;else status=1;fi
   else
    [[ "$(surface_host after)" == "$BEFORE" ]];check=$?;if (( check==0 ));then POST_VERIFIED=1;fi
   fi
   surface_host seal "$status";check=$?;if (( check!=0 && status==0 ));then status=1;fi
  fi
  exit "$status"
 }
 trap surface_finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
 [[ "$(timeout 5s docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')" == "$IMAGE" ]] || exit 1
 EXISTING="$(timeout 5s docker ps -aq --filter "name=^/$NAME$")" || exit 1
 [[ -z "$EXISTING" ]] || exit 1
 BEFORE="$(surface_host before)"
 REMAINING=$((7200-(SECONDS-SURFACE_START)));(( REMAINING>0 )) || exit 1
 CREATED=1
 timeout --signal=TERM --kill-after=1s "${REMAINING}s" docker run --rm --name "$NAME" --cidfile "$CID_FILE" --label "world_reward.cari_prepare.owner=$REV" --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
  --mount "type=bind,src=$ROOT/data,dst=$ROOT/data,readonly" --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" "${HISTORICAL_MOUNTS[@]}" "$IMAGE" python "$CODE/infra/cari_prepare.py" "$@"
 exit 0
fi
docker run --rm --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env PYTHONPATH="$CODE/src" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
  --mount "type=bind,src=$ROOT/data,dst=$ROOT/data,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" \
  ${HISTORICAL_MOUNTS[@]+"${HISTORICAL_MOUNTS[@]}"} \
  world-reward/cari4d-source:0.1 python "$CODE/infra/cari_prepare.py" "$@"
