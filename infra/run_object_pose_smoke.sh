#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Source closure: /infra/mediapipe_cpu_runtime_verify.py /infra/object_pose_smoke.py
# Solid alone uses a narrow inert import closure, distinct from full host provenance.
if [[ " ${*} " == *solid* ]];then
 [[ $# == 5 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ && "$3" == --full-video && "$4" == --mesh-source && "$5" == solid ]] || exit 2
 EPISODE="$2";REV="${WR_CODE_REVISION:?}"
 [[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_object_pose_smoke/code" && "$(uname -s)" == Linux ]] || exit 2
 printf -v PADDED '%06d' "$EPISODE"
 OUT="$ROOT/outputs/episode_$PADDED/object_pose_full_solid";CONTROL="$ROOT/results/object-pose-solid-$PADDED-$REV"
 LOCK="$ROOT/jobs/.world-reward-h100.lock";IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
 NAME="wr-object-pose-solid-$PADDED-$REV";BEFORE='';LOCK_BEFORE='';OPEN=0;OWNED=0;POST_VERIFIED=0;PROOF_SHA=''
 host() {
  timeout --signal=TERM --kill-after=5s 300s python3 -I -B - "$ROOT" "$CODE" "$REV" "$EPISODE" "$CONTROL" "$OUT" "$LOCK" "$1" "${BASH_SOURCE[0]}" <<'PYSOLID'
import ast,hashlib,json,os,re,stat,sys
from pathlib import Path
root,code,rev,episode,control,out,lock,mode,entry=sys.argv[1:];root,code,control,out,lock,entry=map(Path,(root,code,control,out,lock,entry));episode=int(episode)
if os.getuid()!=0 or entry!=code/'infra/run_object_pose_smoke.sh':raise ValueError('Original root-owned Linux wrapper required')
sys.path.insert(0,str(code/'infra'));import mediapipe_cpu_runtime_verify as rt
if Path(rt.__file__).resolve()!=code/'infra/mediapipe_cpu_runtime_verify.py':raise ValueError('Actual host source helper required')
rt.canonical(out);rt.canonical(control);rt.canonical(lock)
s=lock.lstat();rt.require(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'Original cooperative lock required')
if mode=='lock':
 f=os.fstat(9);rt.require((f.st_dev,f.st_ino)==(s.st_dev,s.st_ino),'Original lock inode changed');print(f'{s.st_dev}:{s.st_ino}');sys.exit(0)
if mode=='mounts':
 rt.require(control.is_dir()and not control.is_symlink(),'Owned preflight ledger required');rawproof=(control/'proof.json').read_bytes();rt.require(rt.identity(control/'proof.json',32<<20)['sha256']==os.environ['WR_POSE_PROOF_SHA256'],'Preflight ledger changed');proof=rt.strict(rawproof);print('\n'.join(proof['mounts']));sys.exit(0)
source=rt.source(root,code,rev,'run_object_pose_smoke',('infra/run_object_pose_smoke.sh','infra/object_pose_smoke.py'))
def raw(p,maximum=32<<20,empty=False):return rt.identity(p,maximum,readonly=False,empty=empty)
def read(p):return rt.strict(Path(p).read_bytes())
def same(p,pin):rt.require(raw(p,32<<30)==pin,'Pinned input changed')
base=root/f'outputs/episode_{episode:06d}';pinpath=code/f'configs/solid_mesh_{episode:06d}_pins.json';pin=rt.identity(pinpath);pins=read(pinpath)
rt.require(set(pins)=={'schema','episode_index','input_sha256','metric_scale_baked_once','report','files','source_helpers'} and pins['schema']=='world_reward.solid_mesh_pins.v1' and type(pins['episode_index'])is int and pins['episode_index']==episode,'Actual solid pins required')
producer=pins['report'];qfile='configs/solid_chart_v2_qualification_pins.json';q=read(code/qfile)
rt.require(re.fullmatch('[0-9a-f]{40}',producer['producer_revision']) and re.fullmatch('[0-9a-f]{40}',q['producer_revision']),'Exact original producer revisions required')
proposal=base/('object_budget_solid_'+producer['producer_revision']);qualification=root/'results'/('solid-chart-v2-qualify-'+q['producer_revision'])
expected={proposal/n for n in ('report.json','native.json','geometry.npz','object_fixed_canonical.glb')}|{base/'object_grounded'/n for n in ('report.json','object.glb','transform.json','intrinsics.json')}|{base/'scale_smoke/report.json',qualification/'report.json',qualification/'native.json'}
rt.require(set(pins['files'])=={str(p.relative_to(root))for p in expected},'Exactly eleven canonical solid artifacts required')
inputs={str(p):raw(p)for p in expected}
for name,value in pins['files'].items():same(root/name,value)
rt.require(inputs[str(proposal/'report.json')]=={k:producer[k]for k in ('bytes','sha256')},'Independent producer report differs')
for directory in(proposal,qualification):
 rt.canonical(directory);rt.require(stat.S_IMODE(directory.stat().st_mode)==0o555 and all(p.is_file() and not p.is_symlink() and not p.stat().st_mode&0o111 for p in directory.iterdir()),'Sealed metadata-only proposal/qualification directory required')
manifest=root/'results/input-manifest.json';meta=root/'data/track_1/meta/episodes.jsonl';m=read(manifest)
rt.require((m.get('track'),m.get('repo_id'),m.get('revision'))==('track_1','nvidia/video_to_data_challenge','5f68335f3acc802033d1e80728c1633197521de8'),'Track1 manifest required')
video=root/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4'
for p in(meta,video):
 rows=[r for r in m['files']if r['path']==str(p.relative_to(root/'data'))];rt.require(len(rows)==1,'Selected original input required');same(p,{k:rows[0][k]for k in('bytes','sha256')});inputs[str(p)]=raw(p,32<<30)
rows=[r for r in [rt.strict(line)for line in meta.read_bytes().splitlines()]if r['episode_index']==episode];rt.require(len(rows)==1 and type(rows[0]['length'])is int and rows[0]['length']>=3,'Original fullT metadata required');total=rows[0]['length']
rt.require(inputs[str(video)]['sha256']==pins['input_sha256'],'Solid proposal belongs to original video')
leaves=[manifest,base/'automatic_masks/report.json',base/'automatic_masks/prompts.json',base/'body_full/report.json',base/'body_smoke/report.json',base/'depth_smoke/report.json']
depth=base/'depth_full';dr=read(depth/'report.json');br=read(base/'body_full/report.json')
for report,stage in((dr,'monocular_moge2_full_video'),(br,'sam3d_body_full_video_initializer')):
 rt.require(all(type(report.get(k))is type(v)and report[k]==v for k,v in dict(stage=stage,status='pass',episode_index=episode,total_video_frames=total,input_track='track_1',input_sha256=pins['input_sha256'],ground_truth_used=False,hand_labeled_test=False,oracle_modes=[]).items()),'Full original body/depth proof required')
 rt.require([r['frame_index']for r in report['frames']]==list(range(total)),'Full original frame records required')
directories=[base/'automatic_masks/masks/0',base/'automatic_masks/masks/1',depth,proposal,qualification]
for directory in directories[:3]:
 rt.canonical(directory);names={f'{i:06d}'+('.npz'if directory==depth else'.png')for i in range(total)}|({'report.json'}if directory==depth else set());rt.require({p.name for p in directory.iterdir()}==names,'Exact fullT input directory required')
 for p in sorted(directory.iterdir()):inputs[str(p)]=raw(p,1<<30)
for r in dr['frames']:rt.require(inputs[str(depth/f"{r['frame_index']:06d}.npz")]['sha256']==r['output_sha256'],'Original depth artifact differs')
for p in leaves:inputs[str(p)]=raw(p)
configs=[pinpath,code/qfile,code/'configs/solid_chart_v2_build_pins.json',code/'configs/certified_solid_qualification_pins.json']
# GPU closure follows imports, not unrelated provenance literals in inert helpers.
paths=set(configs);pending=[code/'infra/object_pose_smoke.py',code/'infra/solid_geometry_loader.py',*(code/n for n in pins['source_helpers'])]
def module(name):return next((p for p in (code/'src'/Path(name.replace('.','/')).with_suffix('.py'),code/'src'/Path(name.replace('.','/'))/'__init__.py',code/'infra'/(name.split('.')[0]+'.py'))if p.is_file()),None)
while pending:
 p=pending.pop()
 if p in paths:continue
 paths.add(p)
 if p.suffix!='.py':continue
 for n in ast.walk(ast.parse(p.read_bytes())):
  names=[a.name for a in n.names]if isinstance(n,ast.Import)else[n.module or'']if isinstance(n,ast.ImportFrom)else[]
  if isinstance(n,ast.ImportFrom)and n.level:raise ValueError('Unexpected relative runtime import')
  for name in names:
   dep=module(name)
   if dep is not None:pending.append(dep)
for name,value in pins['source_helpers'].items():rt.require(rt.identity(code/name,empty=True)==value,'Actual reused math source differs')
paths.add(code/'src/world_reward/__init__.py')
rt.require(not any(p.parent==code/'infra'and p.suffix=='.py'and p.name.startswith(('certified_solid','object_budget_solid','mesh_serialization_compile','oriented_solid_compiler','mesh_conditioned_cache','solid_chart_v2'))for p in paths),'No solver/model/qualification driver mounted')
paths.update(code.parent/n for n in('revision','source-sha256'))
official=root/'vendor/v2d_submission_kit/v2dlb/mesh_budget.py';rt.require(raw(official)==dict(bytes=2031,sha256='42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0'),'Actual unmodified official helper required');leaves.append(official)
init=official.with_name('__init__.py')
if init.exists():leaves.append(init)
for p in leaves:inputs[str(p)]=raw(p,empty=p==init)
mounts=sorted({str(p)for p in paths}|{str(p)for p in directories}|{str(p)for p in expected if not p.is_relative_to(proposal)and not p.is_relative_to(qualification)}|{str(p)for p in leaves}|{str(meta),str(video)})
proof=dict(source=source,inputs=inputs,selected_source={str(p):rt.identity(p,empty=True)for p in paths},solid_pins_identity=pin,total_frames=total,video_sha256=pins['input_sha256'],mounts=mounts)
if mode=='before':
 rt.require(not out.exists()and not out.is_symlink()and not control.exists()and not control.is_symlink(),'Fresh output/control only');control.mkdir(mode=0o755);(control/'proof.json').write_text(json.dumps(proof,sort_keys=True));(control/'proof.json').chmod(0o444)
elif mode in('after','complete'):
 rt.require(proof==read(control/'proof.json'),'Post-run source/inputs changed')
 if mode=='complete':
  rt.require({p.name for p in out.iterdir()}=={'report.json','geometry_and_poses.npz','object_fixed_canonical.glb'},'Exactly three fullT native outputs required');r=read(out/'report.json')
  fields=dict(stage='fixed_scale_full_object_pose_initializer',status='pass',episode_index=episode,input_sha256=pins['input_sha256'],mesh_source='solid',execution_verified=True,original_frame_coverage_verified=True,fixed_shape=True,ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],challenge_performance_verified=False)
  rt.require(all(type(r.get(k))is type(v)and r[k]==v for k,v in fields.items())and[r0['frame_index']for r0 in r['frames']]==list(range(total))and len(r['temporal_selection']['candidate_indices'])==total,'Complete original native pose proof required')
  rt.require(raw(out/'geometry_and_poses.npz',1<<30)['sha256']==r['geometry_and_poses_sha256']and raw(out/'object_fixed_canonical.glb')['sha256']==r['fixed_canonical_mesh_sha256']==inputs[str(proposal/'object_fixed_canonical.glb')]['sha256'],'Frozen original geometry/pose artifacts differ')
  rt.require(r['topology_budget']['committed_pins_sha256']==pin['sha256']and r['script_sha256']==proof['selected_source'][str(code/'infra/object_pose_smoke.py')]['sha256'],'Native pinned source differs')
  for p in out.iterdir():p.chmod(0o444)
  out.chmod(0o555)
print(hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest())if mode in('before','after','complete')else None
PYSOLID
 }
 finish() {
  STATUS=$?;trap - EXIT INT TERM;set +e
  if (( OWNED ));then
   LIVE="$(timeout 5s docker ps -aq --filter "name=^/$NAME$")";CHECK=$?
   if [[ "$CHECK" != 0 ]];then STATUS=1
   elif [[ -n "$LIVE" ]];then
    META="$(timeout 5s docker inspect "$LIVE" --format '{{.Id}} {{.Image}} {{index .Config.Labels "world_reward.object_pose_solid.owner"}}')";CHECK=$?
    CID="$(cat "$CONTROL/.container.cid" 2>/dev/null)"
    if [[ "$CHECK" == 0 && "$META" == "$CID $IMAGE $REV" ]];then timeout 15s docker rm -f "$CID" >/dev/null 2>&1 || STATUS=1;else STATUS=1;fi
   fi
  fi
  if [[ -n "$BEFORE" && "$POST_VERIFIED" == 0 ]];then AFTER="$(host after)";[[ $? == 0 && "$AFTER" == "$BEFORE" ]] || STATUS=1;fi
  if (( OPEN ));then [[ "$(host lock)" == "$LOCK_BEFORE" ]] || STATUS=1;exec 9<&-;fi
  exit "$STATUS"
 }
 trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
 BEFORE="$(host before)";PROOF_SHA="$(sha256sum "$CONTROL/proof.json")";export WR_POSE_PROOF_SHA256="${PROOF_SHA%% *}";exec 9<"$LOCK";OPEN=1;LOCK_BEFORE="$(host lock)"
 flock --timeout 43200 9
 [[ "$(host lock)" == "$LOCK_BEFORE" && "$(host after)" == "$BEFORE" ]] || exit 1
 APPS="$(timeout 5s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)";[[ -z "${APPS//[[:space:]]/}" ]] || exit 1
 [[ "$(timeout 5s docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]] || exit 1
 EXISTING="$(timeout 5s docker ps -aq --filter "name=^/$NAME$")";[[ -z "$EXISTING" ]] || exit 1
 SOURCES="$(host mounts)";MOUNTS=();while IFS= read -r path;do MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly");done <<< "$SOURCES"
 [[ ! -e "$OUT" && ! -L "$OUT" ]] || exit 1;mkdir "$OUT";chmod 755 "$OUT";chown 1000:1000 "$OUT";OWNED=1
 timeout --signal=TERM --kill-after=10s 7200s docker run --rm --name "$NAME" --cidfile "$CONTROL/.container.cid" --label "world_reward.object_pose_solid.owner=$REV" \
  --gpus all --read-only --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --cpus 4 --memory 16g \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=128m "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" \
  --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" --env WR_POSE_OUTPUT_RESERVED=1 \
  --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp --env TMPDIR=/tmp \
  --entrypoint python "$IMAGE" "$CODE/infra/object_pose_smoke.py" "$@"
 [[ "$(host complete)" == "$BEFORE" ]] || exit 1;POST_VERIFIED=1;exit 0
fi
docker run --rm --gpus all --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
  --mount "type=bind,src=$ROOT/data,dst=$ROOT/data,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
  --mount "type=bind,src=$ROOT/validation,dst=$ROOT/validation,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" \
  world-reward/cari4d-source:0.1 python "$CODE/infra/object_pose_smoke.py" "$@"
