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
