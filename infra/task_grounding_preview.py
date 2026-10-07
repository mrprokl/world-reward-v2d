"""CPU-only preview of already frozen masks; never reruns models or corrects IDs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mediapipe_cpu_runtime_verify import identity, require, source, strict
from task_grounding_pilot import BASE, ROOT, inputs, preview, record

PRODUCER = 'a9419e0332e3c6258a8022e9286eaafb04dce223'
PIN_CONFIG = 'configs/task_grounding_saved_pins.json'
ENTRY = 'run_task_grounding_preview'
HELPERS = ('infra/task_grounding_preview.py','infra/run_task_grounding_preview.sh',PIN_CONFIG,
           'infra/task_grounding_pilot.py','src/world_reward/task_grounding.py')


def verify(code, old):
    pins = strict((code/PIN_CONFIG).read_bytes())
    require(pins['producer_revision']==PRODUCER,'Original frozen producer required')
    for name,pin in pins['files'].items():
        require(identity(old/name,200000)==pin,'Original saved run differs')
    initial = strict((old/'report.json').read_bytes())
    require(initial['status']=='fail' and initial['phase']=='preview'
            and initial['source_rehashed_after'] is True,'Preserve original preview failure')
    require(inputs()==initial['inputs'],'Original RGB/metadata conditioning changed')
    for row in pins['episodes']:
        dest=old/f"episode_{row['episode']:06d}"
        for key in ('grounding','tracking'):
            require(identity(dest/(key+'.json'),200000)==row[key],'Original per-clip predictions changed')
        ground = strict((dest/'grounding.json').read_bytes()); tracked=strict((dest/'tracking.json').read_bytes())
        require(ground['status']=='seed_available' and ground['calls']==1 and tracked['frames']==row['frames']
                and tracked['status']=='full_T_complete','No inference/track rescue permitted')
        baseline=ROOT/f"outputs/episode_{row['episode']:06d}/automatic_masks"
        require(identity(baseline/'report.json',200000,readonly=False)==row['baseline_report'],'Frozen baseline differs')
        base = strict((baseline/'report.json').read_bytes())
        require(base['ground_truth_used'] is False and base['hand_labeled_test'] is False and base['oracle_modes']==[]
                and base['input_sha256']==ground['video_pin']['sha256'],'Automatic baseline provenance required')
        for folder in (dest,baseline):
            for obj in (0,1):
                paths=sorted((folder/'masks'/str(obj)).glob('*.png'))
                require([p.stem for p in paths]==[f'{f:06d}'for f in range(row['frames'])],
                        'All original mask indices required; no missing-baseline fallback')
    return pins


def native(code, old, output):
    require({p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Offline CPU-only preview')
    before=verify(code,old)
    preview(code,old,destination=output,width=320)
    require(verify(code,old)==before and 'torch' not in sys.modules,'No model import/prediction mutation')


def run():
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01','Azure CPU observer required')
    binding=source(ROOT,code,revision,ENTRY,HELPERS)
    old=ROOT/('results/task-grounding-pilot-'+PRODUCER)
    output=ROOT/('results/task-grounding-preview-'+revision)
    require(not output.exists(),'Fresh CPU preview namespace required');output.mkdir(mode=0o700)
    pins=verify(code,old);started=time.monotonic()
    report=dict(status='fail',stage='task_grounding_saved_preview',producer_revision=PRODUCER,
                preview_revision=revision,source_binding=binding,gpu_used=False,model_calls=0,quality_verified=False)
    name='world-reward-task-preview-'+revision[:12]
    require(not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$'],text=True).strip(),'Fresh owned container')
    mounts=[(code,True),(old,True),(output,False),(ROOT/'results/input-manifest.json',True),(ROOT/'data/track_1/meta',True)]
    for ep in (8,9,26):
        mounts.extend([(ROOT/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4',True),
                       (ROOT/f'outputs/episode_{ep:06d}/automatic_masks',True)])
    command=['docker','run','--rm','--name',name,'--label','world_reward.task_preview.owner='+revision,
             '--network','none','--read-only','--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges',
             '--memory','4g','--cpus','4','--tmpfs','/tmp:rw,nosuid,size=128m']
    for path,readonly in mounts:command+=['--mount',f'type=bind,src={path},dst={path}'+(',readonly'if readonly else'')]
    command+=['--entrypoint','/usr/bin/env',BASE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp',
              'PYTHONPATH='+str(code/'src')+':'+str(code/'infra'),'PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=-1',
              '/opt/conda/bin/python','-B',str(code/'infra/task_grounding_preview.py'),'--native',str(code),str(old),str(output)]
    try:
        with (output/'preview.log').open('xb')as log:
            result=subprocess.run(command,stdout=log,stderr=log,timeout=180,check=False)
        require(result.returncode==0,'Saved-only native preview failed')
        require(verify(code,old)==pins and source(ROOT,code,revision,ENTRY,HELPERS)==binding,'Original receipts/source changed')
        report.update(status='complete_visual_qa_not_quality_pass',saved_inputs_reverified=True,
                      previews=identity(output/'previews.json'))
    except Exception as exc:report['error_type']=type(exc).__name__
    finally:
        found=subprocess.run(['docker','inspect',name],capture_output=True,timeout=15,check=False)
        if found.returncode==0:
            actual=json.loads(found.stdout)[0]
            require(actual['Config']['Labels'].get('world_reward.task_preview.owner')==revision
                    and actual['Image']==BASE,'Foreign container must not be removed')
            subprocess.run(['docker','rm','-f',actual['Id']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20,check=True)
        report['elapsed_seconds']=time.monotonic()-started;record(output/'report.json',report)
    print(json.dumps({k:report[k]for k in ('status','elapsed_seconds','gpu_used','model_calls')}))
    require(report['status']=='complete_visual_qa_not_quality_pass','CPU preview failed closed')


if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--native':native(*map(Path,sys.argv[2:]))
    else:
        require(len(sys.argv)==1,'No arbitrary preview arguments');run()
