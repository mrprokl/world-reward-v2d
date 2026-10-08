"""CPU-only display recovery of immutable VL results; never rerun inference."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict
from vl_localization import ROOT, BASE, settings, preview, save

PRODUCER = '4aa5b0a24cab871ac66529605dd0fb12751f959b'
INFERENCE = ROOT/'results'/('vl-localization-'+PRODUCER)
FAILED_REPORT_SHA = 'b2612ea9bbebf32e197abbefcf4d501ccdfb17cff670c69cce0cfd719c78ddc1'
ENTRY = 'run_vl_localization_preview'
HELPERS = ('infra/vl_localization_preview.py','infra/run_vl_localization_preview.sh',
           'infra/vl_localization.py','src/world_reward/vl_localization.py','configs/vl_localization_v1.json')


def pins():
    r = strict((INFERENCE/'report.json').read_bytes())
    require(identity(INFERENCE/'report.json')['sha256']==FAILED_REPORT_SHA
            and r['status']=='fail' and r['phase']=='preview'
            and r['producer_revision']==PRODUCER and r['source_rehashed_after'] is True,
            'Exact original failed display report required')
    i = strict((INFERENCE/'inference.json').read_bytes())
    require(i['status']=='complete_diagnostic_not_quality_pass' and i['producer_revision']==PRODUCER
            and i['requested_images']==152 and i['sam_executed'] is False
            and i['baseline_modified'] is False and i['quality_verified'] is False
            and i['inputs_rehashed_after'] is True and i['model_rehashed_after'] is True,
            'Complete original native inference required')
    paths = [INFERENCE/name for name in ('report.json','inference.json','batch-control.json','observations-inputs.json')]
    for ep, names in i['episodes'].items():
        for name, pin in names.items():
            p = INFERENCE/f'episode_{int(ep):06d}'/name
            require(identity(p)==pin,'Original observations changed')
            paths.append(p)
    return {str(p):identity(p) for p in paths}


def native(code, out):
    require(sys.platform=='linux' and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
            and os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Offline CPU-only preview required')
    before = pins()
    # Only display RGB width/compression can adapt to the hard180kB sheet cap.
    # The128x96 object inset, all30 original frames and literal boxes are kept.
    preview(code,INFERENCE,destination=out,display_widths=(256,224,192))
    require(pins()==before,'Original inference changed during saved display recovery')
    save(out/'input-pins.json',dict(inference_producer=PRODUCER,files=before,
         predictions_modified=False,model_calls=0,ground_truth_used=False,gpu_used=False))


def run():
    require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01',
            'Azure-only preview host required')
    code = canonical(Path(os.environ['WR_CODE']));revision=os.environ['WR_CODE_REVISION']
    binding=source(ROOT,code,revision,ENTRY,HELPERS);c=settings(code);before=pins()
    out=ROOT/'results'/('vl-localization-preview-'+revision);out.mkdir(mode=0o755)
    start=time.monotonic();report=dict(status='fail',phase='saved_display',producer_revision=revision,
        inference_producer=PRODUCER,source_binding=binding,original_inference_files=before,
        model_calls=0,gpu_used=False,baseline_modified=False,predictions_modified=False,
        sam_executed=False,quality_verified=False,ground_truth_used=False)
    name='wr-vl-preview-'+revision[:12]
    require(subprocess.run(['docker','inspect',name],capture_output=True,timeout=15).returncode!=0,
            'Fresh CPU preview container name required')
    try:
        cmd=['docker','run','--rm','--name',name,'--label','world_reward.vl_preview.owner='+revision,
            '--network','none','--read-only','--user','0:0',
            '--cap-drop','ALL','--security-opt','no-new-privileges','--memory','16g','--cpus','16',
            '--shm-size','2g','--tmpfs','/tmp:rw,nosuid,size=1g']
        mounts=[(code.parent,True),(INFERENCE,True),(out,False),
                (ROOT/'results/input-manifest.json',True),(ROOT/'data/track_1/meta',True)]
        mounts += [(ROOT/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4',True)
                   for ep in c['episodes']]
        for p,ro in mounts:
            canonical(p);cmd += ['--mount',f'type=bind,src={p},dst={p}'+(',readonly' if ro else '')]
        cmd += ['--entrypoint','/usr/bin/env',BASE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp',
                'PYTHONPATH='+str(code/'src')+':'+str(code/'infra'),'PYTHONDONTWRITEBYTECODE=1',
                'CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',
                '/opt/conda/bin/python','-B',str(code/'infra/vl_localization_preview.py'),'--native',str(code),str(out)]
        with (out/'preview.log').open('xb') as stream:
            completed=subprocess.run(cmd,stdout=stream,stderr=stream,timeout=240)
        require(completed.returncode==0,'Saved CPU display failed; no inference retry')
        require(pins()==before,'Original predictions changed')
        from full4d_publish import PrivatePreviews
        client=PrivatePreviews();client.require_private();files=[]
        for view,filename,offset in (('first30','first30.jpg',0),('fullclip9','fullclip9.jpg',1000)):
            for ep in c['episodes']:
                p=out/f'episode_{ep:06d}'/filename;identity(p,c['max_jpeg_bytes'])
                row=client.upload(f'full4d-{revision}/episode_{ep+offset:06d}.jpg',p.read_bytes(),'image/jpeg',revision)
                row.update(episode_index=ep,view=view);files.append(row)
                save(out/f'published_{view}_{ep:06d}.json',row);client.head(row['name'],row,row['etag'])
        report.update(status='complete_saved_only_qa',phase='complete',files=files,
            previews=identity(out/'previews.json'),input_pins=identity(out/'input-pins.json'),
            original_inference_rehashed_after=True)
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error)[:300])
    finally:
        inspected=subprocess.run(['docker','inspect',name],capture_output=True,timeout=15)
        if inspected.returncode==0:
            actual=strict(inspected.stdout)[0]
            require(actual['Image']==BASE and actual['Name']=='/'+name
                    and actual['Config']['Labels'].get('world_reward.vl_preview.owner')==revision,
                    'Only owned CPU container removal')
            subprocess.run(['docker','rm','-f',actual['Id']],capture_output=True,check=True,timeout=20)
        require(source(ROOT,code,revision,ENTRY,HELPERS)==binding,'Preview source changed')
        report['source_rehashed_after']=True;report['elapsed_seconds']=time.monotonic()-start
        save(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','phase','producer_revision','elapsed_seconds')}),flush=True)
    require(report['status']=='complete_saved_only_qa','Saved display recovery failed closed')


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--native':native(Path(sys.argv[2]),Path(sys.argv[3]))
    else:run()
