"""Saved Azure CPU QA of all frozen hybrid clips, including abstentions."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mediapipe_cpu_runtime_verify import canonical,identity,require,source,strict,write
from hybrid_pair_select import ROOT,BASE,settings,Bank,decode,save
from world_reward.hybrid_pair import indices,mask_box

ENTRY='run_hybrid_pair_preview'
CONFIG='configs/hybrid_pair_v1.json'
HELPERS=('infra/hybrid_pair_preview.py','infra/run_hybrid_pair_preview.sh',CONFIG,
         'infra/hybrid_pair_select.py','src/world_reward/hybrid_pair.py')


def native(code,out,sam_producer,selection_producer):
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from io import BytesIO
    from task_grounding_pilot import inputs
    require(sys.platform=='linux' and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
            and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Offline CPU-only QA required')
    c=settings(code);selected=inputs(c['episodes']);sam=ROOT/'results'/('hybrid-pair-'+sam_producer)
    result=ROOT/'results'/('hybrid-pair-selection-'+selection_producer)
    summary=strict((result/'selection-report.json').read_bytes())
    require(summary['status']=='complete_diagnostic_not_quality_pass' and summary['sam_producer']==sam_producer,
            'Actual saved selection cohort required')
    pins={r['episode_index']:r['selection'] for r in summary['episodes']}
    require(identity(sam/'sam-report.json',20<<20)==summary['sam_report_pin'],
            'QA must use the same SAM report as saved selection')
    sam_report=strict((sam/'sam-report.json').read_bytes())
    sam_pins={r['episode_index']:{b['role']:b['bank'] for b in r['banks']} for r in sam_report['episodes']}
    require(set(pins)==set(c['episodes']),'No success-only visual subset')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    def jpeg(im):
        for q in (72,60,48,36,24,16,12):
            b=BytesIO();im.save(b,format='JPEG',quality=q,optimize=True);raw=b.getvalue()
            if len(raw)<=c['max_jpeg_bytes']:return raw,q
        raise ValueError('Bounded QA display only; never drop frames')
    def one(item):
        ep=item['episode'];directory=out/f'episode_{ep:06d}';directory.mkdir(mode=0o755)
        p=result/f'episode_{ep:06d}'/'selection.json';require(identity(p,20<<20)==pins[ep],'Frozen selected IDs changed')
        r=strict(p.read_bytes());banks={}
        for role in ('person','object'):
            folder=sam/f'episode_{ep:06d}'/'candidates'/role
            if (folder/'bank.json').exists():
                banks[role]=Bank(folder,role,ep,item['total'])
                require(banks[role].pin==sam_pins[ep][role],'Published QA bank differs from actual SAM producer')
        chosen={}
        if r['status']=='selected':
            require(set(banks)=={'person','object'},'Both selected role banks required')
            for role,b in banks.items():
                matches=[t for t in b.tracks if t['id']==r[role+'_id']]
                require(len(matches)==1,'Selected existing native ID required');chosen[role]=matches[0]
        interaction=chosen.get('object',{}).get('best_frame',item['total']//2)
        wanted=set(range(30))|set(indices(item['total']))|{interaction}
        images=decode(item,wanted,list(banks.values()))
        def view(index,width,height):
            raw=np.asarray(images[index]).copy();small=cv2.resize(raw,(width,height),interpolation=cv2.INTER_AREA)
            masks={}
            for role,t in chosen.items():
                m=banks[role].mask(t,index);require(m.shape==raw.shape[:2],'Original selected mask geometry required')
                masks[role]=m;binary=cv2.resize(m.astype('uint8'),(width,height),interpolation=cv2.INTER_NEAREST)
                contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(small,contours,-1,(40,220,240) if role=='person' else (255,155,45),1)
            return Image.fromarray(small),raw,masks
        def sheet(ids,columns,main_w,label):
            main_h=main_w*3//4;tw,th=main_w+128,main_h+28
            canvas=Image.new('RGB',(tw*columns,48+th*((len(ids)+columns-1)//columns)),(24,27,33));d=ImageDraw.Draw(canvas)
            d.text((6,5),f"World Reward ep{ep:02d} | {label} | {r['status']} | cyan person / orange object",font=font,fill='white')
            d.text((6,24),'Actual SAM3.1 masks + Qwen ID choice | no manual labels | diagnostic, not ground truth',font=font,fill='white')
            for pos,index in enumerate(ids):
                x,y=(pos%columns)*tw,48+(pos//columns)*th;im,raw,masks=view(index,main_w,main_h);canvas.paste(im,(x,y))
                if 'object' in masks:
                    box=mask_box(masks['object'])
                    if box:
                        x0,y0,x1,y1=box;h,w=raw.shape[:2]
                        unit=max(16,int(np.ceil(max((x1-x0)*1.5/4,(y1-y0)*1.5/3))))
                        unit=min(unit,w//4,h//3);ww,hh=4*unit,3*unit
                        left=max(0,min(w-ww,int((x0+x1)/2-ww/2)));top=max(0,min(h-hh,int((y0+y1)/2-hh/2)))
                        a=raw[top:top+hh,left:left+ww].copy();m=masks['object'][top:top+hh,left:left+ww]
                        contours,_=cv2.findContours(m.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(a,contours,-1,(255,155,45),2)
                        canvas.paste(Image.fromarray(cv2.resize(a,(128,96),interpolation=cv2.INTER_AREA)),(x+main_w,y))
                d.text((x+main_w+3,y+101),'Native mask zoom',font=font,fill='white')
                d.text((x+main_w+3,y+119),'object: '+('visible' if masks.get('object',np.zeros((1,1),bool)).any() else 'NONE'),font=font,fill='white')
                d.text((x+4,y+main_h+3),f'frame {index:04d} | {index/30:.2f}s',font=font,fill='white')
            return canvas
        first,q=jpeg(sheet(list(range(30)),6,192,'ALL frames0..29'));write(directory/'first30.jpg',first,mode=0o444)
        context,q2=jpeg(sheet(list(indices(item['total'])),4,192,'16 uniform full-clip frames'));write(directory/'fullclip16.jpg',context,mode=0o444)
        panel=Image.new('RGB',(512,220),(24,27,33));d=ImageDraw.Draw(panel)
        d.text((6,4),f"ep{ep:02d} | {r['status']} | {r.get('person_id')} + {r.get('object_id')}",font=font,fill='white')
        for col,index in enumerate((0,interaction)):
            im,_,_=view(index,256,192);panel.paste(im,(col*256,26))
            ImageDraw.Draw(panel).text((col*256+3,27),f'frame{index}',font=font,fill='white',stroke_width=1,stroke_fill='black')
        for b in banks.values():b.verify()
        require(identity(p,20<<20)==pins[ep],'Selection changed during QA')
        return dict(episode_index=ep,status=r['status'],first30=identity(directory/'first30.jpg'),
            fullclip16=identity(directory/'fullclip16.jpg'),first30_frames=list(range(30)),
            fullclip_frames=list(indices(item['total'])),jpeg_quality=q,context_jpeg_quality=q2,
            selected_ids={role:r.get(role+'_id') for role in ('person','object')},overview_frames=[0,interaction],panel=panel)
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,selected))
    canvas=Image.new('RGB',(2048,36+220*3),(24,27,33));d=ImageDraw.Draw(canvas)
    d.text((8,6),'World Reward | ALL12 frozen clips | original frame0 + max-visible selected object frame | actual native masks',font=font,fill='white')
    for i,row in enumerate(rows):canvas.paste(row.pop('panel'),((i%4)*512,36+(i//4)*220))
    raw,q=jpeg(canvas);write(out/'overview.jpg',raw,mode=0o444)
    require(inputs(c['episodes'])==selected,'Original sources changed')
    save(out/'previews.json',dict(status='complete_saved_only_qa',episodes=rows,overview=identity(out/'overview.jpg'),
        overview_jpeg_quality=q,sam_producer=sam_producer,selection_producer=selection_producer,
        model_calls=0,gpu_used=False,ground_truth_used=False,quality_verified=False,predictions_modified=False,
        original_frames_preserved=True,source_rehashed_after=True))


def run():
    require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01','Azure host only')
    code=canonical(Path(os.environ['WR_CODE']));revision=os.environ['WR_CODE_REVISION'];c=settings(code)
    sam_producer=c['sam_producer_revision'];selection_producer=c['selection_producer_revision']
    binding=source(ROOT,code,revision,ENTRY,HELPERS);out=ROOT/'results'/('hybrid-pair-preview-'+revision);out.mkdir(mode=0o755)
    name='wr-hybrid-preview-'+revision[:12];start=time.monotonic();report=dict(status='fail',phase='preview',
        producer_revision=revision,sam_producer=sam_producer,selection_producer=selection_producer,
        source_binding=binding,model_calls=0,gpu_used=False,quality_verified=False,predictions_modified=False)
    require(subprocess.run(['docker','inspect',name],capture_output=True,timeout=15).returncode!=0,'Fresh CPU name required')
    try:
        cmd=['docker','run','--rm','--name',name,'--label','world_reward.hybrid_preview.owner='+revision,
            '--network','none','--read-only','--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--memory','24g','--cpus','16','--shm-size','2g','--tmpfs','/tmp:rw,nosuid,size=1g']
        mounts=[(code.parent,True),(out,False),(ROOT/'results'/('hybrid-pair-'+sam_producer),True),
            (ROOT/'results'/('hybrid-pair-selection-'+selection_producer),True),
            (ROOT/'results/input-manifest.json',True),(ROOT/'data/track_1/meta',True)]
        mounts.extend((ROOT/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4',True) for ep in c['episodes'])
        for p,ro in mounts:
            canonical(p);cmd+=['--mount',f'type=bind,src={p},dst={p}'+(',readonly' if ro else '')]
        cmd+=['--entrypoint','/usr/bin/env',BASE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp',
            'PYTHONPATH='+str(code/'src')+':'+str(code/'infra'),'PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=',
            'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','/opt/conda/bin/python','-B',
            str(code/'infra/hybrid_pair_preview.py'),'--native',str(code),str(out),sam_producer,selection_producer]
        with (out/'preview.log').open('xb') as stream:r=subprocess.run(cmd,stdout=stream,stderr=stream,timeout=c['preview_budget_seconds'])
        require(r.returncode==0,'Saved QA failed; never rerun inference for display')
        from full4d_publish import PrivatePreviews
        client=PrivatePreviews();client.require_private();files=[]
        for ep in c['episodes']:
            for view,name,offset in (('first30','first30.jpg',0),('fullclip16','fullclip16.jpg',1000)):
                p=out/f'episode_{ep:06d}'/name;identity(p,c['max_jpeg_bytes'])
                row=client.upload(f'full4d-{revision}/episode_{ep+offset:06d}.jpg',p.read_bytes(),'image/jpeg',revision)
                row.update(episode_index=ep,view=view);save(out/f'published_{view}_{ep:06d}.json',row)
                client.head(row['name'],row,row['etag']);files.append(row)
        p=out/'overview.jpg';identity(p,c['max_jpeg_bytes'])
        row=client.upload(f'full4d-{revision}/episode_002000.jpg',p.read_bytes(),'image/jpeg',revision)
        row['view']='all12_overview';save(out/'published_overview.json',row);client.head(row['name'],row,row['etag']);files.append(row)
        report.update(status='complete_saved_only_qa',phase='complete',files=files,previews=identity(out/'previews.json',20<<20))
    except Exception as error:report.update(error_type=type(error).__name__,error=str(error)[:250])
    finally:
        inspected=subprocess.run(['docker','inspect',name],capture_output=True,timeout=15)
        if inspected.returncode==0:
            a=strict(inspected.stdout)[0];require(a['Image']==BASE and a['Config']['Labels'].get('world_reward.hybrid_preview.owner')==revision,'Owned CPU cleanup only')
            subprocess.run(['docker','rm','-f',a['Id']],capture_output=True,check=True,timeout=20)
        require(source(ROOT,code,revision,ENTRY,HELPERS)==binding,'Source changed')
        report.update(source_rehashed_after=True,elapsed_seconds=time.monotonic()-start);save(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','phase','elapsed_seconds')}),flush=True)
    require(report['status']=='complete_saved_only_qa','Hybrid QA failed closed')


if __name__=='__main__':
    if len(sys.argv)==6 and sys.argv[1]=='--native':native(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],sys.argv[5])
    else:run()
