"""Offline saved-only Azure CPU display; no Vertex request or secret access."""
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys

from mediapipe_cpu_runtime_verify import canonical,identity,require,source,strict,write
from task_grounding_pilot import ROOT,BASE,inputs

ENTRY='run_gemini_initial_zoom'
CONFIG='configs/gemini_initial_v1.json'
HELPERS=('infra/gemini_initial_zoom.py','infra/run_gemini_initial_zoom.sh',CONFIG)

def native(code,out):
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Offline CPU saved QA only')
    c=strict((code/CONFIG).read_bytes());old=ROOT/'results'/('gemini-initial-'+c['saved_producer']);p=old/'native-report.json';pin=identity(p,200000)
    report=strict(p.read_bytes());require(report['status']=='complete_diagnostic_not_quality_pass','Complete saved requests required')
    rows={(r['ep'],r['index']):r for r in report['rows']};require(len(rows)==24,'All saved failures required')
    selected=inputs(c['episodes']);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    canvas=Image.new('RGB',(1440,48+218*8),(24,27,33));d=ImageDraw.Draw(canvas)
    d.text((8,5),'World Reward | Gemini3.5Flash | original frames0,14,29 + automatic object zoom | SAVED OUTPUTS ONLY',font=font,fill='white')
    d.text((8,25),'Cyan person / orange object | NONE = abstention or timeout | no new model calls or manual correction',font=font,fill='white')
    for pos,item in enumerate(selected):
        cap=cv2.VideoCapture(item['video']);require(cap.isOpened(),'Original prefix required')
        try:
            for index in range(30):
                ok,a=cap.read();require(ok,'Original prefix decode failed')
                if index not in c['frames']:continue
                r=rows[(item['episode'],index)];rgb=cv2.cvtColor(a,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
                require(hashlib.sha256(rgb.tobytes()).hexdigest()==r['rgb_sha256'],'Exact original call image required')
                x,y=c['frames'].index(index)*480,48+pos*218;small=Image.fromarray(cv2.resize(rgb,(288,180),interpolation=cv2.INTER_AREA));sd=ImageDraw.Draw(small)
                for key,color in (('person_bbox',(40,220,240)),('object_bbox',(255,155,45))):
                    b=r.get(key)
                    if b:sd.rectangle((b[0]*288/w,b[1]*180/h,b[2]*288/w,b[3]*180/h),outline=color,width=2)
                canvas.paste(small,(x,y));box=r.get('object_bbox')
                if box:
                    x0,y0,x1,y1=box;unit=max(12,int(np.ceil(max((x1-x0)*1.5/4,(y1-y0)*1.5/3))));unit=min(unit,w//4,h//3);ww,hh=4*unit,3*unit
                    left=max(0,min(w-ww,int((x0+x1)/2-ww/2)));top=max(0,min(h-hh,int((y0+y1)/2-hh/2)))
                    im=Image.fromarray(rgb[top:top+hh,left:left+ww]);draw=ImageDraw.Draw(im);draw.rectangle((x0-left,y0-top,x1-left,y1-top),outline=(255,155,45),width=2)
                    canvas.paste(im.resize((192,144)),(x+288,y))
                d.text((x+3,y+184),f"ep{item['episode']:02d} frame{index} | "+r['status'],font=font,fill='white')
                label=r.get('object',{}).get('label','')
                d.text((x+290,y+147),label[:22] if box else 'NO OBJECT OUTPUT',font=font,fill='white')
        finally:cap.release()
    for q in (72,60,48,36,24,16,12):
        b=BytesIO();canvas.save(b,format='JPEG',quality=q,optimize=True);raw=b.getvalue()
        if len(raw)<=180000:break
    require(len(raw)<=180000,'Tiny zoom JPEG gate failed');write(out/'zoom.jpg',raw,mode=0o444)
    require(identity(p,200000)==pin and inputs(c['episodes'])==selected,'Saved source changed')
    write(out/'zoom-report.json',(json.dumps(dict(status='complete_saved_only_qa',source_report=pin,quality_verified=False,new_model_calls=0,gpu_used=False,bytes=len(raw)))+'\n').encode(),mode=0o444)

def run():
    require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01','Azure only')
    code=canonical(Path(os.environ['WR_CODE']));rev=os.environ['WR_CODE_REVISION'];c=strict((code/CONFIG).read_bytes());binding=source(ROOT,code,rev,ENTRY,HELPERS)
    out=ROOT/'results'/('gemini-initial-zoom-'+rev);out.mkdir(mode=0o755)
    cmd=['docker','run','--rm','--network','none','--read-only','--cap-drop','ALL','--memory','8g','--cpus','8','--tmpfs','/tmp:rw,nosuid,size=1g']
    mounts=[(code.parent,True),(out,False),(ROOT/'results'/('gemini-initial-'+c['saved_producer']),True),(ROOT/'results/input-manifest.json',True),(ROOT/'data/track_1/meta',True)]
    mounts += [(ROOT/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4',True) for ep in c['episodes']]
    for p,ro in mounts:cmd+=['--mount',f'type=bind,src={canonical(p)},dst={p}'+(',readonly' if ro else '')]
    cmd+=['--entrypoint','/usr/bin/env',BASE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','CUDA_VISIBLE_DEVICES=','PYTHONPATH='+str(code/'src')+':'+str(code/'infra'),'PYTHONDONTWRITEBYTECODE=1','/opt/conda/bin/python','-B',str(code/'infra/gemini_initial_zoom.py'),'--native',str(code),str(out)]
    with (out/'zoom.log').open('xb') as f:require(subprocess.run(cmd,stdout=f,stderr=f,timeout=90).returncode==0,'Saved QA only failed')
    from full4d_publish import PrivatePreviews
    client=PrivatePreviews();client.require_private();row=client.upload(f'full4d-{rev}/episode_002000.jpg',(out/'zoom.jpg').read_bytes(),'image/jpeg',rev);client.head(row['name'],row,row['etag'])
    require(source(ROOT,code,rev,ENTRY,HELPERS)==binding,'Immutable QA code changed')
    write(out/'published.json',(json.dumps(dict(file=row,new_model_calls=0,source_binding=binding))+'\n').encode(),mode=0o444)

if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--native':native(Path(sys.argv[2]),Path(sys.argv[3]))
    else:run()
