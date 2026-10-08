"""Azure CPU → Vertex single-image initial grounding; saved comparisons only.

No GPU, full video decoding, tracking, learned adaptation or manual labels.
An encrypted one-shot OAuth envelope is delivered separately, never Git/logs.
"""
from __future__ import annotations
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict, write
from task_grounding_pilot import ROOT, BASE, inputs
from hybrid_pair_select import Bank
from world_reward.gemini_localization import build_request, parse_response
from world_reward.vertex_retry import call_with_retry, RetryExhausted, PermanentFailure

ENTRY='run_gemini_initial'
CONFIG='configs/gemini_initial_v1.json'
HELPERS=('infra/gemini_initial.py','infra/run_gemini_initial.sh',CONFIG,'src/world_reward/gemini_localization.py','src/world_reward/vertex_retry.py')

def save(p,d):
    write(p,(json.dumps(d,sort_keys=True,allow_nan=False)+'\n').encode(),mode=0o444)

def settings(code):
    c=strict((code/CONFIG).read_bytes())
    require(c['schema']=='world_reward.gemini_initial.v1' and c['frames']==[0,14,29]
        and c['episodes']==[9,1,14,7,28,13,18,6] and c['model']=='gemini-3.5-flash'
        and c['location']=='global' and c['ground_truth_used'] is False
        and c['manual_labels'] is False and c['quality_verified'] is False,'Frozen initial-image diagnostic required')
    return c

def native(code,out,recovery=None):
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    c=settings(code);start=time.monotonic();token=sys.stdin.buffer.read(16385).decode().strip()
    saved=None;saved_pin=None;preserved={};original_config=dict(c)
    if recovery:
        old=ROOT/'results'/('gemini-initial-'+recovery['saved_producer'])
        saved_pin=identity(old/'native-report.json',200000)
        header=strict((old/'report.json').read_bytes())
        require(header['producer_revision']==recovery['saved_producer'] and header['native_report']==saved_pin,'Original report pin required')
        saved=strict((old/'native-report.json').read_bytes())
        require(saved['config']=={k:v for k,v in c.items() if k!='saved_producer'}
            and saved['status']=='complete_diagnostic_not_quality_pass','Original frozen request configuration required')
        preserved={(r['ep'],r['index']):r for r in saved['rows']}
        failed=[key for key,r in preserved.items() if r['status']=='invalid_response' and r.get('failure_type')=='TimeoutError']
        require(len(preserved)==24 and sorted(failed)==sorted(map(tuple,recovery['expected_failed_calls']))
            and all(r['status']=='pair_returned' or key in failed for key,r in preserved.items()),'Only original four technical failures may be recovered')
        c.update(concurrency=recovery['concurrency'],budget_seconds=recovery['budget_seconds'],call_timeout_seconds=recovery['attempt_timeout_seconds'])
    require(0<len(token)<16384 and not any(x.isspace() for x in token),'RAM-only existing OAuth required')
    require(sys.platform=='linux' and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Azure CPU only')
    selected=inputs(c['episodes']);sam=ROOT/'results'/('hybrid-pair-'+c['sam_producer'])
    selection=ROOT/'results'/('hybrid-pair-selection-'+c['selection_producer'])
    qwen=ROOT/'results'/('vl-localization-'+c['qwen_producer'])
    sp=identity(sam/'sam-report.json',200000);sr=strict((sam/'sam-report.json').read_bytes())
    qp=identity(selection/'selection-report.json',200000);qr=strict((selection/'selection-report.json').read_bytes())
    require(sr['producer_revision']==c['sam_producer'] and qr['sam_report_pin']==sp
        and qr['producer_revision']==c['selection_producer'],'Actual immutable references required')
    qip=identity(qwen/'inference.json',200000);qi=strict((qwen/'inference.json').read_bytes())
    tracks={r['episode_index']:r for r in sr['episodes']}
    choices={r['episode_index']:r for r in qr['episodes']};tasks=[];banks=[];pins={};orig={}
    for item in selected:
        ep=item['episode'];rolebanks={}
        for role in ('person','object'):
            b=Bank(sam/f'episode_{ep:06d}'/'candidates'/role,role,ep,item['total'])
            require(b.pin==next(r['bank'] for r in tracks[ep]['banks'] if r['role']==role),'Native reference bank pin differs')
            rolebanks[role]=b;banks.append(b)
        p=selection/f'episode_{ep:06d}'/'selection.json';pins[p]=identity(p,200000)
        require(pins[p]==choices[ep]['selection'],'Saved selected ID pin differs');choice=strict(p.read_bytes())
        observations={}
        if ep in c['qwen_episodes']:
            p=qwen/f'episode_{ep:06d}'/'observations.json';pins[p]=identity(p,2000000)
            require(pins[p]==qi['episodes'][str(ep)]['observations.json'],'Saved independent Qwen pin differs')
            observations={r['frame_index']:r for r in strict(p.read_bytes())['records']}
        cap=cv2.VideoCapture(item['video']);require(cap.isOpened(),'Original video required')
        try:
            for i in range(max(c['frames'])+1):
                ok,a=cap.read();require(ok,'Original prefix decode failed')
                if i not in c['frames']:continue
                rgb=cv2.cvtColor(a,cv2.COLOR_BGR2RGB);sha=hashlib.sha256(rgb.tobytes()).hexdigest()
                require(all(b.frames[i]['decoded_rgb_sha256']==sha for b in rolebanks.values()),'Exact same original SAM evidence required')
                im=Image.fromarray(rgb);buf=BytesIO();im.save(buf,format='PNG');raw=buf.getvalue()
                previous=observations.get(i)
                if previous:require(previous['decoded_rgb_sha256']==sha,'Exact same original Qwen evidence required')
                originals={}
                for role,b in rolebanks.items():
                    id=choice.get(role+'_id')
                    matches=[t for t in b.tracks if t['id']==id]
                    originals[role]=b.mask(matches[0],i) if choice['status']=='selected' and len(matches)==1 else None
                key=(ep,i);orig[key]=dict(image=im,masks=originals,qwen=previous,choice=choice)
                request=build_request(raw,i,item['object_prompt'],item['action'],thinking_level=c['thinking_level'],media_resolution=c['media_resolution'],max_output_tokens=c['max_output_tokens'])
                tasks.append(dict(ep=ep,index=i,width=im.width,height=im.height,rgb_sha256=sha,
                    png_sha256=hashlib.sha256(raw).hexdigest(),request=request))
        finally:cap.release()
    if recovery:
        # Audit the entire cohort BEFORE making any billable request.
        for t in tasks:
            previous=preserved[(t['ep'],t['index'])]
            body=json.dumps(t['request'],separators=(',',':')).encode()
            require(previous['request_sha256']==hashlib.sha256(body).hexdigest()
                and all(previous[k]==t[k] for k in ('ep','index','width','height','rgb_sha256','png_sha256')),
                'All original requests must pass the cohort preflight')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*a,**k):raise RuntimeError('Vertex redirects prohibited')
    url=f"https://aiplatform.googleapis.com/v1/projects/{c['project']}/locations/{c['location']}/publishers/google/models/{c['model']}:generateContent"
    def ask(t):
        begin=time.monotonic();body=json.dumps(t['request'],separators=(',',':')).encode()
        row={k:v for k,v in t.items() if k!='request'};row.update(request_sha256=hashlib.sha256(body).hexdigest(),status='invalid_schema',person_bbox=None,object_bbox=None)
        if recovery:
            previous=preserved[(t['ep'],t['index'])]
            require(all(previous[k]==row[k] for k in ('ep','index','width','height','rgb_sha256','png_sha256','request_sha256')),'Retries must use the identical image and request bytes')
            if previous['status']=='pair_returned':return previous
            row['original_attempt']=dict(status='transport_timeout',seconds=previous['seconds'],failure_type=previous['failure_type'])
        def transport(payload,timeout):
            op=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
            req=urllib.request.Request(url,data=payload,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            with op.open(req,timeout=timeout) as response:
                data=response.read(200001);require(len(data)<=200000,'Bounded Vertex response required')
            return data
        try:
            result=call_with_retry(body,transport,deadline=start+c['budget_seconds']-5,attempt_timeout=c['call_timeout_seconds'],
                max_attempts=recovery['max_total_attempts']-1 if recovery else 3,
                initial_delay=recovery['initial_delay_seconds'] if recovery else 1,
                max_delay=recovery['max_delay_seconds'] if recovery else 30)
            row['attempts']=list(result.attempts);d=strict(result.response)
            candidates=d.get('candidates',[])
            if not candidates and d.get('promptFeedback',{}).get('blockReason'):
                row['status']='blocked_or_incomplete'
            require(len(candidates)==1,'One Vertex candidate required')
            candidate=candidates[0];row.update(model_version=d.get('modelVersion'),response_id=d.get('responseId'),usage=d.get('usageMetadata',{}),finish_reason=candidate.get('finishReason'))
            if candidate.get('finishReason')!='STOP':
                row['status']='blocked_or_incomplete';raise ValueError('Vertex output incomplete/blocked')
            text=''.join(p.get('text','') for p in candidate.get('content',{}).get('parts',[]) if not p.get('thought',False))
            row['response_text']=text
            row.update(parse_response(text,target_frame=t['index'],height=t['height'],width=t['width']))
            row['status']='pair_returned' if row['person_bbox'] and row['object_bbox'] else 'abstained'
        except RetryExhausted as e:row.update(status='transport_retry_exhausted',retry_reason=e.reason,attempts=list(e.attempts))
        except PermanentFailure as e:row.update(status='transport_permanent_error',retry_reason=e.reason,attempts=list(e.attempts))
        except Exception as e:row.update(failure_type=type(e).__name__)
        row['seconds']=time.monotonic()-begin;save(out/f"call_{t['ep']:06d}_{t['index']:06d}.json",row);return row
    with ThreadPoolExecutor(max_workers=c['concurrency']) as pool:rows=list(pool.map(ask,tasks))
    token=None;require(len(rows)==24,'All frozen images, including failures, required')
    require(time.monotonic()-start<c['budget_seconds'],'Inclusive initial-image stage budget exceeded')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    overview=Image.new('RGB',(1280,48+228*8),(24,27,33));draw=ImageDraw.Draw(overview)
    draw.text((8,6),'World Reward | INITIAL frames0,14,29 | cyan person / orange object | NO NEW TRACKING',font=font,fill='white')
    draw.text((8,26),'Gemini sees ONE original image | SAM/Qwen-ID reference is retrospective, not a fair causal baseline',font=font,fill='white')
    def panel(r,kind,w=320,h=240):
        o=orig[(r['ep'],r['index'])];image=o['image'];small=cv2.resize(np.asarray(image),(w,h),interpolation=cv2.INTER_AREA)
        if kind=='SAM':
            for role,m in o['masks'].items():
                if m is None:continue
                contours,_=cv2.findContours(cv2.resize(m.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(small,contours,-1,(40,220,240) if role=='person' else (255,155,45),2)
            im=Image.fromarray(small);label='SAM saved | '+o['choice']['status']
        else:
            data=r if kind=='Gemini' else o['qwen'];im=Image.fromarray(small);d=ImageDraw.Draw(im)
            if data:
                for key,color in (('person_bbox',(40,220,240)),('object_bbox',(255,155,45))):
                    b=data.get(key)
                    if b:d.rectangle((b[0]*w/image.width,b[1]*h/image.height,b[2]*w/image.width,b[3]*h/image.height),outline=color,width=2)
            label=kind+' | '+(data.get('status','saved') if data else 'NOT AVAILABLE')
        return im,label
    def jpeg(im):
        for quality in (72,60,48,36,24,16,12):
            b=BytesIO();im.save(b,format='JPEG',quality=quality,optimize=True);raw=b.getvalue()
            if len(raw)<=c['max_jpeg_bytes']:return raw
        raise ValueError('Tiny display gate failed; no inference rerun')
    previews=[]
    for pos,ep in enumerate(c['episodes']):
        sheet=Image.new('RGB',(1280,48+268*3),(24,27,33));d=ImageDraw.Draw(sheet)
        d.text((8,8),f'ep{ep:02d} | exact original initial frames | boxes are proposals, not segmentation',font=font,fill='white')
        byframe={r['index']:r for r in rows if r['ep']==ep}
        for j,index in enumerate(c['frames']):
            r=byframe[index]
            for col,kind in enumerate(('Original','Qwen','SAM','Gemini')):
                im,label=panel(r,kind) if kind!='Original' else (orig[(ep,index)]['image'].resize((320,240)),'Original')
                x,y=col*320,48+j*268;sheet.paste(im,(x,y));d.text((x+4,y+241),f'{label} | frame{index}',font=font,fill='white')
        raw=jpeg(sheet);write(out/f'episode_{ep:06d}.jpg',raw,mode=0o444);previews.append((ep,out/f'episode_{ep:06d}.jpg'))
        for col,index in enumerate(c['frames']):
            im,label=panel(byframe[index],'Gemini',320,200);overview.paste(im,(col*320,48+pos*228))
        draw.text((965,48+pos*228),f'ep{ep:02d} Gemini single-image\n0 / 14 / 29\n'+str([byframe[i]['status'] for i in c['frames']]),font=font,fill='white')
    raw=jpeg(overview);write(out/'overview.jpg',raw,mode=0o444);previews.append((2000,out/'overview.jpg'))
    for b in banks:b.verify()
    require(all(identity(p,2000000)==pin for p,pin in pins.items()) and identity(sam/'sam-report.json',200000)==sp and identity(selection/'selection-report.json',200000)==qp and identity(qwen/'inference.json',200000)==qip,'Saved references changed')
    require(inputs(c['episodes'])==selected,'Original inputs changed')
    if recovery:
        require(identity(old/'native-report.json',200000)==saved_pin,'Original report changed')
        require(all(r==preserved[(r['ep'],r['index'])] for r in rows if (r['ep'],r['index']) not in failed),'Successful original responses must be preserved byte-for-byte logically')
    save(out/'native-report.json',dict(status='complete_diagnostic_not_quality_pass',config=original_config,rows=rows,
        recovery_config=recovery,original_report_pin=saved_pin,
        new_api_attempts=sum(len(r.get('attempts',[])) for r in rows if not recovery or (r['ep'],r['index']) in failed),
        preserved_successful_calls=20 if recovery else 0,
        elapsed_seconds=time.monotonic()-start,new_tracking=False,new_sam_inference=False,gpu_used=False,
        full_video_decode=False,ground_truth_used=False,manual_labels=False,quality_verified=False,
        original_inputs=selected,reference_pins=dict(sam=sp,selection=qp,qwen=qip),
        previews=[dict(episode=ep,file=p.name,**identity(p)) for ep,p in previews]))

def run(recovery=None,entry=ENTRY,helpers=HELPERS):
    require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01','Azure host only')
    code=canonical(Path(os.environ['WR_CODE']));rev=os.environ['WR_CODE_REVISION'];c=settings(code)
    binding=source(ROOT,code,rev,entry,helpers);out=ROOT/'results'/('gemini-initial-'+rev);out.mkdir(mode=0o755)
    if recovery:c.update(budget_seconds=recovery['budget_seconds'])
    key=ROOT/'.secrets/gemini-initial-v1-key.pem';envelope=ROOT/'.secrets/gemini-initial-v1.enc'
    start=time.monotonic();report=dict(status='fail',producer_revision=rev,source_binding=binding)
    name='wr-gemini-initial-'+rev[:12]
    try:
        require(key.stat().st_mode&0o077==0 and envelope.stat().st_mode&0o077==0,'Private one-shot envelope required')
        r=subprocess.run(['openssl','pkeyutl','-decrypt','-inkey',str(key),'-in',str(envelope),'-pkeyopt','rsa_padding_mode:oaep','-pkeyopt','rsa_oaep_md:sha256'],capture_output=True,timeout=15)
        require(r.returncode==0 and 0<len(r.stdout)<16384,'One-shot OAuth decryption failed')
        cmd=['docker','run','--rm','-i','--name',name,'--label','world_reward.gemini_initial.owner='+rev,'--network','host','--read-only','--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','8g','--cpus','8','--tmpfs','/tmp:rw,nosuid,size=1g']
        mounts=[(code.parent,True),(out,False),(ROOT/'results/input-manifest.json',True),(ROOT/'data/track_1/meta',True)]
        mounts += [(ROOT/'results'/(prefix+producer),True) for prefix,producer in (('hybrid-pair-',c['sam_producer']),('hybrid-pair-selection-',c['selection_producer']),('vl-localization-',c['qwen_producer']))]
        mounts += [(ROOT/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4',True) for ep in c['episodes']]
        if recovery:mounts.append((ROOT/'results'/('gemini-initial-'+recovery['saved_producer']),True))
        for p,ro in mounts:canonical(p);cmd+=['--mount',f'type=bind,src={p},dst={p}'+(',readonly' if ro else '')]
        cmd+=['--entrypoint','/usr/bin/env',BASE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','CUDA_VISIBLE_DEVICES=',
            'PYTHONPATH='+str(code/'src')+':'+str(code/'infra'),'PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=2',
            '/opt/conda/bin/python','-B',str(code/('infra/gemini_initial_recover.py' if recovery else 'infra/gemini_initial.py')),'--native',str(code),str(out)]
        with (out/'native.log').open('xb') as log:
            os.fchmod(log.fileno(),0o400);done=subprocess.run(cmd,input=r.stdout,stdout=log,stderr=log,timeout=c['budget_seconds'])
        r=None;require(done.returncode==0,'Initial localization failed; inspect Azure-only log')
        from full4d_publish import PrivatePreviews
        client=PrivatePreviews();client.require_private();files=[]
        for ep in [*c['episodes'],2000]:
            p=out/(f'episode_{ep:06d}.jpg' if ep!=2000 else 'overview.jpg')
            row=client.upload(f'full4d-{rev}/episode_{ep:06d}.jpg',p.read_bytes(),'image/jpeg',rev);row['episode_index']=ep;client.head(row['name'],row,row['etag']);files.append(row)
        report.update(status='complete_diagnostic_not_quality_pass',native_report=identity(out/'native-report.json'),files=files)
    except Exception as e:report['error_type']=type(e).__name__
    finally:
        inspected=subprocess.run(['docker','inspect',name],capture_output=True,timeout=15)
        if inspected.returncode==0:
            actual=strict(inspected.stdout)[0]
            require(actual['Image']==BASE and actual['Config']['Labels'].get('world_reward.gemini_initial.owner')==rev,'Owned CPU cleanup only')
            subprocess.run(['docker','rm','-f',actual['Id']],capture_output=True,check=True,timeout=20)
        for p in (key,envelope):
            if p.exists():require(p.resolve()==p and p.stat().st_nlink==1,'Owned auth cleanup only');p.unlink()
        require(source(ROOT,code,rev,entry,helpers)==binding,'Immutable code changed during run')
        report.update(elapsed_seconds=time.monotonic()-start,credential_files_removed=True,baseline_modified=False,ground_truth_used=False,quality_verified=False)
        save(out/'report.json',report)
    require(report['status']=='complete_diagnostic_not_quality_pass','Gemini diagnostic failed closed')

if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--native':native(Path(sys.argv[2]),Path(sys.argv[3]))
    else:run()
