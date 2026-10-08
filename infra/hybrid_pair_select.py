"""Azure-only candidate-track→VL principal couple selection, no VL geometry."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mediapipe_cpu_runtime_verify import canonical,identity,require,source,strict,write
from task_grounding_pilot import ROOT,BASE,inputs,config as model_config,verify_model
from world_reward.hybrid_pair import indices,uncertainty_indices,prompt,decision,screening,layout,mask_box

ENTRY='run_hybrid_pair_select'
CONFIG='configs/hybrid_pair_v1.json'
HELPERS=('infra/hybrid_pair_select.py','infra/run_hybrid_pair_select.sh',
         'src/world_reward/hybrid_pair.py',CONFIG)


class EvidenceIntegrityError(RuntimeError):
    """Never turn provenance/transport failure into a semantic abstention."""


def evidence(condition,message):
    if not condition:raise EvidenceIntegrityError(message)


def save(path,value):
    write(path,(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode(),mode=0o444)


def settings(code):
    c=strict((code/CONFIG).read_bytes())
    require(c['schema']=='world_reward.hybrid_pair.v1' and c['episodes']==[9,1,14,7,28,13,3,11,18,10,22,6]
            and c['qwen_batch_size']==1 and c['evidence_views']==16 and c['selection_max_rounds']==3
            and c['manual_labels'] is False and c['ground_truth_used'] is False
            and c['quality_verified'] is False,'Frozen12-clip hybrid experiment required')
    return c


class Bank:
    def __init__(self,folder,role,episode,total):
        self.folder=canonical(folder);self.role=role;self.pin=identity(folder/'bank.json',20<<20)
        self.data=strict((folder/'bank.json').read_bytes())
        require(self.data['episode_index']==episode and self.data['total_frames']==total
                and len(self.data['frames'])==total,'Full original SAM bank required')
        require([r['frame_index'] for r in self.data['frames']]==list(range(total)),
                'Original bank indices required')
        self.frames={r['frame_index']:r for r in self.data['frames']}
        self.tracks=[dict(t,role=role) for t in self.data['tracks']]
        require(len({t['id'] for t in self.tracks})==len(self.tracks),'Unique native candidate identities')
        self.touched={folder/'bank.json':self.pin}

    def frame(self,index):
        import numpy as np
        row=self.frames[index];path=canonical(self.folder/row['file'])
        require(path.is_relative_to(self.folder),'Bank-relative frame only')
        pin={k:row[k] for k in ('bytes','sha256')}
        evidence(identity(path,256<<20)==pin,'Literal saved candidate mask changed')
        self.touched[path]=pin
        with np.load(path,allow_pickle=False) as data:
            a={k:data[k] for k in data.files}
        ids=a['ids'];h,w=map(int,a['mask_shape'])
        require(ids.ndim==1 and ids.dtype.kind in 'iu' and a['packed_masks'].shape==(len(ids),(h*w+7)//8),
                'Native packed mask format required')
        masks=np.unpackbits(a['packed_masks'],axis=1,count=h*w).reshape(len(ids),h,w).astype(bool)
        return ids,masks

    def mask(self,track,index):
        import numpy as np
        row=self.frames[index];p=canonical(self.folder/row['file']);pin={k:row[k] for k in ('bytes','sha256')}
        require(p.is_relative_to(self.folder),'Bank-relative frame only')
        if p not in self.touched:evidence(identity(p,256<<20)==pin,'Literal candidate bytes changed')
        self.touched[p]=pin
        with np.load(p,allow_pickle=False) as a:
            ids=a['ids'];h,w=map(int,a['mask_shape']);match=np.flatnonzero(ids==track['native_id'])
            if hasattr(self,'expected_shape'):evidence((h,w)==self.expected_shape,'Original mask grid changed')
            evidence(ids.ndim==1 and ids.dtype.kind in 'iu' and len(set(ids.tolist()))==len(ids)
                     and a['packed_masks'].dtype==np.uint8
                     and a['packed_masks'].shape==(len(ids),(h*w+7)//8),'Native packed mask dimensions required')
            packed=a['packed_masks'][int(match[0])] if len(match)==1 else None
        require(len(match)<=1,'Unique native frame identity required')
        return np.unpackbits(packed,count=h*w).reshape(h,w).astype(bool) if packed is not None else np.zeros((h,w),bool)

    def frame_shape(self,index):
        import numpy as np
        row=self.frames[index];p=self.folder/row['file']
        with np.load(p,allow_pickle=False) as a:return a['mask_shape'].tolist()

    def verify(self):
        evidence(all(identity(p,256<<20)==pin for p,pin in self.touched.items()),'Candidate evidence changed')


def decode(item,wanted,banks):
    """Original RGB only; frame hashes compare SAM's original decoded evidence."""
    import cv2
    from PIL import Image
    cap=cv2.VideoCapture(item['video']);images={};wanted=set(wanted)
    require(cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==item['total'],'Original clip required')
    try:
        for i in range(item['total']):
            ok,bgr=cap.read();require(ok,'Original RGB decode failed')
            if i not in wanted:continue
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);digest=hashlib.sha256(rgb.tobytes()).hexdigest()
            for bank in banks:
                evidence(bank.frames[i]['decoded_rgb_sha256']==digest,'SAM→VL original RGB transport differs')
            images[i]=Image.fromarray(rgb)
    finally:cap.release()
    require(set(images)==wanted,'Missing original evidence frames')
    return images


def appearance(track,bank,images):
    """Untinted appearance+native contour context/tight zoom, never prediction crop."""
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    index=track['best_frame'];im=images[index];mask=bank.mask(track,index);box=mask_box(mask)
    evidence(mask.shape==(im.height,im.width),'Appearance mask is not original RGB grid')
    require(box is not None,'Max-visible native candidate frame required')
    w,h=im.size;x0,y0,x1,y1=box
    pad_x=max(1,(x1-x0)//4);pad_y=max(1,(y1-y0)//4)
    left,top=max(0,x0-pad_x),max(0,y0-pad_y)
    right,bottom=min(w,x1+pad_x),min(h,y1+pad_y)
    a=np.asarray(im).copy();contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(a,contours,-1,(0,220,240) if track['role']=='person' else (255,155,45),2)
    canvas=Image.new('RGB',(768,320),(24,27,33));canvas.paste(Image.fromarray(a).resize((384,288)),(0,32))
    crop=Image.fromarray(a[top:bottom,left:right]);crop.thumbnail((384,288))
    canvas.paste(crop,(384+(384-crop.width)//2,32+(288-crop.height)//2))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    ImageDraw.Draw(canvas).text((8,6),f"{track['id']} | {track['role']} | original frame {index}",font=font,fill='white')
    return canvas


def tracked_scene(index,image,tracks,banks):
    """Expose actual instance history to VL, not only an isolated best crop."""
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    a=np.asarray(image).copy();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    labels=[]
    for t in tracks:
        m=banks[t['role']].mask(t,index)
        evidence(m.shape==(image.height,image.width),'Tracking mask is not original RGB grid')
        box=mask_box(m)
        if box is None:continue
        contours,_=cv2.findContours(m.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        color=(40,220,240) if t['role']=='person' else (255,155,45)
        cv2.drawContours(a,contours,-1,color,2);labels.append((box,t['id'],color))
    im=Image.fromarray(a);draw=ImageDraw.Draw(im)
    for box,id,color in labels:
        draw.text((box[0],max(0,box[1]-22)),id,font=font,fill=color,stroke_width=2,stroke_fill='black')
    return im


def encode(processor,text,visuals):
    content=[{'type':'text','text':text}]
    for label,im in visuals:content.extend(({'type':'text','text':label},{'type':'image','image':im}))
    result=processor.apply_chat_template([[{'role':'user','content':content}]],tokenize=True,
        add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True)
    require('video_grid_thw' not in result,'Explicit original still-image evidence required')
    binding=layout(result['input_ids'].numpy(),result['attention_mask'].numpy(),result['image_grid_thw'].numpy(),
        int(result['pixel_values'].shape[0]),int(processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')),
        [len(visuals)],processor.image_processor.merge_size)
    return result,binding


def infer(code,out,producer):
    import torch
    import transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    c=settings(code);cfg=model_config(code);start=time.monotonic();source_out=ROOT/'results'/('hybrid-pair-'+producer)
    require(torch.__version__=='2.5.1+cu124' and transformers.__version__=='5.3.0','Pinned native Qwen runtime required')
    # Source SAM report plus each independently saved bank binds final evidence;
    # a per-clip failure stays in the cohort, rather than aborting/filling it.
    sam_report=strict((source_out/'sam-report.json').read_bytes());sam_pin=identity(source_out/'sam-report.json',20<<20)
    require(sam_report['producer_revision']==producer and sam_report['ground_truth_used'] is False,
            'Actual original SAM execution required')
    sam_episodes={r['episode_index']:r for r in sam_report['episodes']}
    evidence(set(sam_episodes)==set(c['episodes']),'Exact12-clip SAM cohort required')
    model_folder=ROOT/c['qwen_model_folder'];pins=verify_model(model_folder,cfg);selected=inputs(c['episodes'])
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    processor=AutoProcessor.from_pretrained(model_folder,trust_remote_code=False,local_files_only=True,
        min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels']);processor.tokenizer.padding_side='left'
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_folder,trust_remote_code=False,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();torch.cuda.reset_peak_memory_stats()
    complete=[];ledger=out/'calls';ledger.mkdir(mode=0o755);calls=0;generated_total=0
    def deadline():
        if time.monotonic()-start>=c['qwen_budget_seconds']:raise TimeoutError('Inclusive Qwen budget exceeded')
    def ask(text,visuals,ep,kind):
        nonlocal calls,generated_total
        deadline()
        require(len(visuals)<=c['max_selection_images'],'No image/candidate tail dropping to fit context')
        before=time.monotonic();encoded,binding=encode(processor,text,visuals);length=encoded['input_ids'].shape[1]
        encoded=encoded.to('cuda')
        with torch.inference_mode():a=model.generate(**encoded,max_new_tokens=c['output_capacity'],do_sample=False,num_beams=1)
        tokens=a[:,length:].cpu().tolist()[0];response=processor.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        del a,encoded
        record=dict(episode_index=ep,kind=kind,response=response,generated_tokens=len(tokens),
            prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),input_layout_sha256=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest(),
            visual_inputs=[dict(label=label,width=im.width,height=im.height,rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest()) for label,im in visuals],
            elapsed_seconds=time.monotonic()-before)
        save(ledger/f'{calls:04d}.json',record);calls+=1;generated_total+=len(tokens)
        deadline()
        return response
    for item in selected:
        deadline()
        ep=item['episode'];dest=out/f'episode_{ep:06d}';dest.mkdir(mode=0o755);clip_start=time.monotonic()
        result=dict(episode_index=ep,status='abstained',state='uncertain',person_id=None,object_id=None,
            original_indices=list(range(item['total'])),total_frames=item['total'],action=item['action'],object_prompt=item['object_prompt'],
            sam_producer=producer,quality_verified=False,semantic_accuracy_verified=False,sam_executed=True,
            ground_truth_used=False,manual_labels=False,geometry_generated_by_vl=False,candidate_selection_only=True)
        banks=[]
        try:
            for role in ('person','object'):
                folder=source_out/f'episode_{ep:06d}'/'candidates'/role
                require((folder/'bank.json').is_file(),'SAM candidate bank unavailable; do not substitute')
                b=Bank(folder,role,ep,item['total']);banks.append(b)
                original=[r for r in sam_episodes[ep]['banks'] if r['role']==role]
                evidence(len(original)==1 and original[0]['bank']==b.pin,'SAM report→candidate bank identity differs')
                require(b.data['num_obj_dropped']==0 and b.data['status']=='complete_diagnostic_not_quality_pass',
                        'Capacity-truncated candidates are not adopted')
            tracks=[t for b in banks for t in b.tracks]
            result['candidate_counts']={b.role:len(b.tracks) for b in banks}
            require(all(b.tracks for b in banks),'No matching role candidates; abstain')
            rolebank={b.role:b for b in banks};wanted=set(range(30))|set(indices(item['total']))
            for t in tracks:wanted.add(t['best_frame'])
            images=decode(item,wanted,banks)
            for bank in banks:bank.expected_shape=(images[0].height,images[0].width)
            profiles={t['id']:appearance(t,rolebank[t['role']],images) for t in tracks}
            scene_ids=list(indices(item['total']));limit=c['max_selection_images']-len(scene_ids)
            def visuals(active,frames):
                return [(f'Original full scene frame {i}, native contours and candidate IDs',
                         tracked_scene(i,images[i],active,rolebank)) for i in frames]+[(f"Candidate {t['id']} appearance",profiles[t['id']]) for t in active]
            active=tracks;screened=[]
            if len(active)>limit:
                kept=[]
                for offset in range(0,len(tracks),c['candidate_page_size']):
                    page=tracks[offset:offset+c['candidate_page_size']]
                    response=ask(prompt(item['object_prompt'],item['action'],page,scene_ids,screening=True),
                        visuals(page,scene_ids),ep,'candidate_page_screening')
                    states=screening(response,page);screened.extend(dict(id=k,state=v) for k,v in states.items())
                    kept.extend(t for t in page if states[t['id']]!='rejected')
                active=kept
            result['screened_candidates']=screened;result['retained_candidate_ids']=[t['id'] for t in active]
            require(active and len(active)<=limit,'Unresolved candidate capacity; abstain without top-k tail deletion')
            chosen=None
            for round_index in range(c['selection_max_rounds']):
                response=ask(prompt(item['object_prompt'],item['action'],active,scene_ids),
                    visuals(active,scene_ids),ep,'joint_pair_round_'+str(round_index))
                chosen=decision(response,active,scene_ids)
                if chosen['state']!='uncertain':break
                uncertain=set(chosen['uncertain_person_ids']+chosen['uncertain_object_ids'])
                if not uncertain:break
                narrowed=[t for t in active if t['id'] in uncertain]
                # Do not eliminate an entire role from uncertain-only sampling.
                if {t['role'] for t in narrowed}!={'person','object'}:narrowed=active
                next_ids=list(uncertainty_indices(narrowed,c['evidence_views']))
                if not next_ids or (next_ids==scene_ids and narrowed==active):break
                missing=set(next_ids)-set(images)
                if missing:images.update(decode(item,missing,banks))
                active=narrowed;scene_ids=next_ids
            require(chosen is not None,'Explicit joint decision required')
            result.update(chosen);result['status']='selected' if chosen['state']=='accepted' else 'abstained'
            result['original_candidate_bank_pins']={b.role:b.pin for b in banks}
            # Full original timeline is retained. Missing native masks remain
            # missing; no physical-identity swaps or interpolation to fill gaps.
            if chosen['state']=='accepted':
                byid={t['id']:t for t in tracks};timeline=[]
                for i in range(item['total']):
                    record={'frame_index':i}
                    for role in ('person','object'):
                        t=byid[chosen[role+'_id']];m=rolebank[role].mask(t,i)
                        evidence(m.shape==rolebank[role].expected_shape,'Selected mask is not original video grid')
                        record[role+'_bbox']=mask_box(m)
                        record[role+'_visible']=bool(m.any())
                    timeline.append(record)
                save(dest/'selected-timeline.json',dict(episode_index=ep,records=timeline,
                    original_frame_indices=list(range(item['total'])),pair=chosen,interpolation=False,
                    predicted_boxes_from_final_native_masks=True,quality_verified=False))
                result['selected_timeline']=identity(dest/'selected-timeline.json',20<<20)
                result['selected_visible_frames']={role:sum(r[role+'_visible'] for r in timeline) for role in ('person','object')}
        except (ValueError,FileNotFoundError,KeyError) as error:
            result.update(status='abstained',state='uncertain',person_id=None,object_id=None,
                failure_type=type(error).__name__,failure_reason=str(error)[:250])
        finally:
            for bank in banks:bank.verify()
        result['elapsed_seconds']=time.monotonic()-clip_start;save(dest/'selection.json',result)
        complete.append(dict(episode_index=ep,status=result['status'],state=result['state'],selection=identity(dest/'selection.json',20<<20)))
        print(json.dumps(dict(stage='hybrid_qwen',episode=ep,status=result['status'],calls=calls,elapsed_seconds=time.monotonic()-start)),flush=True)
    require(inputs(c['episodes'])==selected and verify_model(model_folder,cfg)==pins
            and identity(source_out/'sam-report.json',20<<20)==sam_pin,'Original evidence/weights changed')
    deadline()
    save(out/'selection-report.json',dict(status='complete_diagnostic_not_quality_pass',producer_revision=os.environ['WR_CODE_REVISION'],
        sam_producer=producer,sam_report_pin=sam_pin,episodes=complete,model=cfg['model'],model_revision=cfg['model_revision'],
        model_files=pins,model_calls=calls,generated_tokens=generated_total,batch_size=1,
        elapsed_seconds=time.monotonic()-start,peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
        inputs_rehashed_after=True,model_rehashed_after=True,baseline_modified=False,quality_verified=False,
        ground_truth_used=False,manual_labels=False,sam_executed=True,geometry_generated_by_vl=False))


def run():
    require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01','Azure only')
    code=canonical(Path(os.environ['WR_CODE']));revision=os.environ['WR_CODE_REVISION'];c=settings(code)
    producer=c.get('sam_producer_revision',revision)
    binding=source(ROOT,code,revision,ENTRY,HELPERS);out=ROOT/'results'/('hybrid-pair-selection-'+revision);out.mkdir(mode=0o755)
    model=ROOT/c['qwen_model_folder'];bank=ROOT/'results'/('hybrid-pair-'+producer);name='wr-hybrid-qwen-'+revision[:12]
    start=time.monotonic();report=dict(status='fail',phase='qwen',producer_revision=revision,sam_producer=producer,
        source_binding=binding,baseline_modified=False,quality_verified=False,ground_truth_used=False)
    try:
        with (ROOT/'jobs/.world-reward-h100.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not subprocess.check_output(['docker','ps','-q'],text=True).strip(),'No duplicate GPU jobs')
            cmd=['docker','run','--rm','--name',name,'--label','world_reward.hybrid.owner='+revision,
                '--gpus','all','--network','none','--read-only','--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges',
                '--memory','48g','--cpus','16','--shm-size','2g','--tmpfs','/tmp:rw,nosuid,size=1g']
            mounts=[(code.parent,True),(out,False),(bank,True),(model,True),
                (ROOT/'results/input-manifest.json',True),(ROOT/'data/track_1/meta',True)]
            mounts.extend((ROOT/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4',True) for ep in c['episodes'])
            for p,ro in mounts:
                canonical(p);cmd+=['--mount',f'type=bind,src={p},dst={p}'+(',readonly' if ro else '')]
            cmd+=['--entrypoint','/usr/bin/env',BASE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp',
                'PYTHONPATH='+str(code/'src')+':'+str(code/'infra'),'PYTHONDONTWRITEBYTECODE=1',
                'WR_CODE_REVISION='+revision,'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','WANDB_MODE=disabled',
                'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8',
                '/opt/conda/bin/python','-B',str(code/'infra/hybrid_pair_select.py'),'--native',str(code),str(out),producer]
            with (out/'qwen.log').open('xb') as stream:finished=subprocess.run(cmd,stdout=stream,stderr=stream,timeout=c['qwen_budget_seconds']+30)
            require(finished.returncode==0,'Native hybrid selection failed; retain committed evidence')
        report.update(status='complete_diagnostic_not_quality_pass',phase='complete',selection_report=identity(out/'selection-report.json',20<<20))
    except Exception as error:report.update(error_type=type(error).__name__,error=str(error)[:250])
    finally:
        inspected=subprocess.run(['docker','inspect',name],capture_output=True,timeout=15)
        if inspected.returncode==0:
            actual=strict(inspected.stdout)[0];require(actual['Image']==BASE and actual['Config']['Labels'].get('world_reward.hybrid.owner')==revision,'Owned cleanup only')
            subprocess.run(['docker','rm','-f',actual['Id']],capture_output=True,check=True,timeout=20)
        require(source(ROOT,code,revision,ENTRY,HELPERS)==binding,'Original code changed')
        report.update(source_rehashed_after=True,elapsed_seconds=time.monotonic()-start);save(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','phase','elapsed_seconds')}),flush=True)
    require(report['status']=='complete_diagnostic_not_quality_pass','Hybrid selection failed closed')


if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--native':infer(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4])
    else:run()
