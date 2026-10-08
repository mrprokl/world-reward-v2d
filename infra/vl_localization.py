"""Azure-only independent Qwen grounding, native transport controls, tiny QA.

One read-only native model load; no SAM, shape/pose inference, judge or tuning.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict, write
from task_grounding_pilot import ROOT, BASE, config as model_config, inputs, verify_model
from world_reward.vl_localization import requested_frames, question, observation, sam_handoff, encoded_layout

ENTRY = 'run_vl_localization'
CONFIG = 'configs/vl_localization_v1.json'
HELPERS = ('infra/vl_localization.py','infra/run_vl_localization.sh',CONFIG,
           'src/world_reward/vl_localization.py')


def save(path, value):
    write(path,(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode(),mode=0o444)


def settings(code):
    c = strict((code/CONFIG).read_bytes())
    require(c['schema']=='world_reward.vl_localization.v1' and c['episodes']==[9,1,14,7]
            and c['batch_size']==8 and c['control_frames']==[0,29] and c['sam_executed'] is False
            and c['manual_labels'] is False and c['ground_truth_used'] is False
            and c['quality_verified'] is False,'Frozen independent localization contract required')
    return c


def decode(item):
    """Sequential original decode once; only requested originals retained in RAM."""
    import cv2
    from PIL import Image
    wanted = requested_frames(item['total']); result = []
    cap = cv2.VideoCapture(item['video'])
    require(cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==item['total'],'Original frame count differs')
    try:
        for index in range(item['total']):
            ok,bgr = cap.read(); require(ok,'Original RGB decode failed')
            if index not in wanted:
                continue
            rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            result.append(dict(episode=item['episode'],index=index,image=image,width=image.width,height=image.height,
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                prompt=question(item['object_prompt'],item['action'],index)))
    finally:
        cap.release()
    require(tuple(r['index'] for r in result)==wanted,'Missing/reordered original RGB')
    return result


def encode(processor, rows):
    messages = [[{'role':'user','content':[{'type':'text','text':r['prompt']},
                 {'type':'text','text':'Original frame_index='+str(r['index'])},
                 {'type':'image','image':r['image']}]}] for r in rows]
    result = processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt',padding=True)
    require('video_grid_thw' not in result and 'pixel_values_videos' not in result,'Still images only')
    layout = encoded_layout(result['input_ids'].numpy(),result['attention_mask'].numpy(),
        result['image_grid_thw'].numpy(),int(result['pixel_values'].shape[0]),
        int(processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')),processor.image_processor.merge_size)
    require(len(layout)==len(rows),'Original observation count differs')
    return result,layout


def trim_tokens(tokens, eos_ids, pad_id):
    end = next((i+1 for i,x in enumerate(tokens) if x in eos_ids),None)
    if end is not None:
        return tokens[:end]
    # Capacity exhaustion is not silently turned into a repaired JSON response.
    return tokens


def infer(code,out):
    import torch
    import transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    c=settings(code);cfg=model_config(code);model_folder=ROOT/c['model_folder']
    start=time.monotonic();pins=verify_model(model_folder,cfg);selected=inputs(c['episodes'])
    require(torch.__version__=='2.5.1+cu124' and transformers.__version__=='5.3.0'
            and torch.cuda.is_available(),'Qualified native Qwen GPU runtime required')
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        decoded=list(pool.map(decode,selected))
    rows=[r for clip in decoded for r in clip]
    by_key={(r['episode'],r['index']):r for r in rows}
    require(len(by_key)==len(rows),'Duplicate observation inputs')
    save(out/'observations-inputs.json',dict(inputs=selected,images=[{k:r[k] for k in
        ('episode','index','width','height','rgb_sha256')} for r in rows],
        requested_count=len(rows),ground_truth_used=False,quality_verified=False))
    processor=AutoProcessor.from_pretrained(model_folder,trust_remote_code=False,local_files_only=True,
        min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'])
    require(processor.image_processor.size=={'shortest_edge':cfg['min_pixels'],'longest_edge':cfg['max_pixels']}
        and processor.image_processor.patch_size==16 and processor.image_processor.merge_size==2,
        'Original native image resolution required')
    processor.tokenizer.padding_side='left'
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_folder,trust_remote_code=False,
        local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    eos=model.generation_config.eos_token_id
    eos_ids={eos} if type(eos) is int else set(eos)
    control=[by_key[(ep,index)] for ep in c['episodes'] for index in c['control_frames']]
    batch,batch_layout=encode(processor,control)
    single_encodings=[];single_layout=[]
    for i,r in enumerate(control):
        one,layout=encode(processor,[r]);s=layout[0];b=batch_layout[i]
        lo,hi=b['patch_slice']
        require(s['tokens']==b['tokens'] and s['grid']==b['grid']
            and torch.equal(one['pixel_values'],batch['pixel_values'][lo:hi]),
            'Batch/singleton image-token transport differs; reject, no correction')
        single_encodings.append(one);single_layout.append(s)
    require(time.monotonic()-start<c['gpu_budget_seconds'],'Inclusive preflight/model deadline exceeded')
    torch.cuda.reset_peak_memory_stats()
    def generate(encoded):
        before=time.monotonic();padded_length=encoded['input_ids'].shape[1]
        encoded=encoded.to('cuda')
        with torch.inference_mode():
            generated=model.generate(**encoded,max_new_tokens=cfg['max_new_tokens'],do_sample=False,num_beams=1)
        tokens=[trim_tokens(row,eos_ids,model.generation_config.pad_token_id) for row in
                generated[:,padded_length:].cpu().tolist()]
        text=processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        del generated,encoded
        require(time.monotonic()-start<c['gpu_budget_seconds'],'Inclusive GPU budget exceeded; no retry')
        return list(zip(tokens,text)),time.monotonic()-before
    batch_out,batch_seconds=generate(batch);del batch
    single_out=[];single_seconds=0.
    for one in single_encodings:
        generated,seconds=generate(one);single_out.extend(generated);single_seconds+=seconds
    del single_encodings
    agree=[a[0]==b[0] for a,b in zip(batch_out,single_out)]
    effective_batch=c['batch_size'] if all(agree) else 1
    control_report=dict(encoding_exact=True,generation_tokens_exact=agree,effective_batch_size=effective_batch,
        keys=[(r['episode'],r['index']) for r in control],batch_seconds=batch_seconds,
        singleton_seconds=single_seconds,batch_outputs=[dict(tokens=t,text=x) for t,x in batch_out],
        singleton_outputs=[dict(tokens=t,text=x) for t,x in single_out],
        layout=batch_layout,quality_verified=False,scientific_output_selection=False)
    save(out/'batch-control.json',control_report)
    complete={};batches=[];(out/'committed_batches').mkdir(mode=0o755)
    def commit(current,outputs,seconds):
        require(len(current)==len(outputs),'Exact response row binding required')
        for r,(tokens,text) in zip(current,outputs):
            key=(r['episode'],r['index']);require(key not in complete,'Repeated prediction row forbidden')
            o=observation(text,episode=r['episode'],frame_index=r['index'],width=r['width'],height=r['height'],
                          rgb_sha256=r['rgb_sha256'])
            o.update(response_text=text,prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest(),
                     generated_tokens=len(tokens),input_frame_binding_explicit=True)
            complete[key]=o
        batch_record=dict(keys=[(r['episode'],r['index']) for r in current],seconds=seconds,
                          output_tokens=sum(len(t) for t,x in outputs))
        save(out/'committed_batches'/f'{len(batches):04d}.json',dict(**batch_record,
            records=[complete[(r['episode'],r['index'])] for r in current]))
        batches.append(batch_record)
    commit(control,batch_out if all(agree) else single_out,batch_seconds if all(agree) else single_seconds)
    todo=[r for r in rows if (r['episode'],r['index']) not in complete]
    for offset in range(0,len(todo),effective_batch):
        current=todo[offset:offset+effective_batch];encoded,_=encode(processor,current)
        generated,seconds=generate(encoded);commit(current,generated,seconds)
        print(json.dumps(dict(stage='vl_independent_progress',committed=len(complete),total=len(rows),
            effective_batch_size=effective_batch,elapsed_seconds=time.monotonic()-start)),flush=True)
    output_pins={}
    for item in selected:
        ep=item['episode'];indices=requested_frames(item['total']);records=[complete[(ep,i)] for i in indices]
        dest=out/f'episode_{ep:06d}';dest.mkdir(mode=0o755)
        save(dest/'observations.json',dict(episode_index=ep,total_frames=item['total'],original_indices=list(indices),
            records=records,model=cfg['model'],model_revision=cfg['model_revision'],input_track='track_1',
            action=item['action'],object_prompt=item['object_prompt'],ground_truth_used=False,
            quality_verified=False,semantic_accuracy_verified=False))
        save(dest/'sam-proposed-input.json',sam_handoff(records,episode=ep,expected_indices=indices))
        output_pins[str(ep)]={name:identity(dest/name) for name in ('observations.json','sam-proposed-input.json')}
    require(inputs(c['episodes'])==selected and verify_model(model_folder,cfg)==pins,'Original inputs/weights changed')
    elapsed=time.monotonic()-start
    require(elapsed<c['gpu_budget_seconds'],'Inclusive inference/posthash deadline exceeded')
    save(out/'inference.json',dict(status='complete_diagnostic_not_quality_pass',producer_revision=os.environ['WR_CODE_REVISION'],
        model=cfg['model'],model_revision=cfg['model_revision'],model_files=pins,torch=torch.__version__,
        transformers=transformers.__version__,requested_images=len(rows),effective_batch_size=effective_batch,
        batch_control=identity(out/'batch-control.json'),episodes=output_pins,batches=batches,
        elapsed_seconds=elapsed,peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
        generated_tokens=sum(r['generated_tokens'] for r in complete.values()),
        missing_objects=sum(r['object_bbox'] is None for r in complete.values()),
        invalid_responses=sum(r['status']=='invalid_response' for r in complete.values()),
        masks_generated=False,sam_executed=False,baseline_modified=False,quality_verified=False,
        training_overlap_verified=False,challenge_overlap_verified=False,inputs_rehashed_after=True,
        model_rehashed_after=True))


def preview(code,out):
    import cv2
    from io import BytesIO
    from PIL import Image,ImageDraw,ImageFont
    from world_reward.task_grounding import fixed_frame_indices
    c=settings(code);inference=strict((out/'inference.json').read_bytes())
    require(inference['status']=='complete_diagnostic_not_quality_pass','Complete actual VL outputs required')
    selected=inputs(c['episodes'])
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    def sheet(ep,records,images,label):
        # Full RGB miniature plus separate box-derived context inset. No manual crop.
        tile_w,tile_h=448,270;cols=6 if len(records)==30 else 3;row_count=(len(records)+cols-1)//cols
        canvas=Image.new('RGB',(tile_w*cols,50+tile_h*row_count),(24,27,33));draw=ImageDraw.Draw(canvas)
        draw.text((8,7),f'World Reward | ep{ep:02d} | {label} | cyan person / orange object',font=font,fill='white')
        draw.text((8,26),'Independent Qwen8B | literal proposed boxes to SAM | SAM NOT RUN | no manual labels',font=font,fill='white')
        for position,(r,rgb) in enumerate(zip(records,images)):
            x,y=(position%cols)*tile_w,50+(position//cols)*tile_h
            view=Image.fromarray(cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA))
            vd=ImageDraw.Draw(view)
            for key,color in (('person_bbox',(40,220,240)),('object_bbox',(255,155,45))):
                b=r[key]
                if b is not None:vd.rectangle([b[0]*320/r['width'],b[1]*240/r['height'],b[2]*320/r['width'],b[3]*240/r['height']],outline=color,width=2)
            canvas.paste(view,(x,y))
            box=r['object_bbox']
            if box is not None:
                cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
                unit=max(16,int(__import__('math').ceil(max((box[2]-box[0])*1.5/4,(box[3]-box[1])*1.5/3))))
                unit=min(unit,r['width']//4,r['height']//3);w,h=4*unit,3*unit
                left=max(0,min(r['width']-w,int(cx-w/2)));top=max(0,min(r['height']-h,int(cy-h/2)))
                inset=Image.fromarray(cv2.resize(rgb[top:top+h,left:left+w],(128,96),interpolation=cv2.INTER_AREA))
                ImageDraw.Draw(inset).rectangle([(box[0]-left)*128/w,(box[1]-top)*96/h,
                    (box[2]-left)*128/w,(box[3]-top)*96/h],outline=(255,155,45),width=2)
                canvas.paste(inset,(x+320,y))
            draw.text((x+323,y+100),'Object-box zoom',font=font,fill='white')
            draw.text((x+323,y+120),'person: '+('box' if r['person_bbox'] is not None else 'NULL'),font=font,fill='white')
            draw.text((x+323,y+138),'object: '+('box' if box is not None else 'NULL'),font=font,fill='white')
            draw.text((x+5,y+244),f"frame {r['frame_index']:04d} | {r['frame_index']/30:.2f}s | {r['status']}",font=font,fill='white')
        for quality in (72,60,48,36,24,16):
            buffer=BytesIO();canvas.save(buffer,format='JPEG',quality=quality,optimize=True);raw=buffer.getvalue()
            if len(raw)<=c['max_jpeg_bytes']:break
        require(len(raw)<=c['max_jpeg_bytes'],'Tiny sheet limit; never drop frames')
        return raw,quality
    def one(item):
        ep=item['episode'];dest=out/f'episode_{ep:06d}'
        for name,pin in inference['episodes'][str(ep)].items():require(identity(dest/name)==pin,'Frozen observation bytes changed')
        o=strict((dest/'observations.json').read_bytes());h=strict((dest/'sam-proposed-input.json').read_bytes())
        records=o['records'];require(h==sam_handoff(records,episode=ep,expected_indices=tuple(o['original_indices'])),'Handoff boxes differ')
        decoded=decode(item);rgb=[__import__('numpy').asarray(r['image']) for r in decoded]
        require([r['rgb_sha256'] for r in decoded]==[r['decoded_rgb_sha256'] for r in records],'Displayed RGB differs from inference')
        raw,q=sheet(ep,records[:30],rgb[:30],'ALL original frames0..29')
        write(dest/'first30.jpg',raw,mode=0o444)
        indices=fixed_frame_indices(item['total'],9);where=[o['original_indices'].index(i) for i in indices]
        raw2,q2=sheet(ep,[records[i] for i in where],[rgb[i] for i in where],'9 uniform full-clip views')
        write(dest/'fullclip9.jpg',raw2,mode=0o444)
        return dict(episode_index=ep,first30=identity(dest/'first30.jpg'),fullclip9=identity(dest/'fullclip9.jpg'),
            first30_frame_indices=list(range(30)),fullclip_frame_indices=list(indices),
            jpeg_quality=q,fullclip_jpeg_quality=q2,display_only_zoom=True,sam_executed=False,quality_verified=False)
    with ThreadPoolExecutor(max_workers=4) as pool:previews=list(pool.map(one,selected))
    require(inputs(c['episodes'])==selected,'Original source changed during QA')
    save(out/'previews.json',dict(status='complete_saved_only_qa',episodes=previews,gpu_used=False,
        model_calls=0,source_rehashed_after=True,original_frame_indices_preserved=True,quality_verified=False))


def docker(code,out,mode,seconds,model_folder,revision):
    name=f'wr-vl-localization-{revision[:12]}-{mode}'
    command=['docker','run','--rm','--name',name,'--label','world_reward.vl.owner='+revision,
        '--network','none','--read-only','--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges',
        '--memory','48g','--cpus','16','--shm-size','2g','--tmpfs','/tmp:rw,nosuid,size=1g']
    if mode=='infer':command+=['--gpus','all']
    mounts=[(code.parent,True),(out,False),(ROOT/'results/input-manifest.json',True),(ROOT/'data/track_1/meta',True)]
    if mode=='infer':mounts.append((model_folder,True))
    for ep in settings(code)['episodes']:
        mounts.append((ROOT/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4',True))
    for p,ro in mounts:
        canonical(p);command+=['--mount',f'type=bind,src={p},dst={p}'+(',readonly' if ro else '')]
    command+=['--entrypoint','/usr/bin/env',BASE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp',
        'PYTHONPATH='+str(code/'src')+':'+str(code/'infra'),'PYTHONDONTWRITEBYTECODE=1',
        'WR_CODE_REVISION='+revision,'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','WANDB_MODE=disabled',
        'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8',
        '/opt/conda/bin/python','-B',str(code/'infra/vl_localization.py'),'--native',mode,str(code),str(out)]
    try:
        with (out/(mode+'.log')).open('xb') as f:r=subprocess.run(command,stdout=f,stderr=f,timeout=seconds)
        require(r.returncode==0,'Native '+mode+' failed; no scientific retry')
    finally:
        r=subprocess.run(['docker','inspect',name],capture_output=True,timeout=15)
        if r.returncode==0:
            actual=strict(r.stdout)[0]
            require(actual['Image']==BASE and actual['Config']['Labels'].get('world_reward.vl.owner')==revision,'Owned container only')
            subprocess.run(['docker','rm','-f',actual['Id']],capture_output=True,timeout=20,check=True)


def run():
    require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01','Azure host only')
    code=canonical(Path(os.environ['WR_CODE']));revision=os.environ['WR_CODE_REVISION'];c=settings(code)
    binding=source(ROOT,code,revision,ENTRY,HELPERS)
    out=ROOT/'results'/('vl-localization-'+revision);out.mkdir(mode=0o755)
    start=time.monotonic();report=dict(status='fail',phase='preflight',producer_revision=revision,
        source_binding=binding,baseline_modified=False,ground_truth_used=False,hand_labeled_test=False,
        quality_verified=False,sam_executed=False)
    def interrupted(*_):raise TimeoutError('Bounded localization interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        with (ROOT/'jobs/.world-reward-h100.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not subprocess.check_output(['docker','ps','-q'],text=True).strip(),'No duplicate jobs')
            require(strict(subprocess.check_output(['docker','image','inspect',BASE]))[0]['Id']==BASE,'Exact offline runtime required')
            report['phase']='infer';docker(code,out,'infer',c['gpu_budget_seconds'],ROOT/c['model_folder'],revision)
        report['phase']='preview';docker(code,out,'preview',c['preview_budget_seconds'],None,revision)
        from full4d_publish import PrivatePreviews
        client=PrivatePreviews();client.require_private();files=[]
        for ep in c['episodes']:
            p=out/f'episode_{ep:06d}/first30.jpg';identity(p,c['max_jpeg_bytes'])
            row=client.upload(f'full4d-{revision}/episode_{ep:06d}.jpg',p.read_bytes(),'image/jpeg',revision)
            row['episode_index']=ep;files.append(row);save(out/f'published_{ep:06d}.json',row)
            client.head(row['name'],row,row['etag'])
        # Full-clip movement context: publish separately under the same fixed
        # lightweight naming policy; offset IDs are display files, not episodes.
        for ep in c['episodes']:
            p=out/f'episode_{ep:06d}/fullclip9.jpg';identity(p,c['max_jpeg_bytes'])
            row=client.upload(f'full4d-{revision}/episode_{ep+1000:06d}.jpg',p.read_bytes(),'image/jpeg',revision)
            row.update(episode_index=ep,view='fullclip9');files.append(row)
            save(out/f'published_context_{ep:06d}.json',row);client.head(row['name'],row,row['etag'])
        report.update(status='complete_diagnostic_not_quality_pass',phase='complete',files=files,
            inference=identity(out/'inference.json'),previews=identity(out/'previews.json'))
    except Exception as e:
        report.update(error_type=type(e).__name__,error=str(e)[:300])
    finally:
        try:require(source(ROOT,code,revision,ENTRY,HELPERS)==binding,'Source changed');report['source_rehashed_after']=True
        except Exception:report.update(status='fail',source_rehashed_after=False)
        report['elapsed_seconds']=time.monotonic()-start;save(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','phase','producer_revision','elapsed_seconds')}),flush=True)
    require(report['status']=='complete_diagnostic_not_quality_pass','Localization diagnostic failed closed')


if __name__=='__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--native':
        require({p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Offline native only')
        mode,code,out=sys.argv[2:]
        if mode=='infer':infer(Path(code),Path(out))
        elif mode=='preview':preview(Path(code),Path(out))
        else:raise ValueError('Unknown native phase')
    else:run()
