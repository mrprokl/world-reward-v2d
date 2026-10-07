"""Bounded Azure-only task-conditioned grounding screen; not quality validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict, write
from world_reward.task_grounding import build_task_grounding_prompt, choose_seed, fixed_frame_indices, parse_response

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_task_grounding_pilot'
CONFIG = 'configs/task_grounding_pilot_v1.json'
BASE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
SAM = 'sha256:53b33bc4b60e0e3e8f83b401775b4701b18eef54408fd585fbe3a5d376c042e1'
HELPERS = ('infra/task_grounding_pilot.py', 'infra/run_task_grounding_pilot.sh',
           'src/world_reward/task_grounding.py', CONFIG)
DATASET_REVISION = '5f68335f3acc802033d1e80728c1633197521de8'


def record(path, value):
    write(path, (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode())


def config(code):
    value = strict((code/CONFIG).read_bytes())
    require(value['schema'] == 'world_reward.task_grounding_pilot.v1'
            and value['episodes'] == [8, 9, 26] and value['views'] == 9
            and value['do_sample'] is False and value['training_overlap_verified'] is False
            and value['challenge_overlap_verified'] is False, 'Frozen pilot scope required')
    require(len(value['files']) == 15 and len({r['name'] for r in value['files']}) == 15
            and sum(r['bytes'] for r in value['files']) == 17545914364, 'Exact publisher whitelist required')
    return value


def verify_model(folder, cfg):
    require({p.name for p in folder.iterdir()} == {r['name'] for r in cfg['files']}, 'Exclusive model whitelist required')
    rows = {}
    for row in cfg['files']:
        path = folder/row['name']; actual = identity(path, 5_000_000_000)
        require(actual['bytes'] == row['bytes'], 'Model size differs')
        if 'sha256' in row:
            require(actual['sha256'] == row['sha256'], 'Publisher model digest differs')
        else:
            git = hashlib.sha1(b'blob ' + str(row['bytes']).encode() + b'\0')
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1 << 20), b''): git.update(block)
            require(git.hexdigest() == row['git_blob_sha1'], 'Publisher tokenizer Git identity differs')
        rows[row['name']] = actual
    return rows


def acquire(folder, cfg):
    """Only exact public publisher URLs; credential/proxy-free, Azure disks only."""
    started = time.monotonic(); deadline = started + cfg['acquire_budget_seconds']
    folder.mkdir(mode=0o700); partials = []
    base = 'https://huggingface.co/' + cfg['model'] + '/resolve/' + cfg['model_revision'] + '/'
    class Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            parsed = urllib.parse.urlsplit(newurl)
            require(parsed.scheme == 'https' and not parsed.username and not parsed.password
                    and (parsed.hostname == 'huggingface.co' or parsed.hostname.endswith('.hf.co')
                         or parsed.hostname.endswith('.huggingface.co')), 'Unexpected publisher redirect')
            return super().redirect_request(req, fp, code, msg, headers, newurl)
    def fetch(row):
        path = folder/(row['name'] + '.part'); partials.append(path)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirect())
        req = urllib.request.Request(base + row['name'], headers={'Accept-Encoding': 'identity'})
        with opener.open(req, timeout=60) as response, path.open('xb') as target:
            os.fchmod(target.fileno(), 0o400)
            require(response.status == 200 and response.headers.get('Content-Encoding', 'identity') == 'identity', 'Non-original model response')
            count = 0
            while True:
                require(time.monotonic() < deadline, 'Acquisition inclusive budget exceeded')
                block = response.read(min(1 << 20, row['bytes'] + 1 - count))
                if not block: break
                count += len(block); require(count <= row['bytes'], 'Oversized publisher artifact'); target.write(block)
            target.flush(); os.fsync(target.fileno())
            require(count == row['bytes'], 'Short publisher artifact')
        # Destination is owned/fresh; no pre-existing cache or implicit adoption.
        require(not (folder/row['name']).exists(), 'Model destination occupied')
        path.rename(folder/row['name'])
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, cfg['files']))
        rows = verify_model(folder, cfg)
        require(time.monotonic() - started <= cfg['acquire_budget_seconds'], 'Acquisition/posthash budget exceeded')
        folder.chmod(0o555)
        return dict(files=rows, elapsed_seconds=time.monotonic()-started, bytes=sum(r['bytes'] for r in rows.values()))
    finally:
        # Only our exact single-link partial paths; never remove original assets.
        for path in partials:
            if path.exists():
                canonical(path); require(path.is_file() and path.stat().st_nlink == 1, 'Owned partial replaced')
                path.unlink()


def inputs():
    manifest = strict((ROOT/'results/input-manifest.json').read_bytes())
    require((manifest['track'], manifest['repo_id'], manifest['revision']) ==
            ('track_1', 'nvidia/video_to_data_challenge', DATASET_REVISION), 'Allowed original Track1 inputs required')
    def file(relative):
        rows = [r for r in manifest['files'] if r['path'] == relative]
        require(len(rows) == 1, 'Unique original input required'); row = rows[0]
        path = ROOT/'data'/relative
        require(identity(path, 1_000_000_000, readonly=False) == {k: row[k] for k in ('bytes', 'sha256')}, 'Original input identity differs')
        return path, {k: row[k] for k in ('bytes', 'sha256')}
    metadata = {}; pins = {}
    for name in ('episodes.jsonl', 'episodes_metadata.jsonl', 'tasks.jsonl'):
        path, pins[name] = file('track_1/meta/' + name)
        metadata[name] = [strict(line) for line in path.read_bytes().splitlines() if line.strip()]
    result = []
    for episode in (8, 9, 26):
        def one(name, key, val):
            rows = [r for r in metadata[name] if type(r.get(key)) is int and r[key] == val]
            require(len(rows) == 1, 'Unique official task/episode required'); return rows[0]
        ep = one('episodes.jsonl', 'episode_index', episode)
        meta = one('episodes_metadata.jsonl', 'episode_index', episode)
        # LeRobot episodes give a list of original task texts, or task indices.
        tasks = ep.get('tasks')
        if type(tasks) is list and len(tasks) == 1 and type(tasks[0]) is str:
            matches = [r for r in metadata['tasks.jsonl'] if r.get('task') == tasks[0]]
            require(len(matches) == 1, 'Episode action must match original task catalog'); action = matches[0]['task']
        else:
            action = one('tasks.jsonl', 'task_index', ep['task_index'])['task']
        require(type(ep['length']) is int and ep['length'] >= 9 and type(action) is str
                and type(meta['object_prompt']) is str and action.strip() and meta['object_prompt'].strip(), 'Original conditioning required')
        relative = f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4'
        video, pin = file(relative)
        result.append(dict(episode=episode, total=ep['length'], video=str(video), video_pin=pin,
                           object_prompt=meta['object_prompt'], action=action, metadata_pins=pins))
    return result


def views(video, total, indices):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video)
    require(cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == total, 'Original full-T decoder count differs')
    images = []
    try:
        for index in indices:
            require(cap.set(cv2.CAP_PROP_POS_FRAMES, index), 'Original frame seek failed')
            ok, bgr = cap.read()
            require(ok and int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) == index+1, 'Wrong original frame decoded')
            images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    finally: cap.release()
    require(len({im.size for im in images}) == 1, 'Clip-constant original grid required')
    return images


def infer(code, out, model_folder):
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    cfg = config(code); started = time.monotonic(); before = verify_model(model_folder, cfg)
    require(torch.__version__ == '2.5.1+cu124' and transformers.__version__ == '5.3.0'
            and torch.cuda.is_available(), 'Qualified native Qwen CUDA runtime required')
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    processor = AutoProcessor.from_pretrained(model_folder, trust_remote_code=False, local_files_only=True,
                                             min_pixels=cfg['min_pixels'], max_pixels=cfg['max_pixels'])
    require(processor.image_processor.size == {'shortest_edge': cfg['min_pixels'], 'longest_edge': cfg['max_pixels']}
            and processor.image_processor.patch_size == 16 and processor.image_processor.merge_size == 2,
            'Effective native full-frame processor differs')
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_folder, trust_remote_code=False,
              local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    rows = inputs(); reports = []
    for item in rows:
        index = item['episode']; dest = out/f'episode_{index:06d}'; dest.mkdir(mode=0o700)
        indices = fixed_frame_indices(item['total'], cfg['views']); images = views(item['video'], item['total'], indices)
        prompt = build_task_grounding_prompt(item['object_prompt'], item['action'], indices)
        content = [{'type': 'text', 'text': prompt}]
        for frame, image in zip(indices, images):
            content.extend([{'type': 'text', 'text': 'Original frame_index=' + str(frame)}, {'type': 'image', 'image': image}])
        encoded = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True,
                   add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
        require('video_grid_thw' not in encoded and 'pixel_values_videos' not in encoded
                and tuple(encoded['image_grid_thw'].shape) == (9,3)
                and bool((encoded['image_grid_thw'][:,0] == 1).all()), 'Nine still-image grids required; no invented FPS')
        called = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=cfg['max_new_tokens'], do_sample=False, num_beams=1)
        text = processor.batch_decode(generated[:, encoded['input_ids'].shape[1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
        write(dest/'response.txt', text.encode())
        report = dict(**item, indices=list(indices), width=images[0].width, height=images[0].height,
                      prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), calls=1,
                      generated_tokens=int(generated.shape[1]-encoded['input_ids'].shape[1]),
                      image_grid_thw=encoded['image_grid_thw'].cpu().tolist(),
                      decoding=dict(do_sample=False,num_beams=1,max_new_tokens=cfg['max_new_tokens']),
                      elapsed_seconds=time.monotonic()-called, status='invalid_response')
        try:
            parsed = parse_response(text, indices, images[0].width, images[0].height); seed = choose_seed(parsed)
            report.update(records=[asdict(r) for r in parsed], seed=asdict(seed) if seed else None,
                          status='seed_available' if seed else 'abstained')
        except ValueError: report['records'] = []; report['seed'] = None
        record(dest/'grounding.json', report); reports.append(report)
        del encoded, generated, images; gc.collect(); torch.cuda.empty_cache()
        require(time.monotonic()-started <= cfg['inference_budget_seconds'], 'Model inference budget exceeded; no retry')
    require(verify_model(model_folder, cfg) == before, 'Weights changed during inference')
    require(time.monotonic()-started <= cfg['inference_budget_seconds'], 'Inclusive inference/posthash budget exceeded')
    record(out/'inference.json', dict(status='complete', episodes=reports, model=cfg['model'], model_revision=cfg['model_revision'],
           torch=torch.__version__, transformers=transformers.__version__, elapsed_seconds=time.monotonic()-started,
           quality_verified=False, training_overlap_verified=False, challenge_overlap_verified=False))


def track(code, out):
    import torch
    import numpy as np
    from PIL import Image
    from v2d.sam2.lib.video_to_masks import video_to_masks
    cfg = config(code); started = time.monotonic(); rows = inputs(); reports = []
    for name, expected in (('video_to_masks.py','5193404292cfc7e66053e261049e58b76d3484f92ac1124c481f9951cc4907ec'),
                           ('sam2_utils.py','603a9cea368098fd50137ad6bf4c34c81289c042d15ad1bc4a99992f744b2408')):
        require(identity(Path('/workspace/v2d_sam2/lib')/name,100000,readonly=False)['sha256']==expected,
                'Qualified direct SAM2 source differs; no GT/HF wrapper allowed')
    checkpoint = ROOT/'weights/sam2'/cfg['sam2_checkpoint']['name']
    pin = {k:cfg['sam2_checkpoint'][k] for k in ('bytes','sha256')}
    require(identity(checkpoint,1_000_000_000,readonly=False)==pin,'Original SAM2 checkpoint differs')
    for item in rows:
        dest = out/f"episode_{item['episode']:06d}"; grounding = strict((dest/'grounding.json').read_bytes())
        require(grounding['video_pin'] == item['video_pin'], 'Grounding source differs')
        seed = grounding['seed']
        if seed is None:
            reports.append(dict(episode=item['episode'], status=grounding['status'], frames=item['total'])); continue
        prompts = {'prompts': [dict(frame_index=seed['frame_index'], object_id=obj, points=None, point_labels=None,
           mask_path=None, box=dict(zip(('x0', 'y0', 'x1', 'y1'), seed[key])))
           for obj, key in ((0, 'person_bbox'), (1, 'object_bbox'))]}
        record(dest/'prompts.json', prompts)
        with torch.inference_mode():
            video_to_masks(item['video'], str(dest/'prompts.json'), str(dest/'masks'), str(ROOT/'weights/sam2'))
        areas = {}
        for obj in (0, 1):
            paths = sorted((dest/'masks'/str(obj)).glob('*.png'))
            require([p.stem for p in paths] == [f'{f:06d}' for f in range(item['total'])], 'Every original frame must be retained')
            counts = [int(np.count_nonzero(np.asarray(Image.open(p)))) for p in paths]
            require(max(counts) > 0, 'Required entity wholly missing')
            areas[str(obj)] = counts
        record(dest/'tracking.json', dict(episode=item['episode'], status='full_T_complete', frames=item['total'], areas=areas))
        reports.append(dict(episode=item['episode'], status='full_T_complete', frames=item['total'],
                            empty_frames={k:sum(v == 0 for v in a) for k,a in areas.items()}))
        gc.collect(); torch.cuda.empty_cache()
        require(time.monotonic()-started <= cfg['tracking_budget_seconds'], 'Full-T tracking budget exceeded')
    require(identity(checkpoint,1_000_000_000,readonly=False)==pin,'SAM2 checkpoint changed during tracking')
    require(time.monotonic()-started <= cfg['tracking_budget_seconds'],'Inclusive SAM2/posthash budget exceeded')
    record(out/'tracking.json', dict(episodes=reports, elapsed_seconds=time.monotonic()-started,
                                   checkpoint=pin,quality_verified=False))


def preview(code, out):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    cfg = config(code); reports = []
    font = ImageFont.load_default(size=15)
    for item in inputs():
        dest = out/f"episode_{item['episode']:06d}"; ground = strict((dest/'grounding.json').read_bytes())
        fixed = list(fixed_frame_indices(item['total'], 3)); extra = None
        tracking = dest/'tracking.json'
        if tracking.exists():
            areas = strict(tracking.read_bytes())['areas']
            # Largest relative mask-area change, independent of human labels.
            changes = [(max(abs(a[f]-a[f-1])/max(1,a[f],a[f-1]) for a in areas.values()), f)
                       for f in range(1,item['total']) if f not in fixed]
            extra = max(changes, key=lambda r:(r[0],-r[1]))[1]
        else:
            extra = next((r['frame_index'] for r in ground['records'] if r['status'] != 'both_boxes' and r['frame_index'] not in fixed), None)
        if extra is not None: fixed.append(extra)
        images = views(item['video'], item['total'], fixed)
        width = 480; height = round(images[0].height*width/images[0].width)
        sheet = Image.new('RGB', (width*2, 54+len(fixed)*(height+25)), (20,20,20)); draw = ImageDraw.Draw(sheet)
        draw.text((8,5), f"Episode {item['episode']} | person=cyan, object=orange", fill='white', font=font)
        draw.text((8,28), 'Existing baseline', fill='white', font=font)
        draw.text((width+8,28), 'Qwen3-VL + SAM2 / automatic', fill='white', font=font)
        for row,(f,image) in enumerate(zip(fixed,images)):
            for col,folder in enumerate((ROOT/f"outputs/episode_{item['episode']:06d}/automatic_masks",dest)):
                rgb = np.array(image.resize((width,height))).astype(float)
                for obj,color in ((0,(0,220,240)),(1,(255,140,30))):
                    path = folder/'masks'/str(obj)/f'{f:06d}.png'
                    if path.exists():
                        mask = np.asarray(Image.open(path).convert('L').resize((width,height),Image.Resampling.NEAREST)) > 0
                        rgb[mask] = .55*rgb[mask]+.45*np.asarray(color)
                sheet.paste(Image.fromarray(rgb.astype('uint8')),(col*width,54+row*(height+25)))
                label = f'original frame {f}' + (' | auto area-change' if f == extra else '')
                if col == 1 and not (dest/'masks').exists(): label += ' | '+ground['status']
                draw.text((col*width+8,54+row*(height+25)+height+3), label, fill='white', font=font)
        import io
        raw = io.BytesIO(); sheet.save(raw, format='JPEG', quality=68, optimize=True)
        require(len(raw.getvalue()) <= cfg['preview_bytes_per_episode'], 'Preview byte cap exceeded; no heavy transfer')
        write(dest/'comparison.jpg',raw.getvalue()); reports.append(dict(episode=item['episode'],frames=fixed,preview=identity(dest/'comparison.jpg')))
    record(out/'previews.json',dict(episodes=reports,manual_labels=False,quality_verified=False))


def docker(code, out, model_folder, mode, image, seconds):
    name = 'world-reward-task-pilot-'+out.name[-12:]+'-'+mode
    owner = out.name
    require(not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$'],text=True).strip(),
            'Fresh owned container namespace required')
    mounts = [(code,code,True),(out,out,False),(ROOT/'results/input-manifest.json',ROOT/'results/input-manifest.json',True)]
    mounts.append((ROOT/'data/track_1/meta',ROOT/'data/track_1/meta',True))
    for episode in (8,9,26):
        video = ROOT/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4'
        mounts.append((video,video,True))
        if mode == 'preview':
            path = ROOT/f'outputs/episode_{episode:06d}/automatic_masks'; mounts.append((path,path,True))
    if mode == 'infer': mounts.append((model_folder,model_folder,True))
    if mode == 'track': mounts.append((ROOT/'weights/sam2',ROOT/'weights/sam2',True))
    command = ['docker','run','--rm','--name',name,'--label','world_reward.task_pilot.owner='+owner,
      '--network','none','--read-only','--user','0:0','--cap-drop','ALL',
      '--security-opt','no-new-privileges','--memory','160g','--cpus','16','--shm-size','2g','--tmpfs','/tmp:rw,nosuid,size=4g']
    if mode != 'preview': command += ['--gpus','all']
    for src,dst,readonly in mounts:
        canonical(src); command += ['--mount',f'type=bind,src={src},dst={dst}' + (',readonly' if readonly else '')]
    command += ['--entrypoint','/usr/bin/env',image,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp',
      'PYTHONPATH='+str(code/'src')+':'+str(code/'infra'),'PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1',
      'TRANSFORMERS_OFFLINE=1','WANDB_MODE=disabled','OMP_NUM_THREADS=8','MPLBACKEND=Agg',
      '/opt/conda/bin/python','-B',str(code/'infra/task_grounding_pilot.py'),'--native',mode,str(code),str(out),str(model_folder)]
    try:
        with (out/(mode+'.log')).open('xb') as log:
            result = subprocess.run(command,stdout=log,stderr=log,timeout=seconds,check=False)
        require(result.returncode == 0, mode+' native execution failed; no scientific retry')
    finally:
        found = subprocess.run(['docker','inspect',name],capture_output=True,timeout=15,check=False)
        if found.returncode == 0:
            actual = json.loads(found.stdout)[0]
            require(actual['Config']['Labels'].get('world_reward.task_pilot.owner')==owner
                    and actual['Image']==image,'Container ownership differs; do not remove')
            subprocess.run(['docker','rm','-f',actual['Id']],stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,timeout=20,check=True)


def run():
    root=canonical(ROOT); code=canonical(Path(os.environ['WR_CODE'])); revision=os.environ['WR_CODE_REVISION']
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01', 'Azure VM01 driver only')
    binding = source(root,code,revision,ENTRY,HELPERS); cfg=config(code)
    out=root/'results'/('task-grounding-pilot-'+revision); require(not out.exists(),'Fresh frozen run only'); out.mkdir(mode=0o700)
    model_folder=out/'model'; report=dict(status='fail',stage='task_grounding_pilot',phase='preflight',producer_revision=revision,
       source_binding=binding,quality_verified=False,ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],
       training_overlap_verified=False,challenge_overlap_verified=False,scope=cfg['scope'])
    started=time.monotonic()
    def interrupted(*_): raise TimeoutError('Frozen pilot interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        with (root/'docker/jobs/.world-reward-h100.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not subprocess.check_output(['docker','ps','-q'],text=True).strip(), 'No duplicate GPU jobs')
            for image in (BASE,SAM):
                actual=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
                require(actual['Id']==image,'Exact pre-existing native images required')
            report['inputs']=inputs();report['phase']='acquire';report['acquisition']=acquire(model_folder,cfg)
            report['phase']='infer';docker(code,out,model_folder,'infer',BASE,cfg['inference_budget_seconds'])
            report['phase']='track';docker(code,out,model_folder,'track',SAM,cfg['tracking_budget_seconds'])
            report['phase']='preview';docker(code,out,model_folder,'preview',BASE,cfg['preview_budget_seconds'])
            report.update(status='complete_diagnostic_not_quality_pass',phase='complete',
              inference=identity(out/'inference.json'),tracking=identity(out/'tracking.json'),previews=identity(out/'previews.json'))
    except Exception as exc:
        report['error_type']=type(exc).__name__;report['error']=str(exc)[:220]
    finally:
        try: require(source(root,code,revision,ENTRY,HELPERS)==binding,'Immutable source changed');report['source_rehashed_after']=True
        except Exception: report['status']='fail';report['source_rehashed_after']=False
        report['elapsed_seconds']=time.monotonic()-started;record(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','phase','producer_revision','elapsed_seconds')}))
    require(report['status']=='complete_diagnostic_not_quality_pass','Pilot failed closed; retain actual diagnostic')


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--native':
        require(len(sys.argv)==6 and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Offline native worker only')
        mode,code,out,model_folder=sys.argv[2:];code,out,model_folder=map(Path,(code,out,model_folder))
        if mode=='infer':infer(code,out,model_folder)
        elif mode=='track':track(code,out)
        elif mode=='preview':preview(code,out)
        else:raise ValueError('Unknown frozen phase')
    else:
        require(len(sys.argv)==1,'No arbitrary pilot arguments');run()
