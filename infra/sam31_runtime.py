"""Pinned Azure-only SAM3.1 candidate banks; no Qwen, fitting or manual labels.

The public September code has an init-state argument mismatch and broken CV2
normalization. Compatibility is general: filter accepted initialization keywords
and supply losslessly decoded RGB PIL frames to the native, correct PIL loader.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict, write
from task_grounding_pilot import ROOT, inputs

ENTRY = 'run_sam31_runtime'
CONFIG = 'configs/sam31_runtime_v1.json'
COHORT = 'configs/hybrid_pair_v1.json'
HELPERS = ('infra/sam31_runtime.py', 'infra/run_sam31_runtime.sh',
           'infra/Dockerfile.sam31', CONFIG, COHORT)
EXPECTED_EPISODES = [9, 1, 14, 7, 28, 13, 3, 11, 18, 10, 22, 6]


def save(path, value):
    write(path, (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode(), mode=0o444)


def settings(code):
    c = strict((code/CONFIG).read_bytes()); h = strict((code/COHORT).read_bytes())
    require(c['schema'] == 'world_reward.sam31_runtime.v1'
            and c['source_revision'] == '2345a4ad109ac29c569da749c91d84f10dc08c40'
            and c['model_revision'] == 'daa63191845a41281374e725f4c9e51c7a824460'
            and c['use_fa3'] is False and c['compile'] is False and c['warm_up'] is False
            and h['schema'] == 'world_reward.hybrid_pair.v1' and h['episodes'] == EXPECTED_EPISODES
            and h['sam_capacity'] == 64 and h['sam_multiplex_count'] == 16
            and h['sam_budget_seconds'] == 3600 and h['sam_episode_budget_seconds'] == 600
            and h['manual_labels'] is False and h['ground_truth_used'] is False
            and c['ground_truth_used'] is False, 'Frozen SAM3.1/source/cohort contract required')
    return c, h


def git_blob(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def verify_git_tree(rows, expected):
    """Verify the complete public Git tree, before fetching code-only whitelist."""
    require(len({r['path'] for r in rows}) == len(rows), 'Duplicate publisher source path')
    directories = {'': expected} | {r['path']: r['sha'] for r in rows if r['type'] == 'tree'}
    for path, sha in directories.items():
        children = [r for r in rows if ('' if str(Path(r['path']).parent) == '.' else str(Path(r['path']).parent)) == path]
        children.sort(key=lambda r: (Path(r['path']).name + ('/' if r['type'] == 'tree' else '')).encode())
        raw = b''.join(r['mode'].lstrip('0').encode()+b' '+Path(r['path']).name.encode()+b'\0'
                       +bytes.fromhex(r['sha']) for r in children)
        actual = hashlib.sha1(b'tree '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        require(actual == sha, 'Publisher Git tree does not match frozen commit')


def allowed_source(row):
    p = row['path']
    return row['type'] == 'blob' and (p.startswith('sam3/') and p.endswith('.py')
        or p in ('LICENSE', 'pyproject.toml', 'sam3/assets/bpe_simple_vocab_16e6.txt.gz'))


def fetch_public(url, maximum):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, headers={'Accept-Encoding': 'identity'}), timeout=60) as r:
        require(r.status == 200, 'Public publisher source response failed')
        raw = r.read(maximum+1); require(len(raw) <= maximum, 'Oversized source metadata')
        return raw


def acquire_source(folder, c, deadline):
    base = 'https://api.github.com/repos/'+c['source_repo']+'/git/trees/'+c['source_tree']+'?recursive=1'
    tree = strict(fetch_public(base, 2_000_000))
    require(tree['sha'] == c['source_tree'] and tree['truncated'] is False, 'Complete pinned publisher tree required')
    verify_git_tree(tree['tree'], c['source_tree'])
    rows = [r for r in tree['tree'] if allowed_source(r)]
    require(len(rows) == 155 and sum(r['size'] for r in rows) == 4184994, 'Code-only source whitelist differs')
    folder.mkdir(mode=0o755)
    def one(row):
        require(time.monotonic() < deadline and row['mode'] == '100644', 'Source acquisition deadline/type differs')
        p = folder/row['path']; p.parent.mkdir(parents=True, exist_ok=True)
        raw = fetch_public('https://raw.githubusercontent.com/'+c['source_repo']+'/'+c['source_revision']+'/'+row['path'], row['size'])
        require(len(raw) == row['size'] and git_blob(raw) == row['sha'], 'Pinned publisher source blob differs')
        write(p, raw, mode=0o444)
        return dict(file=row['path'], git_blob_sha1=row['sha'], **identity(p, 2_000_000))
    with ThreadPoolExecutor(max_workers=8) as pool: result = list(pool.map(one, rows))
    require(time.monotonic() < deadline, 'Source posthash deadline exceeded')
    return result


def acquire_weight(path, c, deadline):
    token_path = canonical(ROOT/'.secrets/hf_token'); before = token_path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and not before.st_mode & 0o077,
            'Private token file permissions differ')
    token = token_path.read_text().strip()
    require(token.startswith('hf_') and not any(ch.isspace() for ch in token), 'Credential format invalid')
    class SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            u = urllib.parse.urlsplit(newurl)
            require(u.scheme == 'https' and not u.username and not u.password
                    and (u.hostname == 'huggingface.co' or u.hostname.endswith('.hf.co')
                         or u.hostname.endswith('.huggingface.co')), 'Unexpected credential publisher redirect')
            new = super().redirect_request(req, fp, code, msg, headers, newurl)
            if u.hostname != 'huggingface.co':
                new.remove_header('Authorization')
            return new
    url = 'https://huggingface.co/'+c['model_repo']+'/resolve/'+c['model_revision']+'/'+c['weight_file']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), SafeRedirect())
    request = urllib.request.Request(url, headers={'Authorization': 'Bearer '+token, 'Accept-Encoding': 'identity'})
    partial = path.with_suffix('.part'); digest = hashlib.sha256(); count = 0
    try:
        with opener.open(request, timeout=60) as r, partial.open('xb') as f:
            os.fchmod(f.fileno(), 0o400)
            require(r.status == 200 and r.headers.get('Content-Encoding', 'identity') == 'identity', 'Weight response differs')
            while True:
                require(time.monotonic() < deadline, 'Weight acquisition deadline exceeded')
                block = r.read(min(1 << 20, c['weight_bytes']+1-count))
                if not block: break
                count += len(block); require(count <= c['weight_bytes'], 'Oversized publisher weight')
                digest.update(block); f.write(block)
            f.flush(); os.fsync(f.fileno())
        require(count == c['weight_bytes'] and digest.hexdigest() == c['weight_sha256'], 'Frozen weight bytes/SHA differ')
        require(not path.exists(), 'Weight destination occupied'); partial.rename(path)
        return identity(path, c['weight_bytes'])
    finally:
        token = None; request = None
        if partial.exists():
            canonical(partial); require(partial.stat().st_nlink == 1, 'Owned partial replaced'); partial.unlink()


def control(args, timeout=30):
    r = subprocess.run(args, capture_output=True, timeout=timeout)
    require(r.returncode == 0 and len(r.stdout) < 2_000_000, 'Bounded container control failed')
    return r.stdout


def prepare(code, out, revision, binding):
    c, _ = settings(code); start = time.monotonic(); deadline = start+c['prepare_budget_seconds']
    prep = out/'runtime'; prep.mkdir(mode=0o700); context = prep/'context'; context.mkdir(mode=0o700)
    tag = 'world-reward/sam31:'+revision
    require(subprocess.run(['docker', 'image', 'inspect', tag], capture_output=True).returncode != 0,
            'Fresh runtime image tag required')
    source_rows = acquire_source(context/'upstream', c, deadline)
    weight = prep/c['weight_file']; weight_pin = acquire_weight(weight, c, deadline)
    dockerfile=Path(__file__).with_name('Dockerfile.sam31')
    require(dockerfile == code/'infra/Dockerfile.sam31', 'Frozen sibling Dockerfile required')
    write(context/'Dockerfile', dockerfile.read_bytes(), mode=0o444)
    with (prep/'build.log').open('xb') as log:
        os.fchmod(log.fileno(), 0o400)
        r = subprocess.run(['docker', 'build', '--network', 'host', '--tag', tag,
            '--build-arg', 'SAM_BASE='+c['docker_base'], '--build-arg', 'WR_REVISION='+revision,
            '--build-arg', 'WR_SOURCE_SHA256='+binding['closure_sha256'], str(context)],
            stdout=log, stderr=log, timeout=max(1, deadline-time.monotonic()))
    require(r.returncode == 0 and time.monotonic() < deadline, 'Runtime build failed/deadline')
    image = strict(control(['docker', 'image', 'inspect', tag]))[0]
    require(re.fullmatch('sha256:[0-9a-f]{64}', image['Id']) and image['Architecture'] == 'amd64'
            and image['Os'] == 'linux' and image['Config']['Labels']['world_reward.sam31.revision'] == revision,
            'Actual pinned SAM child image required')
    actual = control(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', '/usr/bin/env', image['Id'],
        '-i', 'PATH=/usr/local/bin:/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE=1',
        '/usr/local/bin/python', '-B', '-c', 'import sys,torch,torchvision;from pathlib import Path;'
        'print(sys.version.split()[0]);print(torch.__version__);print(torchvision.__version__);'
        'print(Path("/opt/requirements-actual.txt").read_text())'], 60).decode()
    lines = actual.splitlines()
    require(lines[:3] == [c['python'], c['torch'], c['torchvision']], 'Actual Python/Torch runtime pin differs')
    write(prep/'requirements-actual.txt', actual.encode(), mode=0o444)
    receipt = dict(status='pass', producer_revision=revision, image_id=image['Id'], docker_base=c['docker_base'],
        source_revision=c['source_revision'], source_tree=c['source_tree'], source_files=source_rows,
        model_revision=c['model_revision'], model= c['model_repo'], weight_file=str(weight), weight=weight_pin,
        source_binding=binding, requirements=identity(prep/'requirements-actual.txt'),
        elapsed_seconds=time.monotonic()-start, gpu_used=False, inference_executed=False,
        training_overlap_verified=False, challenge_overlap_verified=False)
    save(prep/'prepare-report.json', receipt)
    shutil.rmtree(context)  # exact newly owned context; code remains in immutable image
    return receipt


def compatible_init(model, resource_path, *, async_loading_frames=False):
    """General signature adapter; no heuristics, geometry or input modifications."""
    kwargs = dict(resource_path=resource_path, offload_video_to_cpu=False,
                  offload_state_to_cpu=False, async_loading_frames=async_loading_frames)
    signature = inspect.signature(model.init_state)
    filtered = {k:v for k,v in kwargs.items() if k in signature.parameters}
    require('resource_path' in filtered, 'Native init resource API absent')
    return model.init_state(**filtered), sorted(set(kwargs)-set(filtered))


def full_grid_stream(model, state):
    """Native one-pass final batch; public base API drops the final-batch flag."""
    require('is_last_batch' in inspect.signature(model.propagate_in_video).parameters,
            'Pinned final-batch native API required')
    for index,output in model.propagate_in_video(inference_state=state,start_frame_idx=0,
            max_frame_num_to_track=None,reverse=False,is_last_batch=True):
        state.get('cached_frame_outputs',{}).pop(index,None)
        yield dict(frame_index=index,outputs=output)


def decode_original(item):
    import cv2
    import numpy as np
    from PIL import Image
    cap = cv2.VideoCapture(item['video']); frames = []; hashes = []
    require(cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == item['total'], 'Original video count differs')
    try:
        for i in range(item['total']):
            ok, bgr = cap.read(); require(ok, 'Original full-grid decode failed')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); require(rgb.dtype == np.uint8, 'Original uint8 RGB required')
            frames.append(Image.fromarray(rgb)); hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
        require(not cap.read()[0], 'More source frames than official count')
    finally: cap.release()
    require(len({im.size for im in frames}) == 1, 'Clip dimensions must be constant')
    return frames, hashes


def verify_runtime_source(prep):
    folder=Path('/opt/sam3')
    for row in prep['source_files']:
        require(identity(folder/row['file'],2_000_000) == {k:row[k] for k in ('bytes','sha256')},
                'Native imported publisher source identity differs')


def verify_checkpoint_coverage(model, weight):
    """The upstream merged-checkpoint loader is non-strict: require full coverage."""
    import torch
    checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
    if 'model' in checkpoint and isinstance(checkpoint['model'],dict):checkpoint=checkpoint['model']
    require(not any(k.startswith(('sam3_model.','sam2_predictor.')) for k in checkpoint),
            'Official already-remapped checkpoint keys required')
    native=model.state_dict()
    require(set(checkpoint)==set(native)
        and all(tuple(checkpoint[k].shape)==tuple(native[k].shape) for k in native),
        'Non-strict upstream loader left missing/unexpected model parameters')
    return dict(checkpoint_parameters=len(checkpoint),model_parameters=len(native),complete_key_shape_coverage=True)


def mask_records(output, height, width):
    """Copy exact visible native masks and derive exclusive full-image bounds."""
    import numpy as np
    ids = np.asarray(output['out_obj_ids']); masks = np.asarray(output['out_binary_masks'])
    probs = np.asarray(output['out_probs']); boxes = np.asarray(output['out_boxes_xywh'])
    require(ids.dtype == np.int64 and ids.ndim == 1 and len(set(ids.tolist())) == len(ids)
        and masks.dtype == bool and masks.shape == (len(ids), height, width)
        and probs.shape == (len(ids),) and boxes.shape == (len(ids), 4)
        and np.isfinite(probs).all() and np.isfinite(boxes).all(), 'Native full-grid output schema differs')
    rows = []
    for native, mask in zip(ids.tolist(), masks):
        yy, xx = np.nonzero(mask); require(len(xx) > 0, 'Native visible mask unexpectedly empty')
        rows.append(dict(native_id=native, area=int(len(xx)), box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]))
    return dict(ids=ids.copy(), probs=probs.copy(), boxes_xywh=boxes.copy(),
        mask_shape=np.array([height,width],dtype=np.int64),
        packed_masks=np.packbits(masks.reshape(len(ids), height*width), axis=1)), rows


def persist_frame(folder, output, index, rgb_sha, height, width):
    import numpy as np
    arrays, records = mask_records(output, height, width)
    p = folder/f'frame_{index:06d}.npz'
    with p.open('xb') as f:
        os.fchmod(f.fileno(), 0o444); np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    return dict(frame_index=index, file=p.name, decoded_rgb_sha256=rgb_sha, **identity(p, 512_000_000)), records


def native(code, out):
    import torch
    import numpy as np
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    c, h = settings(code); revision = os.environ['WR_CODE_REVISION']; start = time.monotonic()
    require(sys.version.split()[0] == c['python'] and torch.__version__ == c['torch'] and torch.cuda.is_available(),
            'Exact qualified SAM3.1 GPU runtime required')
    require({p.name for p in Path('/sys/class/net').iterdir()} == {'lo'}, 'Offline native inference required')
    prep = strict((out/'runtime/prepare-report.json').read_bytes()); weight = Path(prep['weight_file'])
    require(identity(weight,c['weight_bytes']) == prep['weight'] == dict(bytes=c['weight_bytes'],sha256=c['weight_sha256']),
            'Frozen checkpoint identity differs before inference')
    verify_runtime_source(prep)
    selected = inputs(h['episodes']); save(out/'sam-inputs.json',dict(inputs=selected,ground_truth_used=False))
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(weight), max_num_objects=h['sam_capacity'],
        multiplex_count=h['sam_multiplex_count'], use_fa3=c['use_fa3'], use_rope_real=c['use_rope_real'],
        compile=c['compile'], warm_up=c['warm_up'], async_loading_frames=False)
    checkpoint_coverage=verify_checkpoint_coverage(predictor.model,weight)
    torch.cuda.reset_peak_memory_stats(); results = []; torch.manual_seed(20261008)
    def budget(clip_start=None):
        require(time.monotonic()-start < h['sam_budget_seconds'], 'Inclusive SAM GPU deadline exceeded')
        if clip_start is not None:
            require(time.monotonic()-clip_start < h['sam_episode_budget_seconds'], 'Predeclared episode deadline exceeded')
    pool = ThreadPoolExecutor(max_workers=c['prefetch_clips']); pending = {}
    try:
        for i in range(min(c['prefetch_clips'],len(selected))): pending[i] = pool.submit(decode_original,selected[i])
        for position,item in enumerate(selected):
            clip_start=time.monotonic(); frames, hashes=pending.pop(position).result()
            original_width,original_height=frames[0].size
            nxt=position+c['prefetch_clips']
            if nxt<len(selected): pending[nxt]=pool.submit(decode_original,selected[nxt])
            budget(clip_start); state, omitted = compatible_init(predictor.model,frames)
            require(omitted == ['offload_state_to_cpu'], 'Pinned init compatibility behavior differs')
            video = state['input_batch'].img_batch.tensors
            require(video.dtype == torch.float16 and tuple(video.shape) == (item['total'],3,1008,1008)
                and state['orig_width'] == frames[0].width and state['orig_height'] == frames[0].height
                and float(video.min()) >= -1.01 and float(video.max()) <= 1.01, 'Native PIL normalized-image transport gate failed')
            sid='episode-'+str(item['episode']);predictor._all_inference_states[sid]={'state':state,'session_id':sid,
                'start_time':time.time(),'last_use_time':time.time()}
            # Validated original RGB stays RAM-only on Azure; subsequent concepts reuse video tensor.
            del frames, video, state; banks = []; fail_clip = False
            for role,query in (('person','person'),('object',item['object_prompt'])):
                budget(clip_start); folder=out/f"episode_{item['episode']:06d}"/'candidates'/role
                folder.mkdir(parents=True,mode=0o755); prefix='p' if role=='person' else 'o'; tracks={}; frame_rows=[]; dropped=0
                predictor.handle_request(dict(type='reset_session',session_id=sid))
                predictor.handle_request(dict(type='add_prompt',session_id=sid,frame_index=0,text=query))
                expected=0; stage_start=time.monotonic()
                for response in full_grid_stream(predictor.model,predictor._all_inference_states[sid]['state']):
                    index=response['frame_index']; require(index == expected, 'Missing/duplicate/reordered original SAM frame')
                    output=response['outputs']; stats=output.get('frame_stats') or {}
                    require('num_obj_dropped' in stats, 'Explicit native capacity diagnostic required')
                    dropped += int(stats['num_obj_dropped']); pin,records=persist_frame(folder,output,index,hashes[index],
                        original_height,original_width)
                    frame_rows.append(pin)
                    for r in records:
                        native=r['native_id'];track=tracks.setdefault(native,dict(id=prefix+':'+str(native),role=role,
                            native_id=native,visible_frames=[],best_frame=index,best_area=0,best_box=None))
                        track['visible_frames'].append(index)
                        if r['area']>track['best_area']:track.update(best_frame=index,best_area=r['area'],best_box=r['box'])
                    expected+=1; budget(clip_start)
                require(expected == item['total'], 'Original full trajectory frame grid incomplete')
                bank=dict(schema='world_reward.sam31_candidate_bank.v1',episode_index=item['episode'],
                    query=query,namespace=prefix,total_frames=item['total'],frames=frame_rows,
                    tracks=[tracks[k] for k in sorted(tracks)],num_obj_dropped=dropped,
                    capacity=h['sam_capacity'],status='fail_capacity' if dropped else 'complete_diagnostic_not_quality_pass',
                    elapsed_seconds=time.monotonic()-stage_start,original_frame_indices_preserved=True,
                    mask_bbox_xyxy_exclusive=True,probs_are_initial_detection_confidence=True,
                    model_revision=c['model_revision'],source_revision=c['source_revision'],producer_revision=revision,
                    ground_truth_used=False,manual_labels=False,quality_verified=False)
                save(folder/'bank.json',bank);banks.append(dict(role=role,bank=identity(folder/'bank.json'),
                    tracks=len(tracks),num_obj_dropped=dropped,status=bank['status']))
                fail_clip |= dropped > 0
            predictor.handle_request(dict(type='close_session',session_id=sid)); del hashes
            results.append(dict(episode_index=item['episode'],banks=banks,elapsed_seconds=time.monotonic()-clip_start,
                                status='fail_capacity' if fail_clip else 'complete_diagnostic_not_quality_pass'))
            print(json.dumps(dict(stage='sam31_progress',completed=len(results),total=len(selected),
                episode=item['episode'],status=results[-1]['status'],elapsed_seconds=time.monotonic()-start)),flush=True)
        verify_runtime_source(prep)
        require(inputs(h['episodes']) == selected and identity(weight,c['weight_bytes']) == prep['weight'],
                'Original input/checkpoint changed after SAM inference')
        budget(); report=dict(status='fail_capacity' if any(r['status']=='fail_capacity' for r in results)
            else 'complete_diagnostic_not_quality_pass',producer_revision=revision,episodes=results,
            elapsed_seconds=time.monotonic()-start,peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            input_receipt=identity(out/'sam-inputs.json'),image_id=prep['image_id'],
            model=c['model_repo'],model_revision=c['model_revision'],source_revision=c['source_revision'],
            torch=torch.__version__,use_fa3=False,compile=False,normalized_pil_loader=True,
            checkpoint_coverage=checkpoint_coverage,
            init_unsupported_argument_filtered=['offload_state_to_cpu'],full_original_frame_grid=True,
            explicit_native_final_batch=True,cached_outputs_evicted_after_delivery=True,
            source_rehashed_after=True,inputs_rehashed_after=True,weight_rehashed_after=True,
            baseline_modified=False,ground_truth_used=False,manual_labels=False,quality_verified=False,
            training_overlap_verified=False,challenge_overlap_verified=False)
        save(out/'sam-report.json',report)
    finally:
        for future in pending.values():future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        predictor.shutdown()
        if getattr(predictor,'bf16_context',None) is not None:
            predictor.bf16_context.__exit__(None,None,None);predictor.bf16_context=None
        tracker=getattr(predictor.model,'tracker',None)
        if getattr(tracker,'bf16_context',None) is not None:
            tracker.bf16_context.__exit__(None,None,None);tracker.bf16_context=None


def run(code, revision):
    require(sys.platform=='linux' and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01',
            'Azure VM01 owner runtime required')
    c,h=settings(code);binding=source(ROOT,code,revision,ENTRY,HELPERS)
    out=canonical(ROOT/'results'/('hybrid-pair-'+revision));require(not out.exists(),'Fresh frozen result namespace required')
    out.mkdir(mode=0o755);lock=ROOT/'jobs/.world-reward-h100.lock';start=time.monotonic();failure=None
    def interrupted(*_):raise TimeoutError('SAM runtime interrupted; original baseline retained')
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        prep=prepare(code,out,revision,binding)
        with lock.open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            name='wr-sam31-'+revision[:12]
            require(not control(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip(),'Owned native container occupied')
            command=['docker','run','--rm','--name',name,'--label','world_reward.sam31.owner='+revision,
                '--gpus','all','--network','none','--read-only','--user','0:0','--cap-drop','ALL',
                '--security-opt','no-new-privileges','--memory','64g','--cpus','16','--shm-size','2g',
                '--tmpfs','/tmp:rw,nosuid,size=2g']
            for p,ro in [(code.parent,True),(out,False),(ROOT/'results/input-manifest.json',True),
                         (ROOT/'data/track_1/meta',True)]+[(ROOT/f'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4',True) for ep in h['episodes']]:
                canonical(p);command+=['--mount',f'type=bind,src={p},dst={p}'+(',readonly' if ro else '')]
            command+=['--entrypoint','/usr/bin/env',prep['image_id'],'-i','PATH=/usr/local/bin:/usr/bin:/bin',
                'HOME=/tmp','PYTHONPATH=/opt/sam3:'+str(code/'src')+':'+str(code/'infra'),
                'PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','WANDB_MODE=disabled',
                'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','WR_CODE_REVISION='+revision,
                '/usr/local/bin/python','-B',str(code/'infra/sam31_runtime.py'),'--native',str(code),str(out)]
            try:
                with (out/'sam-native.log').open('xb') as log:
                    os.fchmod(log.fileno(),0o400)
                    r=subprocess.run(command,stdout=log,stderr=log,timeout=h['sam_budget_seconds']+60)
                require(r.returncode==0,'Native SAM candidate inference failed; partial banks preserved')
            finally:
                if control(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip():
                    owner=control(['docker','inspect',name,'--format','{{index .Config.Labels "world_reward.sam31.owner"}}']).decode().strip()
                    require(owner==revision,'Cannot clean foreign GPU container');control(['docker','rm','-f',name])
        require(source(ROOT,code,revision,ENTRY,HELPERS)==binding,'Original code changed after execution')
        receipt=strict((out/'sam-report.json').read_bytes())
        print(json.dumps(dict(status=receipt['status'],producer_revision=revision,elapsed_seconds=time.monotonic()-start,
            sam_elapsed_seconds=receipt['elapsed_seconds'],episodes=len(receipt['episodes']))),flush=True)
    except Exception as exc:
        failure=exc
        save(out/'sam-runtime-failure.json',dict(status='fail',phase='prepare_or_infer',error_type=type(exc).__name__,
            producer_revision=revision,elapsed_seconds=time.monotonic()-start,baseline_modified=False,
            ground_truth_used=False,quality_verified=False))
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)
    if failure:raise RuntimeError('SAM runtime failed; see Azure-only stage logs') from None


if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--native':native(Path(sys.argv[2]),Path(sys.argv[3]))
    else:run(Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'])
