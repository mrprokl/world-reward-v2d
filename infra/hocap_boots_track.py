"""Two real full-T Boots calls on a frozen public all-seed bank, not evaluation.

Historical bank source is hash-only on host. Native consumes its independently
pinned public receipt, ten opaque-to-semantics banks and original RGB only.
Coordinates, query births and pre-birth visibility are never rewritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time
import sys

import mediapipe_cpu_runtime_verify as rt
import hocap_amg_bank as bank
import robotap_boots_infer as boots
import tracker_noise_experiment as noise

ROOT = rt.ROOT
ENTRY = 'run_hocap_boots_track'
PROTOCOL = 'configs/hocap_boots_protocol_v1.json'
SCIENCE = 'configs/hocap_point_retention_protocol_v1.json'
PROTOCOL_PIN = dict(bytes=2095,sha256='f7a1ca0fb25b3543e126216f87866784906f8c835969e7d0e0114ff503215ca6')
HELPERS = ('infra/hocap_boots_track.py','infra/run_hocap_boots_track.sh',PROTOCOL,SCIENCE,
    'infra/mediapipe_cpu_runtime_verify.py','infra/hocap_amg_bank.py',bank.PROTOCOL,
    'infra/robotap_boots_infer.py','infra/robotap_boots_acquire.py','infra/tracker_noise_experiment.py',
    'configs/robotap_boots_inference_pins.json','configs/robotap_boots_protocol.json',
    'infra/bridge_frontend_bindings.py','infra/frontend_selected_assets.py',
    'infra/frontend_sam2_kernel_gate.py',bank.binding.CONFIG,bank.binding.selected.PINS)
BANK_KEYS = {'frame_index','anchor_frame_index','image_size','packed_masks','mask_ids','mask_sha256',
    'native_mask_indices','native_bbox_xywh','native_point_coords_xy','native_crop_box_xywh',
    'predicted_iou','stability_score','mask_area','query_points','query_offsets','query_owner_indices',
    'query_owner_ids','query_birth_frame_indices'}
require = rt.require


def check(deadline):
    if time.monotonic() >= deadline: raise TimeoutError('Inclusive full-bank Boots deadline exhausted')


def same_facts(value, facts):
    require(all(type(value.get(k)) is type(v) and value[k] == v for k,v in facts.items()),
            'Exact complete public producer facts required')


def protocol(code):
    p = rt.pinned(code/PROTOCOL,PROTOCOL_PIN,16384)
    require(p['image_id'] == boots.IMAGE and p['budget_seconds'] == 1800 and p['cleanup_grace_seconds'] == 60,
            'Frozen actual Boots runtime and resource policy required')
    rt.pinned(code/SCIENCE,p['retention_protocol'],16384)
    rt.pinned(code/bank.PROTOCOL,p['bank_protocol'],16384)
    return p


def source(code, revision):
    require(Path(__file__).resolve() == code/'infra/hocap_boots_track.py','Actual new driver source required')
    return rt.source(ROOT,code,revision,ENTRY,HELPERS)


def authenticate_bank(p, pin, *, historical):
    folder = rt.canonical(Path(p['bank']))
    r = rt.pinned(folder/'report.json',pin,p['maximum_report_bytes'])
    same_facts(r,dict(schema='world_reward.hocap_amg_bank.v1',stage='hocap_native_all_mask_bank',
        status='pass',phase='complete',private_labels_read=False,calibration_read=False,ground_truth_used=False,
        opaque_metadata_read=False,challenge_inputs_used=False,oracle_modes=[],network='none',
        quality_verified=False,adoption=False,boots_executed=False,association_executed=False,
        all_native_returned_masks=True,model_loads=1,rgb_decodes=1493,amg_calls=10,amg_attempts=10,
        anchors_retained=10,all_original_jpegs_decoded_before_model=True,
        original_source_public_models_rehashed_after=True,public_manifest_identity=p['public_manifest']))
    rev = r.get('producer_revision'); require(type(rev) is str and re.fullmatch('[0-9a-f]{40}',rev), 'Bank producer revision required')
    host_pin = p['bank_host_report']; host = rt.pinned(folder/'host.json',host_pin,p['maximum_report_bytes'])
    same_facts(host,dict(stage='hocap_amg_host',status='pass',producer_revision=rev,
        image_id=r['image_id'],native_exit_status=0,owned_cleanup_verified=True,
        source_public_rehashed_after=True,native_report_identity=pin,opaque_input_receipt_mounted=False,
        private_labels_read=False,quality_verified=False,sanitized_input_proof_identity=r['sanitized_input_proof_identity']))
    safe_pin = r['sanitized_input_proof_identity']
    safe = rt.pinned(folder/'sanitized_input_proof.json',safe_pin,p['maximum_report_bytes'])
    bank.validate_sanitized(safe,safe['input_report_identity'],p)
    require(safe['public_manifest_identity'] == p['public_manifest'] and safe['frames'] == 1493
        and bank.json_digest(safe['source_binding']) == r['source_binding_sha256']
        and safe['frontend_proof_sha256'] == r['frontend_proof_sha256'], 'Bank public source proof differs')
    if historical:
        old = ROOT/'jobs'/rev/bank.ENTRY/'code'
        require(bank.binding.identity(old/'infra/frontend_sam2_kernel_gate.py',200000) == bank.binding.KERNEL_SOURCE_PIN,
                'Original immutable bank digest helper differs')
        original = bank.binding.kernel_helper().closure(old,rev,bank.ENTRY,bank.HELPERS)
        require(original == safe['source_binding'] and rt.identity(old/bank.PROTOCOL,16384) == p['bank_protocol'],
                'Complete historical bank source/markers/protocol differ')
        current=Path(bank.__file__).resolve().parent.parent
        require(all(rt.identity(current/n,2_000_000)==identity for n,identity in original['helpers'].items()
                    if n.endswith('.py')), 'Reused bank numerical/source helpers differ from actual producer')
    rows = r.get('banks'); require(type(rows) is list and len(rows) == 10,'Complete ten-bank inventory required')
    wanted = set(); total = 0
    for clip, frames in zip(p['clips'],p['frames']):
        for k in range(5):
            row = rows[len(wanted)]; anchor = k*(frames-1)//4
            name = f'bank/{clip}_anchor_{anchor:06d}.npz'
            same_facts(row,dict(clip=clip,frames=frames,anchor_frame_index=anchor,file=name,
                original_seed_ids_retained=True,physical_identities_certified=False))
            require(type(row['width']) is type(row['height']) is int and min(row['width'],row['height']) > 0
                and row['width']*row['height'] <= p['maximum_image_pixels']
                and r['image_size'] == [row['height'],row['width']], 'Constant original bank image grid required')
            require(rt.identity(folder/name,p['maximum_bank_bytes']) == {q:row[q] for q in ('bytes','sha256')},
                    'Frozen bank bytes differ'); wanted.add(name); total += row['bytes']
    require(total == r['bank_bytes'] <= p['maximum_bank_bytes']
        and {str(v.relative_to(folder/'bank')) for v in (folder/'bank').iterdir()} == {n[5:] for n in wanted},
        'Exact bank inventory and total bytes required')
    expected={'report.json','host.json','sanitized_input_proof.json','bank'}
    if historical: expected.add('.container.cid')
    require({v.name for v in folder.iterdir()} == expected,
            'No unknown/failed bank files accepted')
    return dict(report=pin,host=host_pin,sanitized=safe_pin,source_binding=safe['source_binding'],
                banks=rows,image_size=r['image_size'],producer_revision=rev)


def bank_arrays(path, row, p):
    """Validate every original seed and regenerate its fixed grid, one mask at a time."""
    import numpy as np
    with np.load(path,allow_pickle=False) as z:
        require(len(z.files) == len(BANK_KEYS) and set(z.files) == BANK_KEYS,'Exact public bank arrays required')
        a = {n:z[n] for n in z.files}
    h,w = row['height'],row['width']; m,q = row['native_masks'],row['queries']; t=row['frames']; anchor=row['anchor_frame_index']
    require(type(m) is type(q) is int and 0 <= m <= p['maximum_masks_per_anchor'] and 0 <= q <= m*16,
            'Finite bounded full seed/query counts required')
    specifications = {'frame_index':(np.int64,(t,)), 'anchor_frame_index':(np.int64,()), 'image_size':(np.int64,(2,)),
        'packed_masks':(np.uint8,(m,(h*w+7)//8)), 'mask_ids':(np.dtype('U128'),(m,)),
        'mask_sha256':(np.dtype('U64'),(m,)), 'native_mask_indices':(np.int64,(m,)),
        'native_bbox_xywh':(np.float64,(m,4)), 'native_point_coords_xy':(np.float64,(m,2)),
        'native_crop_box_xywh':(np.float64,(m,4)), 'predicted_iou':(np.float64,(m,)),
        'stability_score':(np.float64,(m,)), 'mask_area':(np.int64,(m,)),
        'query_points':(np.float64,(q,3)), 'query_offsets':(np.int64,(m+1,)),
        'query_owner_indices':(np.int64,(q,)), 'query_owner_ids':(np.dtype('U128'),(q,)),
        'query_birth_frame_indices':(np.int64,(q,))}
    for n,(dtype,shape) in specifications.items():
        require(a[n].dtype == dtype and a[n].shape == shape and not np.ma.isMaskedArray(a[n]), 'Exact bank dtype/shape required')
        if a[n].dtype.kind == 'f': require(np.isfinite(a[n]).all(),'Nonfinite public bank evidence')
    offsets = a['query_offsets']
    require(np.array_equal(a['frame_index'],np.arange(t,dtype=np.int64)) and a['anchor_frame_index'].item() == anchor
        and a['image_size'].tolist() == [h,w] and np.array_equal(a['native_mask_indices'],np.arange(m,dtype=np.int64))
        and offsets[0] == 0 and offsets[-1] == q and np.all(np.diff(offsets) >= 0)
        and np.array_equal(a['query_owner_indices'],np.repeat(np.arange(m,dtype=np.int64),np.diff(offsets)))
        and np.array_equal(a['query_owner_ids'],a['mask_ids'][a['query_owner_indices']])
        and np.all(a['query_birth_frame_indices'] == anchor), 'Original seed order/owners/births/timeline differ')
    require(np.all((a['predicted_iou'] >= 0)&(a['predicted_iou'] <= 1))
        and np.all((a['stability_score'] >= 0)&(a['stability_score'] <= 1))
        and np.array_equal(a['native_crop_box_xywh'],np.tile([0,0,w,h],(m,1))), 'Native scores/crop provenance differs')
    for i in range(m):
        bits = np.unpackbits(a['packed_masks'][i],bitorder='little'); require(not bits[h*w:].any(),'Nonzero packed mask padding')
        mask = bits[:h*w].reshape(h,w).astype(bool); digest=hashlib.sha256(mask.tobytes()).hexdigest()
        require(a['mask_sha256'][i] == digest and a['mask_ids'][i] == f"{row['clip']}:{anchor:06d}:{i:06d}:{digest}"
            and a['mask_area'][i] == int(mask.sum()),'Original mask bytes/hash/ID/area differ')
        x0,y0,bw,bh = a['native_bbox_xywh'][i]; require(min(bw,bh) >= 0 and np.isfinite([x0+bw,y0+bh]).all(),'Native XYWH endpoints required')
        x0,x1=np.clip([x0,x0+bw],0,w); y0,y1=np.clip([y0,y0+bh],0,h); points=np.empty((0,3),np.float64)
        if x1>x0 and y1>y0:
            xx=np.floor(x0+(np.arange(4)+.5)*(x1-x0)/4).astype(np.int64); yy=np.floor(y0+(np.arange(4)+.5)*(y1-y0)/4).astype(np.int64)
            x,y=np.meshgrid(xx,yy); pixels=np.unique(np.column_stack((y.ravel(),x.ravel())),axis=0)
            pixels=pixels[(pixels[:,0]>=0)&(pixels[:,0]<h)&(pixels[:,1]>=0)&(pixels[:,1]<w)]
            pixels=pixels[mask[pixels[:,0],pixels[:,1]]]
            points=np.column_stack((np.full(len(pixels),anchor),pixels.astype(np.float64)+.5))
        require(np.array_equal(points,a['query_points'][offsets[i]:offsets[i+1]]),'Fixed inside-mask queries changed; no refill')
    require(row['zero_query_seeds'] == int((np.diff(offsets)==0).sum())
        and row['duplicate_mask_bytes'] == m-len(set(a['mask_sha256'].tolist()))
        and row['packed_mask_bytes'] == a['packed_masks'].nbytes,'Full bank diagnostic counts differ')
    return a


def prepare_queries(rows, p, deadline):
    import numpy as np
    result=[]
    for clip,frames in zip(p['clips'],p['frames']):
        banks=[]
        for row in [r for r in rows if r['clip']==clip]:
            check(deadline); banks.append(bank_arrays(Path(p['bank'])/row['file'],row,p))
        require(len(banks)==5,'Five original seed banks per clip required')
        q=np.concatenate([a['query_points'] for a in banks]); seeds=np.concatenate([a['mask_ids'] for a in banks])
        require(len(q)>0,'NO_OBSERVATIONS: full clip has no fixed automatic queries')
        require(len(q)<=p['maximum_queries_per_clip'],'Whole-bank query resource cap exceeded; no truncation')
        seed_offsets=np.r_[0,np.cumsum([len(a['mask_ids']) for a in banks])].astype(np.int64)
        offsets=np.r_[0,np.cumsum(np.concatenate([np.diff(a['query_offsets']) for a in banks]))].astype(np.int64)
        owners=np.repeat(np.arange(len(seeds),dtype=np.int64),np.diff(offsets))
        metadata=dict(query_owner_ids=seeds[owners],query_owner_indices=owners,
            query_birth_frame_indices=q[:,0].astype(np.int64),seed_ids=seeds,
            seed_birth_frame_indices=np.concatenate([np.full(len(a['mask_ids']),a['anchor_frame_index'],np.int64) for a in banks]),
            seed_mask_sha256=np.concatenate([a['mask_sha256'] for a in banks]),
            seed_native_mask_indices=np.concatenate([a['native_mask_indices'] for a in banks]),
            seed_query_offsets=offsets,seed_anchor_offsets=seed_offsets,
            anchor_frame_indices=np.array([a['anchor_frame_index'] for a in banks],np.int64),image_size=banks[0]['image_size'].copy())
        # Original outputs plus metadata: conservative full array bytes, zip/header reserve.
        predicted=len(q)*frames*25+q.nbytes+len(q)*8+frames*8+sum(a.nbytes for a in metadata.values())+65536
        require(predicted<=p['maximum_prediction_bytes'],'Full prediction byte cap exceeded before model')
        result.append(dict(clip=clip,frames=frames,height=int(metadata['image_size'][0]),width=int(metadata['image_size'][1]),
            query_points=q,point_indices=np.arange(len(q),dtype=np.int64),query_count=len(q),metadata=metadata))
    return result


def decode_all(clips, prepared, deadline, report):
    import numpy as np
    from PIL import Image
    for clip,target in zip(clips,prepared):
        require(clip['clip']==target['clip'] and clip['num_frames']==target['frames'],'Full original clips differ')
        video=np.empty((target['frames'],target['height'],target['width'],3),np.uint8)
        for t,row in enumerate(clip['records']):
            check(deadline)
            with Image.open(row['path']) as image:
                require(image.format=='JPEG' and image.mode=='RGB' and image.size==(target['width'],target['height']),
                        'Original JPEG RGB and constant bank grid required; no conversions')
                image.load(); video[t]=np.asarray(image)
            report['rgb_decodes']+=1
        target['video']=video


def track_all(prepared, predict, save, report, p, deadline):
    require(len(prepared)==2 and all('video' in c for c in prepared),'All original clips decoded before model calls')
    report['predictions']=[]
    for clip in prepared:
        check(deadline); report.update(phase='native_full_t',current_clip=clip['clip'])
        data={k:clip[k] for k in ('video','query_points','point_indices')}
        arrays=predict(data,report); check(deadline)
        row={k:clip[k] for k in ('frames','height','width','query_count')}; row['point_indices']=clip['point_indices'].tolist()
        boots.validate_prediction(arrays,row,clip['query_points'],clip['point_indices'])
        del arrays['static_tracks'],arrays['static_visible']
        arrays.update(clip['metadata']); record=save(clip['clip'],arrays)
        report['predictions'].append(dict(clip=clip['clip'],frames=clip['frames'],queries=clip['query_count'],
            seeds=len(clip['metadata']['seed_ids']),zero_query_seeds=int((clip['metadata']['seed_query_offsets'][1:]
                ==clip['metadata']['seed_query_offsets'][:-1]).sum()),**record))
        report['native_calls_completed']+=1; del clip['video']; check(deadline)
    require(report['native_calls_attempted']==report['native_calls_returned']==report['native_calls_completed']==2,
            'Exactly two complete real full-T native calls required')


def write_json(path, value, *, deadline=None, seal_parent=False):
    raw=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    require(len(raw)<=2<<20,'Bounded scalar receipt required')
    with path.open('xb') as stream:
        try:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(),0o444)
            if seal_parent: path.parent.chmod(0o555)
            if deadline is not None: check(deadline)
        except Exception:
            value={**value,'status':'fail','phase':'receipt_sealing','error_type':'ReceiptDeadlineOrWriteFailure'}
            stream.seek(0); stream.truncate(); stream.write((json.dumps(value,sort_keys=True)+'\n').encode())
            stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(),0o444); raise
    return rt.identity(path,2<<20)


def preflight(code, revision, p, pin, deadline, *, historical):
    current=source(code,revision); runtime_pins,runtime=noise.runtime_evidence(code)
    original=authenticate_bank(p,pin,historical=historical)
    clips,_=bank.public_inputs(Path(p['inputs']),p['public_manifest'],p,deadline)
    require([c['num_frames'] for c in clips]==p['frames'],'Pinned original full frame counts required')
    check(deadline)
    return dict(source_binding=current,runtime=runtime,bank=original,protocol_identity=PROTOCOL_PIN,
                retention_protocol_identity=p['retention_protocol'],public_manifest_identity=p['public_manifest']),clips,runtime_pins


def native(args, code, revision, p):
    started=time.monotonic(); deadline=float(os.environ['WR_BOOTS_DEADLINE']); out=rt.canonical(Path(p['output']))
    r=dict(schema='world_reward.hocap_boots_tracks.v1',stage='hocap_native_all_seed_full_t_tracks',status='fail',phase='authentication',
        producer_revision=revision,image_id=p['image_id'],proof_identity=args.proof_pin,budget_seconds=p['budget_seconds'],
        native_calls_attempted=0,native_calls_returned=0,native_calls_completed=0,model_loads=0,rgb_decodes=0,
        private_labels_read=False,opaque_metadata_read=False,calibration_read=False,ground_truth_used=False,
        challenge_inputs_used=False,oracle_modes=[],quality_verified=False,association_verified=False,
        adoption=False,training_overlap_verified=False,static_diagnostic_arrays_published=False,
        all_source_public_bank_assets_rehashed_after=False,predictions=[])
    failure=None; before=None
    try:
        require(os.environ.get('WR_IMAGE_ID')==p['image_id'] and os.geteuid()==0
            and {v.name for v in Path('/sys/class/net').iterdir()}=={'lo'},'Offline original GPU container required')
        require({v.name for v in out.iterdir()}=={'proof.json','.container.cid','native.log'},'Fresh owned native output required')
        require(not (Path(p['inputs']).parent/'report.json').exists() and not (Path(p['inputs']).parent/'metadata_private').exists()
            and not (ROOT/'data').exists() and not (ROOT/'vendor').exists(),'No private/challenge mounts permitted')
        safe=rt.pinned(out/'proof.json',args.proof_pin,p['maximum_report_bytes'])
        before,clips,_=preflight(code,revision,p,args.bank_pin,deadline,historical=False)
        require(before==safe,'Authenticated host/native source/runtime/public proof differs')
        prepared=prepare_queries(before['bank']['banks'],p,deadline)
        r['phase']='all_original_jpeg_decode'; decode_all(clips,prepared,deadline,r)
        require(r['rgb_decodes']==1493,'Every original RGB decoded before model required')
        r['all_original_jpegs_decoded_before_model']=True; r['phase']='native_model_load'; check(deadline)
        import numpy as np
        import torch
        import random
        require(torch.__version__=='2.5.1+cu124' and np.__version__=='1.26.3' and torch.cuda.is_available()
            and 'H100' in torch.cuda.get_device_name(),'Original qualified native CUDA runtime required')
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        require(not torch.are_deterministic_algorithms_enabled(),'No new strict determinism policy')
        assets=ROOT/noise.ORIGINAL/'assets'; native_protocol=noise.acquisition.read_protocol(code/'configs/robotap_boots_protocol.json')
        tapir,utils=boots.native_modules(assets/'tapnet_source'); model=boots.load_model(torch,tapir,assets/native_protocol['checkpoint']['file'])
        r['model_loads']=1; check(deadline)
        def save(clip,arrays):
            path=out/(clip+'_tracks.npz')
            require(sum(v.nbytes for v in arrays.values())+65536<=p['maximum_prediction_bytes'],'Complete output cap exceeded')
            with path.open('xb') as stream:
                np.savez(stream,**arrays); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(),0o444)
            pin=rt.identity(path,p['maximum_prediction_bytes'])
            with np.load(path,allow_pickle=False) as z:
                require(len(z.files)==len(arrays) and set(z.files)==set(arrays)
                    and all(z[n].dtype==v.dtype and np.array_equal(z[n],v) for n,v in arrays.items()),'Lossless all-query output roundtrip required')
            return dict(file=path.name,**pin)
        track_all(prepared,lambda d,report:boots.native_prediction(torch,utils,model,d,report),save,r,p,deadline)
        torch.cuda.synchronize(); del model
        r.update(status='pass',phase='complete',source_binding_sha256=bank.json_digest(before['source_binding']),
            bank_report_identity=args.bank_pin,public_manifest_identity=p['public_manifest'],native_parameters=p['native'])
    except Exception as error:
        failure=error; r.update(status='fail',error_type=type(error).__name__,error=str(error)[:400])
    finally:
        # FAIL-only grace enables evidence after compute timeout; never a late PASS.
        signal.alarm(p['cleanup_grace_seconds'])
        try:
            if before is not None:
                after,_,_=preflight(code,revision,p,args.bank_pin,time.monotonic()+p['cleanup_grace_seconds'],historical=False)
                require(after==before,'Native source/public/bank/assets changed')
                require(rt.identity(out/'proof.json',p['maximum_report_bytes'])==args.proof_pin,'Native proof changed')
                r['all_source_public_bank_assets_rehashed_after']=True
            if failure is None: check(deadline)
        except Exception as error:
            failure=failure or error; r.update(status='fail',post_error_type=type(error).__name__)
        r['elapsed_seconds']=time.monotonic()-started
        write_json(out/'report.json',r,deadline=None if failure else deadline); signal.alarm(0)
    require(failure is None,'Native Boots failed; bounded immutable partial report retained')
    return r


def command(args, seconds=5):
    result=subprocess.run(args,capture_output=True,timeout=seconds,check=False)
    require(result.returncode==0 and len(result.stdout)<=32768 and len(result.stderr)<=32768,'Bounded host control failed')
    return result.stdout.decode().strip()


def cleanup(cidfile, name, revision, image):
    s=rt.canonical(cidfile).lstat(); require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==0 and s.st_size<=65,'Owned CID required')
    raw=cidfile.read_bytes(); require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Exact CID bytes required'); cid=raw.decode().rstrip('\n')
    result=subprocess.run(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],capture_output=True,timeout=5,check=False)
    if result.returncode==0:
        require(result.stderr==b'' and result.stdout.decode().rstrip('\n')==image+'|/'+name+'|'+ENTRY+'|'+revision,'Foreign container not removed')
        require(command(['docker','rm','-f',cid],15)==cid,'Owned removal failed')
    else:
        require(result.returncode==1 and result.stdout==b'' and result.stderr in tuple(
            prefix+f'Error: No such {kind}: {cid}\n'.encode() for prefix in (b'',b'\n') for kind in ('object','container')),
            'Daemon/timeout/ambiguous absence is not cleanup')
    require(command(['docker','ps','-aq','--no-trunc','--filter','id='+cid])==''
        and command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'])=='','Independent owned CID/name absence required')
    cidfile.chmod(0o444)


def mounts(code, p, proof, runtime_pins):
    assets=ROOT/noise.ORIGINAL/'assets'
    paths=[code.parent,Path(p['inputs']),Path(p['bank'])/'report.json',Path(p['bank'])/'host.json',
        Path(p['bank'])/'sanitized_input_proof.json',ROOT/runtime_pins['runtime']['report_path']]
    paths.append(Path(p['bank'])/'bank')
    paths += [assets/'tapnet_source'/n for n in proof['runtime']['native_sources']]
    protocol=noise.acquisition.read_protocol(code/'configs/robotap_boots_protocol.json')
    paths.append(assets/protocol['checkpoint']['file'])
    require(len(paths)==len(set(paths)),'Duplicate public/runtime mount')
    for path in paths:
        rt.canonical(path); require(path.exists() and ',' not in str(path) and '\n' not in str(path)
            and not any(n in path.parts for n in ('metadata_private','eval_private','data','vendor')),'Public/runtime source-only mounts required')
    # Only the exact public bank directory, not its CID or broader validation.
    return paths


def dispatch(args, code, revision, p):
    import fcntl
    started=time.monotonic(); deadline=started+p['budget_seconds']
    before,_,runtime_pins=preflight(code,revision,p,args.bank_pin,deadline,historical=True)
    out=rt.canonical(Path(p['output'])); require(not out.exists() and out.parent.is_dir(),'Fresh fixed output namespace required')
    name='world-reward-hocap-boots-'+revision[:12]; require(not command(['docker','ps','-aq','--filter','name=^/'+name+'$']),'Name already occupied')
    require(command(['docker','image','inspect',p['image_id'],'--format','{{.Id}}'])==p['image_id'],'Actual exact qualified image required')
    lock=rt.canonical(ROOT/'jobs/.world-reward-h100.lock'); st=lock.lstat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink==1,'Existing cooperative lock required')
    fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW); owned=False; failure=None; safe_pin=None
    host=dict(stage='hocap_boots_host',status='fail',producer_revision=revision,image_id=p['image_id'],
        native_exit_status=None,owned_cleanup_verified=False,source_public_bank_assets_rehashed_after=False,
        private_labels_read=False,quality_verified=False,adoption=False)
    try:
        require((os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(st.st_dev,st.st_ino),'Lock inode changed')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']),'GPU must be idle after lock')
        require(preflight(code,revision,p,args.bank_pin,deadline,historical=True)[0]==before,'Pre-model proof changed')
        paths=mounts(code,p,before,runtime_pins); out.mkdir(mode=0o700); owned=True
        safe_pin=write_json(out/'proof.json',before,deadline=deadline)
        cmd=['docker','run','--rm','--name',name,'--cidfile',str(out/'.container.cid'),
            '--label','world-reward.job='+ENTRY,'--label','world-reward.revision='+revision,
            '--gpus','all','--network','none','--user','0:0','--memory','64g','--cpus','4','--read-only',
            '--cap-drop','ALL','--security-opt','no-new-privileges','--tmpfs','/tmp:rw,noexec,nosuid,size=512m',
            '--entrypoint','/usr/bin/env']
        for path in paths: cmd+=['--mount',f'type=bind,src={path},dst={path},readonly']
        cmd+=['--mount',f'type=bind,src={out},dst={out}',p['image_id'],'-i',
            'PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','PYTHONPATH='+str(code/'infra'),
            'WR_ROOT='+str(ROOT),'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_IMAGE_ID='+p['image_id'],
            'WR_BOOTS_DEADLINE='+format(deadline,'.17g'),'PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1',
            'TRANSFORMERS_OFFLINE=1','WANDB_MODE=disabled','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',
            '/opt/conda/bin/python','-B',str(code/'infra/hocap_boots_track.py'),'--run',
            '--bank-report-bytes',str(args.bank_pin['bytes']),'--bank-report-sha256',args.bank_pin['sha256'],
            '--proof-bytes',str(safe_pin['bytes']),'--proof-sha256',safe_pin['sha256']]
        with (out/'native.log').open('xb') as log:
            os.fchmod(log.fileno(),0o400)
            result=subprocess.run(cmd,stdout=log,stderr=log,timeout=max(.1,deadline-time.monotonic()),check=False)
        host['native_exit_status']=result.returncode; require(result.returncode==0,'Native all-query Boots failed')
        r=rt.strict((out/'report.json').read_bytes())
        same_facts(r,dict(status='pass',phase='complete',producer_revision=revision,proof_identity=safe_pin,
            model_loads=1,rgb_decodes=1493,native_calls_attempted=2,native_calls_returned=2,native_calls_completed=2,
            all_original_jpegs_decoded_before_model=True,all_source_public_bank_assets_rehashed_after=True))
        require(len(r['predictions'])==2,'Both complete prediction receipts required')
        for row,clip,frames in zip(r['predictions'],p['clips'],p['frames']):
            require(row['clip']==clip and row['frames']==frames and 0<row['queries']<=p['maximum_queries_per_clip']
                and row['file']==clip+'_tracks.npz'
                and rt.identity(out/row['file'],p['maximum_prediction_bytes'])=={k:row[k] for k in ('bytes','sha256')},'Complete output bytes differ')
        require({v.name for v in out.iterdir()}=={'proof.json','.container.cid','native.log','report.json',
            *(clip+'_tracks.npz' for clip in p['clips'])},'Exclusive complete all-query output inventory required')
        host['native_report_identity']=rt.identity(out/'report.json',p['maximum_report_bytes']); check(deadline)
    except Exception as error:
        failure=error; host.update(error_type=type(error).__name__,error=str(error)[:400])
    finally:
        signal.alarm(p['cleanup_grace_seconds'])
        try:
            if owned:
                cleanup(out/'.container.cid',name,revision,p['image_id']); host['owned_cleanup_verified']=True
                require(rt.identity(out/'proof.json',p['maximum_report_bytes'])==safe_pin,'Host proof changed')
                if failure is None:
                    require(rt.identity(out/'report.json',p['maximum_report_bytes'])==host['native_report_identity'],
                            'Native sealed report changed')
                    for row in r['predictions']:
                        require(rt.identity(out/row['file'],p['maximum_prediction_bytes'])=={k:row[k] for k in ('bytes','sha256')},
                                'Sealed fullT predictions changed after cleanup')
            require(preflight(code,revision,p,args.bank_pin,time.monotonic()+p['cleanup_grace_seconds'],historical=True)[0]==before
                and (lock.lstat().st_dev,lock.lstat().st_ino)==(st.st_dev,st.st_ino),'Full post-run source/public/bank/runtime/lock differs')
            require(not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']),'GPU cleanup incomplete')
            host['source_public_bank_assets_rehashed_after']=True
            if failure is None: check(deadline)
        except Exception as error:
            failure=failure or error; host.update(post_error_type=type(error).__name__)
        finally:
            os.close(fd)
            if owned:
                host.update(status='fail' if failure else 'pass',elapsed_seconds=time.monotonic()-started)
                if not failure:
                    try: (out/'native.log').unlink()
                    except Exception as error:
                        failure=error; host.update(status='fail',post_error_type=type(error).__name__)
                write_json(out/'host.json',host,deadline=None if failure else deadline,seal_parent=True)
            signal.alarm(0)
    require(failure is None,'Boots host lifecycle failed; immutable partial outputs retained')
    return host


def main(argv=None):
    parser=argparse.ArgumentParser(); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dispatch',action='store_true'); mode.add_argument('--run',action='store_true')
    parser.add_argument('--bank-report-bytes',type=int,required=True); parser.add_argument('--bank-report-sha256',required=True)
    parser.add_argument('--proof-bytes',type=int); parser.add_argument('--proof-sha256'); args=parser.parse_args(argv)
    code=Path(os.environ['WR_CODE']); revision=os.environ['WR_CODE_REVISION']
    require(sys.platform=='linux' and os.geteuid()==0 and Path(os.environ['WR_ROOT'])==ROOT,'Owned immutable Linux runtime required')
    p=protocol(code); args.bank_pin=dict(bytes=args.bank_report_bytes,sha256=args.bank_report_sha256)
    boots.pin(args.bank_pin)
    if args.dispatch:
        require(args.proof_bytes is args.proof_sha256 is None and os.uname().nodename=='world-reward-ncc-h100-02','Original VM02 dispatch required')
    else:
        args.proof_pin=dict(bytes=args.proof_bytes,sha256=args.proof_sha256); boots.pin(args.proof_pin)
    def expired(*_): raise TimeoutError('Inclusive Boots compute/cleanup deadline exhausted')
    signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,expired)
    signal.alarm(p['budget_seconds'] if args.dispatch else max(1,math.ceil(float(os.environ['WR_BOOTS_DEADLINE'])-time.monotonic())))
    try: result=dispatch(args,code,revision,p) if args.dispatch else native(args,code,revision,p)
    finally: signal.alarm(0)
    print(json.dumps(dict(stage=result['stage'],status=result['status'],quality_verified=False)),flush=True)


if __name__=='__main__':
    try: main()
    except Exception as error:
        print(json.dumps(dict(stage='hocap_boots_tracks',status='fail',error_type=type(error).__name__)),flush=True)
        raise SystemExit(1) from None
