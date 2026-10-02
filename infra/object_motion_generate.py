"""Three RGB-derived anchor meshes; no truth, metric correction or shape fit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np
import object_synthetic_generate as native
from world_reward.data import sha256

STAGE='public_object_motion_fixed_anchor_generation'


def load_anchors(root):
    base=root/'validation/object_motion_v1/observations';path=base/'report.json';digest=sha256(path)
    native.require_hash(path,digest);r=json.loads(path.read_text())
    expected={'schema':'world-reward-object-motion-observations-v1','stage':'public_object_motion_rgb_automatic_observations',
              'status':'pass','object_cases':3,'frames_per_object':8,'outputs_completed':24,'private_truth_read':False,'challenge_inputs_used':False}
    if any(type(r.get(k)) is not type(v) or r[k]!=v for k,v in expected.items()):raise ValueError('Require actual public motion observations')
    records=r.get('outputs',[])
    if [(x.get('object_index'),x.get('frame_index')) for x in records]!=[(o,f) for o in range(3) for f in range(8)]:raise ValueError('Original motion frame order differs')
    anchors=[]
    for obj in range(3):
        record=records[obj*8];p=base/f'object_{obj:02d}_frame_000.npz'
        if record['file']!=p.name:raise ValueError('Anchor public filename differs')
        native.require_hash(p,record['sha256'])
        with np.load(p,allow_pickle=False) as d:
            arrays=native.observation_arrays(d,obj,0)
            for key in d.files:
                a=d[key];proof=record.get('arrays',{}).get(key,{})
                if proof!={'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest(),'dtype':str(a.dtype),'shape':list(a.shape)}:raise ValueError('Decoded anchor public array differs')
        anchors.append(arrays)
    return anchors,digest


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:raise RuntimeError('Require isolated remote GPU')
    root=Path(os.environ['WR_ROOT']);out=root/'validation/object_motion_v1/anchors';path=out/'report.json'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()):raise FileExistsError('Require new reserved anchor output')
    revision,image=os.environ['WR_CODE_REVISION'],os.environ['WR_IMAGE_ID']
    if not re.fullmatch('[0-9a-f]{40}',revision) or not re.fullmatch('sha256:[0-9a-f]{64}',image):raise ValueError('Immutable source/image required')
    report={'stage':STAGE,'status':'fail','code_revision':revision,'image_id':image,'script_sha256':sha256(Path(__file__)),
            'private_truth_read':False,'challenge_inputs_used':False,'adoption_performed':False,'accuracy_verified':False,
            'generation_calls':3,'budget_seconds':240,'geometry_metric_accuracy_verified':False,'proposals':[]}
    started=time.perf_counter();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Frozen240s anchor generation deadline')));signal.alarm(240)
    try:
        import torch
        anchors,digest=load_anchors(root);report['observation_report_sha256']=digest
        pipe=native.build_pipeline(root,report,torch)
        settings=dict(seed=42,stage1_inference_steps=50,stage2_inference_steps=25,decode_formats=['gaussian','mesh'],
            use_stage1_distillation=False,use_stage2_distillation=False,stage1_only=False,with_mesh_postprocess=False,
            with_texture_baking=False,use_vertex_color=True,with_layout_postprocess=False)
        report['configuration']=settings
        for obj,(rgb,mask,pointmap) in enumerate(anchors):
            report['active_object']=obj;native.persist(path,report);native.full.ss._seed(torch)
            with torch.no_grad():result=pipe.run(rgb,mask,pointmap=torch.from_numpy(pointmap),**settings)
            proof=native.export_actual(result,out,f'object_{obj:02d}_anchor',torch)
            report['proposals'].append({'object_index':obj,**proof});del result
            torch.cuda.synchronize();native.persist(path,report)
            if torch.cuda.max_memory_allocated()>80*1024**3:raise RuntimeError('Frozen80GiB GPU allocation exceeded')
        native.require_hash(root/'validation/object_motion_v1/observations/report.json',digest)
        report.update(status='pass',actual_raw_decoder_parity_verified=True,native_camera_transform_parity_verified=True,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:report.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:signal.alarm(0);report['elapsed_seconds']=time.perf_counter()-started;native.persist(path,report)
    print(json.dumps({k:report[k] for k in ('stage','status','elapsed_seconds')}))

if __name__=='__main__':main()
