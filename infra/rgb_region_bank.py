"""Reusable RGB-only SAM2.1 region bank; no object/owner or target selection.

Host authenticates the full dispatch closure and qualified original runtime.
Native sees only explicit helper files, model proofs, public RGB and fresh output,
never the renderer, scene recipe, split, labels or the full current source tree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import warnings

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import bridge_frontend_bindings as binding
import hocap_amg_bank as mechanics
import mediapipe_cpu_runtime_verify as rt

ROOT = binding.ROOT
ENTRY = 'run_rgb_region_bank'
CONFIG = 'configs/rgb_region_bank_v1.json'
NATIVE_FILES = ('infra/rgb_region_bank.py', 'infra/hocap_amg_bank.py',
    'infra/bridge_frontend_bindings.py', 'infra/frontend_selected_assets.py',
    'infra/frontend_sam2_kernel_gate.py', 'infra/mediapipe_cpu_runtime_verify.py',
    CONFIG, binding.CONFIG, binding.selected.PINS)
HELPERS = (*NATIVE_FILES, 'infra/run_rgb_region_bank.sh')
INPUTS = (ROOT/'validation/proposal_stress_v1/inputs',
          Path('/srv/world-reward-data/proposal_external_rgb_v1/inputs'))


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    rt.write(path, raw, 0o400)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def configuration(code):
    p = rt.strict((code/CONFIG).read_bytes())
    rt.require(p['schema'] == 'world_reward.rgb_region_bank.v1'
        and p['image_id'] == binding.IMAGE and p['all_native_returned_masks'] is True
        and p['quality_verified'] is p['adoption'] is False, 'Frozen native region protocol required')
    # Original author defaults, not selected on either new cohort.
    expected = dict(points_per_side=32, points_per_batch=64, pred_iou_thresh=.8,
        stability_score_thresh=.95, stability_score_offset=1., mask_threshold=0.,
        box_nms_thresh=.7, crop_n_layers=0, crop_nms_thresh=.7,
        crop_overlap_ratio=512/1500, crop_n_points_downscale_factor=1,
        min_mask_region_area=0, output_mode='binary_mask', use_m2m=False, multimask_output=True)
    rt.require(p['amg'] == expected and p['build_apply_postprocessing'] is True
        and p['packed_mask_bitorder'] == 'little'
        and p['budget_seconds'] == 1200 and p['maximum_images'] == 64
        and p['maximum_masks'] == 3072 and p['maximum_bank_bytes'] == 1 << 30,
        'Original complete AMG defaults and fixed nontruncating bounds required')
    return p


def public_inputs(directory, pin, p, count):
    directory = rt.canonical(directory)
    value = rt.pinned(directory/'manifest.json', pin, 1 << 20)
    rt.require(type(value) is dict and set(value) == {'schema','images'}
        and value['schema'] == 'world_reward.rgb_proposal_inputs.v1'
        and type(value['images']) is list and len(value['images']) == count
        and type(count) is int and 0 < count <= p['maximum_images'], 'Exact frozen public RGB inventory required')
    files, ids = {'manifest.json'}, set()
    for row in value['images']:
        rt.require(type(row) is dict and set(row) == {'image_id','file','bytes','sha256','width','height'}
            and type(row['image_id']) is str and re.fullmatch('[0-9a-f]{16,64}',row['image_id'])
            and row['image_id'] not in ids and type(row['file']) is str
            and re.fullmatch(r'image_[0-9]{6}\.(png|jpg)',row['file']) and row['file'] not in files
            and all(type(row[k]) is int and row[k] > 0 for k in ('width','height'))
            and row['width']*row['height'] <= p['maximum_pixels'], 'Only opaque distinct RGB records; no private fields')
        wanted = {k:row[k] for k in ('bytes','sha256')}
        binding.validate_file_pins({row['file']:wanted}, {row['file']}, p['maximum_rgb_bytes'])
        rt.require(rt.identity(directory/row['file'],p['maximum_rgb_bytes']) == wanted,
            'Frozen original RGB changed')
        files.add(row['file']); ids.add(row['image_id'])
    rt.require({x.name for x in directory.iterdir()} == files
        and not directory.lstat().st_mode & 0o222, 'Readonly public directory, no extra recipe or private payload')
    return value


def decode(directory, row):
    import numpy as np
    from PIL import Image
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with Image.open(directory/row['file']) as im:
            rt.require(im.format == ('PNG' if row['file'].endswith('.png') else 'JPEG')
                and im.mode == 'RGB' and im.size == (row['width'],row['height']), 'Native RGB header/grid required')
            im.load(); rgb = np.asarray(im).copy()
    rt.require(rgb.dtype == np.uint8 and rgb.shape == (row['height'],row['width'],3), 'Complete native uint8 RGB required')
    return rgb


def region_arrays(records, row, p):
    """Retain native order, duplicates, raw unbounded predicted-IoU and zero banks."""
    import numpy as np
    rt.require(type(records) is list and len(records) <= p['maximum_masks'], 'Native mask cap; never truncate')
    n,h,w = len(records),row['height'],row['width']
    arrays = dict(packed_masks=np.empty((n,(h*w+7)//8),np.uint8),
        native_bbox_xywh=np.empty((n,4),np.float64), native_crop_box_xywh=np.empty((n,4),np.float64),
        native_point_coords_xy=np.empty((n,2),np.float64), predicted_iou=np.empty(n,np.float64),
        stability_score=np.empty(n,np.float64), mask_area=np.empty(n,np.int64),
        native_mask_indices=np.arange(n,dtype=np.int64), image_size=np.array([h,w],np.int64))
    hashes = []
    for i,r in enumerate(records):
        rt.require(type(r) is dict and set(r) == {'segmentation','bbox','area','predicted_iou',
            'point_coords','stability_score','crop_box'}, 'Original native AMG fields required')
        mask = r['segmentation']
        rt.require(type(mask) is np.ndarray and mask.dtype == np.bool_ and mask.shape == (h,w)
            and type(r['area']) is int and r['area'] == int(mask.sum()), 'Exact native original-grid mask/area required')
        for key,shape in (('bbox',(4,)),('crop_box',(4,)),('point_coords',(1,2))):
            v = np.asarray(r[key])
            rt.require(not np.ma.isMaskedArray(r[key]) and v.shape == shape
                and v.dtype.kind in 'fiu' and np.isfinite(v).all(), 'Finite original native coordinates required')
        rt.require(np.array_equal(r['crop_box'],[0,0,w,h]) and np.all(np.asarray(r['bbox'])[2:] >= 0), 'Frozen crop0 XYWH required')
        for key in ('predicted_iou','stability_score'):
            rt.require(type(r[key]) in (int,float) and np.isfinite(r[key]), 'Finite native quality required')
            arrays[key][i] = r[key]
        rt.require(0 <= r['stability_score'] <= 1, 'Native stability intersection/union ratio required')
        arrays['packed_masks'][i] = np.packbits(mask.ravel(),bitorder='little')
        arrays['native_bbox_xywh'][i],arrays['native_crop_box_xywh'][i] = r['bbox'],r['crop_box']
        arrays['native_point_coords_xy'][i] = r['point_coords'][0]
        arrays['mask_area'][i] = r['area']; hashes.append(hashlib.sha256(mask.tobytes()).hexdigest())
    arrays['mask_sha256'] = np.asarray(hashes,dtype='U64')
    arrays['region_ids'] = np.asarray([f"{row['image_id']}:{i:06d}:{s}" for i,s in enumerate(hashes)],dtype='U136')
    return arrays


def observe(images, directory, out, p, generate, report, persist):
    import numpy as np
    used = 0
    for row in images:
        rgb = decode(directory,row); digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        report['rgb_decodes'] += 1; report['amg_attempts'] += 1; persist()
        started = time.monotonic(); records = generate(rgb); report['amg_calls'] += 1
        rt.require(hashlib.sha256(rgb.tobytes()).hexdigest() == digest, 'Native generator mutated RGB')
        arrays = region_arrays(records,row,p)
        rt.require(used+sum(a.nbytes for a in arrays.values())+65536 <= p['maximum_bank_bytes'], 'Complete bank cap; no truncation')
        path = out/(row['image_id']+'.npz')
        with path.open('xb') as stream:
            np.savez(stream,**arrays); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(),0o400)
        pin = rt.identity(path,p['maximum_bank_bytes']); used += pin['bytes']
        rt.require(used <= p['maximum_bank_bytes'], 'Actual complete byte cap exceeded')
        report['banks'].append(dict(image_id=row['image_id'],file=path.name,**pin,
            width=row['width'],height=row['height'],native_masks=len(records),decoded_rgb_sha256=digest,
            native_generate_seconds=time.monotonic()-started))
        report['bank_bytes'] = used; persist()


def native(code, revision, p, directory, out, count, pin, proof_pin):
    started = time.monotonic(); safe = rt.pinned(out/'proof.json',proof_pin,1 << 20)
    report = dict(schema=p['schema'],stage='native_rgb_region_bank',status='fail',phase='authentication',
        producer_revision=revision,image_id=binding.IMAGE,proof_identity=proof_pin,
        native_model_loads=0,amg_attempts=0,amg_calls=0,rgb_decodes=0,banks=[],
        all_native_returned_masks=True,reference_metadata_read=False,split_read=False,challenge_inputs_used=False,
        object_identities_verified=False,ownership_verified=False,quality_verified=False,training_overlap_verified=False,adoption=False)
    def persist():
        rt.require(time.monotonic() < float(os.environ['WR_REGION_DEADLINE']), 'Inclusive native budget exhausted')
    failure = None
    try:
        rt.require({x.name for x in Path('/sys/class/net').iterdir()} == {'lo'}
            and os.environ['WR_IMAGE_ID'] == binding.IMAGE, 'Original offline native image required')
        rt.require(set(safe) == {'source','native_files','frontend_sha256','manifest','images','directory'}
            and safe['manifest'] == pin and safe['images'] == count and safe['directory'] == str(directory)
            and safe['source']['producer_revision'] == revision, 'Exact public-only host proof required')
        mounted = {str(x.relative_to(code)) for x in code.rglob('*') if x.is_file()}
        rt.require(mounted == set(NATIVE_FILES) == set(safe['native_files']), 'Only whitelisted helpers; no renderer/recipe/configs')
        for name,wanted in safe['native_files'].items():
            rt.require(rt.identity(code/name,2_000_000) == wanted, 'Mounted original helper changed')
        before = public_inputs(directory,pin,p,count); model = mechanics.frontend_proof(code,p)
        rt.require(mechanics.json_digest(model) == safe['frontend_sha256'], 'Qualified original model proof differs')
        import torch
        rt.require(torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(), 'Actual H100 required')
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        installed = binding.installed_sam2(model)
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        sam = build_sam2(p['installed_config']['path'],str(binding.DEST/'weights'/p['checkpoint']['relative_path']),
            device='cuda',mode='eval',apply_postprocessing=True)
        report['native_model_loads'] += 1
        rt.require(not sam.training and not hasattr(sam.image_encoder.forward,'_torchdynamo_orig_callable')
            and sam.sam_mask_decoder.dynamic_multimask_via_stability is True
            and sam.sam_mask_decoder.dynamic_multimask_stability_delta == .05
            and sam.sam_mask_decoder.dynamic_multimask_stability_thresh == .98, 'Unchanged native multimask policy required')
        generator = SAM2AutomaticMaskGenerator(sam,**p['amg'])
        rt.require(generator.predictor._transforms.max_hole_area == generator.predictor._transforms.max_sprinkle_area == 0,
            'Native hole/sprinkle0, no fallback')
        def generate(rgb):
            with warnings.catch_warnings(),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                warnings.simplefilter('error'); result = generator.generate(rgb)
            torch.cuda.synchronize(); return result
        report['phase'] = 'native_amg'; observe(before['images'],directory,out,p,generate,report,persist)
        rt.require(public_inputs(directory,pin,p,count) == before
            and binding.installed_sam2(model) == installed and mechanics.frontend_proof(code,p) == model,
            'Original RGB/model/runtime changed after native calls')
        for name,wanted in safe['native_files'].items(): rt.require(rt.identity(code/name,2_000_000) == wanted, 'Original source changed')
        rt.require(report['amg_calls'] == report['amg_attempts'] == report['rgb_decodes'] == len(report['banks']) == count,
            'All fixed RGB/native calls required')
        report['source_rgb_model_rehashed_after'] = True
    except Exception as e: failure = e; report['error_type'] = type(e).__name__
    report.update(status='fail' if failure else 'pass',phase=report['phase'] if failure else 'complete',elapsed_seconds=time.monotonic()-started)
    write(out/'native.json',report); rt.require(failure is None, 'Native region bank failed; receipt retained')
    return report


def dispatch(code, revision, p, directory, out, count, pin):
    import fcntl
    deadline = time.monotonic()+p['budget_seconds']; source = rt.source(ROOT,code,revision,ENTRY,HELPERS)
    inputs = public_inputs(directory,pin,p,count); model = mechanics.frontend_proof(code,p,live=True)
    safe = dict(source=source,native_files={n:rt.identity(code/n,2_000_000) for n in NATIVE_FILES},
        frontend_sha256=mechanics.json_digest(model),manifest=pin,images=count,directory=str(directory))
    rt.require(not out.exists() and out.parent.is_dir(), 'Fresh fixed output required')
    name = 'world-reward-regions-'+revision[:12]
    cmdout = mechanics._command
    rt.require(not cmdout(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$']), 'Owned name already exists')
    lock = rt.canonical(ROOT/'jobs/.world-reward-h100.lock'); lockstat = lock.lstat()
    rt.require(stat.S_ISREG(lockstat.st_mode) and lockstat.st_nlink == 1, 'Original cooperative single-link lock required')
    fd = os.open(lock,os.O_RDONLY|os.O_NOFOLLOW); failure = None; created = False
    host = dict(stage='rgb_region_bank_host',status='fail',producer_revision=revision,source_binding=source,
        image_id=binding.IMAGE,private_mounts=False,quality_verified=False,adoption=False)
    try:
        rt.require((os.fstat(fd).st_dev,os.fstat(fd).st_ino) == (lockstat.st_dev,lockstat.st_ino), 'Cooperative lock inode differs')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rt.require(not cmdout(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']), 'Other GPU compute present')
        out.mkdir(mode=0o700); created = True; proof_pin = write(out/'proof.json',safe)
        mounts = [code/n for n in NATIVE_FILES]+list(binding.control_paths())+[directory,
            binding.DEST/'weights'/p['checkpoint']['relative_path']]
        cmd = ['docker','run','--rm','--name',name,'--cidfile',str(out/'.container.cid'),
            '--label','world-reward.job='+ENTRY,'--label','world-reward.revision='+revision,
            '--gpus','all','--network','none','--user','0:0','--memory','64g','--cpus','4','--pids-limit','256',
            '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--tmpfs','/tmp:rw,noexec,nosuid,size=512m','--entrypoint','/usr/bin/env']
        for path in mounts:
            rt.canonical(path); rt.require(',' not in str(path) and '\n' not in str(path), 'Safe individual mount required')
            cmd.extend(('--mount',f'type=bind,src={path},dst={path},readonly'))
        cmd.extend(('--mount',f'type=bind,src={out},dst={out}',binding.IMAGE,'-i',
            'PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','PYTHONDONTWRITEBYTECODE=1',
            'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',
            'WR_ROOT='+str(ROOT),'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_IMAGE_ID='+binding.IMAGE,
            'WR_REGION_DEADLINE='+format(deadline,'.17g'),'/opt/conda/bin/python','-I','-B',str(code/'infra/rgb_region_bank.py'),
            '--native','--inputs',str(directory),'--output',str(out),'--count',str(count),
            '--manifest-bytes',str(pin['bytes']),'--manifest-sha256',pin['sha256'],
            '--proof-bytes',str(proof_pin['bytes']),'--proof-sha256',proof_pin['sha256']))
        with (out/'native.log').open('xb') as log:
            os.fchmod(log.fileno(),0o400)
            run = subprocess.run(cmd,stdout=log,stderr=log,timeout=max(1,deadline-time.monotonic()),check=False)
        host['native_exit_status'] = run.returncode; rt.require(run.returncode == 0, 'Native region bank failed')
        receipt = rt.strict((out/'native.json').read_bytes())
        rt.require(receipt['status'] == 'pass' and receipt['amg_calls'] == count
            and receipt['source_rgb_model_rehashed_after'] is True and receipt['proof_identity'] == proof_pin,
            'Complete native receipt required')
        host['native_report_identity'] = rt.identity(out/'native.json',1 << 20)
    except Exception as e: failure = e; host['error_type'] = type(e).__name__
    finally:
        try:
            if created:
                cidfile = rt.canonical(out/'.container.cid'); cid = cidfile.read_text().strip()
                rt.require(re.fullmatch('[0-9a-f]{64}',cid), 'Exact owned CID required')
                if cmdout(['docker','ps','-aq','--no-trunc','--filter','id='+cid]):
                    actual = cmdout(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'])
                    rt.require(actual == binding.IMAGE+'|/'+name+'|'+ENTRY+'|'+revision, 'Cannot remove foreign container')
                    cmdout(['docker','rm','-f',cid])
                rt.require(not cmdout(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$']), 'Owned cleanup incomplete')
                cidfile.chmod(0o400); host['owned_cleanup_verified'] = True
            rt.require(rt.source(ROOT,code,revision,ENTRY,HELPERS) == source and public_inputs(directory,pin,p,count) == inputs
                and mechanics.frontend_proof(code,p) == model, 'Full source/input/model changed after dispatch')
            rt.require((lock.lstat().st_dev,lock.lstat().st_ino) == (lockstat.st_dev,lockstat.st_ino), 'Original lock changed')
            if created: rt.require(not cmdout(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']), 'Owned GPU cleanup incomplete')
            host['source_rgb_model_rehashed_after'] = True
        except Exception as e: failure = failure or e; host['post_error_type'] = type(e).__name__
        os.close(fd)
        if created:
            host['status'] = 'fail' if failure else 'pass'; write(out/'host.json',host)
            if not failure: (out/'native.log').unlink()
    rt.require(failure is None, 'Region dispatch failed; immutable receipts retained')
    return host


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--native',action='store_true'); parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--count',type=int,required=True)
    parser.add_argument('--manifest-bytes',type=int,required=True); parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--proof-bytes',type=int); parser.add_argument('--proof-sha256')
    a = parser.parse_args(); code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.environ['WR_ROOT'] == str(ROOT)
        and re.fullmatch('[0-9a-f]{40}',revision) and code == ROOT/'jobs'/revision/ENTRY/'code', 'Actual immutable source namespace required')
    rt.require(a.inputs in INPUTS and a.output == ROOT/'results'/('proposal-stress-regions-v1' if a.inputs == INPUTS[0]
        else 'proposal-external-regions-v1'), 'Only prospectively frozen fresh validation namespaces allowed')
    p = configuration(code); pin = dict(bytes=a.manifest_bytes,sha256=a.manifest_sha256)
    def expired(*_): raise TimeoutError('Inclusive region bank bound exceeded')
    signal.signal(signal.SIGTERM,expired); signal.signal(signal.SIGALRM,expired); signal.alarm(p['budget_seconds']+60)
    if a.native:
        result = native(code,revision,p,a.inputs,a.output,a.count,pin,dict(bytes=a.proof_bytes,sha256=a.proof_sha256))
    else:
        rt.require(a.proof_bytes is a.proof_sha256 is None and os.uname().nodename == 'world-reward-ncc-h100-02', 'VM02 dispatch only')
        result = dispatch(code,revision,p,a.inputs,a.output,a.count,pin)
    print(json.dumps(dict(stage=result['stage'],status=result['status'],quality_verified=False)))


if __name__ == '__main__':
    try: main()
    except Exception as e:
        print(json.dumps(dict(stage='rgb_region_bank',status='fail',error_type=type(e).__name__)))
        raise SystemExit(1) from None
