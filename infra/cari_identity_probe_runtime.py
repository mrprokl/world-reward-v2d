"""Source-bound shared-identity inverse probe; never changes a submitted mesh.

Reuses the immutable failed original conversion, not another cold conversion.
Only original predicted vertices are targets. No private evaluation or RGB fit,
no full-frame output, and no adoption even when a small representation gate passes.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np
import cari_converter as convert
from cari_converter_diagnostic import PINS, _converted
from world_reward.data import sha256

BUDGET = 900
SEALED_SHA = 'dbcb2177c20c213ee68cf026a89b8f493dc4851930907af0d6753992a5ccdce2'


def sealed_inputs(root):
    base=root/'outputs/episode_000000';folder=base/'cari_converter_diagnostic_refined_v1'
    if folder.resolve()!=folder.absolute() or not folder.is_dir():raise ValueError('Canonical sealed source directory required')
    p=folder/'report.json';convert._require_hash(p,SEALED_SHA);receipt=json.loads(p.read_text())
    expected={'stage':'world_reward_cari_converter_runtime_diagnostic','status':'pass','phase':'complete',
              'episode_index':0,'bundle_source':'refined','frames':790,'research_only':True,'network':'none',
              'ground_truth_used':False,'hand_labeled_test':False,'oracle_modes':[],
              'submission_produced':False,'adoption_authorized':False,'original_gate_pass':False,
              'source_bindings_runtime_verified':True,'native_full_frame_decode_verified':True}
    if any(type(receipt.get(k)) is not type(v) or receipt[k]!=v for k,v in expected.items()):
        raise ValueError('Complete sealed original refined0 diagnostic required')
    if any(receipt.get('provenance',{}).get(k)!=v for k,v in PINS.items()):
        raise ValueError('Original converter/model/checkpoint pins differ')
    paths=[(p,SEALED_SHA)]
    for name,digest in [('sealed_original/original_report.json',receipt['original_report_sha256']),
                        ('sealed_original/original_converter.npz',receipt['original_archive']['sha256']),
                        ('native_parameters.npz',receipt['native_parameters_sha256'])]:
        target=folder/name;convert._require_hash(target,digest);paths.append((target,digest))
    with np.load(folder/'sealed_original/original_converter.npz',allow_pickle=False) as f:
        if set(f.files)!={'pose','scales','shape','valid_input','per_frame_vertex_error_mm','report'}:
            raise ValueError('Exact original official conversion array inventory required')
        original={k:f[k].copy() for k in f.files if k!='report'};original['report']=json.loads(str(f['report'].item()))
    if _converted(original)[0]!=790:raise ValueError('Every original frame required')
    with np.load(folder/'native_parameters.npz',allow_pickle=False) as f:params={k:f[k].copy() for k in f.files}
    indices=params.get('frame_index')
    if (set(params)!={*convert.PARAMETER_DIMS,'frame_index'} or not isinstance(indices,np.ndarray) or indices.dtype!=np.int64
            or not np.array_equal(indices,np.arange(790))):
        raise ValueError('Exact complete original native parameter inventory required')
    params.pop('frame_index')
    for key,dim in convert.PARAMETER_DIMS.items():
        convert._float_array(params[key],key,(790,dim))
        if convert.canonical_array_identity(params[key])!=receipt['native_parameter_identities'][key]:
            raise ValueError('Original native parameter identity differs')
    return original,params,receipt,paths


def decode_target(root,params,receipt,report,persist):
    from body_smoke import _body_assets, _source_identity, _pinned_checkout
    vendor=root/'vendor/video_to_data';_pinned_checkout(vendor,convert.UPSTREAM_REVISION)
    if _source_identity(root)!=receipt['inference_source_identity']:raise ValueError('Pinned native decoder source changed')
    assets,hashes=_body_assets(root)
    if hashes!=receipt['body_assets'] or (assets/'mhr_buffers.pt').exists():raise ValueError('Pinned original Body decoder assets changed')
    refinement=root/'outputs/episode_000000/cari_refined/report.json'
    convert._require_hash(refinement,receipt['input_report_sha256']['refinement'])
    ref=json.loads(refinement.read_text());bundle=root/'outputs/episode_000000/cari_refined/refined.pth'
    convert._require_hash(bundle,ref['bundle_sha256'])
    if ref['bundle_sha256']!=receipt['provenance']['native_bundle_sha256']:raise ValueError('Original bundle lineage differs')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Original CUDA decoder required')
    actual,_=convert.validate_native_bundle(torch.load(bundle,map_location='cpu',weights_only=False),790)
    if any(not np.array_equal(actual[k],params[k]) for k in params):raise ValueError('Sealed parameters differ from original native bundle')
    native=vendor/'reconstruction/modules/v2d_cari4d/lib/cari4d'
    os.environ.update(MHR_ASSETS_ROOT=str(root/'weights/cari4d/sam3d_body'),MOMENTUM_ENABLED='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    sys.path[:0]=[str(native),'/workspace/v2d_sam3d_body/lib']
    from lib_mhr.mhr_layer import MHRLayer
    from lib_mhr.geometry_provider import decode_mhr_vertices_numpy
    for symbol in (MHRLayer,decode_mhr_vertices_numpy):
        if not Path(sys.modules[symbol.__module__].__file__).resolve().is_relative_to(native/'lib_mhr'):
            raise ValueError('Original native decoder module path differs')
    report.update(phase='native_target_replay');persist()
    layer=MHRLayer.from_mhr_assets(mhr_assets_root=Path('/workspace/v2d_sam3d_body/lib'),checkpoint_path=assets/'model.ckpt',
        buffer_path=root/'never_use_unverified_probe_buffer.pt',mhr_model_path=assets/'assets/mhr_model.pt',device='cuda')
    if layer.decoder_identity()!=receipt['decoder_identity']:raise ValueError('Original native decoder identity differs')
    target=decode_mhr_vertices_numpy(layer,params,batch_size=16);del layer
    if convert.canonical_array_identity(target)!=receipt['native_vertices_identity']:raise ValueError('Original frozen target vertices changed')
    report.update(original_native_target_replay_verified=True,native_vertices_identity=receipt['native_vertices_identity']);persist()
    return torch,target,[(refinement,receipt['input_report_sha256']['refinement']),(bundle,ref['bundle_sha256'])]


def native_callbacks(torch,converter,tool,model,report):
    """Exactly one shared joint solve, one reserved pose solve, two F32 replays."""
    for symbol in (converter.lm_joint,converter.lm_pose,converter.MHR.run):
        if Path(symbol.__code__.co_filename).resolve()!=tool.resolve():raise ValueError('Official callback outside pinned converter')
    f64=converter.MHR(str(model),'cuda',chunk=64,precision='float64')
    f32=converter.MHR(str(model),'cuda',chunk=16,precision='float32')
    if (f64.mdtype!=torch.float64 or f64.fd_step!=1e-6 or f32.mdtype!=torch.float32 or f32.fd_step!=1e-3):
        raise ValueError('Pinned official solver precision differs')
    calls={'joint':0,'pose':0,'replay':0};report['runtime_bindings']=dict(precision='float64',fd_step=1e-6,
        joint_model_chunk=64,joint_frames_per_chunk=1,reserved_pose_frames_per_batch=4,final_model_chunk=16,
        final_reference_precision='float32',joint_prior=0.,pmask=None,weights=None,calls=calls)
    def tensor(array):
        if (not isinstance(array,np.ndarray) or array.dtype not in (np.float32,np.float64)
                or not np.isfinite(array).all()):raise ValueError('Finite owned NumPy callback inputs required')
        return torch.tensor(array.copy(),device='cuda',dtype=torch.float64)
    def joint(target,pose,identity,*,iters,tol,pmask,w,prior):
        if (len(pose)!=5 or iters!=40 or tol!=1e-5 or identity.shape!=(1,113)
                or any(v.dtype!=np.float64 for v in [target,pose,identity])
                or pmask is not None or w is not None or prior!=0.):raise ValueError('Predeclared five-frame shared joint protocol required')
        p,z=converter.lm_joint(f64,tensor(target),tensor(pose),tensor(identity),iters=iters,tol=tol,prior=0.,frames_per_chunk=1)
        calls['joint']+=1;return p.detach().cpu().numpy().copy(),z.detach().cpu().numpy().copy()
    def pose(target,poses,identity,*,iters,tol):
        if (not 4<=len(poses)<=5 or iters!=60 or tol!=1e-5 or identity.shape!=(1,113)
                or any(v.dtype!=np.float64 for v in [target,poses,identity])):raise ValueError('Predeclared reserved pose-only protocol required')
        p,e=converter.lm_pose(f64,tensor(target),tensor(poses),tensor(identity),iters=iters,tol=tol,frames_per_batch=4)
        calls['pose']+=1;return p.detach().cpu().numpy().copy(),e.detach().cpu().numpy().copy()
    def replay(poses,identity):
        if not 9<=len(poses)<=10 or poses.dtype!=np.float32 or identity.dtype!=np.float32 or identity.shape!=(1,113):
            raise ValueError('Only complete quantized fit/reserved F32 diagnostic replays required')
        v,_=f32.run(tensor(poses),tensor(identity));calls['replay']+=1;return v.detach().cpu().numpy().copy()
    return joint,pose,replay,calls


def main(argv=None):
    argparse.ArgumentParser(allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:raise RuntimeError('Remote offline CUDA probe required')
    root=Path(os.environ['WR_ROOT']);out=root/'outputs/episode_000000/cari_identity_probe_refined_v1'
    if out.resolve()!=out.absolute() or not out.is_dir() or any(out.iterdir()):raise FileExistsError('Exclusive new probe output required')
    rev,image=os.environ['WR_CODE_REVISION'],os.environ['WR_IMAGE_ID']
    if not re.fullmatch('[0-9a-f]{40}',rev) or not re.fullmatch('sha256:[0-9a-f]{64}',image):raise ValueError('Immutable producing source/image required')
    report=dict(stage='world_reward_cari_shared_identity_runtime_probe',status='fail',phase='provenance',producer_revision=rev,image_id=image,
        script_sha256=sha256(Path(__file__)),budget_seconds=BUDGET,network='none',input_track='track_1',ground_truth_used=False,
        hand_labeled_test=False,oracle_modes=[],research_only=True,submission_produced=False,adoption_authorized=False,
        full_frame_fidelity_verified=False,source_bindings_runtime_verified=False,original_conversion_rerun=False)
    start=time.perf_counter()
    with (out/'report.json').open('x') as handle:
        def persist():
            report['elapsed_seconds']=time.perf_counter()-start;handle.seek(0);json.dump(report,handle,allow_nan=False);handle.write('\n');handle.truncate();handle.flush()
        def expired(*_):raise TimeoutError('Whole shared-identity probe exceeded900s')
        old=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:
            persist();original,params,receipt,frozen=sealed_inputs(root)
            torch,target,more=decode_target(root,params,receipt,report,persist);frozen+=more
            tool=root/'vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py';model=root/'weights/mhr/mhr_model.pt'
            convert._require_hash(tool,convert.CONVERTER_SHA256);convert._require_hash(model,convert.REFERENCE_MODEL_SHA256)
            spec=importlib.util.spec_from_file_location('world_reward_identity_official_converter',tool)
            converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
            from cari_identity_probe import identity_probe
            joint,pose,replay,calls=native_callbacks(torch,converter,tool,model,report)
            report.update(phase='shared_identity_probe',sealed_source_report_sha256=SEALED_SHA);persist()
            result=identity_probe(original,params,target,episode_index=0,frame_index=np.arange(790,dtype=np.int64),
                provenance=receipt['provenance'],joint_fit=joint,pose_polish=pose,reference_f32=replay)
            if calls!={'joint':1,'pose':1,'replay':2}:raise ValueError('Actual joint/pose/two separate replays required')
            for p,h in frozen+[(tool,convert.CONVERTER_SHA256),(model,convert.REFERENCE_MODEL_SHA256)]:convert._require_hash(p,h)
            from body_smoke import _body_assets,_source_identity
            if _body_assets(root)[1]!=receipt['body_assets'] or _source_identity(root)!=receipt['inference_source_identity']:
                raise ValueError('Original native source/assets changed during probe')
            p=out/'diagnostic_proposal.npz'
            with p.open('xb') as f:np.savez_compressed(f,**{k:v for k,v in result.items() if k!='report'})
            result['report'].update(source_bindings_runtime_verified=True,actual_official_joint_verified=True,
                actual_official_reserved_pose_verified=True,final_reference_float32_runtime_verified=True)
            report.update(status='pass',phase='complete',probe_report=result['report'],source_bindings_runtime_verified=True,
                actual_official_joint_and_pose_verified=True,proposal_sha256=sha256(p));persist()
        except Exception as error:report.update(error_type=type(error).__name__,error=str(error));raise
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,old);signal.signal(signal.SIGTERM,term);persist()
    print(json.dumps({k:report[k] for k in ['stage','status','probe_report','elapsed_seconds']}))


if __name__=='__main__':main()
