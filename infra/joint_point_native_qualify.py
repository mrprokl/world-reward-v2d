"""REAL native baseline/zero-point-weight pair; runtime-only, no tracker or quality.

Original constructors, losses, contact/render kernels and301-update run stay
unmodified. Exact mismatch fails; no positive weight or tolerance is allowed.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import random
import re
import signal
import sys
import time

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_joint_point_native_qualify'
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
BUDGET, EPISODE, FRAMES = 7200, 21, 563
PROBES = (0, 181)
REFERENCE = 'runtime-only-zero-weight-no-calibration'
HELPERS = ('infra/joint_point_native_qualify.py', 'infra/run_joint_point_native_qualify.sh',
 'infra/cari_full_refine.py', 'infra/cari_full_forward.py', 'infra/cari_shared_prepare.py',
 'src/world_reward/joint_point_objective.py', 'src/world_reward/joint_point_evidence.py',
 'src/world_reward/point_surface_queries.py', 'src/world_reward/fixed_shape_point_pose.py',
 'configs/cari_clip_000021_input_pins.json', 'configs/cari_clip_000021_shared_prepare_pins.json',
 'configs/cari_clip_000021_shared_forward_pins.json')


def runtime():
    import mediapipe_cpu_runtime_verify as rt
    return rt


def selected_episode(argv):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--episode', choices=('21',), required=True, action='append')
    if p.parse_args(argv).episode != ['21']:
        raise ValueError('Exactly one frozen EP21 qualification required')
    return EPISODE


def historical(root, code):
    """One genuine authenticated snapshot per role; historical code never runs."""
    rt = runtime(); result = {}
    for role, entries in (('prepare', ('run_cari_shared_prepare', 'run_cari_shared_stage_queued')),
                           ('forward', ('run_cari_full_forward', 'run_cari_shared_stage_queued'))):
        pin = rt.strict((code/f'configs/cari_clip_000021_shared_{role}_pins.json').read_bytes())
        revision = pin[role]['producer_revision']
        if not re.fullmatch('[0-9a-f]{40}', str(revision)): raise ValueError('Actual producer revision required')
        candidates = [root/'jobs'/revision/entry/'code' for entry in entries
                      if (root/'jobs'/revision/entry/'code').is_dir()]
        if len(candidates) != 1: raise ValueError('One actual original role snapshot required')
        path = candidates[0]; proof = rt.source(root, path, revision, path.parent.name, ())
        script = 'cari_shared_prepare.py' if role == 'prepare' else 'cari_full_forward.py'
        if rt.identity(path/'infra'/script)['sha256'] != pin[role]['script_sha256']:
            raise ValueError('Original script differs from independent pins')
        result[role] = (path, proof)
    return result


def source_binding(root, code, revision):
    return runtime().source(root, code, revision, ENTRY, HELPERS)


def first_mask_binding(root, code):
    """Bind only the original frame0 PNG through its independent pose receipt."""
    rt=runtime();pin=rt.strict((code/'configs/cari_clip_000021_input_pins.json').read_bytes())
    relative='outputs/episode_000021/object_pose_full/report.json';path=root/relative
    expected=pin['source_files'][relative]
    if rt.identity(path,readonly=False)!=expected:raise ValueError('Pinned original object pose report differs')
    record=rt.strict(path.read_bytes());frames=record.get('frames')
    if (type(frames)is not list or len(frames)!=FRAMES or type(frames[0])is not dict
            or type(frames[0].get('frame_index'))is not int or frames[0]['frame_index']!=0
            or not re.fullmatch('[0-9a-f]{64}',str(frames[0].get('object_mask_sha256')))):
        raise ValueError('Original first-frame automatic mask receipt required')
    mask=root/'outputs/episode_000021/automatic_masks/masks/1/000000.png'
    identity=rt.identity(mask,16<<20,readonly=False)
    if (identity['sha256']!=frames[0]['object_mask_sha256']
            or rt.identity(path,readonly=False)!=expected):
        raise ValueError('Original first-frame mask/report identity differs')
    return mask,identity


def load_first_mask(path, identity, image_size):
    """Decode original binary L PNG, never resize or borrow224x224 observations."""
    import numpy as np
    from PIL import Image
    rt=runtime()
    if rt.identity(path,16<<20,readonly=False)!=identity:raise ValueError('Original mask changed before decode')
    with Image.open(path)as image:
        if image.format!='PNG' or image.mode!='L':raise ValueError('Original binary single-channel PNG required')
        raw=np.array(image,copy=True)
    if (raw.dtype!=np.uint8 or raw.shape!=image_size or not np.isin(raw,[0,255]).all()
            or not np.count_nonzero(raw)):
        raise ValueError('Original full-grid nonempty binary uint8 mask required')
    if rt.identity(path,16<<20,readonly=False)!=identity:raise ValueError('Original mask changed during decode')
    mask=raw>0;mask.flags.writeable=False
    return mask


def host_mounts(root, code):
    """Stdlib metadata only; no broad validation/private/model-cache mounts."""
    rt = runtime(); pin = rt.strict((code/'configs/cari_clip_000021_input_pins.json').read_bytes())
    if (pin.get('schema') != 'world-reward-cari-clip-input-pins-v1'
            or pin.get('clip_spec') != dict(episode_index=21,total_frames=563,
                camera_name='front_stereo_camera_left',height=1152,width=1536)
            or len(pin.get('source_files', {})) != 15): raise ValueError('Frozen complete EP21 pins required')
    paths = [root/name for name in pin['source_files']]
    if any(not str(p).startswith(str(root/'outputs/episode_000021')+'/') or '..' in p.parts for p in paths):
        raise ValueError('Original selected public paths required')
    paths += [code,code.parent/'revision',code.parent/'source-sha256',
        root/'outputs/episode_000021/cari_shared_prepare_v1',root/'outputs/episode_000021/cari_shared_forward_v1',
        root/'vendor/video_to_data',root/'vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py',
        root/'weights/mhr/mhr_model.pt',root/'weights/cari4d/sam3d_body',root/'weights/cari4d/refinement',
        root/'results/cari-refinement-assets.json',root/'results/weights-acquisition.json']
    for snapshot, _ in historical(root,code).values():
        paths += [snapshot,snapshot.parent/'revision',snapshot.parent/'source-sha256']
    paths += [first_mask_binding(root,code)[0]]
    for path in paths: rt.canonical(path)
    return tuple(sorted(set(paths)))


def paired_execution(construct, probe, run, reset, release, initial, validate, report, persist,
                     *, arm_names=('A_original','B_point_weight_zero')):
    """Injectable sequencing tests are not a substitute real-native proof."""
    if (type(arm_names) is not tuple or len(arm_names)!=2
            or any(type(name)is not str or not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,63}',name) for name in arm_names)
            or len(set(arm_names))!=2):
        raise ValueError('Exactly two distinct explicit arm names required')
    baseline = None
    for arm in arm_names:
        reset(); report['phase']=arm+'_constructor';report['constructor_attempts']+=1;persist()
        instance=construct(arm);report['constructor_returns']+=1
        observed={'initial_state':initial(instance),'probes':[]}
        for step in PROBES:
            report['phase']=arm+f'_loss_gradient_{step}';report['probe_attempts']+=1;persist()
            observed['probes'].append(probe(instance,step));report['probe_returns']+=1
        if baseline is not None and observed != baseline['initial']:
            raise ValueError('Exact initial state/loss/gradient parity failed; no tolerance')
        report['phase']=arm+'_301_updates';report['run_attempts']+=1;persist()
        result=run(instance);report['run_returns']+=1;checked=validate(result,arm)
        if baseline is None: baseline={'initial':observed,'result':checked}
        elif checked != baseline['result']:raise ValueError('Exact full native result/history parity failed')
        report[arm]={'initial':observed,'result':checked};persist()
        del instance,result
        release()
    report.update(exact_initial_state_loss_gradient_parity=True,exact_full_result_history_parity=True,
                  actual_native_updates_per_arm=301,actual_native_updates_total=602)


def numerical_control(root, source, vertices, faces, spec, pose, mask):
    import numpy as np
    import cari_clip_inputs as inputs
    from prep.mhr_depth_h5 import read_metric_depth
    from world_reward.point_surface_queries import canonical_surface_queries
    from world_reward.fixed_shape_point_pose import PointTrackEvidence
    from world_reward.joint_point_evidence import bind_joint_point_evidence
    depth=read_metric_depth(root/inputs.relative_paths(spec)['depth_h5'],'aligned',spec.camera_name,source['frames'][0])
    if mask.dtype!=np.bool_ or mask.shape!=(spec.height,spec.width) or np.shape(depth)!=mask.shape:
        raise ValueError('Original automatic mask and inferred depth grid required')
    K=inputs.inferred_camera(spec)
    queries=canonical_surface_queries(vertices,faces,pose[:3,:3],pose[:3,3],K,mask,
        np.isfinite(depth)&(depth>0),image_width=spec.width,image_height=spec.height)
    q=len(queries.face_indices);timeline=np.arange(spec.total_frames,dtype=np.int64)
    xy=queries.query_points[:,[2,1]]*np.array([256/spec.width,256/spec.height])
    tracks=PointTrackEvidence(timeline,timeline,np.arange(q,dtype=np.int64),queries.query_points,
        np.broadcast_to(xy,(spec.total_frames,q,2)).copy(),np.zeros((spec.total_frames,q),bool))
    return bind_joint_point_evidence(queries,tracks,native_vertices=vertices,native_faces=faces,K=K,
        image_size=(spec.height,spec.width),frame_index=timeline,source_frame_ids=timeline,
        native_frame_names=tuple(source['frames']),source_references=(REFERENCE,'NUMERICALCONTROL_NOTTRACKER'))


def run_runtime(root,out,code,report,persist,frozen):
    import numpy as np
    import cari_full_refine as full
    import cari_full_forward as forward
    import cari_clip_inputs as inputs
    rt=runtime();body,contract=full.body,full.contract;histories=historical(root,code)
    pin=code/'configs/cari_clip_000021_shared_forward_pins.json';pins=rt.strict(pin.read_bytes())
    spec=inputs.PublicClipSpec(**pins['clip_spec'])
    if spec.episode_index!=EPISODE or spec.total_frames!=FRAMES:raise ValueError('Frozen EP21 full563 required')
    old=histories['forward'][0]
    for name,row in forward.prepare.source_helpers(histories['prepare'][0]).items():
        if full.identity(old/name)!=row:raise ValueError('Forward snapshot preparation helper identity differs')
    selected=forward.verify_forward_artifacts(root,code,spec,pins,source_code=old)
    frozen.update({Path(p):row for p,row in selected['bindings'].items()})
    mask_path,mask_identity=first_mask_binding(root,code);frozen[mask_path]=mask_identity
    mask=load_first_mask(mask_path,mask_identity,(spec.height,spec.width))
    report['first_frame_automatic_mask']=dict(path=str(mask_path.relative_to(root)),identity=mask_identity,
        frame_index=0,grid=[spec.height,spec.width],source_dtype='uint8',binary_values=[0,255],
        nonzero_pixels=int(np.count_nonzero(mask)),resized=False,cropped_native_mask_used=False)
    for name in HELPERS:frozen[code/name]=rt.identity(code/name)
    report['historical_sources']={role:proof for role,(_,proof) in histories.items()}
    fr,pr=selected['report'],selected['prepare']['report']
    vendor=root/'vendor/video_to_data';body._pinned_checkout(vendor,body.UPSTREAM_REVISION)
    assets,hashes=body._body_assets(root);source_id=body._source_identity(root)
    if (assets!=root/full.BODY_ASSET_RELATIVE or hashes!=pr['body_assets'] or hashes!=fr['body_assets']
            or source_id!=pr['inference_source_identity'] or source_id!=fr['inference_source_identity']
            or (assets/'mhr_buffers.pt').exists()):raise ValueError('Original native source/model identity required')
    native=vendor/'reconstruction/modules/v2d_cari4d/lib/cari4d'
    optimizer_path=native/contract.OPTIMIZER_RELATIVE_PATH;layer_path=native/'lib_mhr/mhr_layer.py'
    for path,digest in ((optimizer_path,contract.OPTIMIZER_SHA256),(layer_path,full.LAYER_SHA)):
        row=full.identity(path,immutable=False)
        if row['sha256']!=digest:raise ValueError('Unchanged native optimizer/MHR source required')
        frozen[path]=row
    receipt=root/'results/cari-refinement-assets.json';contract.require_asset_receipt(rt.strict(receipt.read_bytes()))
    frozen[receipt]=full.identity(receipt,immutable=False);asset_paths={}
    for name,expected in contract.REFINEMENT_ASSETS.items():
        path=root/'weights/cari4d/refinement'/name;row=full.identity(path,immutable=False)
        if row!={k:expected[k] for k in ('bytes','sha256')}:raise ValueError('Original contact/collision assets required')
        asset_paths[name]=path;frozen[path]=row
    frozen.update({assets/name:row for name,row in hashes.items()})
    path=root/'results/weights-acquisition.json';frozen[path]=full.identity(path,immutable=False)
    mesh=root/inputs.relative_paths(spec)['mesh']
    if full.identity(mesh,immutable=False)!=pr['source_files'][str(mesh.relative_to(root))]:raise ValueError('Original aligned mesh required')
    if 'torch' in sys.modules:raise ValueError('Fresh native CUDA process required')
    import torch
    if (not torch.cuda.is_available() or str(torch.__version__)!='2.5.1+cu124' or torch.version.cuda!='12.4'
            or torch.are_deterministic_algorithms_enabled()):raise ValueError('Original CUDA runtime/policy required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path[:0]=[str(native),'/workspace/v2d_sam3d_body/lib']
    os.environ.update(MHR_ASSETS_ROOT=str(root/'weights/cari4d/sam3d_body'),MOMENTUM_ENABLED='0')
    from learning.training import mhr_opt_refineout as optimizer
    from lib_mhr.mhr_layer import MHRLayer
    from world_reward import joint_point_objective as op
    op._native_binding(optimizer)
    if (Path(optimizer.__file__).resolve()!=optimizer_path or
            Path(sys.modules[MHRLayer.__module__].__file__).resolve()!=layer_path):raise ValueError('Actual native imports required')
    def reset():
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    reset()
    layer=MHRLayer.from_mhr_assets(mhr_assets_root=Path('/workspace/v2d_sam3d_body/lib'),checkpoint_path=assets/'model.ckpt',
        buffer_path=out/'never_use_unverified_buffer.pt',mhr_model_path=assets/'assets/mhr_model.pt',device='cuda')
    if layer.decoder_identity()!=pr['decoder_identity']:raise ValueError('Actual native MHR decoder differs')
    from lib_mhr.collision_proxy import load_mhr_collision_proxy
    from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
    human_faces=layer.mesh_faces(device='cuda').detach().cpu().numpy()
    proxy=load_mhr_collision_proxy(human_faces,asset_paths['mhr_collision_proxy_4000v.npz'])
    hand=load_mhr_hand_surface_spec(asset_paths['mhr_hand_surface_spec.npz'],faces=human_faces)
    if proxy.vertex_count!=4000 or hand.mhr_model_sha256!=hashes['assets/mhr_model.pt']['sha256']:
        raise ValueError('Full real contact/collision model binding required')
    source=torch.load(selected['directory']/'coconet.pth',map_location='cpu',weights_only=False)
    full.validate_source_bundle(source,mesh,FRAMES);source_sha=full.fingerprint(source)
    vertices,faces=optimizer._load_object_vertices(mesh)
    evidence=None;extension=None
    config=op.PointObjectiveConfig(1.,0.,REFERENCE)
    cfg=optimizer.MHRParityPostOptConfig(penetration_collision_proxy_path=str(asset_paths['mhr_collision_proxy_4000v.npz']),
        hand_surface_spec_path=str(asset_paths['mhr_hand_surface_spec.npz']),report_every=100,checkpoint_path=None)
    if any(getattr(cfg,k)!=v for k,v in dict(num_steps=300,batch_size=0,freeze_object_rotation=True,
            freeze_body_internal_translations=True,checkpoint_path=None).items()):raise ValueError('Unchanged native configuration required')
    report.update(config=asdict(cfg),point_config=asdict(config),decoder_identity=layer.decoder_identity(),
        future_supported_observations=0,
        evidence_kind='NUMERICALCONTROL_NOTTRACKER',source_bundle_sha256=source_sha,
        runtime=dict(torch=str(torch.__version__),cuda=torch.version.cuda,numpy=np.__version__,
                     tf32=False,deterministic_algorithms=False,seed=0))
    def construct(arm):
        nonlocal evidence,extension
        if full.fingerprint(source)!=source_sha:raise ValueError('Original source changed before constructor')
        if arm=='A_original':
            instance=optimizer.MHRParityPostOptimizer(source,vertices,faces,cfg,mhr_layer=layer)
            indices=torch.arange(FRAMES,dtype=torch.int64,device=instance.device)
            with torch.no_grad():
                rotation,translation,_,_=instance._object_state(indices,include_surface=False)
                pose=optimizer.pose_matrix(rotation,translation)[0].detach().cpu().numpy()
            evidence=numerical_control(root,source,vertices,faces,spec,pose,mask)
            extension=op.native_point_optimizer_class(optimizer,evidence,config)
            report.update(evidence_sha256=evidence.evidence_sha256,query_count=len(evidence.query_ids),
                          attachment_pose_source='actual_original_constructor._object_state')
            return instance
        if extension is None:raise ValueError('Original first-frame attachment must precede B')
        return extension(source,vertices,faces,cfg,mhr_layer=layer)
    def initial(instance):
        tensors={k:v for k,v in vars(instance).items() if torch.is_tensor(v) or isinstance(v,np.ndarray)}
        return dict(state_sha256=full.fingerprint(tensors),state_fields=sorted(tensors),
            optimizer_sha256=full.fingerprint(instance.optimizer.state_dict()),
            scheduler_sha256=full.fingerprint(instance.scheduler.state_dict()))
    def probe(instance,step):
        indices=torch.arange(FRAMES,dtype=torch.int64,device=instance.device)
        instance.optimizer.zero_grad(set_to_none=True)
        total,metrics=instance.loss(indices,step,include_diagnostics=True)
        if not torch.isfinite(total).all():raise ValueError('Nonfinite actual native loss')
        total.backward();torch.cuda.synchronize();gradients=[]
        for group in instance.optimizer.param_groups:
            for parameter in group['params']:
                grad=parameter.grad
                if grad is not None and not torch.isfinite(grad).all():raise ValueError('Nonfinite actual native gradient')
                gradients.append(None if grad is None else full.fingerprint(grad))
        value=dict(step=step,loss_sha256=full.fingerprint(total),metrics_sha256=full.fingerprint(metrics),gradients=gradients)
        instance.optimizer.zero_grad(set_to_none=True)
        return value
    def validate(result,arm):
        if arm=='B_point_weight_zero':
            metadata=result['postopt'].pop('point_objective')
            if metadata['config']!=asdict(config) or any(metadata['after_initializer_support']):raise ValueError('Zero-control metadata differs')
        checked=full.validate_result(source,result,FRAMES)
        if full.fingerprint(source)!=source_sha:raise ValueError('Raw native source modified')
        return dict(complete_result_sha256=full.fingerprint(result),metadata_sha256=full.fingerprint(checked))
    paired_execution(construct,probe,lambda instance:instance.run(),reset,
        lambda:(gc.collect(),torch.cuda.empty_cache()),initial,validate,report,persist)
    op._native_binding(optimizer)
    if body._source_identity(root)!=source_id or body._body_assets(root)[1]!=hashes:raise ValueError('Native transitive source/model changed')


def main(argv=None):
    selected_episode(argv);started=time.monotonic();deadline=started+BUDGET
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Inclusive7200s qualification budget')))
    signal.alarm(BUDGET)
    root=Path(os.environ['WR_ROOT']);code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    rt=runtime();out=root/'results'/('joint-point-native-qualify-'+revision)
    if (platform.system()!='Linux' or root!=ROOT or os.geteuid()!=1000 or os.environ.get('WR_IMAGE_ID')!=IMAGE
            or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'} or not out.is_dir() or any(out.iterdir())):
        raise ValueError('Fresh owned offline native GPU qualification namespace required')
    before=source_binding(root,code,revision);old_before=historical(root,code);frozen={}
    report=dict(stage='joint_point_native_runtime_qualification_v1',status='fail',phase='preflight',episode_index=21,
        frames=563,budget_seconds=BUDGET,constructor_attempts=0,constructor_returns=0,probe_attempts=0,probe_returns=0,
        run_attempts=0,run_returns=0,quality_verified=False,tracker_executed=False,positive_weight_executed=False,
        ground_truth_used=False,private_truth_read=False,oracle_modes=[],input_track='track_1',
        full_original_timeline=True,original_frame_indices=list(range(FRAMES)),object_rotation_fixed=True,
        original_priors_contact_penetration_render_scheduler=True,adoption=False,submission_produced=False,
        source_binding=before)
    destination=out/'report.json'
    def persist():
        temp=out/'report.tmp'
        with temp.open('w')as stream:json.dump(report,stream,sort_keys=True,allow_nan=False)
        temp.replace(destination)
    try:
        persist();run_runtime(root,out,code,report,persist,frozen)
        report.update(status='pass',phase='complete')
    except BaseException as error:
        report.update(status='fail',error_type=type(error).__name__)
        if type(error) in (ValueError,TimeoutError,FileNotFoundError):
            text=str(error)
            if not any(token in text.lower() for token in ('://','token','password','secret','credential')):
                report['error_context']=text[-500:]
        raise
    finally:
        try:
            import cari_full_refine as full
            if (source_binding(root,code,revision)!=before or historical(root,code)!=old_before
                    or any(full.identity(p,immutable=False)!=row for p,row in frozen.items())):
                raise ValueError('Source/inputs/assets changed after qualification')
            report['source_inputs_assets_rehashed_after']=True
            if time.monotonic()>deadline:raise TimeoutError('Inclusive budget including final rehash')
        except BaseException as error:
            report.update(status='fail',postcheck_error_type=type(error).__name__)
        finally:
            report['elapsed_seconds']=time.monotonic()-started;persist();destination.chmod(0o444);signal.alarm(0)
    if report['status']!='pass':raise ValueError('Qualification failed closed')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        print(json.dumps({'status':'fail','error_type':type(error).__name__}),file=sys.stderr);raise SystemExit(1)
