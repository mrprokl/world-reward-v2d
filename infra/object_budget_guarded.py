"""One guarded QEM proposal on predicted episode geometry, never adoption.

Does not run or modify endpoint-QEM, remove components, repair faces, change
metric scale, frame poses or frozen predictions. Only after new controls pass.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np
import guarded_mesh_gate as guarded
import object_budget_endpoint as endpoint
from mesh_link_gate import mesh_topology, _array_hash, _write
from mesh_endpoint_gate import source_intersections, true_hollow_containment
from world_reward.data import sha256

STAGE='world_reward_cpu_guarded_object_mesh'


def export_birth_mapping(candidate, exported, mapping):
    """Float32 storage may quantize but must not merge or change triangles."""
    cv,cf=candidate;ev,ef=exported
    if not np.array_equal(ev[ef],cv[cf].astype(np.float32).astype(np.float64)):
        raise ValueError('GLB export changed oriented triangles beyond native float32 storage')
    unique,inverse=np.unique(cv.astype(np.float32),axis=0,return_inverse=True)
    if len(unique)!=len(cv) or not np.array_equal(unique.astype(np.float64),ev):
        raise ValueError('Float32 export merged or changed meaningful native vertices')
    birth=np.asarray(mapping['I']);new_birth=np.empty(len(ev),np.int64)
    new_birth[inverse]=birth
    return mapping|{'output_vertices':len(ev),'I':new_birth.tolist()}


def prerequisites(root,episode):
    inputs,sources,scale=endpoint.prerequisites(root,episode)
    build=guarded.validate_build(root)
    p=endpoint.regular(root,root/'validation/guarded_qem_v1/report.json');gate=json.loads(p.read_text())
    if (gate.get('stage')!=guarded.STAGE or gate.get('status')!='pass'
            or gate.get('script_sha256')!=sha256(Path(guarded.__file__)) or gate.get('build')!=build
            or gate.get('target_faces')!=4096 or gate.get('target_vertices')!=4096
            or gate.get('challenge_inputs_used') is not False or gate.get('adoption_performed') is not False
            or [x.get('fixture') for x in gate.get('fixtures',[])]!=['new_close_asymmetric_shells','new_disconnected_smooth_asymmetric']
            or any(x.get('independent_intersecting_faces')!=0 or x.get('source_arrays_unchanged') is not True for x in gate['fixtures'])
            or gate['fixtures'][0].get('containment',{}).get('true_containment_verified') is not True):
        raise ValueError('Require actual new guarded controls with exact binary and source')
    return inputs,sources,scale,{'guarded_gate_sha256':sha256(p),**build}


def produce(root,episode,report,path):
    inputs,sources,scale,control=prerequisites(root,episode)
    report.update(source_hashes=sources,control_evidence=control,input_sha256=inputs['video_sha256'])
    helper=endpoint.regular(root,root/'vendor/v2d_submission_kit/v2dlb/mesh_budget.py')
    if sha256(helper)!=endpoint.BUDGET_HELPER_SHA:raise ValueError('Official budget helper changed')
    raw=endpoint._load_mesh(root/f'outputs/episode_{episode:06d}/object_grounded/object.glb')
    v,f,weld=endpoint.exact_weld(*raw);source=v,f;hashes=[_array_hash(x) for x in source]
    report.update(exact_source_welding=weld,source_topology=mesh_topology(*source),source_intersecting_faces=source_intersections(*source));_write(path,report)
    if report['source_intersecting_faces']:raise ValueError('Source intersects; no healing')
    candidate,mapping=guarded.simplify(source,840)
    report['candidate_topology']=mesh_topology(*candidate);report['independent_candidate_intersecting_faces']=source_intersections(*candidate);_write(path,report)
    if report['independent_candidate_intersecting_faces']:raise ValueError('Independent final embedding failed')
    report['candidate_geometry']=guarded.mapped_geometry(source,candidate,mapping);_write(path,report)
    import trimesh
    fixed=path.parent/'object_fixed_canonical.glb'
    trimesh.Trimesh(*candidate,process=False).export(fixed)
    ev,ef,weld=endpoint.exact_weld(*endpoint._load_mesh(fixed))
    emap=export_birth_mapping(candidate,(ev,ef),mapping)
    report['export_geometry']=guarded.mapped_geometry(source,(ev,ef),emap)
    if source_intersections(ev,ef):raise ValueError('Float32 serialization introduced intersections')
    spec=importlib.util.spec_from_file_location('wr_guarded_official_budget',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pv,pf=module.budget_mesh(str(fixed),faces=4096,vertices=4096)
    compact,report['official_pack_fidelity']=endpoint.verify_pack_fidelity((ev,ef),pv,pf)
    report['packed_topology']=mesh_topology(*compact)
    report['packed_intersecting_faces']=source_intersections(*compact)
    if report['packed_intersecting_faces']:raise ValueError('Packed embedding failed')
    signs=sorted(x['volume_sign'] for x in report['packed_topology']['components'])
    report['source_cavity_containment_verified']=False;report['general_cavity_containment_verified']=False
    if signs==[-1,1]:report['packed_two_shell_containment']=true_hollow_containment(*compact,self_intersecting_faces=0)
    metric=pv*scale
    if not np.isfinite(metric).all():raise ValueError('Metric grounding overflow')
    geometry=path.parent/'geometry.npz'
    with geometry.open('xb') as h:np.savez_compressed(h,vertices=metric,faces=pf,episode_index=np.array(episode),object_scale=np.array(1.),grounded_scale_baked=np.array(scale))
    if hashes!=[_array_hash(x) for x in source]:raise ValueError('Source changed')
    if prerequisites(root,episode)[1:]!=(sources,scale,control):raise ValueError('Frozen provenance changed during proposal')
    report.update(status='pass',geometry_sha256=sha256(geometry),canonical_glb_sha256=sha256(fixed),
                  metric_scale_baked_once=scale,source_arrays_unchanged=True,official_helper_sha256=sha256(helper))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--episode',type=int,choices=range(30),required=True);args=p.parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:raise RuntimeError('Require remote CPU network-none')
    root=Path(os.environ['WR_ROOT']);out=root/f'outputs/episode_{args.episode:06d}/object_budget_guarded';path=out/'report.json'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()):raise FileExistsError('Require exclusively reserved guarded proposal output')
    revision,image=os.environ['WR_CODE_REVISION'],os.environ['WR_IMAGE_ID']
    if not re.fullmatch('[0-9a-f]{40}',revision) or not re.fullmatch('sha256:[0-9a-f]{64}',image):raise ValueError('Immutable source/image required')
    report={'stage':STAGE,'status':'fail','episode_index':args.episode,'producer_revision':revision,'image_id':image,
            'script_sha256':sha256(Path(__file__)),'input_track':'track_1','ground_truth_used':False,'hand_labeled_test':False,
            'oracle_modes':[],'adoption_performed':False,'challenge_performance_verified':False,'metric_scale_accuracy_verified':False,
            'budget_seconds':900,'target_faces':4096,'target_vertices':4096,'components_deleted':False,'holes_filled':False,
            'normals_repaired':False,'frame_poses_changed':False,'source_embedding_exact_universal_proof':False}
    started=time.perf_counter();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Frozen900s proposal deadline exceeded')));signal.alarm(900)
    try:_write(path,report);produce(root,args.episode,report,path)
    except Exception as exc:report.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:signal.alarm(0);report['elapsed_seconds']=time.perf_counter()-started;_write(path,report)
    print(json.dumps({k:report[k] for k in ('stage','status','episode_index','elapsed_seconds')}))

if __name__=='__main__':main()
