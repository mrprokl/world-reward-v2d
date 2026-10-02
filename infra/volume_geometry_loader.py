"""Load the frozen CPU-qualified metric mesh, without resimplifying on GPU.

This verifies receipts/arrays/topology. Independent intersection/volume checks
belong to the original CPU producer, not a fabricated GPU revalidation claim.
"""
import json
from pathlib import Path
import numpy as np
from mesh_link_gate import mesh_topology
from object_budget_endpoint import _load_mesh, verify_pack_fidelity
from world_reward.data import sha256
from world_reward.mesh_geometry import normalize_degenerate_faces

PRODUCER_REVISION='5e0a16f1c8541b49d9543d0366473f0462951d24'
PRODUCER_SHA='8166d8e2c2a60123cacc9810645b82da8aa38339a0f9bc008dbd080fabbbe556'
CONTROL_SHA='53faaf1c913c4296251680a598414d1b918d36b795d8a44267037e6d26937187'


def regular(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.resolve()!=path.absolute():
        raise ValueError('Require canonical regular frozen volume artifact')
    return path


def load(root,episode,input_sha,object_report_sha,alignment_sha,scale):
    root=Path(root)
    if type(episode) is not int or not 0 <= episode < 30 or type(scale) is not float or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Require selected episode and finite positive grounding scale')
    base=root/f'outputs/episode_{episode:06d}/object_budget_volume';p=regular(base/'report.json')
    r=json.loads(p.read_text())
    expected={'stage':'world_reward_cpu_volume_constrained_object_mesh','status':'pass','episode_index':episode,
              'producer_revision':PRODUCER_REVISION,'script_sha256':PRODUCER_SHA,'input_track':'track_1',
              'input_sha256':input_sha,'ground_truth_used':False,'hand_labeled_test':False,'oracle_modes':[],
              'target_faces':4096,'target_vertices':4096,'components_deleted':False,'holes_filled':False,
              'normals_repaired':False,'frame_poses_changed':False,'source_shell_volume_relative_limit':.05,
              'native_cost_and_placement_unchanged':True,'independent_candidate_intersecting_faces':0,
              'packed_intersecting_faces':0,'metric_scale_baked_once':scale}
    if any(type(r.get(k)) is not type(v) or r[k]!=v for k,v in expected.items()):
        raise ValueError('Require actual frozen qualified volume producer')
    if r.get('source_hashes',{}).get('object_report')!=object_report_sha or r['source_hashes'].get('alignment_report')!=alignment_sha:
        raise ValueError('Volume mesh human-grounding ancestry differs')
    for field in ('candidate_geometry','export_geometry'):
        measurements=r.get(field,{})
        errors=[x.get('relative_volume_error',np.nan) for x in measurements.get('birthface_matched_shells',[])]
        cd=measurements.get('sampled_bidirectional_chamfer_diagonal_ratio',np.nan)
        net=measurements.get('net_volume_relative_error',np.nan)
        if (not errors or not np.isfinite([cd,net,*errors]).all() or not 0 <= cd <= .01
                or not 0 <= net <= .05 or any(not 0 <= x <= .05 for x in errors)):
            raise ValueError('Frozen geometry fidelity thresholds not met')
    if (r.get('source_arrays_unchanged') is not True
            or r.get('official_pack_fidelity',{}).get('oriented_triangles_exact') is not True
            or r['official_pack_fidelity'].get('official_helper_simplification_invoked') is not False):
        raise ValueError('Frozen export/packing fidelity not proved')
    control_path=regular(root/'validation/volume_qem_v1/report.json');control=json.loads(control_path.read_text())
    build_path=regular(root/'results/image-volume-qem.json');build=json.loads(build_path.read_text())
    evidence=r.get('control_evidence',{})
    if (sha256(control_path)!=evidence.get('volume_gate_sha256') or sha256(build_path)!=evidence.get('build_report_sha256')
            or control.get('stage')!='own_volume_constrained_intersection_qem_geometry' or control.get('status')!='pass'
            or control.get('script_sha256')!=CONTROL_SHA or build.get('stage')!='world_reward_volume_qem_build'
            or build.get('status')!='pass' or build.get('image_id')!=r.get('image_id')
            or control.get('build')!={k:v for k,v in evidence.items() if k!='volume_gate_sha256'}):
        raise ValueError('Volume CPU control/build receipts differ')
    geometry=regular(base/'geometry.npz');glb=regular(base/'object_fixed_canonical.glb')
    if sha256(geometry)!=r.get('geometry_sha256') or sha256(glb)!=r.get('canonical_glb_sha256'):
        raise ValueError('Frozen CPU volume geometry/export differs')
    with np.load(geometry,allow_pickle=False) as data:
        vertices,faces=data['vertices'].copy(),data['faces'].copy()
        if (data['episode_index'].shape!=() or data['episode_index'].dtype.kind not in 'iu'
                or data['object_scale'].shape!=() or data['object_scale'].dtype.kind!='f'
                or data['grounded_scale_baked'].shape!=() or data['grounded_scale_baked'].dtype.kind!='f'
                or int(data['episode_index'])!=episode or float(data['object_scale'])!=1. or float(data['grounded_scale_baked'])!=scale):
            raise ValueError('Metric scale already baked; do not apply another scale')
    if vertices.shape!=(4096,3) or faces.shape!=(4096,3):raise ValueError('Require exact official padded budget')
    canonical_v,canonical_f=_load_mesh(glb)
    # Use the producer's packed arithmetic dtype before the same one-time
    # scalar multiplication (GLB loaders expose float32 coordinates as float64).
    compact,fidelity=verify_pack_fidelity((canonical_v.astype(vertices.dtype)*scale,canonical_f),vertices,faces)
    active,cleanup=normalize_degenerate_faces(vertices,faces)
    topology=mesh_topology(*compact)
    historical=r['packed_topology']
    if (topology['faces']!=historical['faces'] or topology['vertices']!=historical['vertices']
            or [(x['euler'],x['volume_sign']) for x in topology['components']]!=[(x['euler'],x['volume_sign']) for x in historical['components']]):
        raise ValueError('Actual packed metric topology differs from qualified CPU mesh')
    receipt={'backend':'frozen_volume_constrained_qem','cpu_report_sha256':sha256(p),'geometry_sha256':sha256(geometry),
             'control_report_sha256':sha256(control_path),'build_report_sha256':sha256(build_path),
             'metric_scale_already_baked':True,'resimplification_performed':False,'actual_topology_verified':True,
             'metric_oriented_triangle_fidelity':fidelity,
             'independent_embedding_reverified_here':False,'upstream_packed_intersections':0}
    return vertices,faces,active,cleanup,glb,receipt
