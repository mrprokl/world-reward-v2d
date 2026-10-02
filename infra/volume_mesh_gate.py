"""New per-shell-volume QEM controls; same independent geometry thresholds.

Only newly frozen procedural surfaces are inputs. Native volume protection is
not a substitute for independent final embedding, cavity or fidelity checks.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import tempfile
import time

import numpy as np
import guarded_mesh_gate as guarded
from mesh_link_gate import _array_hash, _write, mesh_topology
from mesh_endpoint_gate import source_intersections, true_hollow_containment
from world_reward.data import sha256

STAGE='own_volume_constrained_intersection_qem_geometry'
BINARY=Path('/opt/world-reward/volume-qem/mesh_volume_qem')
FIXTURE_NAMES=['new_thin_asymmetric_volume_cavity','new_multiscale_six_cavities']


def validate_build(root):
    path=root/'results/image-volume-qem.json'
    if path.is_symlink() or not path.is_file():raise ValueError('Require frozen volume QEM build receipt')
    build=json.loads(path.read_text())
    if (build.get('stage')!='world_reward_volume_qem_build' or build.get('status')!='pass' or build.get('image_id')!=os.environ.get('WR_IMAGE_ID')
            or BINARY.is_symlink() or not BINARY.is_file() or sha256(BINARY)!=build.get('binary_sha256')
            or build.get('source_cpp_sha256')!=sha256(Path(__file__).with_name('mesh_volume_qem.cpp'))):
        raise ValueError('Actual volume QEM source/binary/image binding differs')
    info=json.loads(subprocess.check_output([str(BINARY),'--build-info'],text=True,timeout=10))
    if (info.get('source_sha256')!=build['source_cpp_sha256']
            or info.get('base_source_sha256')!=sha256(Path(__file__).with_name('mesh_guarded_qem.cpp'))
            or info.get('libigl_revision')!=guarded.LIBIGL or info.get('eigen_revision')!=guarded.EIGEN
            or info.get('target_faces')!=4096 or info.get('block_intersections') is not True
            or info.get('volume_relative_limit')!=.05 or info.get('native_cost_and_placement_unchanged') is not True):
        raise ValueError('Volume QEM actual configuration differs')
    return {'build_report_sha256':sha256(path),'binary_sha256':build['binary_sha256'],'build_info':info}


def simplify(source,timeout):
    with tempfile.TemporaryDirectory(prefix='volume-qem-') as temporary:
        root=Path(temporary);a,b,m=(root/name for name in ('input.obj','output.obj','mapping.json'))
        guarded.write_obj(a,*source)
        result=subprocess.run([str(BINARY),str(a),str(b),str(m)],capture_output=True,text=True,timeout=timeout)
        if result.returncode:raise RuntimeError('Volume-constrained QEM did not reach fixed budget: '+result.stderr[-800:])
        candidate=guarded.read_obj(b);mapping=json.loads(m.read_text())
        if (mapping.get('final_shell_volumes_verified') is not True or mapping.get('volume_relative_limit')!=.05
                or mapping.get('native_cost_and_placement_unchanged') is not True or mapping.get('cost_normalization') is not False):
            raise ValueError('Actual native volume protection/configuration differs')
    return candidate,mapping


def fixtures():
    import trimesh
    sphere=trimesh.creation.icosphere(subdivisions=4);unit=sphere.vertices.copy();f=sphere.faces.copy()
    x,y,z=unit.T
    outer=unit*(1+.065*x*y-.055*y*z+.045*x*x*z)[:,None]*[.91,.71,.57]
    thin=(np.r_[outer,outer*.947],np.r_[f,f[:,::-1]+len(outer)])
    vertices=[outer];faces=[f];offset=len(outer)
    small=trimesh.creation.icosphere(subdivisions=3);u=small.vertices.copy();sf=small.faces.copy()
    sx,sy,sz=u.T;u=u*(1+.045*sx*sy+.035*sy*sz)[:,None]
    radii=[.055,.072,.088,.062,.094,.067]
    for i,radius in enumerate(radii):
        center=[-.25+.25*(i%3),-.18+.36*(i//3),.013*(-1)**i]
        cavity=u*radius*[1.,.86,1.09]+center
        vertices.append(cavity);faces.append(sf[:,::-1]+offset);offset+=len(cavity)
    multi=(np.concatenate(vertices),np.concatenate(faces))
    return [(FIXTURE_NAMES[0],thin),(FIXTURE_NAMES[1],multi)]


def all_cavities_contained(candidate):
    """Test each original inward shell against the outward boundary alone."""
    v,f=candidate;labels=guarded.component_labels(v,f);groups=[]
    for label in np.unique(labels):
        selected=f[labels==label];ids=np.unique(selected);tri=v[selected]-v[ids].mean(0)
        signed=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
        groups.append((selected,signed))
    outward=[x[0] for x in groups if x[1]>0];inward=[x[0] for x in groups if x[1]<0]
    if len(outward)!=1 or len(inward)!=len(groups)-1:raise ValueError('Require one outer boundary and inward cavity shells')
    checks=[]
    for cavity in inward:
        pair=np.concatenate([outward[0],cavity]);ids,inverse=np.unique(pair,return_inverse=True)
        checks.append(true_hollow_containment(v[ids],inverse.reshape(-1,3),self_intersecting_faces=0))
    return {'cavities_checked':len(checks),'all_cavities_contained':all(x['true_containment_verified'] for x in checks),
            'checks':checks}


def run(root,report,path):
    report['build']=validate_build(root)
    report['shared_geometry_helper_sha256']=sha256(Path(guarded.__file__))
    for name,source in fixtures():
        hashes=[_array_hash(x) for x in source]
        before=mesh_topology(*source)
        if source_intersections(*source):raise ValueError('New source intersects; no healing')
        candidate,mapping=simplify(source,120)
        record={'fixture':name,'source_topology':before,'geometry':{},
                'independent_intersecting_faces':source_intersections(*candidate),
                'source_array_sha256':hashes,'candidate_array_sha256':[_array_hash(x) for x in candidate],
                'native_volume':{k:mapping[k] for k in ('volume_checks','volume_vetoes','committed_collapses','shells')}}
        report['fixtures'].append(record);_write(path,report)
        guarded.mapped_geometry(source,candidate,mapping,record['geometry'])
        if record['independent_intersecting_faces']:raise ValueError('Independent final embedding failed')
        record['containment']=all_cavities_contained(candidate)
        # All cavity signs must remain negative, not shells shrunk/deleted to
        # evade a penetration check. Birth maps already match every source shell.
        record['source_arrays_unchanged']=hashes==[_array_hash(x) for x in source]
        if not record['source_arrays_unchanged']:raise ValueError('Source arrays changed')
        _write(path,report)
    report['status']='pass'


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:
        raise RuntimeError('Require isolated remote CPU')
    root=Path(os.environ['WR_ROOT']);out=root/'validation/volume_qem_v1';path=out/'report.json'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()):raise FileExistsError('Require reserved new volume QEM controls')
    report={'stage':STAGE,'status':'fail','code_revision':os.environ['WR_CODE_REVISION'],'image_id':os.environ['WR_IMAGE_ID'],
            'script_sha256':sha256(Path(__file__)),'challenge_inputs_used':False,'adoption_performed':False,
            'budget_seconds':240,'target_faces':4096,'target_vertices':4096,'fixtures':[],
            'embedding_exact_universal_proof':False,'source_shell_volume_relative_limit':.05,
            'native_cost_and_placement_unchanged':True}
    start=time.perf_counter();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Frozen240s new controls deadline')));signal.alarm(240)
    try:_write(path,report);run(root,report,path)
    except Exception as error:report.update(error_type=type(error).__name__,error=str(error));raise
    finally:signal.alarm(0);report['elapsed_seconds']=time.perf_counter()-start;_write(path,report)
    print(json.dumps({k:report[k] for k in ('stage','status','elapsed_seconds')}))

if __name__=='__main__':main()
