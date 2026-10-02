"""Intersection-blocking QEM: fixed protocol and independent geometry gates.

The third-party floating predicates are not a universal embedding proof. Reject
post-hoc intersections/topology/volume loss; never repair, remove shells or fit.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import tempfile
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from mesh_link_gate import mesh_topology, _surface, _array_hash, _write
from mesh_endpoint_gate import source_intersections, true_hollow_containment
from world_reward.data import sha256

LIBIGL = '40e7900ccbd767f1f360e0eb10f0f1a6432e0993'
EIGEN = '3147391d946bb4b6c68edd901f2add6ac1f31f8c'
BINARY = Path('/opt/world-reward/guarded-qem/mesh_guarded_qem')
STAGE = 'own_intersection_blocking_qem_geometry'


def write_obj(path, vertices, faces):
    mesh_topology(vertices, faces)
    with Path(path).open('x') as stream:
        for row in vertices:stream.write('v '+' '.join(format(float(x), '.17g') for x in row)+'\n')
        for row in faces:stream.write('f '+' '.join(str(int(x)+1) for x in row)+'\n')


def read_obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts=line.split()
        if not parts or parts[0].startswith('#'):continue
        if len(parts)!=4 or parts[0] not in ('v','f'):raise ValueError('Require pure triangular OBJ, no implicit transforms')
        if parts[0]=='v':vertices.append([float(x) for x in parts[1:]])
        else:
            if any(not re.fullmatch('[1-9][0-9]*',x) for x in parts[1:]):raise ValueError('OBJ faces require positive bare indices')
            faces.append([int(x)-1 for x in parts[1:]])
    v,f=np.asarray(vertices,np.float64),np.asarray(faces,np.int64)
    mesh_topology(v,f)
    return v,f


def validate_build(root):
    path=root/'results/image-guarded-qem.json'
    if path.is_symlink() or not path.is_file():raise ValueError('Require frozen regular guarded build receipt')
    build=json.loads(path.read_text())
    if (build.get('stage')!='world_reward_guarded_qem_build' or build.get('status')!='pass'
            or build.get('image_id')!=os.environ.get('WR_IMAGE_ID') or not BINARY.is_file()
            or BINARY.is_symlink() or sha256(BINARY)!=build.get('binary_sha256')
            or build.get('source_cpp_sha256')!=sha256(Path(__file__).with_name('mesh_guarded_qem.cpp'))):
        raise ValueError('Actual guarded binary/image/source integrity differs')
    info=json.loads(subprocess.check_output([str(BINARY),'--build-info'],text=True,timeout=10))
    if (info.get('libigl_revision')!=LIBIGL or info.get('eigen_revision')!=EIGEN
            or info.get('source_sha256')!=build['source_cpp_sha256']
            or info.get('target_faces')!=4096 or info.get('block_intersections') is not True):
        raise ValueError('Actual binary configuration differs')
    return {'build_report_sha256':sha256(path),'binary_sha256':build['binary_sha256'],'build_info':info}


def simplify(source, timeout):
    """One fresh native QEM call; temporary OBJ files never become predictions."""
    with tempfile.TemporaryDirectory(prefix='guarded-qem-') as tmp:
        root=Path(tmp);a,b,m=(root/name for name in ('input.obj','output.obj','mapping.json'))
        write_obj(a,*source)
        child=subprocess.run([str(BINARY),str(a),str(b),str(m)],capture_output=True,text=True,timeout=timeout)
        if child.returncode:raise RuntimeError('Intersection-blocking QEM did not meet its fixed budget: '+child.stderr[-800:])
        candidate=read_obj(b);mapping=json.loads(m.read_text())
    return candidate,mapping


def component_labels(vertices,faces):
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(vertices),len(vertices))).tocsr()
    _,labels=connected_components(graph,directed=False)
    return labels[faces[:,0]]


def mapped_geometry(source,candidate,mapping,diagnostics=None):
    """Birthface provenance matches actual shells, never volume-sorted pairing."""
    sv,sf=source;cv,cf=candidate;before=mesh_topology(sv,sf);after=mesh_topology(cv,cf)
    expected={'source_vertices':len(sv),'source_faces':len(sf),'output_vertices':len(cv),'output_faces':len(cf),
              'target_reached':True,'mapping_complete':True}
    if any(type(mapping.get(k)) is not type(v) or mapping[k]!=v for k,v in expected.items()):raise ValueError('Native birth mapping counts/target mismatch')
    J,I=(np.asarray(mapping.get(key)) for key in ('J','I'))
    if (J.shape!=(len(cf),) or I.shape!=(len(cv),) or J.dtype.kind not in 'iu' or I.dtype.kind not in 'iu'
            or np.any(J<0) or np.any(J>=len(sf)) or np.any(I<0) or np.any(I>=len(sv))):
        raise ValueError('Native birth maps invalid')
    if len(cv)>4096 or len(cf)>4096:raise ValueError('Native fixed budget not met')
    sl,cl=component_labels(sv,sf),component_labels(cv,cf);matched=[];seen=set()
    for label in np.unique(cl):
        selected=cl==label;birth=np.unique(sl[J[selected]])
        if len(birth)!=1 or int(birth[0]) in seen:raise ValueError('Native QEM merged or split an original shell')
        source_label=int(birth[0]);seen.add(source_label)
        # Inward shells are valid but mesh_topology requires positive whole
        # volume. Count/sign/Euler directly per shell; full meshes validated above.
        def shell(v,f):
            ids=np.unique(f);edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
            t=v[f]-v[ids].mean(0);volume=float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)
            return len(ids)-len(edges)+len(f),volume
        ae,av=shell(sv,sf[sl==source_label]);be,bv=shell(cv,cf[selected])
        if ae!=be or np.sign(av)!=np.sign(bv) or not av or not bv:raise ValueError('Original shell Euler/orientation changed')
        # Vertex birth provenance must remain in its corresponding source shell.
        active=np.unique(cf[selected]);source_vertices=np.unique(sf[sl==source_label])
        if not np.isin(I[active],source_vertices).all():raise ValueError('Native vertex birth crosses source shells')
        error=abs(bv-av)/abs(av)
        matched.append({'source_component':source_label,'euler':int(ae),'volume_sign':1 if av>0 else -1,
                        'source_volume':av,'candidate_volume':bv,'relative_volume_error':error})
    if seen!=set(map(int,np.unique(sl))):raise ValueError('Native QEM removed an original shell')
    p,q=_surface(*source),_surface(*candidate)
    cd=float(.5*(cKDTree(p).query(q)[0].mean()+cKDTree(q).query(p)[0].mean())/before['diagonal'])
    av=sum(x['source_volume'] for x in matched);bv=sum(x['candidate_volume'] for x in matched);net=abs(bv-av)/av
    result={'sampled_bidirectional_chamfer_diagonal_ratio':cd,'net_volume_relative_error':net,'birthface_matched_shells':matched,
            'source_topology':before,'candidate_topology':after,'scale_or_pose_fitted':False}
    if diagnostics is not None:diagnostics.update(result)
    if cd>.01 or net>.05 or any(x['relative_volume_error']>.05 for x in matched):raise ValueError('Frozen1% geometry/5% net and per-shell volume gates failed')
    return result


def fixtures():
    """New nonconvex close shells and disconnected controls, not old failures."""
    import trimesh
    unit=trimesh.creation.icosphere(subdivisions=4);v,f=unit.vertices.copy(),unit.faces.copy()
    x,y,z=v.T;v=v*(1+.09*(x*x-y*y)*z+.07*x*y+.06*x**3)[:,None]*[.82,.63,.51]
    hollow=(np.concatenate([v,v*.965]),np.concatenate([f,f[:,::-1]+len(v)]))
    other=v*[.6,.75,.5]
    disconnected=(np.concatenate([v+[-1.2,0,0],other+[1.1,.12,0]]),np.concatenate([f,f+len(v)]))
    return [('new_close_asymmetric_shells',hollow),('new_disconnected_smooth_asymmetric',disconnected)]


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:raise RuntimeError('Require isolated remote CPU')
    root=Path(os.environ['WR_ROOT']);out=root/'validation/guarded_qem_v1';path=out/'report.json'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()):raise FileExistsError('Require reserved new validation output')
    report={'stage':STAGE,'status':'fail','code_revision':os.environ['WR_CODE_REVISION'],'image_id':os.environ['WR_IMAGE_ID'],
            'script_sha256':sha256(Path(__file__)),'challenge_inputs_used':False,'adoption_performed':False,
            'budget_seconds':180,'target_faces':4096,'target_vertices':4096,'fixtures':[],'embedding_exact_universal_proof':False}
    started=time.perf_counter();signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Frozen180s gate budget exceeded')));signal.alarm(180)
    try:
        report['build']=validate_build(root)
        for name,source in fixtures():
            hashes=[_array_hash(x) for x in source];mesh_topology(*source)
            if source_intersections(*source):raise ValueError('Source embedding failed; no healing')
            candidate,mapping=simplify(source,120)
            record={'fixture':name,'geometry':{},'independent_intersecting_faces':source_intersections(*candidate)}
            report['fixtures'].append(record)
            mapped_geometry(source,candidate,mapping,record['geometry'])
            if record['independent_intersecting_faces']:raise ValueError('Independent embedding gate failed')
            if name=='new_close_asymmetric_shells':record['containment']=true_hollow_containment(*candidate,self_intersecting_faces=0)
            if hashes!=[_array_hash(x) for x in source]:raise ValueError('Source arrays changed')
            record['source_arrays_unchanged']=True;_write(path,report)
        report['status']='pass'
    except Exception as exc:report.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:signal.alarm(0);report['elapsed_seconds']=time.perf_counter()-started;_write(path,report)
    print(json.dumps({k:report[k] for k in ('stage','status','elapsed_seconds')}))

if __name__=='__main__':main()
