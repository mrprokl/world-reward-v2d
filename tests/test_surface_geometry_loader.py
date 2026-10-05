"""Manufactured numeric/receipt tests; no native qualification is fabricated."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'infra'),str(REPO/'src')]
import surface_geometry_loader as loader
from world_reward.raw_shape_proposal import _identity
from world_reward.surface_pose_geometry import compact_surface


def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o644)
    path.write_bytes(raw);path.chmod(0o444)
    return loader._source_identity(path) if not raw else loader.identity(path)


def save(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o644)
    with path.open('wb') as stream:np.savez_compressed(stream,**arrays)
    path.chmod(0o444)
    return loader.identity(path)


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'root';code=tmp_path/'code'
    monkeypatch.setattr(loader,'CODE',code)
    # Keep real pure module provenance validation for the mathematical modules.
    original_module=loader._module
    monkeypatch.setattr(loader,'_module',lambda name,path:__import__(name,fromlist=['x']) if name.startswith('world_reward.') else original_module(name,path))
    revision='a'*40;episode=7;scale=.5
    proposal=root/f'outputs/episode_{episode:06d}/object_budget_surface_{revision}'
    names={k:str((proposal/n).relative_to(root)) for k,n in
        (('geometry','geometry.npz'),('candidate','candidate_geometry.npz'),('glb','object_fixed_canonical.glb'),('mapping','mapping.json'),('report','report.json'),('native','native.json'))}
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[2.,2.,2.]],np.float32)
    f=np.array([[0,1,2]],np.int64);rv=v*scale
    pv=np.vstack((rv.astype(np.float64),np.repeat(rv[:1].astype(np.float64),4092,axis=0)))
    pf=np.vstack((f,np.zeros((4095,3),np.int64)))
    save(root/names['candidate'],source_vertices=v,source_faces=f,candidate_vertices=v,candidate_faces=f,canonical_vertices=rv,canonical_faces=f)
    save(root/names['geometry'],vertices=pv,faces=pf,episode_index=np.array(episode,np.int64),object_scale=np.array(1.),grounded_scale_baked=np.array(scale))
    write(root/names['mapping'],json.dumps(dict(schema='surface-identity-mapping-v1',I=list(range(4)),J=[0])).encode())
    write(root/names['glb'],b'fake GLB decoded by an explicit test callback')
    write(root/names['report'],b'{}');write(root/names['native'],b'{}')
    pins=dict(report=dict(producer_revision=revision,script_sha256='b'*64))
    write(code/f'configs/surface_mesh_{episode:06d}_pins.json',json.dumps(pins).encode())
    c=dict(method='identity',source_vertices=_identity(v),source_faces=_identity(f),candidate_vertices=_identity(v),candidate_faces=_identity(f),
        canonical_surface=dict(vertices=_identity(rv),faces=_identity(f)),canonical_vertex_count=4,canonical_face_count=1,
        official_pack_fidelity=dict(raw_oriented_triangles_exact=True))
    from world_reward.surface_identity import SurfaceIdentity
    surface=SurfaceIdentity(rv,f)
    c['canonical_surface'].update(component_keys=list(surface.component_keys),boundary_loops=[list(x)for x in surface.boundary_loops],
        boundary_components=list(surface.boundary_components),components=[dict(x)for x in surface.diagnostics['components']],unused_vertices_preserved=1)
    host=dict(source_binding={'scope':'manufactured'})
    native=dict(compiler=c)
    ledger={root/n:loader.identity(root/n) for n in names.values()}
    monkeypatch.setattr(loader,'verify_pinned_artifacts',lambda *a,**kw:(host,native,ledger,names))
    from types import SimpleNamespace
    precision=SimpleNamespace(raw_glb=lambda path:([(rv,f)],rv.astype(np.float64)[f],[{'node_transform_identity':True}]),
        triangle_hash=lambda a:hashlib.sha256(a.tobytes()).hexdigest())
    monkeypatch.setattr(loader,'_module',lambda name,path:precision if name=='mesh_precision_diagnostic' else __import__(name,fromlist=['x']))
    return root,code,pins,episode,scale,names,pv,pf,rv,f,native


def test_real_numeric_payload_orphan_identity_fullsixreturns(tmp_path,monkeypatch):
    root,code,pins,e,s,names,pv,pf,rv,f,native=fixture(tmp_path,monkeypatch)
    out=loader.load(root,e,'c'*64,'d'*64,'e'*64,s,pins=pins)
    assert len(out)==6 and np.array_equal(out[0],pv) and np.array_equal(out[1],pf)
    assert out[2].tolist()==[0] and out[3]['meaningful_faces_removed']==0
    assert out[4]==root/names['glb'] and out[5]['canonical_vertices_count']==4
    assert out[5]['surface_topology']['canonical_unused_vertices_preserved']==1
    for array in out[:3]:
        assert not array.flags.writeable
        with pytest.raises(ValueError):array.setflags(write=True)


@pytest.mark.parametrize('mutation',['extra','scale','vertices','faces','padding','candidate','mapping','raw','posthash','method','raw_dtype'])
def test_numeric_mutations_failclosed(tmp_path,monkeypatch,mutation):
    root,code,pins,e,s,names,pv,pf,rv,f,native=fixture(tmp_path,monkeypatch)
    if mutation in ('extra','scale','vertices','faces','padding'):
        values=dict(vertices=pv,faces=pf,episode_index=np.array(e,np.int64),object_scale=np.array(1.),grounded_scale_baked=np.array(s))
        if mutation=='extra':values['second_scale']=np.array(1.)
        elif mutation=='scale':values['object_scale']=np.array(s)
        elif mutation=='vertices':values['vertices']=pv.astype(np.float32)
        elif mutation=='faces':values['faces']=pf.astype(np.int32)
        else:values['faces']=pf.copy();values['faces'][-1]=[0,1,1]
        save(root/names['geometry'],**values)
    elif mutation=='candidate':native['compiler']['candidate_faces']['sha256']='0'*64
    elif mutation=='method':native['compiler']['method']='qslim'
    elif mutation=='mapping':write(root/names['mapping'],b'{"schema":"surface-identity-mapping-v1","I":[0,1,2,3],"J":[1]}')
    elif mutation in ('raw','raw_dtype'):
        from types import SimpleNamespace
        vertices=rv[:3] if mutation=='raw' else rv.astype(np.float64)
        precision=SimpleNamespace(raw_glb=lambda p:([(vertices,f)],rv.astype(np.float64)[f],[{'node_transform_identity':True}]),triangle_hash=lambda a:hashlib.sha256(a.tobytes()).hexdigest())
        old=loader._module;monkeypatch.setattr(loader,'_module',lambda name,path:precision if name=='mesh_precision_diagnostic' else old(name,path))
    else:monkeypatch.setattr(loader,'recheck',lambda ledger:(_ for _ in ()).throw(ValueError('source changed')))
    with pytest.raises(ValueError):loader.load(root,e,'c'*64,'d'*64,'e'*64,s,pins=pins)


def test_preflight_fulloriginalposes_and_nonunit_bakedscale(tmp_path,monkeypatch):
    root,code,pins,e,s,names,pv,pf,rv,f,native=fixture(tmp_path,monkeypatch)
    pose=root/'pose.npz';r=np.repeat(np.eye(3)[None],3,axis=0);t=np.zeros((3,3))
    save(pose,vertices=pv,faces=pf,frame_index=np.arange(3),rotation=r,translation=t,object_scale=np.array(1.))
    budget=dict(source_domain='surface',metric_scale_baked_once=s,canonical_vertices_count=4,canonical_faces_count=1)
    out=loader.preflight_geometry_and_poses(pose,expected_v=pv,expected_f=pf,expected_episode=e,expected_scale=1.,topology_budget=budget)
    assert out[3].shape==(3,3,3) and out[5][0].shape==(4,3) and out[6]['canonical_unused_vertices_preserved']==1
    assert out[7]=={pose:loader.identity(pose)}


@pytest.mark.parametrize('field',['frame_index','rotation','translation','object_scale','vertices','extra'])
def test_pose_preflight_malformed_or_changed(tmp_path,monkeypatch,field):
    root,code,pins,e,s,names,pv,pf,rv,f,native=fixture(tmp_path,monkeypatch)
    pose=root/'pose.npz';data=dict(vertices=pv.copy(),faces=pf,frame_index=np.arange(3),rotation=np.repeat(np.eye(3)[None],3,axis=0),translation=np.zeros((3,3)),object_scale=np.array(1.))
    if field=='frame_index':data[field]=np.array([0,2,3])
    elif field=='rotation':data[field][2,0,0]=2
    elif field=='translation':data[field][1,0]=np.nan
    elif field=='object_scale':data[field]=np.array(.5)
    elif field=='vertices':data[field][-1,0]=1
    else:data['episode']=np.array(e)
    save(pose,**data)
    with pytest.raises(ValueError):loader.preflight_geometry_and_poses(pose,expected_v=pv,expected_f=pf,topology_budget=dict(source_domain='surface',metric_scale_baked_once=s,canonical_vertices_count=4,canonical_faces_count=1))


def snapshot(root,rev,entry):
    code=root/'jobs'/rev/entry/'code';write(code/'infra/x.py',b'# source\n');write(code/'src/__init__.py',b'')
    for p in (code,*code.rglob('*')):
        if p.is_dir():p.chmod(0o555)
    write(code.parent/'revision',(rev+'\n').encode());write(code.parent/'source-sha256',('f'*64+'\n').encode())
    entries={str(p.relative_to(code)):({'directory':True} if p.is_dir() else dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())) for p in (code,*sorted(code.rglob('*')))}
    bound=dict(producer_revision=rev,markers={n:loader.identity(code.parent/n) for n in ('revision','source-sha256')},entries=len(entries),helpers={'infra/x.py':entries['infra/x.py']},closure_sha256=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest())
    return code,bound


def test_whole_original_source_emptyinit_and_rehash(tmp_path):
    code,bound=snapshot(tmp_path,'a'*40,'run_object_budget_solid')
    ledger=loader._snapshot(tmp_path,'a'*40,'run_object_budget_solid',bound)
    assert ledger[code/'src/__init__.py']['bytes']==0
    loader.recheck(ledger)
    (code/'infra/x.py').chmod(0o644);(code/'infra/x.py').write_bytes(b'# changed\n')
    with pytest.raises(ValueError):loader.recheck(ledger)


@pytest.mark.parametrize('kind',['extra','wrongrevision','helper','closure','symlink'])
def test_original_snapshot_not_reportselfauthenticated(tmp_path,kind):
    code,bound=snapshot(tmp_path,'a'*40,'run_object_budget_solid')
    if kind=='extra':write(code.parent/'other',b'extra')
    elif kind=='wrongrevision':bound['producer_revision']='b'*40
    elif kind=='helper':bound['helpers']['infra/x.py']['sha256']='0'*64
    elif kind=='closure':bound['closure_sha256']='0'*64
    else:
        (code/'infra').chmod(0o755)
        (code/'infra/x.py').unlink();(code/'infra/x.py').symlink_to(code/'src/__init__.py')
    with pytest.raises(ValueError):loader._snapshot(tmp_path,'a'*40,'run_object_budget_solid',bound)


def test_paths_exact15_no_arbitrary_namespace():
    paths=loader.paths(9,'a'*40,'b'*40,'c'*40)
    assert len(paths)==15 and paths['report'].endswith('/object_budget_surface_'+'b'*40+'/report.json')
    assert paths['object']=='outputs/episode_000009/object_grounded/report.json'
    for e,q,p,i in ((True,'a'*40,'b'*40,'c'*40),(30,'a'*40,'b'*40,'c'*40),(9,'../x','b'*40,'c'*40)):
        with pytest.raises(ValueError):loader.paths(e,q,p,i)


def test_missing_independentpins_fails_before_payload(tmp_path,monkeypatch):
    monkeypatch.setattr(loader,'CODE',tmp_path/'code')
    with pytest.raises(ValueError):loader.verify_pinned_artifacts(tmp_path,{},2,'a'*64,'b'*64,'c'*64,.5)


def test_historical_writable_metadata_and_secrets_not_read(tmp_path):
    path=tmp_path/'metadata.json';path.write_text('{}');pin=loader.identity(path,readonly=False)
    loader.recheck({path:pin})
    with pytest.raises(ValueError):loader.identity(path)
    # No subprocess/native/model import in the consumer or mathematical core.
    import ast
    tree=ast.parse((REPO/'infra/surface_geometry_loader.py').read_text())
    imports={n.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for n in node.names}
    assert not imports&{'subprocess','torch','trimesh','mediapipe','requests'}


def receipt_fixture(tmp_path,monkeypatch):
    """Actual validator ABI, synthetic sealed JSON only, no geometry execution."""
    root=tmp_path/'root';code=tmp_path/'current';monkeypatch.setattr(loader,'CODE',code)
    p,qrev,irev='a'*40,'b'*40,'c'*40;e=6;s=.5
    names=loader.paths(e,qrev,p,irev)
    def seal(role,row):return write(root/names[role],json.dumps(row,sort_keys=True).encode())
    for role in ('glb','geometry','candidate','mapping','source_glb','intrinsics'):
        write(root/names[role],('opaque-'+role).encode())
    source={key:loader.identity(root/names[role])['sha256'] for role,key in
        (('source_glb','object.glb'),('intrinsics','intrinsics.json'))}
    transform={'scale':[s,s,s]};source['transform.json']=seal('transform',transform)['sha256']
    common=dict(status='pass',episode_index=e,input_track='track_1',input_sha256='d'*64,ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
    alignment=common|dict(stage='predicted_human_anchored_moge2_pointmaps',coordinate_frame='OpenCV_x_right_y_down_z_forward',
        pointmap_scale_application='one_clip_scalar_to_MoGe2_XYZ_already_applied')
    ash=seal('alignment',alignment)['sha256']
    obj=common|dict(stage='sam3d_objects_grounded_fixed_frame',frame_index=0,transform=transform,
        pointmap_grounding={'alignment_report_sha256':ash},scale_source='already_human_anchored_MoGe2_no_second_scalar',
        object_sha256=source['object.glb'],transform_sha256=source['transform.json'],intrinsics_sha256=source['intrinsics.json'])
    osh=seal('object',obj)['sha256'];source.update(video='d'*64,object_report=osh,alignment_report=ash)
    _,ps=snapshot(root,p,'run_object_budget_solid')
    ps['helpers'][loader.PROCESSOR]=ps['helpers'].pop('infra/x.py')
    oldcode=root/'jobs'/p/'run_object_budget_solid/code'
    (oldcode/'infra').chmod(0o755);(oldcode/'infra/x.py').rename(oldcode/loader.PROCESSOR);(oldcode/'infra').chmod(0o555)
    # Recompute actual complete source map after the authoritative filename change.
    entries={str(x.relative_to(oldcode)):({'directory':True} if x.is_dir() else loader._source_identity(x)) for x in (oldcode,*sorted(oldcode.rglob('*')))}
    ps['closure_sha256']=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
    _,qs=snapshot(root,qrev,'run_surface_qslim_qualify');_,isource=snapshot(root,irev,'run_surface_identity_qualify')
    firstnative=dict(status='pass',phase='complete',producer_revision=irev,source_proof={'source_binding':isource},
        controls=dict(QEM_calls=0,native_loader_calls_including_authority_replays=6))
    firstpin=seal('phase1_native',firstnative)
    oldproof={'native':{'source_binding':qs}}
    oldnative=dict(status='pass',phase='complete',producer_revision=qrev,source_proof=oldproof['native'])
    oldpin=seal('qualification_native',oldnative)
    oldhost=dict(status='fail',native_report=oldnative,source_proof=oldproof)
    oldhostpin=seal('qualification_host',oldhost)
    audit=dict(stage='world_reward.surface_qslim_independent_receipt_audit.v1',status='pass',producer_revision=qrev,
        original_host_status='fail',original_native_status='pass',original_host_receipt_unchanged=True,
        original_exact_CID_absence_verified=True,source_runtime_rehashed_after=True,geometry_replay=False,retained_receipt_only=True)
    auditpin=seal('qualification_audit',audit)
    first=dict(schema='world_reward.surface_identity_qualification_pins.v1',historical_host_status='fail',producer_revision=irev,
        native_loader_calls=6,official_budget_calls=3,QEM_calls=0,independent_native_audit=True,adoption=False,native=firstpin)
    q=dict(schema='world_reward.surface_qslim_qualification_pins.v1',historical_host_status='fail',producer_revision=qrev,
        native_loader_calls=4,official_budget_calls=2,QEM_calls=2,independent_native_receipt_audit=True,
        independent_geometry_replay=False,actual_original_CID_absence_verified=True,adoption=False,native=oldpin,
        historical_host_report=oldhostpin,independent_audit_report=auditpin,independent_audit_path=names['qualification_audit'])
    b=dict(schema='world_reward.surface_qslim_build_pins.v1',producer_revision='f'*40,
        binary=dict(bytes=100,sha256='f'*64),source_cpp=dict(bytes=100,sha256='e'*64))
    configids={n:write(code/n,json.dumps(row).encode()) for n,row in ((loader.PHASE1,first),(loader.QUALIFICATION,q),(loader.BUILD,b))}
    qual=dict(pins_identity=configids[loader.QUALIFICATION],native=oldpin,historical_host=oldhostpin,independent_audit=auditpin,
        original_host_status='fail',original_source_proof=oldproof,build=dict(binary=b['binary'],source_cpp=b['source_cpp']),phase1=dict(native=firstpin))
    c=dict(schema='world_reward.surface_budget.v1',status='pass',phase='complete',domain='surface',method='identity',qem_calls=0,
        source_arrays_frozen_before_gates=True,source_arrays_unchanged=True,geometry_repaired=False,components_deleted=False,
        orientation_repaired=False,volume_or_closure_required=False,embedding_certified=False,full_surface_fidelity_certified=False,
        adoption=False,reconstruction_accuracy_verified=False,metric_scale_baked_once=s,object_scale=1.,official_budget_calls=1,
        payload_role='native_canonical_arrays_plus_only_repeat_first_vertex_zero_faces',orphan_vertices_dropped=False,
        canonical_glb_identity=loader.identity(root/names['glb']),mapping_identity=loader.identity(root/names['mapping']),
        official_pack_fidelity=dict(raw_oriented_triangles_exact=True,native_fp32_quantization_exact=True,
            original_nonzero_triangles_preserved=True,oriented_triangles_exact=True,additional_scale_or_alignment=False,geometry_repaired=False),
        default8_serialization=dict(serialized_triangles_numerically_preserved_by_weld=True,welded_triangles_exactly_active=True))
    outputs={n:loader.identity(root/names[k]) for n,k in zip(loader.OUTPUTS,('glb','geometry','candidate','mapping'))}
    fields=dict(status='pass',phase='complete',domain='surface',episode_index=e,producer_revision=p,
        source_rehashed_after=True,inputs_qualification_rehashed_after=True,runtime_rehashed_after=True,gpu_used=False,
        ground_truth_used=False,adoption=False,reconstruction_accuracy_verified=False,competition_eligibility_verified=False,
        source_binding=ps,source_binding_after=ps,qualification=qual,input_binding={'video_sha256':'d'*64},outputs=outputs)
    native=fields|dict(stage='world_reward_object_budget_surface_native_v1',input_track='track_1',input_sha256='d'*64,
        hand_labeled_test=False,media_decoded=False,frame_poses_changed=False,input_video_hashed=True,budget_seconds=600,
        maximum_qem_calls=1,object_scale=1.,metric_scale_baked_once=s,oracle_modes=[],elapsed_seconds=1.,source_hashes=source,compiler=c,
        source_geometry=dict(source_identity=loader.identity(root/names['source_glb']),native_float32_conversion=True,
            source_vertices_preserved=True,source_faces_preserved=True,welding_performed=False,geometry_repaired=False))
    npin=seal('native',native)
    host=fields|dict(stage='world_reward_object_budget_surface_host_v1',owned_container_removed=True,owned_scratch_removed=True,
        native=native,native_identity=npin,image_identity={'Id':loader.IMAGE},elapsed_seconds=2.)
    hpin=seal('report',host);(root/names['report']).parent.chmod(0o555)
    helpers={n:write(code/n,('# '+n).encode()) for n in loader.SOURCE_HELPERS}
    pins=dict(schema=loader.SCHEMA,episode_index=e,input_sha256='d'*64,metric_scale_baked_once=s,
        report=hpin|dict(producer_revision=p,script_sha256=ps['helpers'][loader.PROCESSOR]['sha256']),
        source_helpers=helpers,files={n:loader.identity(root/n) for n in names.values()})
    pinpath=code/f'configs/surface_mesh_{e:06d}_pins.json';write(pinpath,json.dumps(pins).encode())
    return root,code,pins,e,s,osh,ash,names


def test_full_manufactured_receipt_authentication_and_three_complete_snapshots(tmp_path,monkeypatch):
    root,code,pins,e,s,osh,ash,names=receipt_fixture(tmp_path,monkeypatch)
    host,native,ledger,returned=loader.verify_pinned_artifacts(root,pins,e,'d'*64,osh,ash,s)
    assert returned==names and host['native']==native
    assert len([p for p in ledger if p.name=='source-sha256'])==3


@pytest.mark.parametrize('mutation',['selfpins','sourcehelper','hostfail','wronginput','extraartifact','oldrelabel'])
def test_full_receipt_no_selfauth_or_legacy_failure_relabel(tmp_path,monkeypatch,mutation):
    root,code,pins,e,s,osh,ash,names=receipt_fixture(tmp_path,monkeypatch)
    if mutation=='selfpins':pins=deepcopy(pins);pins['metric_scale_baked_once']=s+1
    elif mutation=='sourcehelper':write(code/loader.SOURCE_HELPERS[0],b'# changed')
    elif mutation=='wronginput':osh='0'*64
    elif mutation=='extraartifact':
        pins=deepcopy(pins);pins['files']['outputs/episode_000006/foreign.npz']=dict(bytes=1,sha256='0'*64)
        write(code/f'configs/surface_mesh_{e:06d}_pins.json',json.dumps(pins).encode())
    else:
        role='report' if mutation=='hostfail' else 'qualification_host'
        row=json.loads((root/names[role]).read_bytes());row['status']='fail' if mutation=='hostfail' else 'pass'
        pins=deepcopy(pins);pin=write(root/names[role],json.dumps(row).encode());pins['files'][names[role]]=pin
        if role=='report':pins['report'].update(pin)
        write(code/f'configs/surface_mesh_{e:06d}_pins.json',json.dumps(pins).encode())
    with pytest.raises(ValueError):loader.verify_pinned_artifacts(root,pins,e,'d'*64,osh,ash,s)
