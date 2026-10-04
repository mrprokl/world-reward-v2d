"""Manufactured receipt composition only; no query/QEM/geometry qualification."""
from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np
import pytest

from test_solid_geometry_loader import fixture,seal,write,loader


def balanced_fixture(tmp_path,monkeypatch):
    root,code,pins,oldnames,host=fixture(tmp_path,monkeypatch)
    q=loader.strict_json((code/loader.QUALIFICATION).read_bytes())
    g=loader.strict_json((code/loader.CGAL).read_bytes());g.update(producer_revision='c'*40,parent_image_id='sha256:'+'f'*64)
    seal(code/loader.CGAL,g)
    active=deepcopy(g);active.update(producer_revision='e'*40,native_source=dict(bytes=2,sha256='9'*64))
    active.update({n:dict(bytes=2,sha256=str(i)*64)for i,n in enumerate(('report','native','binary'),4)})
    apin=seal(code/loader.BALANCED_CGAL,active)
    cache=dict(pins_identity=dict(bytes=1,sha256='a'*64),artifacts={n:dict(bytes=1,sha256='b'*64)
        for n in ('host_report','native_report','retained_binary')},build_info={'physical_coordinate_cache':True},compiler='native c++')
    qr=dict(original_query=dict(pins_identity=loader.identity(code/loader.CGAL),artifacts={n:g[n]for n in ('report','native','binary')},
            native_source=g['native_source'],producer_revision=g['producer_revision']),
        active_query=dict(pins_identity=apin,artifacts={n:active[n]for n in ('report','native','binary')},
            native_source=active['native_source'],producer_revision=active['producer_revision']),
        original_qualified_inputs=dict(cgal={n:g[n]for n in ('report','native','binary')},cache=cache,historical_image_receipt=dict(bytes=1,sha256='c'*64)),
        original_qem_build_unchanged=True,original_qem_recompiled=False,procedural_controls_required=15)
    names=loader.paths(8,q['producer_revision'],pins['report']['producer_revision'],query_requalification=True)
    for k in ('geometry','glb'):write(root/names[k],(root/oldnames[k]).read_bytes())
    qhost=loader.strict_json((root/oldnames['qualification_host']).read_bytes());qnative=qhost['native']
    build=host['qualification']['build'];build['query_requalification']=qr
    for row in (qhost,qnative):row['qualified_build']=build;row['query_requalification']=qr
    q['report']=seal(root/names['qualification_host'],qhost);q['native']=seal(root/names['qualification_native'],qnative)
    qp=seal(code/loader.BALANCED_QUALIFICATION,q)
    proof=host['qualification'];proof.update(pins_identity=qp,report=q['report'],native=q['native'],query_requalification=qr)
    for row in (host,host['native']):row.update(qualification=proof,query_requalification=qr)
    def replace_sha(row):
        if isinstance(row,dict):
            if 'native_certificate'in row:row['native_certificate']['source_sha256']=active['native_source']['sha256']
            for value in row.values():replace_sha(value)
        elif isinstance(row,list):
            for value in row:replace_sha(value)
    replace_sha(host['native']['compiler'])
    save(root,pins,names,host)
    return root,code,pins,names,host,qr


def save(root,pins,names,host):
    parent=(root/names['report']).parent
    if parent.exists():parent.chmod(0o755)
    seal(root/names['native'],host['native']);hp=seal(root/names['report'],host)
    pins['report'].update(hp);pins['files']={name:loader.identity(root/name)for name in names.values()}
    parent.chmod(0o555)


def verify(data,**kwargs):
    root,_,pins,names,_,_=data
    return loader.verify_pinned_artifacts(root,pins,8,'d'*64,pins['files'][names['object']]['sha256'],
        pins['files'][names['alignment']]['sha256'],.375,**kwargs)


def test_opt_in_balanced_preserves_eleven_files_and_old_build_query(tmp_path,monkeypatch):
    data=balanced_fixture(tmp_path,monkeypatch);host,native,_,names=verify(data,query_requalification=True)
    assert len(data[2]['files'])==11 and host['native']==native
    assert names['report'].split('/')[-2].startswith('object_budget_solid_balanced_')
    assert '/solid-chart-v2-query-requalify-'in names['qualification_host']
    assert host['qualification']['build']['cgal']==host['query_requalification']['original_query']['artifacts']
    assert host['query_requalification']['active_query']['native_source']!=host['query_requalification']['original_query']['native_source']
    with pytest.raises(ValueError):verify(data)


def test_default_legacy_profiles_do_not_require_balanced_pins(tmp_path,monkeypatch):
    root,code,pins,names,_=fixture(tmp_path,monkeypatch)
    loader.verify_pinned_artifacts(root,pins,8,'d'*64,pins['files'][names['object']]['sha256'],pins['files'][names['alignment']]['sha256'],.375)
    assert not(code/loader.BALANCED_QUALIFICATION).exists()
    with pytest.raises(FileNotFoundError):loader.verify_pinned_artifacts(root,pins,8,'d'*64,pins['files'][names['object']]['sha256'],pins['files'][names['alignment']]['sha256'],.375,query_requalification=True)


@pytest.mark.parametrize('flag',[None,0,1,'balanced',[],{}])
def test_selector_is_explicit_bool_before_any_artifact_read(flag):
    with pytest.raises(ValueError):loader.paths(8,'b'*40,'a'*40,query_requalification=flag)
    with pytest.raises(ValueError):loader.verify_pinned_artifacts(Path('/absent'),None,8,'d'*64,'e'*64,'f'*64,.375,query_requalification=flag)
    with pytest.raises(ValueError):loader.load(Path('/absent'),8,'d'*64,'e'*64,'f'*64,.375,query_requalification=flag)


@pytest.mark.parametrize('fault',['original_pin','active_pin','original_artifact','active_artifact','source','revision','old_build_relabel',
    'image','recompile','qem_changed','controls','controls_bool','original_inputs','swapped','missing_qr','qhost','qnative','production_native','old_query_in_compiler'])
def test_wrong_original_active_query_or_proof_cannot_be_relabelled(tmp_path,monkeypatch,fault):
    data=balanced_fixture(tmp_path,monkeypatch);root,code,pins,names,host,qr=data
    if fault=='original_pin':qr['original_query']['pins_identity']['sha256']='0'*64
    elif fault=='active_pin':qr['active_query']['pins_identity']['sha256']='0'*64
    elif fault=='original_artifact':qr['original_query']['artifacts']['binary']['sha256']='0'*64
    elif fault=='active_artifact':qr['active_query']['artifacts']['binary']['sha256']='0'*64
    elif fault=='source':qr['active_query']['native_source']['sha256']='0'*64
    elif fault=='revision':qr['active_query']['producer_revision']='0'*40
    elif fault=='old_build_relabel':host['qualification']['build']['cgal']=deepcopy(qr['active_query']['artifacts'])
    elif fault=='image':
        p=loader.strict_json((code/loader.BALANCED_CGAL).read_bytes());p['child_image_id']='sha256:'+'0'*64;seal(code/loader.BALANCED_CGAL,p)
    elif fault=='recompile':qr['original_qem_recompiled']=True
    elif fault=='qem_changed':qr['original_qem_build_unchanged']=False
    elif fault=='controls':qr['procedural_controls_required']=14
    elif fault=='controls_bool':qr['procedural_controls_required']=True
    elif fault=='original_inputs':qr['original_qualified_inputs']['cgal']=deepcopy(qr['active_query']['artifacts'])
    elif fault=='swapped':qr['original_query'],qr['active_query']=qr['active_query'],qr['original_query']
    elif fault=='missing_qr':host['qualification']['build'].pop('query_requalification')
    elif fault in ('qhost','qnative'):
        qhost=loader.strict_json((root/names['qualification_host']).read_bytes())
        if fault=='qhost':qhost['query_requalification']['original_qem_recompiled']=True
        else:qhost['native']['query_requalification']['original_qem_recompiled']=True
        q=loader.strict_json((code/loader.BALANCED_QUALIFICATION).read_bytes())
        q['report']=seal(root/names['qualification_host'],qhost);q['native']=seal(root/names['qualification_native'],qhost['native'])
        qp=seal(code/loader.BALANCED_QUALIFICATION,q);host['qualification'].update(pins_identity=qp,report=q['report'],native=q['native'])
    elif fault=='production_native':host['native']['query_requalification']=deepcopy(qr);host['native']['query_requalification']['original_qem_recompiled']=True
    else:host['native']['compiler']['stages']['physical_source']['native_certificate']['source_sha256']='c'*64
    save(root,pins,names,host)
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):verify(data,query_requalification=True)


def test_load_transmits_opt_in_without_decoding_when_prerequisite_fails(monkeypatch):
    assert all(inspect.signature(getattr(loader,n)).parameters['query_requalification'].default is False
        for n in ('paths','verify_pinned_artifacts','load'))
    calls=[]
    def fail(*args,**kwargs):calls.append(kwargs);raise ValueError('Pinned prerequisite not ready')
    monkeypatch.setattr(loader,'verify_pinned_artifacts',fail)
    with pytest.raises(ValueError):loader.load('/absent',8,'d'*64,'e'*64,'f'*64,.375,pins={},query_requalification=True)
    assert calls==[{'query_requalification':True}]


@pytest.mark.parametrize('fault',['missing_active','missing_controls','writable_active','writable_controls'])
def test_new_fixed_pin_files_are_mandatory_and_readonly(tmp_path,monkeypatch,fault):
    data=balanced_fixture(tmp_path,monkeypatch);_,code,_,_,_,_=data
    path=code/(loader.BALANCED_CGAL if fault.endswith('active')else loader.BALANCED_QUALIFICATION)
    if fault.startswith('missing'):path.unlink()
    else:path.chmod(0o644)
    with pytest.raises((ValueError,FileNotFoundError)):verify(data,query_requalification=True)


def test_balanced_payload_read_keeps_exact_arrays_and_exposes_bound_query_only(tmp_path,monkeypatch):
    data=balanced_fixture(tmp_path,monkeypatch);root,code,pins,names,host,qr=data
    import exact_mesh_geometry as geometry
    import object_budget_endpoint as endpoint
    v=np.array([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]])
    f=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]],np.int64)
    pv=np.r_[v,np.repeat(v[:1],4092,axis=0)]*.375;pf=np.r_[f,np.zeros((4092,3),np.int64)]
    compact=endpoint.exact_weld(v*.375,f)[:2]
    final=host['native']['compiler']['stages'][loader.STAGES[-1]]
    final['topology']=final['fidelity']['candidate_topology']=geometry.exact_mesh_topology(*compact)
    import hashlib,json
    final['stored_array_sha256']=[hashlib.sha256(json.dumps(dict(dtype=a.dtype.str,shape=a.shape),sort_keys=True).encode()+b'\0'+a.tobytes()).hexdigest()for a in compact]
    path=root/names['geometry'];path.chmod(0o644)
    with path.open('wb')as stream:np.savez(stream,vertices=pv,faces=pf,episode_index=np.array(8,np.int64),object_scale=np.array(1.),grounded_scale_baked=np.array(.375))
    path.chmod(0o444);save(root,pins,names,host)
    calls=[]
    def verified(*_,**kwargs):calls.append(kwargs);return host,host['native'],pins['files'],names
    monkeypatch.setattr(loader,'verify_pinned_artifacts',verified)
    monkeypatch.setattr(geometry,'__file__',str(code/'infra/exact_mesh_geometry.py'))
    monkeypatch.setattr(endpoint,'__file__',str(code/'infra/object_budget_endpoint.py'))
    monkeypatch.setattr(endpoint,'_load_mesh',lambda _: (v.copy(),f.copy()))
    actual=loader.load(root,8,'d'*64,'e'*64,'f'*64,.375,pins=pins,query_requalification=True)
    assert calls==[{'query_requalification':True}]
    np.testing.assert_array_equal(actual[0],pv);np.testing.assert_array_equal(actual[1],pf)
    assert actual[5]['query_requalification']==qr and actual[5]['challenge_performance_verified'] is False
