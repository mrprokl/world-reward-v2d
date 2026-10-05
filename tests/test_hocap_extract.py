"""Authored tiny saved ZIPs only; no public archive, reference or native calls."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name,ROOT/'infra'/f'{name}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def pin(raw): return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def mutate_owned(path,raw):
    assert path.stat().st_uid==os.getuid() and not path.is_symlink()
    mode=stat.S_IMODE(path.stat().st_mode); path.chmod(0o600)
    path.write_bytes(raw); path.chmod(mode)


def saved_zip(rows):
    out = io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for n,raw in rows: z.writestr(n,raw)
    return out.getvalue()


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    m,h,rt = map(module,('hocap_extract','hocap_acquire','mediapipe_cpu_runtime_verify'))
    p = json.loads((ROOT/m.PROTOCOL).read_bytes()); p['source_owner_uid'] = os.getuid(); p['minimum_free_bytes'] = 1
    base = tmp_path/'original'; (base/'quarantine').mkdir(parents=True,mode=0o700); (base/'primary').mkdir()
    p['original'] = str(base); p['output'] = str(tmp_path/'new')
    prior = json.loads((ROOT/'configs/hocap_acquisition_protocol_v1.json').read_bytes()); prior['output'] = str(base)
    raw = b'Creative Commons Attribution 4.0 International License (CC BY 4.0) https://creativecommons.org/licenses/by/4.0/'
    p['primary_sources'] = {'publisher.html':pin(raw)}; (base/'primary/publisher.html').write_bytes(raw); (base/'primary/publisher.html').chmod(0o444)
    rows = []
    for clip in p['clips']:
        prefix = f"subject_5/{clip}/"
        rows.append((prefix+'meta.yaml',b'num_frames: 2\nprivate: !!python/object:NOT_CONSTRUCTED\n'))
        for t in range(2):
            rows += [(prefix+p['camera']+f'/color_{t:06d}.jpg',b'tinyRGB'+bytes([t])),
                     (prefix+p['camera']+f'/depth_{t:06d}.png',b'NEVER_READ_DEPTH')]
    p['archives'] = {}
    for name in ('subject_5.zip','labels.zip','poses.zip','models.zip','calibration.zip'):
        data = rows if name=='subject_5.zip' else [(name[:-4]+'/private.bin',b'NEVER_READ_REFERENCE')]
        raw = saved_zip(data); path=base/'quarantine'/name; path.write_bytes(raw); path.chmod(0o400)
        p['archives'][name] = dict(**pin(raw),members=len(data),expanded_bytes=sum(len(x[1])for x in data),maximum_member_bytes=max(len(x[1])for x in data))
    p['maximum_zip_members'] = max(r['members']for r in p['archives'].values())
    prior['archives'] = [dict(file=n,bytes=r['bytes'])for n,r in p['archives'].items()]
    original = tmp_path/'root/jobs'/p['producer_revision']/'run_hocap_acquire/code'
    (original/'configs').mkdir(parents=True); (original/'infra').mkdir()
    code=tmp_path/'code'; (code/'infra').mkdir(parents=True)
    helpers={}
    for name in ('infra/hocap_acquire.py','infra/mediapipe_cpu_runtime_verify.py'):
        b=(ROOT/name).read_bytes()
        for directory in (original,code): (directory/name).write_bytes(b); (directory/name).chmod(0o444)
        helpers[name]=pin(b)
    b=(json.dumps(prior)+'\n').encode(); (original/'configs/hocap_acquisition_protocol_v1.json').write_bytes(b); (original/'configs/hocap_acquisition_protocol_v1.json').chmod(0o444)
    p['original_protocol']=pin(b)
    source=dict(closure_sha256=p['source_closure_sha256'],helpers=helpers,producer_revision=p['producer_revision'])
    old=dict(schema='world_reward.hocap_acquisition.v1',stage='hocap_rgb_private_reference_acquisition',status='fail',phase='safe_inventory',
        producer_revision=p['producer_revision'],error_type='ValueError',error_context='ZIP member cap exceeded',download_complete=True,
        source_archive_public_rehashed_after=True,public_inventory_qualified=False,label_values_parsed=False,calibration_values_parsed=False,
        model_loaded=False,gpu_used=False,challenge_inputs_used=False,adoption=False,source_before=source,protocol_identity=p['original_protocol'],
        primary_sources=p['primary_sources'],archives={n:{k:r[k]for k in ('bytes','sha256')}for n,r in p['archives'].items()})
    b=(json.dumps(old)+'\n').encode(); (base/'report.json').write_bytes(b); (base/'report.json').chmod(0o400); p['report']=pin(b)
    monkeypatch.setattr(m,'ROOT',tmp_path/'root'); monkeypatch.setattr(rt,'source',lambda *_:copy.deepcopy(source))
    current=dict(closure_sha256='c'*64,producer_revision='d'*40)
    monkeypatch.setattr(m,'source_binding',lambda *_:(copy.deepcopy(p),copy.deepcopy(current)))
    monkeypatch.setattr(h,'fetch',lambda *_a,**_k:pytest.fail('No networking/download authorized'))
    return SimpleNamespace(m=m,h=h,rt=rt,p=p,prior=prior,old=old,base=base,code=code,target=Path(p['output']),current=current)


def test_frozen_original_receipt_census_selectors_and_limits():
    m=module('hocap_extract'); b=(ROOT/m.PROTOCOL).read_bytes(); p=json.loads(b)
    assert pin(b)==m.PROTOCOL_PIN
    assert p['report']==dict(bytes=3224,sha256='4252b1a131bf36ec4257be0c43d334d562f4685ca974bd14f9ac5e33c10bb86e')
    assert p['maximum_zip_members']==583768 and p['budget_seconds']==600 and p['cleanup_grace_seconds']==60
    assert sum(r['bytes']for r in p['archives'].values())==5080226520
    assert p['clips']==['20231027_112303','20231027_113202'] and p['camera']=='105322251564'
    assert 'expected_frames' not in p and p['network_used'] is p['label_values_parsed'] is False


def test_full_saved_only_flow_keeps_original_fail_and_all_frames(fixture,monkeypatch):
    f=fixture; before={n:f.h.identity(Path(n),r['bytes'])for n,r in f.m.authenticate(f.rt,f.h,f.code,f.p)[1]['original_files'].items()}
    original=zipfile.ZipFile.open; opened=[]
    def permitted(self,name,*a,**k):
        key=name.filename if isinstance(name,zipfile.ZipInfo)else name
        assert key.endswith('meta.yaml')or '/color_' in key; opened.append(key); return original(self,name,*a,**k)
    monkeypatch.setattr(zipfile.ZipFile,'open',permitted)
    r=f.m.extract(f.rt,f.h,f.code,'d'*40,f.target)
    assert r['status']=='pass' and r['frames']==4 and len(opened)==6
    assert r['source_archives_public_rehashed_after'] and r['original_ancestry']['original_failure_preserved']
    assert json.loads((f.base/'report.json').read_bytes())['status']=='fail'
    assert before=={n:f.h.identity(Path(n),p['bytes'])for n,p in before.items()}
    manifest=json.loads((f.target/'inputs/manifest.json').read_bytes())
    assert [r['source_frame_id']for r in manifest['images']]==[0,1,0,1]
    assert 'PRIVATE'not in json.dumps(manifest) and not tuple((f.target/'inputs').rglob('*.yaml'))
    assert (f.target/'metadata_private').stat().st_mode&0o777==0o700
    assert all(p.stat().st_mode&0o777==0o400 for p in (f.target/'metadata_private').iterdir())
    assert all((f.target/'inputs'/r['file']).stat().st_mode&0o777==0o444 for r in manifest['images'])
    assert all((f.target/'inputs'/r['clip']).stat().st_mode&0o777==0o555 for r in manifest['clips'])
    assert (f.target/'report.json').stat().st_mode&0o777==0o444
    assert f.prior['maximum_zip_members']==250000


@pytest.mark.parametrize('key,value',[('status','pass'),('phase','public_extract'),('download_complete',1),('gpu_used',True),('error_context','other')])
def test_exact_old_failure_scope_required_before_payload(fixture,key,value):
    f=fixture; f.old[key]=value; b=(json.dumps(f.old)+'\n').encode()
    mutate_owned(f.base/'report.json',b); f.p['report']=pin(b)
    with pytest.raises(ValueError): f.m.authenticate(f.rt,f.h,f.code,f.p)


@pytest.mark.parametrize('role',['report','archive','primary','helper','closure'])
def test_tampered_original_receipts_archives_sources_rejected(fixture,monkeypatch,role):
    f=fixture
    paths={'report':f.base/'report.json','archive':f.base/'quarantine/labels.zip',
        'primary':f.base/'primary/publisher.html','helper':f.code/'infra/hocap_acquire.py'}
    if role=='closure': monkeypatch.setattr(f.rt,'source',lambda *_:dict(closure_sha256='0'*64))
    else: mutate_owned(paths[role],paths[role].read_bytes()+b'tamper')
    with pytest.raises(ValueError): f.m.authenticate(f.rt,f.h,f.code,f.p)


@pytest.mark.parametrize('key',['members','expanded_bytes','maximum_member_bytes'])
def test_independent_census_not_just_larger_generic_capacity(fixture,key):
    f=fixture; f.p['archives']['labels.zip'][key]+=1
    r=f.m.extract(f.rt,f.h,f.code,'d'*40,f.target)
    assert r['status']=='fail' and r['phase']=='safe_inventory' and not r['public_inventory_qualified']
    assert not tuple((f.target/'inputs').iterdir())


def test_original_safe_zip_guards_retained_with_ephemeral_count_capacity(fixture,monkeypatch):
    f=fixture; original=f.h.inventory
    def unsafe(path,p,deadline):
        assert p['maximum_zip_members']==f.p['maximum_zip_members']
        assert p['maximum_member_bytes']==f.prior['maximum_member_bytes']
        raise ValueError('Unsafe ZIP name/type/compression/size')
    monkeypatch.setattr(f.h,'inventory',unsafe)
    r=f.m.extract(f.rt,f.h,f.code,'d'*40,f.target)
    assert r['status']=='fail' and not r['public_inventory_qualified']


def test_posthash_failure_never_promotes_modified_saved_source(fixture,monkeypatch):
    f=fixture; original=f.m.authenticate; calls=[]
    def changing(*a):
        result=original(*a); calls.append(1)
        if len(calls)>1: raise ValueError('Original failed source/archives changed')
        return result
    monkeypatch.setattr(f.m,'authenticate',changing)
    r=f.m.extract(f.rt,f.h,f.code,'d'*40,f.target)
    assert len(calls)==2 and r['status']=='fail' and r['post_error_type']=='ValueError'
    assert not r.get('source_archives_public_rehashed_after',False)
    assert json.loads((f.target/'report.json').read_bytes())['status']=='fail'


def test_foreign_hardlink_and_archive_mode_fail_before_zip_decode(fixture):
    f=fixture; p=f.base/'quarantine/labels.zip'; p.chmod(0o600)
    with pytest.raises(ValueError): f.m.authenticate(f.rt,f.h,f.code,f.p)
    p.chmod(0o400); os.link(p,f.base/'alias.zip')
    with pytest.raises(ValueError): f.m.authenticate(f.rt,f.h,f.code,f.p)


def test_exact_lease_no_overwrite_and_no_late_pass(fixture):
    f=fixture; f.target.mkdir(mode=0o700); s=f.target.stat()
    wrong=dict(device=s.st_dev,inode=s.st_ino+1,source_sha256=f.current['closure_sha256'])
    with pytest.raises(ValueError): f.m.extract(f.rt,f.h,f.code,'d'*40,f.target,reservation=wrong)
    lease={**wrong,'inode':s.st_ino}
    assert f.m.extract(f.rt,f.h,f.code,'d'*40,f.target,reservation=lease)['status']=='pass'
    with pytest.raises(ValueError): f.m.extract(f.rt,f.h,f.code,'d'*40,f.target)
    r={'status':'pass'}; f.h.write_report(f.target/'late.json',r,0o444,deadline=0)
    assert json.loads((f.target/'late.json').read_bytes())['status']=='fail'


def test_source_and_shell_only_saved_source_text_and_no_network():
    src=(ROOT/'infra/hocap_extract.py').read_text(); shell=(ROOT/'infra/run_hocap_extract.sh').read_text()
    for token in ('urllib','requests','h.fetch(','.extractall(','import torch','import numpy','safe_load','np.load'):
        assert token not in src
    assert "local['maximum_zip_members'] = p['maximum_zip_members']"in src
    assert 'rt.source(ROOT, original, revision'in src and 'h.extract_public('in src
    assert 'ulimit -v 3145728'in shell and '660s'in shell and 'runuser -u scenesmith'in shell
    assert 'docker'not in shell and 'nvidia'not in shell
