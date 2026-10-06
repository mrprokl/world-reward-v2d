"""Opaque manufactured bytes only: no actual poses, RGB, models or network."""
import ast
import base64
import copy
import hashlib
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest
import vcoco_full_pose_replica as v

R, S = '1'*40, '2'*40


def readonly(path, raw):
    path.write_bytes(raw); path.chmod(0o400); return v.pin(raw)


def local_private(path, mode, names=None):
    s = v.rt.canonical(path).lstat()
    v.rt.require(stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode) == mode
        and s.st_uid == os.getuid() and s.st_gid == os.getgid()
        and (names is None or {p.name for p in path.iterdir()} == set(names)), 'Authored process-owned namespace')


@pytest.fixture(autouse=True)
def fixture_group(monkeypatch, tmp_path):
    monkeypatch.setattr(v.os, 'getgid', lambda: tmp_path.lstat().st_gid)


def fixture():
    rows, payload = [], {}
    for i in range(48):
        n = i % 3; iid = f'{i:032x}'; name = f'image_{i:06d}.npz'; raw = f'opaque authored pose {i}'.encode()
        ids = [f'image:{iid}/person/retained:{j:06d}' for j in range(n)]
        shapes = dict(person_ids=[n], boxes_original_xyxy=[n,4], detector_scores=[n], keypoints_original_xy=[n,133,2],
            raw_scores=[n,133], native_valid=[n,133], in_original_image=[n,133], image_size=[2],
            original_frame_index=[], original_slot=[], acquired_ordinal=[])
        dtypes = dict(person_ids='<U70', boxes_original_xyxy='<f8', detector_scores='<f8', keypoints_original_xy='<f8',
            raw_scores='<f4', native_valid='|b1', in_original_image='|b1', image_size='<i8', original_frame_index='<i8', original_slot='<i8', acquired_ordinal='<i8')
        row = dict(image_id=iid, original_slot=i, acquired_ordinal=i, original_frame_index=0, image_size=[12,16],
            file=name, identity=v.pin(raw), endpoint_bank_identity=v.pin(b'original endpoint'+bytes([i])),
            source_person_ids=ids, person_ids=ids, owl_patches=3600, persons=n,
            arrays={k:dict(shape=s, dtype=dtypes[k], sha256=hashlib.sha256((k+str(i)).encode()).hexdigest()) for k,s in shapes.items()})
        rows.append(row); payload['banks/'+name] = raw
    public = dict(schema=v.PUBLIC_SCHEMA, rows=rows); payload['public-pose.json'] = v.encode(public)
    pins = {name:v.pin(name.encode()) for name in ('report.json','native.json','proof.json')}
    manifest = dict(schema=v.SCHEMA, export_revision=R, original_source_declaration=v.DECLARATION,
        pose_receipt_pins=pins, files={n:v.pin(p) for n,p in payload.items()})
    return manifest, public, payload


def archive_raw(manifest, payload, members=None, format=tarfile.USTAR_FORMAT):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w', format=format) as archive:
        for name, raw, kind in members or [('manifest.json', v.encode(manifest), tarfile.REGTYPE),
                *[(n,payload[n],tarfile.REGTYPE) for n in sorted(payload)]]:
            row = tarfile.TarInfo(name); row.size = len(raw); row.mode = 0o400; row.type = kind
            if kind in (tarfile.SYMTYPE,tarfile.LNKTYPE): row.linkname = 'foreign'
            archive.addfile(row, io.BytesIO(raw))
    return stream.getvalue()


def test_complete49_leaves50_members_full11_and_zero_person(tmp_path):
    m,p,payload = fixture(); before = copy.deepcopy((m,p)); raw = archive_raw(m,payload); path = tmp_path/'archive'; ap = readonly(path,raw)
    assert len(v.NAMES) == len(m['files']) == 49 and v.validate_manifest(m,R) == m and v.validate_projection(p,m['files']) == p
    assert p['rows'][0]['persons'] == 0 and all(len(r['arrays']) == 11 for r in p['rows'])
    result,table = v.verify_archive(path,ap,v.pin(v.encode(m)),R,time.monotonic()+10)
    assert result == m and len(table) == 50
    with path.open('rb') as stream:
        for name,offset,size in table[1:]: stream.seek(offset); assert stream.read(size) == payload[name]
    assert (m,p) == before and not any(n.endswith(('.jpg','report.json','native.json','proof.json')) for n in v.NAMES)


@pytest.mark.parametrize('change',['hole','extra','alias','revision','source','zero','bool','hash','large','receipt'])
def test_strict_manifest_rejects(change):
    m,_,_ = fixture()
    if change == 'hole': del m['files']['banks/image_000047.npz']
    elif change == 'extra': m['files']['private.json'] = v.pin(b'x')
    elif change == 'alias': m['files']['banks/./image_000000.npz'] = m['files'].pop('banks/image_000000.npz')
    elif change == 'revision': m['export_revision'] = S
    elif change == 'source': m['original_source_declaration'] = dict(v.DECLARATION,files=374)
    elif change == 'zero': m['files']['banks/image_000000.npz']['bytes'] = 0
    elif change == 'bool': m['files']['banks/image_000000.npz']['bytes'] = True
    elif change == 'hash': m['files']['banks/image_000000.npz']['sha256'] = 'bad'
    elif change == 'large': m['files']['banks/image_000000.npz']['bytes'] = (2 << 20)+1
    else: del m['pose_receipt_pins']['proof.json']
    with pytest.raises(ValueError): v.validate_manifest(m,R)


@pytest.mark.parametrize('change',['private','hole','duplicate','slot','ordinal','frame','persons','id','owl','shape','bool_shape','dtype','extra_array','bank_pin','public_pin'])
def test_public_projection_only_exact_all48(change):
    m,p,_ = fixture(); r = p['rows'][1]
    if change == 'private': p['roles'] = []
    elif change == 'hole': p['rows'].pop()
    elif change == 'duplicate': r['image_id'] = p['rows'][0]['image_id']
    elif change == 'slot': r['original_slot'] = True
    elif change == 'ordinal': r['acquired_ordinal'] = 0
    elif change == 'frame': r['original_frame_index'] = 1
    elif change == 'persons': r['persons'] = 2
    elif change == 'id': r['source_person_ids'] = ['private-native-id']
    elif change == 'owl': r['owl_patches'] = 128
    elif change == 'shape': r['arrays']['keypoints_original_xy']['shape'] = [1,132,2]
    elif change == 'bool_shape': r['arrays']['person_ids']['shape'] = [True]
    elif change == 'dtype': r['arrays']['detector_scores']['dtype'] = '<f4'
    elif change == 'extra_array': r['arrays']['selected_owner'] = r['arrays']['detector_scores']
    elif change == 'bank_pin': r['identity'] = v.pin(b'other')
    else: m['files']['public-pose.json'] = v.pin(b'other')
    with pytest.raises(ValueError): v.validate_projection(p,m['files'])


@pytest.mark.parametrize('change',['symlink','hardlink','duplicate','unknown','escape','alias','pax','mode','uid','size','hash','tail','truncated','padding','first'])
def test_fixed_archive_firewall(tmp_path,change):
    m,_,payload = fixture(); members = [('manifest.json',v.encode(m),tarfile.REGTYPE), *[(n,payload[n],tarfile.REGTYPE) for n in sorted(payload)]]
    if change in ('symlink','hardlink'): members[1] = (members[1][0],b'',tarfile.SYMTYPE if change == 'symlink' else tarfile.LNKTYPE)
    elif change == 'duplicate': members.append(members[1])
    elif change == 'unknown': members.append(('roles.json',b'x',tarfile.REGTYPE))
    elif change == 'escape': members[1] = ('../other',members[1][1],tarfile.REGTYPE)
    elif change == 'alias': members[1] = ('./'+members[1][0],members[1][1],tarfile.REGTYPE)
    elif change == 'pax': members[1] = ('x'*101,b'x',tarfile.REGTYPE)
    elif change == 'hash': members[1] = (members[1][0],b'bad',tarfile.REGTYPE)
    elif change == 'first': members[0],members[1] = members[1],members[0]
    raw = archive_raw(m,payload,members,tarfile.PAX_FORMAT if change == 'pax' else tarfile.USTAR_FORMAT)
    if change in ('mode','uid','size'):
        header = tarfile.TarInfo.frombuf(raw[:512],'utf-8','strict')
        if change == 'mode': header.mode = 0o444
        elif change == 'uid': header.uid = 1
        else: header.size += 1
        raw = header.tobuf(format=tarfile.USTAR_FORMAT)+raw[512:]
    elif change == 'tail': raw += b'x'+b'\0'*511
    elif change == 'truncated': raw = raw[:700]
    elif change == 'padding': offset = 512+len(v.encode(m)); raw = raw[:offset]+b'x'+raw[offset+1:]
    path = tmp_path/'archive'; ap = readonly(path,raw)
    with pytest.raises((ValueError,tarfile.TarError,UnicodeError)): v.verify_archive(path,ap,v.pin(v.encode(m)),R,time.monotonic()+10)


def local_install(monkeypatch,tmp_path):
    tmp_path.chmod(0o700); monkeypatch.setattr(v,'DEST',tmp_path/'replica'); monkeypatch.setattr(v,'private_directory',local_private)
    def rename(stage,dest):
        if dest.exists(): raise FileExistsError()
        stage.rename(dest)
    monkeypatch.setattr(v.atomic,'rename_noreplace',rename)
    m,p,payload = fixture(); path = tmp_path/'archive'; ap = readonly(path,archive_raw(m,payload))
    m,table = v.verify_archive(path,ap,v.pin(v.encode(m)),R,time.monotonic()+10)
    return m,p,path,table


def test_atomic_all49_install_sealed_nooverwrite(monkeypatch,tmp_path):
    m,p,path,table = local_install(monkeypatch,tmp_path); stage = tmp_path/'stage'; v.install(path,stage,m,table,time.monotonic()+10)
    assert not stage.exists() and len(list(v.DEST.rglob('*'))) == 50 and stat.S_IMODE(v.DEST.stat().st_mode) == 0o700
    assert stat.S_IMODE((v.DEST/'banks').stat().st_mode) == 0o500 and all(v.rt.identity(v.DEST/n) == q for n,q in m['files'].items())
    assert v.rt.pinned(v.DEST/'public-pose.json',m['files']['public-pose.json']) == p
    with pytest.raises(ValueError): v.install(path,stage,m,table,time.monotonic()+10)


@pytest.mark.parametrize('fault',['timeout','race','foreign_stage'])
def test_install_partial_or_race_never_removes_foreign(monkeypatch,tmp_path,fault):
    m,_,path,table = local_install(monkeypatch,tmp_path); stage = tmp_path/'stage'
    if fault == 'timeout':
        calls = 0
        def stop(_):
            nonlocal calls
            calls += 1
            if calls == 3: raise TimeoutError()
        monkeypatch.setattr(v,'check',stop); wanted = TimeoutError
    elif fault == 'race':
        def race(a,b): b.mkdir(); (b/'foreign').write_bytes(b'untouched'); raise FileExistsError()
        monkeypatch.setattr(v.atomic,'rename_noreplace',race); wanted = FileExistsError
    else: stage.mkdir(); (stage/'foreign').write_bytes(b'untouched'); wanted = FileExistsError
    with pytest.raises(wanted): v.install(path,stage,m,table,time.monotonic()+10)
    if fault == 'race': assert (v.DEST/'foreign').read_bytes() == b'untouched' and not stage.exists()
    elif fault == 'foreign_stage': assert (stage/'foreign').read_bytes() == b'untouched'
    else: assert not stage.exists() and not v.DEST.exists()


def receipt(m):
    return dict(schema=v.SCHEMA,phase='export',status='pass',producer_revision=R,source_binding=dict(producer_revision=R),
        original_source_declaration=v.DECLARATION,pose_receipt_pins=m['pose_receipt_pins'],files=49,budget_seconds=v.BUDGET,
        source_inputs_rehashed_after=True,outputs_sealed=True,sender_source_live_verified=True,sender_runtime_live_verified=True,
        blob_etag='"unique-etag"',archive_identity=v.pin(b'archive'),manifest_identity=v.pin(v.encode(m)),**{k:False for k in v.FLAGS})


@pytest.mark.parametrize('fault',['fail','unsealed','count','private','source','sender','receipt_pin','error','etag'])
def test_independent_export_receipt_cannot_promote_unknown_or_fail(fault):
    m,_,_ = fixture(); r = receipt(m)
    if fault == 'fail': r['status'] = 'fail'
    elif fault == 'unsealed': r['outputs_sealed'] = False
    elif fault == 'count': r['files'] = 48
    elif fault == 'private': r['reference_metadata_read'] = True
    elif fault == 'source': r['original_source_declaration'] = {}
    elif fault == 'sender': r['sender_source_live_verified'] = False
    elif fault == 'receipt_pin': del r['pose_receipt_pins']['proof.json']
    elif fault == 'error': r['post_failure_stage'] = 'post'
    else: r['blob_etag'] = 'not quoted'
    raw = v.encode(r)
    with pytest.raises(ValueError): v.export_receipt(raw,v.pin(raw),R)


def cli_pins(pins):
    return [x for kind,name in (('host','report.json'),('native','native.json'),('proof','proof.json'))
        for x in ('--pose-'+kind+'-bytes',str(pins[name]['bytes']),'--pose-'+kind+'-sha256',pins[name]['sha256'])]


def test_cli_all_independent_pose_pins_and_duplicate_no_fallback():
    m,_,_ = fixture(); base = cli_pins(m['pose_receipt_pins']); assert v.arguments(['--phase','export',*base]).pose_pins == m['pose_receipt_pins']
    for args in (['--phase','export'],['--phase','export',*base,'--phase=export'],['--phase','export',*base,'--archive-bytes','1'],['--phase','import',*base]):
        with pytest.raises(ValueError): v.arguments(args)
    raw = v.encode(receipt(m)); args = ['--phase','import',*base,'--export-revision',R,'--export-receipt-base64',base64.b64encode(raw).decode()]
    for name,p in [('archive',v.pin(b'archive')),('manifest',v.pin(v.encode(m))),('export-receipt',v.pin(raw))]: args += ['--'+name+'-bytes',str(p['bytes']),'--'+name+'-sha256',p['sha256']]
    assert v.arguments(args).export_receipt_pin == v.pin(raw)
    with pytest.raises(ValueError): v.arguments(args+['--pose-host-bytes='+str(m['pose_receipt_pins']['report.json']['bytes'])])


def lifecycle(monkeypatch,tmp_path):
    root = tmp_path/'root'; (root/'results').mkdir(parents=True); data = tmp_path/'data'; data.mkdir(mode=0o700)
    monkeypatch.setattr(v,'ROOT',root); monkeypatch.setattr(v,'DEST',data/'replica'); monkeypatch.setattr(v,'private_directory',local_private)
    parent_identity=(1,2,stat.S_IFDIR|0o700,0,0)
    def parent(deadline,*,expected=None,verify_only=False):
        local_private(data,0o700); assert expected is None or expected==parent_identity
        return parent_identity
    monkeypatch.setattr(v,'bootstrap_parent',parent)
    m,p,payload = fixture(); originals = tmp_path/'originals'; originals.mkdir(); paths = {}
    for i,(n,raw) in enumerate(payload.items()):
        if n == 'public-pose.json': continue
        paths[n] = originals/str(i); readonly(paths[n],raw)
    monkeypatch.setattr(v,'source',lambda code,rev:dict(binding=dict(producer_revision=rev,helpers={}),states={}))
    calls = []; peers = []; stored = {}; blocks = {}; delete = [202]; post = []
    def sender(code,binding,pins,deadline): post.append('sender'); return p,dict(paths),m['files'],dict(stable=True)
    monkeypatch.setattr(v,'sender_inputs',sender); monkeypatch.setattr(v.transport,'verify_azure_peer',lambda phase:peers.append(phase))
    def observed(manifest):
        local_private(v.DEST,0o700,{'banks','public-pose.json'}); local_private(v.DEST/'banks',0o500,{Path(n).name for n in v.NAMES if n.startswith('banks/')})
        files = {str(v.DEST/n):v.rt.identity(v.DEST/n,2 << 20) for n in v.NAMES}; assert files == {str(v.DEST/n):q for n,q in manifest['files'].items()}
        assert all(stat.S_IMODE(Path(n).stat().st_mode) == 0o400 for n in files)
        return dict(files=files,states=v.tree_state(v.DEST),projection=v.validate_projection(v.rt.pinned(v.DEST/'public-pose.json',manifest['files']['public-pose.json']),manifest['files']))
    monkeypatch.setattr(v,'installed',observed)
    def rename(stage,dest):
        if dest.exists(): raise FileExistsError()
        stage.rename(dest)
    monkeypatch.setattr(v.atomic,'rename_noreplace',rename)
    class Response(io.BytesIO):
        def __init__(self,raw=b'',status=201,headers=None): super().__init__(raw); self.status=status; self.headers=headers or {}
    class Blob:
        def __init__(self,url,revision,managed_identity):
            assert managed_identity and url == 'https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/articulated-runtime-'+revision+'.tar'; self.rev=revision
        def request(self,method,data=None,query='',headers=None):
            calls.append((method,query,headers))
            if query.startswith('comp=block&'): blocks.setdefault(self.rev,[]).append(bytes(data)); return Response()
            if query == 'comp=blocklist': assert headers['If-None-Match']=='*'; stored[self.rev]=b''.join(blocks[self.rev]); return Response()
            if method == 'HEAD': return Response(status=200,headers={'Content-Length':str(len(stored[self.rev])),'ETag':'"unique-etag"'})
            if method == 'GET': return Response(stored[self.rev],200,{'Content-Length':str(len(stored[self.rev]))})
            if method == 'DELETE':
                assert headers == {'If-Match':'"unique-etag"'} and stat.S_IMODE((root/f'results/vcoco-full-pose-replica-import-{S}').stat().st_mode)==0o500
                post.append('delete'); return Response(status=delete[0])
            raise AssertionError(method)
    monkeypatch.setattr(v.transport,'Blob',Blob)
    return root,m,p,calls,peers,delete,post


def export_import(monkeypatch,tmp_path):
    state = lifecycle(monkeypatch,tmp_path); root,m,*_ = state
    export = v.run(v.arguments(['--phase','export',*cli_pins(m['pose_receipt_pins'])]),tmp_path/'code',R)
    raw = (root/f'results/vcoco-full-pose-replica-export-{R}'/'report.json').read_bytes()
    assert export['status']=='pass' and v.export_receipt(raw,v.pin(raw),R)==export
    args = SimpleNamespace(phase='import',export_revision=R,export_receipt_base64=base64.b64encode(raw).decode(),export_receipt_pin=v.pin(raw),
        archive_pin=export['archive_identity'],manifest_pin=export['manifest_identity'],pose_pins=m['pose_receipt_pins'])
    return state,args


def test_full_export_import49_reverse_peers_one_aftersealed_DELETE(monkeypatch,tmp_path):
    state,args = export_import(monkeypatch,tmp_path); root,m,p,calls,peers,_,post = state
    report = v.run(args,tmp_path/'code',S)
    assert report['status']=='pass' and report['source_inputs_rehashed_after'] and report['outputs_sealed'] and report['archive_removed']
    assert report['single_etag_DELETE_202'] and report['delete_attempts']==1 and report['blob_cleanup_verified']
    assert report['sender_source_live_verified'] is report['sender_runtime_live_verified'] is False and peers==['import','export']
    assert v.rt.pinned(v.DEST/'public-pose.json',m['files']['public-pose.json'])==p
    out = root/f'results/vcoco-full-pose-replica-import-{S}'; assert {q.name for q in out.iterdir()}=={'report.json','manifest.json','export-receipt.json'}
    assert all(stat.S_IMODE(q.stat().st_mode)==0o400 for q in out.iterdir()) and post[-1]=='delete'
    assert len([c for c in calls if c[0]=='DELETE'])==1 and not(out/'archive.tar').exists()


@pytest.mark.parametrize('failure',['sender','delete','post'])
def test_stage_fail_closed_no_exception_text_no_retry(monkeypatch,tmp_path,failure):
    if failure=='sender':
        root,m,_,calls,*_ = lifecycle(monkeypatch,tmp_path)
        def bad(*_): raise ValueError('SECRET_HTTP_TOKEN_AND_PRIVATE_TEXT')
        monkeypatch.setattr(v,'sender_inputs',bad); report=v.run(v.arguments(['--phase','export',*cli_pins(m['pose_receipt_pins'])]),tmp_path/'code',R); rev=R; phase='export'
        assert report['failure_stage']=='sender' and report['error_type']=='ValueError' and not calls
    else:
        state,args=export_import(monkeypatch,tmp_path); root,_,_,calls,_,delete,_=state
        if failure=='delete': delete[0]=500
        else:
            original=v.installed; count=0
            def drift(m):
                nonlocal count
                count+=1; value=original(m)
                return dict(value,changed=True) if count>=2 else value
            monkeypatch.setattr(v,'installed',drift)
        report=v.run(args,tmp_path/'code',S); rev=S; phase='import'
        assert report['archive_removed'] and v.DEST.exists()
        if failure=='delete': assert report['publication_failure_stage']=='after_seal' and report['delete_attempts']==1
        else: assert report['post_failure_stage']=='post' and report['delete_attempts']==0
        assert len([c for c in calls if c[0]=='DELETE'])==(1 if failure=='delete' else 0)
    raw=(root/f'results/vcoco-full-pose-replica-{phase}-{rev}'/'report.json').read_bytes()
    assert report['status']=='fail' and report['outputs_sealed'] and b'SECRET' not in raw and v.rt.strict(raw)==report


def test_receiver_full49_plus3_technical_and_original_pose_pins(monkeypatch,tmp_path):
    state,args=export_import(monkeypatch,tmp_path); root,m,p,_,_,_,_=state; imported=v.run(args,tmp_path/'code',S)
    out=root/f'results/vcoco-full-pose-replica-import-{S}'; receipt_pin=v.pin((out/'report.json').read_bytes())
    monkeypatch.setattr(v.rt,'source',lambda *args:imported['source_binding'])
    # Production uses root UID/GID; only this authored boundary substitutes the
    # already strict process-owned directory/leaf assertions on macOS fixtures.
    real_stat=Path.stat
    def owner_stat(path,*args,**kwargs):
        s=real_stat(path,*args,**kwargs)
        if kwargs.get('follow_symlinks',True) and (path.is_relative_to(root) or path.is_relative_to(v.DEST)):
            return SimpleNamespace(st_uid=0,st_gid=0,st_mode=s.st_mode)
        return s
    monkeypatch.setattr(Path,'stat',owner_stat)
    # canonical/identity use lstat; public metadata checks above still read real
    # inode/mode/link/hash state. Avoid Path.is_file's stat view in fake source.
    result=v.authenticate_receiver(tmp_path/'code',S,receipt_pin)
    assert result['rows']==p['rows'] and len(result['files'])==52 and result['original_pose_revision']==v.POSE_REV
    assert result['pose_receipt_pins']==m['pose_receipt_pins'] and result['import_identity']==receipt_pin
    assert result['sender_source_live_verified'] is result['sender_runtime_live_verified'] is False
    assert imported['status']=='pass' and result['projection_identity']==m['files']['public-pose.json']


@pytest.mark.parametrize('fault',['none','hostfail','source','helper','proof','cid'])
def test_actual_sender_shape_whole_source_and_all52_hashes(monkeypatch,tmp_path,fault):
    root=tmp_path/'root'; old=root/'jobs'/v.POSE_REV/v.pose.ENTRY/'code'; old.mkdir(parents=True)
    output=root/'results/pose'; output.mkdir(parents=True); monkeypatch.setattr(v,'ROOT',root); monkeypatch.setattr(v.pose,'OUTPUT',output)
    monkeypatch.setattr(v,'private_directory',local_private)
    m,p,payload=fixture(); rows=p['rows']; banks=[]; images=[]
    for row in rows:
        arrays={origin:row['arrays'][target] for target,origin in (('person_ids','person_retained_ids'),('boxes_original_xyxy','person_retained_boxes'),('detector_scores','person_retained_scores'))}
        banks.append(dict(person_retained_rows=row['persons'],person_ids=row['person_ids'],arrays=arrays,image_size=row['image_size'],identity=row['endpoint_bank_identity']))
        images.append(dict(image_id=row['image_id'])); readonly(output/row['file'],payload['banks/'+row['file']])
    projection=dict(banks=banks,inputs=dict(images=images)); projection_path=tmp_path/'endpoint.json'; projection_pin=readonly(projection_path,v.encode(projection))
    inputs=dict(projection_path=str(projection_path),projection_identity=projection_pin,import_source=dict(original=True))
    (old.parent/'source-sha256').write_text(v.DECLARATION['source_XZ_sha256']+'\n'); (old.parent/'revision').write_text(v.POSE_REV+'\n')
    for i in range(v.DECLARATION['files']): readonly(old/str(i),b'authored source')
    old.chmod(0o555)
    helpers={n:v.pin(n.encode()) for n in v.pose.helpers()}; binding=dict(helpers=helpers)
    source=dict(producer_revision=v.POSE_REV,entries=v.DECLARATION['entries'],closure_sha256=v.DECLARATION['closure_sha256'],helpers=copy.deepcopy(helpers),markers={})
    models=dict(source=dict(dwpose=True),assets=[]); monkeypatch.setattr(v.pose,'asset_context',lambda code:models); monkeypatch.setattr(v.pose,'image_state',lambda deadline:dict(Id=v.pose.IMAGE))
    call=dict(run_completed=True,supplied_container='list',supplied_array=dict(dtype='float64',shape=[3,384,288]),effective_feed_shape=[1,3,384,288],
        delegated_unmodified=True,effective_runtime_conversion_observed=False,raw_simcc=[dict(dtype='float32',shape=[1,133,576]),dict(dtype='float32',shape=[1,133,768])])
    native=dict(schema=v.pose.SCHEMA,stage='native_public48_person_pose',status='pass',phase='complete',producer_revision=v.POSE_REV,image_id=v.pose.IMAGE,
        models_loaded=1,models_released=True,source_inputs_assets_runtime_rehashed_after=True,private_prefix_removed=True,images=rows,
        persons=sum(r['persons'] for r in rows),native_calls=[call for r in rows for _ in range(r['persons'])],**{k:False for k in v.pose.FLAGS})
    proof=dict(schema=v.pose.SCHEMA,producer_revision=v.POSE_REV,image_id=v.pose.IMAGE,helpers={n:helpers[n] for n in v.pose.NATIVE_FILES},markers=source['markers'],
        public=dict(path=str(projection_path),identity=projection_pin),assets=models['assets'])
    if fault=='proof': proof['assets']=[dict(forged=True)]
    pp=readonly(output/'proof.json',v.encode(proof)); native['proof_identity']=pp; np=readonly(output/'native.json',v.encode(native))
    host=dict(schema=v.pose.SCHEMA,stage='public48_person_pose_host',status='pass',phase='complete',producer_revision=v.POSE_REV,source_binding=source,
        native_identity=np,native_exit_status=0,owned_cleanup_verified=True,source_inputs_assets_runtime_rehashed_after=True,outputs_sealed=True,
        image_id=v.pose.IMAGE,replica_revision=R,replica_identity=v.pin(b'original import'),dwpose_source=models['source'],input_import_source=inputs['import_source'],
        images=rows,persons=native['persons'],**{k:False for k in v.pose.FLAGS})
    if fault=='hostfail': host['status']='fail'
    hp=readonly(output/'report.json',v.encode(host)); readonly(output/'.container.cid',b'd'*64); output.chmod(0o500)
    monkeypatch.setattr(v.rt,'source',lambda *args:source)
    expected_helpers=v.pose.helpers(); monkeypatch.setattr(v.pose,'helpers',lambda:expected_helpers)
    monkeypatch.setattr(v.pose,'host_modules',lambda:(SimpleNamespace(authenticate_receiver=lambda *args:inputs),None,None))
    monkeypatch.setattr(v.pose.public.endpoint.original,'command',lambda *args:'d'*64 if fault=='cid' else '')
    real_stat=Path.stat
    def owner_stat(path,*args,**kwargs):
        s=real_stat(path,*args,**kwargs)
        return SimpleNamespace(st_mode=s.st_mode,st_uid=0,st_gid=0) if kwargs.get('follow_symlinks',True) and path.parent==output else s
    monkeypatch.setattr(Path,'stat',owner_stat)
    if fault=='source': source['closure_sha256']='0'*64
    if fault=='helper': binding['helpers']=dict(helpers); binding['helpers'][v.pose.NATIVE_FILES[0]]=v.pin(b'changed')
    pins=dict(zip(('report.json','native.json','proof.json'),(hp,np,pp)))
    if fault!='none':
        with pytest.raises(ValueError): v.sender_inputs(tmp_path/'code',binding,pins,time.monotonic()+10)
    else:
        public,paths,files,original=v.sender_inputs(tmp_path/'code',binding,pins,time.monotonic()+10)
        assert public==p and len(paths)==48 and len(files)==49 and len(original['files'])==52 and original['source']==source


def test_stdlib_host_import_and_no_qualified_global_mutation():
    source=Path(v.__file__).read_text(); tree=ast.parse(source)
    assert not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name)
        and t.value.id in ('shared','pose','transport','atomic','publication') for t in n.targets) for n in ast.walk(tree))
    assert 'pose.validate_native(n, projection, POSE_REV' in source and 'pose.cpu(' not in source and 'pose.observe(' not in source
    assert 'pose.asset_context(code)' in source and 'import dwpose_smoke' not in source and 'import keypoint_rgb_dwpose' not in source
    asset_tree=ast.parse(Path(v.pose.__file__).read_text())
    asset=next(n for n in asset_tree.body if isinstance(n,ast.FunctionDef) and n.name=='asset_context')
    assert not any(isinstance(n,ast.Import) and any(a.name in ('dwpose_smoke','keypoint_rgb_dwpose','numpy') for a in n.names) for n in ast.walk(asset))
    script='''import sys,importlib.abc
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('numpy','torch','PIL','onnxruntime','transformers'): raise ImportError('denied heavy import')
sys.meta_path.insert(0,Deny());sys.path[:0]=['infra','src'];import vcoco_full_pose_replica
'''
    result=subprocess.run([sys.executable,'-I','-B','-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_partial_control_write_prospectively_owned_and_never_foreign(monkeypatch,tmp_path):
    allowed=set(); real_fsync=v.os.fsync
    def stop(_): raise OSError('secret text never serialized')
    monkeypatch.setattr(v.os,'fsync',stop)
    with pytest.raises(OSError): v.control_write(tmp_path,'public-pose.json',b'authored',allowed)
    assert allowed=={'public-pose.json'} and (tmp_path/'public-pose.json').read_bytes()==b'authored'
    assert stat.S_IMODE((tmp_path/'public-pose.json').stat().st_mode)==0o400
    monkeypatch.setattr(v.os,'fsync',real_fsync); untouched=set()
    with pytest.raises(FileExistsError): v.control_write(tmp_path,'public-pose.json',b'foreign replacement',untouched)
    assert not untouched and (tmp_path/'public-pose.json').read_bytes()==b'authored'


def test_wrapper_own_namespace_bounds_and_bash_syntax():
    path=Path(v.__file__).with_name('run_vcoco_full_pose_replica.sh'); source=path.read_text()
    assert v.ENTRY in source and '315s' in source and 'ulimit -v 8388608' in source and 'env -i' in source and 'python3 -I -B' in source
    subprocess.run(['bash','-n',str(path)],check=True)


def anchored_fixture(monkeypatch,tmp_path):
    anchor=tmp_path/'srv'; anchor.mkdir(mode=0o755)
    parent=anchor/'world-reward-public-replicas'; destination=parent/'vcoco_full_pose_replica_v1'
    monkeypatch.setattr(v,'ANCHOR',anchor);monkeypatch.setattr(v,'PRIVATE_PARENT',parent);monkeypatch.setattr(v,'DEST',destination)
    real_lstat=Path.lstat; real_fstat=v.os.fstat; ancestors={anchor,*anchor.parents}
    def root_view(s,ancestor=False):
        fields={n:getattr(s,n)for n in dir(s)if n.startswith('st_')}
        fields.update(st_uid=0,st_gid=0)
        if ancestor: fields['st_mode'] &= ~0o022
        return SimpleNamespace(**fields)
    def lstat(path):
        s=real_lstat(path)
        return root_view(s,path in ancestors)if path in ancestors or path==parent else s
    monkeypatch.setattr(Path,'lstat',lstat)
    monkeypatch.setattr(v.os,'fstat',lambda fd:root_view(real_fstat(fd)))
    return anchor,parent,destination,real_lstat,lstat


def test_bootstrap_exclusive_fixed_root700_then_accept_same_inode(monkeypatch,tmp_path):
    anchor,parent,dest,real_lstat,_=anchored_fixture(monkeypatch,tmp_path)
    before=real_lstat(anchor); value=v.bootstrap_parent(time.monotonic()+10)
    assert not dest.exists() and stat.S_IMODE(real_lstat(parent).st_mode)==0o700
    assert v.bootstrap_parent(time.monotonic()+10,expected=value)==value
    assert (real_lstat(anchor).st_dev,real_lstat(anchor).st_ino,real_lstat(anchor).st_mode)==(before.st_dev,before.st_ino,before.st_mode)
    public_parent=tmp_path/'shared-data';public_parent.mkdir(mode=0o755)
    monkeypatch.setattr(v,'PRIVATE_PARENT',public_parent);monkeypatch.setattr(v,'DEST',public_parent/'vcoco_full_pose_replica_v1')
    with pytest.raises(ValueError):v.bootstrap_parent(time.monotonic()+10)
    assert stat.S_IMODE(public_parent.stat().st_mode)==0o755


def test_receiver_parent_verify_only_never_creates_fsyncs_or_repairs(monkeypatch,tmp_path):
    _,parent,_,_,_=anchored_fixture(monkeypatch,tmp_path)
    with pytest.raises(ValueError):v.bootstrap_parent(time.monotonic()+10,verify_only=True)
    assert not parent.exists()
    value=v.bootstrap_parent(time.monotonic()+10)
    monkeypatch.setattr(v.os,'mkdir',lambda *a,**k:pytest.fail('read-only verifier creates namespace'))
    monkeypatch.setattr(v.os,'fsync',lambda *a:pytest.fail('read-only verifier fsyncs'))
    assert v.bootstrap_parent(time.monotonic()+10,expected=value,verify_only=True)==value
    parent.rename(parent.with_name('original'));assert not parent.exists()
    with pytest.raises(ValueError):v.bootstrap_parent(time.monotonic()+10,expected=value,verify_only=True)
    assert not parent.exists()


@pytest.mark.parametrize('fault',['ancestor_owner','ancestor_writable','parent_owner','parent_group','parent_mode','parent_symlink','ancestor_symlink','mkdir_race','path_change','expected_inode'])
def test_bootstrap_rejects_foreign_and_races_without_repair(monkeypatch,tmp_path,fault):
    anchor,parent,dest,real_lstat,view=anchored_fixture(monkeypatch,tmp_path)
    expected=None
    if fault.startswith('parent_'):
        if fault=='parent_symlink':parent.symlink_to(anchor,target_is_directory=True)
        else:parent.mkdir(mode=0o755 if fault=='parent_mode'else 0o700)
    if fault=='ancestor_symlink':
        other=tmp_path/'other';other.mkdir();anchor.rmdir();anchor.symlink_to(other,target_is_directory=True)
    def drift(path):
        s=view(path)
        if (fault=='ancestor_owner'and path==anchor)or(fault=='parent_owner'and path==parent):s.st_uid=99
        if fault=='ancestor_writable'and path==anchor:s.st_mode|=0o020
        if fault=='parent_group'and path==parent:s.st_gid=99
        return s
    monkeypatch.setattr(Path,'lstat',drift)
    if fault=='mkdir_race':
        real_mkdir=v.os.mkdir
        def race(path,mode=0o777,*,dir_fd=None):
            real_mkdir(path,mode,dir_fd=dir_fd);raise FileExistsError('authored competing namespace')
        monkeypatch.setattr(v.os,'mkdir',race)
    if fault=='path_change':
        real_fsync=v.os.fsync;changed=False
        def change(fd):
            nonlocal changed
            real_fsync(fd)
            if not changed:
                changed=True;parent.rename(anchor/'original-owned-parent');parent.mkdir(mode=0o700)
        monkeypatch.setattr(v.os,'fsync',change)
    if fault=='expected_inode':
        expected=v.bootstrap_parent(time.monotonic()+10)
        parent.rename(anchor/'original-owned-parent');parent.mkdir(mode=0o700)
    with pytest.raises((ValueError,FileExistsError,OSError)):
        v.bootstrap_parent(time.monotonic()+10,expected=expected)
    assert not dest.exists()
    if fault=='parent_mode':assert stat.S_IMODE(real_lstat(parent).st_mode)==0o755
    if fault in('parent_symlink','ancestor_symlink'):assert real_lstat(parent if fault=='parent_symlink'else anchor).st_mode&stat.S_IFLNK
    if fault in('mkdir_race','path_change','expected_inode'):assert parent.exists()


@pytest.mark.parametrize('failure_stage',['receipt_decode','incoming_pins','control_write','namespace'])
def test_import_subgate_failure_retains_control_without_get_delete_or_retry(monkeypatch,tmp_path,failure_stage):
    state,args=export_import(monkeypatch,tmp_path);root,_,_,calls,_,_,_=state
    if failure_stage=='receipt_decode':args.export_receipt_base64='!'
    elif failure_stage=='incoming_pins':args.archive_pin=v.pin(b'not-original')
    elif failure_stage=='control_write':
        original=v.control_write
        def fail(out,name,raw,allowed):
            original(out,name,raw,allowed);raise OSError('authored control-fsync failure')
        monkeypatch.setattr(v,'control_write',fail)
    else:
        monkeypatch.setattr(v,'bootstrap_parent',lambda *a,**k:(_ for _ in()).throw(ValueError('authored foreign namespace')))
    before=len(calls);report=v.run(args,tmp_path/'code',S)
    assert report['status']=='fail'and report['failure_stage']==failure_stage and report['outputs_sealed']
    assert not v.DEST.exists()and not calls[before:]and report['delete_attempts']==0
    out=root/f'results/vcoco-full-pose-replica-import-{S}'
    assert (out/'export-receipt.json').exists()is(failure_stage in('control_write','namespace'))
    assert v.rt.strict((out/'report.json').read_bytes())==report


def test_repair_changes_only_namespace_run_and_keeps_original_export_schema_helpers():
    original=subprocess.check_output(['rtk','proxy','git','show','be9087a031f3755f3947276a2cb3bb5f8dcb0fd7:infra/vcoco_full_pose_replica.py'])
    old=ast.parse(original);new=ast.parse(Path(v.__file__).read_bytes())
    functions=lambda tree:{n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)}
    left,right=functions(old),functions(new)
    assert set(right)-set(left)=={'bootstrap_parent'}
    assert all(right[name]==node for name,node in left.items()if name not in('run','authenticate_receiver'))
    values=lambda tree:{n.targets[0].id:ast.dump(n.value,include_attributes=False)for n in tree.body
        if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    a,b=values(old),values(new)
    assert all(a[n]==b[n]for n in('SCHEMA','PUBLIC_SCHEMA','POSE_REV','DECLARATION','NAMES','ROW_KEYS','HELPERS','FLAGS'))
