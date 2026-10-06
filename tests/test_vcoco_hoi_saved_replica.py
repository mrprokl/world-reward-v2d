"""Tiny authored bytes/receipt controls; never read native data or call HTTP."""
import base64
import copy
import hashlib
import io
from pathlib import Path
import stat
import sys
import tarfile
import time
from types import SimpleNamespace
import pytest
import vcoco_hoi_saved_replica as v

R = '1'*40


def tiny(monkeypatch):
    payload = {n: ('authored opaque '+n).encode() for n in v.NAMES}
    monkeypatch.setattr(v, 'PINS', {n: v.pin(payload[n]) for n in v.PINS})
    manifest = dict(schema=v.SCHEMA, export_revision=R, original_source_declaration=v.DECLARATION,
                    files={n: v.pin(raw) for n, raw in payload.items()})
    return manifest, payload


def archive_raw(m, payload, members=None, *, format=tarfile.USTAR_FORMAT):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w', format=format) as archive:
        for name, raw, kind in members or [('manifest.json', v.encode(m), tarfile.REGTYPE),
                *[(n, payload[n], tarfile.REGTYPE) for n in sorted(payload)]]:
            row = tarfile.TarInfo(name); row.size = len(raw); row.mode = 0o400; row.type = kind
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE): row.linkname = 'foreign'
            archive.addfile(row, io.BytesIO(raw))
    return output.getvalue()


def readonly(path, raw):
    path.write_bytes(raw); path.chmod(0o400); return v.pin(raw)


def test_full19_opaque_banks_exact_catalog(monkeypatch, tmp_path):
    m, payload = tiny(monkeypatch); path = tmp_path/'incoming.tar'; ap = readonly(path, archive_raw(m, payload))
    result, table = v.verify_archive(path, ap, v.pin(v.encode(m)), R, time.monotonic()+5)
    assert result == m and len(table) == 20 and table[0][0] == 'manifest.json'
    assert {r[0] for r in table[1:]} == v.NAMES
    with path.open('rb') as f:
        for n, offset, size in table[1:]:
            f.seek(offset); assert f.read(size) == payload[n]


@pytest.mark.parametrize('change', ['hole', 'alias', 'extra', 'schema', 'source', 'revision', 'zero', 'boolean', 'hash', 'oversize', 'receipt'])
def test_manifest_frozen_exactly(monkeypatch, change):
    m, _ = tiny(monkeypatch); m = copy.deepcopy(m)
    if change == 'hole': del m['files']['image_000015.npz']
    elif change == 'alias': m['files']['./image_000000.npz'] = m['files'].pop('image_000000.npz')
    elif change == 'extra': m['files']['reference.json'] = v.pin(b'x')
    elif change == 'schema': m['schema'] = 'other'
    elif change == 'source': m['original_source_declaration'] = dict(v.DECLARATION, files=343)
    elif change == 'revision': m['export_revision'] = '2'*40
    elif change == 'zero': m['files']['image_000000.npz']['bytes'] = 0
    elif change == 'boolean': m['files']['image_000000.npz']['bytes'] = True
    elif change == 'hash': m['files']['image_000000.npz']['sha256'] = 'bad'
    elif change == 'oversize': m['files']['image_000000.npz']['bytes'] = v.MAXIMUM
    elif change == 'receipt': m['files']['model_proof.json']['sha256'] = '0'*64
    with pytest.raises(ValueError): v.validate_manifest(m, R)


@pytest.mark.parametrize('change', ['symlink', 'hardlink', 'dupe', 'unknown', 'escape', 'alias', 'prefix', 'pax', 'mode', 'uid', 'size', 'hash', 'tail', 'truncated', 'padding', 'first'])
def test_archive_rejects_before_install(monkeypatch, tmp_path, change):
    m, payload = tiny(monkeypatch)
    members = [('manifest.json', v.encode(m), tarfile.REGTYPE), *[(n, payload[n], tarfile.REGTYPE) for n in sorted(payload)]]
    if change in ('symlink', 'hardlink'): members[1] = (members[1][0], b'', tarfile.SYMTYPE if change == 'symlink' else tarfile.LNKTYPE)
    elif change == 'dupe': members.append(members[1])
    elif change == 'unknown': members.append(('unknown', b'x', tarfile.REGTYPE))
    elif change == 'escape': members[1] = ('../other', members[1][1], tarfile.REGTYPE)
    elif change == 'alias': members[1] = ('./'+members[1][0], members[1][1], tarfile.REGTYPE)
    elif change in ('prefix', 'pax'): members[1] = ('a'*101+'/other' if change == 'prefix' else 'a'*101, b'x', tarfile.REGTYPE)
    elif change == 'hash': members[1] = (members[1][0], b'bad', tarfile.REGTYPE)
    elif change == 'first': members[0], members[1] = members[1], members[0]
    raw = archive_raw(m, payload, members, format=tarfile.PAX_FORMAT if change == 'pax' else tarfile.USTAR_FORMAT)
    if change in ('mode', 'uid', 'size'):
        header = tarfile.TarInfo.frombuf(raw[:512], 'utf-8', 'strict')
        if change == 'mode': header.mode = 0o444
        elif change == 'uid': header.uid = 1
        else: header.size = v.MAX_MANIFEST+1
        raw = header.tobuf(format=tarfile.USTAR_FORMAT)+raw[512:]
    elif change == 'tail': raw += b'x'+b'\0'*511
    elif change == 'truncated': raw = raw[:700]
    elif change == 'padding':
        at = 512+len(v.encode(m)); raw = raw[:at]+b'x'+raw[at+1:]
    path = tmp_path/'archive'; ap = readonly(path, raw)
    with pytest.raises((ValueError, tarfile.HeaderError, IndexError)): v.verify_archive(path, ap, v.pin(v.encode(m)), R, time.monotonic()+5)


def test_pin_and_deadline_before_archive_read(monkeypatch, tmp_path):
    m, payload = tiny(monkeypatch); p = tmp_path/'archive'; ap = readonly(p, archive_raw(m, payload))
    with pytest.raises(ValueError): v.verify_archive(p, dict(ap, sha256='0'*64), v.pin(v.encode(m)), R, time.monotonic()+5)
    with pytest.raises(ValueError): v.verify_archive(p, ap, dict(v.pin(v.encode(m)), bytes=v.MAX_MANIFEST+1), R, time.monotonic()+5)
    with pytest.raises(ValueError): v.verify_archive(p, ap, v.pin(v.encode(m)), R, time.monotonic()-1)


def test_pack_stream_bounded_and_identical(monkeypatch, tmp_path):
    m, payload = tiny(monkeypatch); paths = {}
    for n, raw in payload.items(): p = tmp_path/n; readonly(p, raw); paths[n] = p
    writer = io.BytesIO(); v.pack(writer, m, paths, time.monotonic()+5)
    apath = tmp_path/'archive'; ap = readonly(apath, writer.getvalue())
    assert v.verify_archive(apath, ap, v.pin(v.encode(m)), R, time.monotonic()+5)[0] == m
    bw = v.BoundedWriter(io.BytesIO(), time.monotonic()+5); bw.size = v.MAXIMUM
    with pytest.raises(ValueError): bw.write(b'x')
    with pytest.raises(ValueError): v.BoundedWriter(io.BytesIO(), time.monotonic()-1).write(b'x')


def metadata():
    images = [dict(image_id='opaque%02d'%i, height=20, width=30) for i in range(16)]
    banks = [dict(original_slot=i, image_size=[20, 30], identity=v.pin(('endpoint%d'%i).encode()), person_ids=['p%d'%i]) for i in range(16)]
    sb = dict(producer_revision=v.REV, entries=349, closure_sha256=v.CLOSURE, helpers=v.hoi.FROZEN)
    proof = dict(source=sb, image_id=v.hoi.IMAGE, images=images, banks=banks, runtime=dict(report_identity=v.pin(b'runtime')))
    rows = []; counts = [1,12,25,6,3,6,2,4,2,12,9,8,12,12,50,1]
    ints = {'native_nms_keep','retained_nms_positions','query_ids','class_ids','hand_object_pairs','object_target_pairs','image_size','original_frame_index','original_slot','acquired_ordinal'}
    files = {}
    for i, k in enumerate(counts):
        n = 2; t = 0; file = f'image_{i:06d}.npz'; files[file] = v.pin(file.encode())
        shapes = dict(query_logits=[1500,3],query_boxes_cxcywh=[1500,4],query_tokens=[1500,256],native_nms_detections=[2,5],native_nms_keep=[2],retained_nms_positions=[n],
            query_ids=[n],class_ids=[n],boxes_original_xyxy=[n,4],raw_scores=[n],decayed_scores=[n],hand_object_pairs=[k,2],hand_object_logits=[k,2],
            object_target_pairs=[t,2],object_target_logits=[t,2],image_size=[2],original_frame_index=[],original_slot=[],acquired_ordinal=[])
        arrays = {a:dict(shape=s,dtype='<i8' if a in ints else '<f4',sha256='1'*64) for a,s in shapes.items()}
        rows.append(dict(image_id=images[i]['image_id'],original_slot=i,acquired_ordinal=i,original_frame_index=0,image_size=[20,30],endpoint_bank_identity=banks[i]['identity'],
            source_person_ids=banks[i]['person_ids'],owl_patches=3600,file=file,identity=files[file],native_detections=n,hand_object_pairs=k,object_target_pairs=t,arrays=arrays))
    flags=dict(reference_metadata_read=False,FIT_performed=False,ownership_verified=False,quality_verified=False,adoption=False,source_inputs_runtime_assets_rehashed_after=True,elapsed_seconds=1)
    model=dict(flags,schema=v.hoi.SCHEMA,stage='native_vcoco_hoi_model',status='pass',phase='complete',producer_revision=v.REV,image_id=v.hoi.IMAGE,
        proof_identity=v.PINS['model_proof.json'],images=rows,models_loaded=1,native_forward_calls=16,AMP_used=False,TF32_used=False,
        model=dict(strict_checkpoint=dict(keys=1796,strict=True,weights_only=True),checkpoint_buffer_schema=dict(ema_swapped=False)))
    host=dict(flags,schema=v.hoi.SCHEMA,stage='vcoco_hoi_observations_host',status='pass',producer_revision=v.REV,image_id=v.hoi.IMAGE,source_binding=sb,
        native_identity=v.PINS['model.json'],images=rows,public_inputs_identity=v.public.PUBLIC,endpoint_producer_revision=v.public.REV,endpoint_receipt_pins=v.public.PINS,
        qualified_model=v.hoi.QUALIFIED,runtime_report_identity=proof['runtime']['report_identity'],outputs_sealed=True,owned_cleanup_verified=True)
    return host, model, proof, files, dict(images=images,banks=banks)


def test_complete_actual19_field_schema_manufactured():
    assert v.validate_metadata(*metadata()) is None


@pytest.mark.parametrize('change', ['query', 'field', 'dtype', 'arrayhash', 'source', 'slot', 'ordinal', 'boolslot', 'image', 'grid', 'endpoint', 'person', 'owl', 'paircount', 'missing', 'ema', 'strict', 'TF32', 'failure', 'runtime'])
def test_original_abi_and_alias_identity_rejected(change):
    host, model, proof, files, endpoints = copy.deepcopy(metadata()); r=model['images'][0]
    if change == 'query': r['arrays']['query_logits']['shape'][0] = 1499
    elif change == 'field': del r['arrays']['query_tokens']
    elif change == 'dtype': r['arrays']['hand_object_logits']['dtype'] = '<f8'
    elif change == 'arrayhash': r['arrays']['query_tokens']['sha256'] = 'bad'
    elif change == 'source': proof['source'] = dict(proof['source'], closure_sha256='0'*64)
    elif change == 'slot': r['original_slot'] = 1
    elif change == 'ordinal': r['acquired_ordinal'] = 1
    elif change == 'boolslot': r['original_slot'] = False
    elif change == 'image': r['image_id'] = 'other'
    elif change == 'grid': r['image_size'] = [30,20]
    elif change == 'endpoint': r['endpoint_bank_identity'] = v.pin(b'other')
    elif change == 'person': r['source_person_ids'] = []
    elif change == 'owl': r['owl_patches'] = 3599
    elif change == 'paircount': r['hand_object_pairs'] = 0
    elif change == 'missing': model['images'].pop()
    elif change == 'ema': model['model']['checkpoint_buffer_schema']['ema_swapped'] = True
    elif change == 'strict': model['model']['strict_checkpoint']['keys'] = 1795
    elif change == 'TF32': model['TF32_used'] = True
    elif change == 'failure': host['publication_failed'] = True
    elif change == 'runtime': host['runtime_report_identity'] = v.pin(b'other')
    with pytest.raises((ValueError, KeyError)): v.validate_metadata(host, model, proof, files, endpoints)


def local_install_policy(monkeypatch, tmp_path):
    dest=tmp_path/'replica';monkeypatch.setattr(v,'DEST',dest)
    # Local authored fixture owner is not root; production policy stays strict.
    def directory(path,mode,names=None):
        s=path.lstat(); assert stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode)==mode
        if names is not None: assert {p.name for p in path.iterdir()}==set(names)
    monkeypatch.setattr(v,'private_directory',directory);tmp_path.chmod(0o700)
    monkeypatch.setattr(v.atomic,'rename_noreplace',lambda a,b:a.rename(b) if not b.exists() else (_ for _ in()).throw(FileExistsError()))
    return dest


def test_install_raw_bytes_no_overwrite(monkeypatch,tmp_path):
    m,payload=tiny(monkeypatch);dest=local_install_policy(monkeypatch,tmp_path);p=tmp_path/'archive';ap=readonly(p,archive_raw(m,payload))
    _,table=v.verify_archive(p,ap,v.pin(v.encode(m)),R,time.monotonic()+5);stage=tmp_path/'stage'
    v.install(p,stage,m,table,time.monotonic()+5)
    assert not stage.exists() and stat.S_IMODE(dest.stat().st_mode)==0o700 and stat.S_IMODE((dest/'banks').stat().st_mode)==0o500
    assert {p.name for p in (dest/'banks').iterdir()}==v.NAMES
    assert all((dest/'banks'/n).read_bytes()==raw and stat.S_IMODE((dest/'banks'/n).stat().st_mode)==0o400 for n,raw in payload.items())
    with pytest.raises(ValueError):v.install(p,stage,m,table,time.monotonic()+5)
    (dest/'banks').chmod(0o700)


def test_bad_install_owns_only_partial(monkeypatch,tmp_path):
    m,payload=tiny(monkeypatch);dest=local_install_policy(monkeypatch,tmp_path);p=tmp_path/'archive';ap=readonly(p,archive_raw(m,payload));_,table=v.verify_archive(p,ap,v.pin(v.encode(m)),R,time.monotonic()+5)
    bad=copy.deepcopy(m);bad['files'][table[1][0]]['sha256']='0'*64;stage=tmp_path/'stage'
    with pytest.raises(ValueError):v.install(p,stage,bad,table,time.monotonic()+5)
    assert not dest.exists() and not stage.exists() and p.exists()


def test_argument_bounds_no_hidden_urls():
    assert v.arguments(['--phase','export']).phase=='export'
    with pytest.raises(ValueError):v.arguments(['--phase','export','--export-revision',R])
    with pytest.raises(ValueError):v.arguments(['--phase','import','--export-revision',R,'--export-receipt-base64','x'*(384<<10|1)])
    pins=['--'+n+'-'+s for n in ('archive','manifest','export-receipt')for s in ('bytes','sha256')]
    args=['--phase','import','--export-revision',R,'--export-receipt-base64',base64.b64encode(b'x').decode()]
    for p in pins:args.extend([p,'1' if p.endswith('bytes') else '1'*64])
    assert v.arguments(args).archive_pin==dict(bytes=1,sha256='1'*64)


def test_stdlib_only_and_frozen_source_count():
    assert not any(n in sys.modules for n in ('torch','onnxruntime'))
    assert len(v.NAMES)==19 and v.DECLARATION['files']==344 and v.DECLARATION['entries']==349


@pytest.mark.parametrize('revision,script,executables', [
    ('5191c27f6820825116ffea4207f3f7cb5876a142', 'infra/run_vcoco_hoi_saved_replica.sh', v.CURRENT_EXECUTABLES),
    (v.REV, 'infra/run_vcoco_hoi_observations.sh', v.SENDER_EXECUTABLES),
])
def test_real_frozen_git_archives_have_exact_executable_leaves(revision, script, executables):
    import subprocess
    import azure_job
    raw = subprocess.check_output(['rtk', 'proxy', 'git', 'archive', '--format=tar', revision,
                                   'infra', 'src', 'configs', 'pyproject.toml'])
    frozen, _ = azure_job.runtime_archive(raw, script)
    with tarfile.open(fileobj=io.BytesIO(frozen)) as archive:
        assert {row.name for row in archive if row.mode & 0o111} == executables


@pytest.mark.parametrize('fault', [None, 'extra_exec', 'missing_exec', 'writable', 'symlink', 'hardlink'])
def test_exact_git_modes_reject_drift_without_repair(tmp_path, monkeypatch, fault):
    code = tmp_path/'code'; code.mkdir(); (code/'infra').mkdir()
    for name in v.CURRENT_EXECUTABLES:
        leaf=code/name; leaf.write_bytes(b'authored source\n'); leaf.chmod(0o555)
    regular=code/'infra/regular.py'; regular.write_bytes(b'authored\n'); regular.chmod(0o444)
    for name in ('revision','source-sha256'):
        leaf=tmp_path/name; leaf.write_bytes(b'authored marker\n'); leaf.chmod(0o444)
    target=code/'infra/run_keypoint_rgb_dwpose.sh'
    if fault=='extra_exec': regular.chmod(0o555)
    elif fault=='missing_exec': target.chmod(0o444)
    elif fault=='writable': regular.chmod(0o644)
    elif fault=='symlink': target.unlink(); target.symlink_to(regular)
    elif fault=='hardlink': os=__import__('os'); os.link(regular, code/'infra/alias.py')
    for folder in (code/'infra',code):folder.chmod(0o555)
    original=Path.lstat
    def root_stat(path):
        s=original(path)
        class RootStat:
            st_uid=st_gid=0
            def __getattr__(self,name):return getattr(s,name)
        return RootStat()
    monkeypatch.setattr(Path,'lstat',root_stat)
    try:
        if fault is None:v.source_modes(code,v.CURRENT_EXECUTABLES)
        else:
            with pytest.raises(ValueError):v.source_modes(code,v.CURRENT_EXECUTABLES)
    finally:
        code.chmod(0o700); (code/'infra').chmod(0o700)


def run_import_fixture(monkeypatch, tmp_path, *, delete_failure=False):
    code=tmp_path/'code'; code.mkdir(); monkeypatch.setattr(v,'ROOT',tmp_path)
    monkeypatch.setattr(v.sys,'platform','linux');monkeypatch.setattr(v.os,'geteuid',lambda:0)
    (tmp_path/'results').mkdir(); before={'binding':{'producer_revision':R}}
    monkeypatch.setattr(v,'source',lambda *_:before)
    monkeypatch.setattr(v.transport,'verify_azure_peer',lambda *_:None)
    monkeypatch.setattr(v,'private_directory',lambda *_:None)
    monkeypatch.setattr(v.pose,'completed_replica_inputs',lambda *_:dict(images=[],banks=[]))
    monkeypatch.setattr(v,'validate_metadata',lambda *_:None)
    m,payload=tiny(monkeypatch);raw=archive_raw(m,payload);ap=v.pin(raw);mp=v.pin(v.encode(m))
    def download(blob,path,expected):
        assert expected==ap
        with path.open('xb')as f:f.write(raw)
    monkeypatch.setattr(v.transport,'download',download)
    def install(path,stage,manifest,table,deadline):
        assert manifest==m and len(table)==20
    monkeypatch.setattr(v,'install',install);monkeypatch.setattr(v,'installed',lambda *_:dict(files=m['files'],states={}))
    # Opaque fixture JSONs are intentionally not native qualification evidence.
    monkeypatch.setattr(v.rt,'strict',lambda raw:{} if raw.startswith(b'authored opaque')else __import__('json').loads(raw))
    export=dict(archive_identity=ap,manifest_identity=mp,blob_etag='"fixed"')
    monkeypatch.setattr(v,'export_receipt',lambda *_:export)
    calls=[]
    class Response:
        status=202
        def __enter__(self):return self
        def __exit__(self,*_):pass
    class Blob:
        def __init__(self,url,revision,**kwargs):
            assert url.endswith('/articulated-runtime-'+revision+'.tar') and '?'not in url and kwargs=={'managed_identity':True}
        def request(self,method,**kwargs):
            calls.append((method,kwargs));assert method=='DELETE'and kwargs=={'headers':{'If-Match':'"fixed"'}}
            if delete_failure:raise RuntimeError('SECRET never serialized')
            return Response()
    monkeypatch.setattr(v.transport,'Blob',Blob)
    args=SimpleNamespace(phase='import',export_revision=R,export_receipt_base64=base64.b64encode(b'{}').decode(),
        export_receipt_pin=v.pin(b'{}'),archive_pin=ap,manifest_pin=mp)
    result=v.run(args,code,R)
    out=tmp_path/'results'/('vcoco-hoi-saved-replica-import-'+R)
    saved=__import__('json').loads((out/'report.json').read_text())
    assert saved==result and len(calls)==1 and not(out/'archive.tar').exists()
    assert stat.S_IMODE(out.stat().st_mode)==0o500 and all(stat.S_IMODE(p.stat().st_mode)==0o400 for p in out.iterdir())
    assert 'SECRET'not in(out/'report.json').read_text()
    out.chmod(0o700)
    return result


def test_samefd_callback_cleanup_success_persisted(monkeypatch,tmp_path):
    report=run_import_fixture(monkeypatch,tmp_path)
    assert report['status']=='pass'and report['blob_cleanup_verified']is True and report['single_etag_DELETE_202']is True
    assert report['delete_attempts']==1 and report['source_inputs_rehashed_after']is True


def test_callback_cleanup_failure_persisted_no_retry(monkeypatch,tmp_path):
    report=run_import_fixture(monkeypatch,tmp_path,delete_failure=True)
    assert report['status']=='fail'and report['publication_failed']is True and report['blob_cleanup_verified']is False
    assert report['publication_failure_stage']=='after_seal'and report['publication_error_type']=='RuntimeError'and report['delete_attempts']==1
