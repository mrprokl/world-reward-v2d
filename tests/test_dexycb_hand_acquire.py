"""Tiny gzip/JPEG/opaque-label fixtures; no dataset, pixels or annotation decoding."""
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tarfile
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'infra'))
import dexycb_hand_acquire as hand


def jpeg(width=640):
    return b'\xff\xd8\xff\xc0'+struct.pack('>H',17)+struct.pack('>BHHB',8,480,width,3)+bytes(9)+b'\xff\xd9'


def tar_bytes(change=None, subject=hand.SUBJECT):
    rows = []
    for i in range(100):
        prefix = f'{subject}/20200820_{i:06d}/{hand.CAMERA}/'
        rows += [(prefix+f'color_{f:06d}.jpg', jpeg(), tarfile.REGTYPE) for f in range(2)]
        rows += [(prefix+f'labels_{f:06d}.npz', b'OPAQUE_NO_NPZ_PARSE', tarfile.REGTYPE) for f in range(2)]
        rows.append((f'{subject}/20200820_{i:06d}/meta.yml', b'NEVER_PARSED_OR_RETAINED', tarfile.REGTYPE))
        rows.append((prefix+'aligned_depth_to_color_000000.png', b'NOT_RETAINED', tarfile.REGTYPE))
    if change: rows = change(rows)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name, data, kind in rows:
            item = tarfile.TarInfo(name); item.type = kind; item.size = len(data)
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE): item.linkname = '/outside'; item.size = 0
            archive.addfile(item, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
    return stream.getvalue()


class Response:
    status = 200
    def __init__(self, raw, url):
        import email.message
        self.raw = io.BytesIO(raw); self.url = url; self.headers = email.message.Message()
        self.headers['Content-Type'] = 'application/gzip'; self.headers['Content-Length'] = str(len(raw))
    def __enter__(self): return self
    def __exit__(self, *args): self.raw.close()
    def geturl(self): return self.url
    def read(self, count): return self.raw.read(count)


def fixture(tmp_path, monkeypatch, raw=None, protocol_path=hand.PROTOCOL_V1):
    root=tmp_path/'root';root.mkdir();(root/'validation').mkdir()
    incoming=tmp_path/'incoming';output=root/hand.expected_protocol(protocol_path)['base']
    incoming.mkdir(mode=0o700);output.mkdir(mode=0o700);incoming.chmod(0o700);output.chmod(0o700)
    os.chown(incoming,os.getuid(),os.getgid());os.chown(output,os.getuid(),os.getgid())
    monkeypatch.setattr(hand,'INCOMING',incoming)
    raw=raw or tar_bytes(subject=hand.expected_protocol(protocol_path)['subject'])
    protocol=copy.deepcopy(hand.expected_protocol(protocol_path));protocol['archive']['bytes']=len(raw)
    if protocol_path == hand.PROTOCOL_V1: monkeypatch.setattr(hand,'EXPECTED_PROTOCOL',protocol)
    else: monkeypatch.setattr(hand,'expected_protocol',lambda selected=hand.PROTOCOL_V1: protocol if selected==protocol_path else hand.EXPECTED_PROTOCOL)
    monkeypatch.setattr(hand,'profile_paths',lambda root,selected=hand.PROTOCOL_V1:[output,incoming])
    # Linux NOREPLACE is unavailable on macOS; publication ABI is already tested by its owner.
    def publish(part,target):
        assert not target.exists();part.rename(target)
    monkeypatch.setattr(hand.download,'publish',publish)
    before={'closure_sha256':'a'*64, 'producer_revision':'b'*40}
    monkeypatch.setattr(hand,'source_binding',lambda *args: before.copy())
    lease={'schema':'world_reward.fresh_namespace_lease.v1','source_closure_sha256':'a'*64,'directories':[]}
    for path in (output,incoming):
        s=path.stat();lease['directories'].append(dict(path=str(path),device=s.st_dev,inode=s.st_ino,
            uid=os.getuid(),gid=os.getgid(),mode=0o700))
    class Opener:
        def open(self, request, timeout): return Response(raw,request.full_url)
    return root,output,incoming,lease,Opener()


def test_frozen_protocol_helper_hashes_and_actual_no_model_imports():
    config=Path(__file__).parents[1]/'configs/dexycb_hand_protocol_v1.json'
    hand.dex.exact(json.loads(config.read_bytes()),hand.EXPECTED_PROTOCOL)
    assert hand.INDICES==[4,39,74] and hand.SUBJECT=='20200820-subject-03'
    assert hand.EXPECTED_PROTOCOL['archive']['bytes']==12197037343
    assert hand.INCOMING==Path('/srv/world-reward-data/dexycb_hand_v1')
    for name,pin in hand.HELPER_PINS.items():
        raw=(config.parents[1]/'infra'/name).read_bytes()
        assert pin==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def test_whole_header_inventory_fresh3_full_ids_before_any_values(tmp_path):
    path=tmp_path/'tiny.gz';path.write_bytes(tar_bytes())
    selected,wanted,audit=hand.inspect_archive(path,time.monotonic()+10)
    assert [s['sequence_lex_index'] for s in selected]==[4,39,74]
    assert all(s['frames']==2 for s in selected) and len(wanted)==12
    assert audit['gzip_crc_verified'] and audit['member_count']==600
    assert all('meta.yml' not in n and 'depth' not in n for n in wanted)


@pytest.mark.parametrize('case',['duplicate','missing_label','missing_zero','symlink','root','ancestor','extra_sequence'])
def test_header_failclosed_before_extraction(tmp_path,case):
    def change(rows):
        if case=='duplicate':return rows+[rows[0]]
        if case=='missing_label':return [r for r in rows if not r[0].endswith('labels_000001.npz')]
        if case=='missing_zero':return [r for r in rows if not r[0].endswith('color_000000.jpg')]
        if case=='symlink':return rows+[(hand.SUBJECT+'/link',b'',tarfile.SYMTYPE)]
        if case=='root':return [('wrapper/'+n,b,k) for n,b,k in rows]
        if case=='ancestor':return rows+[(hand.SUBJECT+'/20200820_000000',b'x',tarfile.REGTYPE)]
        return rows+[(hand.SUBJECT+'/20200820_000100/meta.yml',b'x',tarfile.REGTYPE)]
    path=tmp_path/'bad.gz';path.write_bytes(tar_bytes(change))
    with pytest.raises(ValueError):hand.inspect_archive(path,time.monotonic()+10)


def test_gzip_crc_and_explicit_deadline_guards(tmp_path):
    path=tmp_path/'bad.gz';raw=bytearray(tar_bytes());raw[-8]^=1;path.write_bytes(raw)
    with pytest.raises(gzip.BadGzipFile):hand.inspect_archive(path,time.monotonic()+10)
    path.write_bytes(tar_bytes())
    with pytest.raises(TimeoutError):hand.inspect_archive(path,0)


def test_real_reused_download_and_extract_full3_no_private_decode(tmp_path,monkeypatch):
    root,out,incoming,lease,opener=fixture(tmp_path,monkeypatch)
    report=hand.acquire(root,tmp_path/'code','b'*40,lease,opener=opener)
    assert report['status']=='pass' and report['frames']==6 and report['sequences']==3
    assert report['source_rehashed_after'] and report['disposable_archive_removed']
    assert not report['annotation_values_parsed'] and not report['gpu_used']
    assert list(incoming.iterdir())==[]
    manifest=json.loads((out/'inputs/manifest.json').read_bytes())
    assert len(manifest['images'])==6 and all(i['frame_position']==i['source_frame_id'] for i in manifest['images'])
    assert [s['sequence_lex_index'] for s in manifest['sequences']]==[4,39,74]
    assert (out/'eval_private').stat().st_mode&0o777==0o700
    assert out.stat().st_mode&0o777==0o755 and (out/'inputs').stat().st_mode&0o777==0o755
    labels=list((out/'eval_private').rglob('*.npz'));assert len(labels)==6
    assert all(p.read_bytes()==b'OPAQUE_NO_NPZ_PARSE' and p.stat().st_mode&0o777==0o400 for p in labels)
    assert not list((out/'eval_private').rglob('meta.yml'))
    assert all(p.stat().st_mode&0o777==0o444 for p in (out/'inputs').iterdir())
    assert (out/'report.json').stat().st_mode&0o777==0o444
    assert all(x not in json.dumps(manifest) for x in ['joint_2d','joint_3d','intrinsics','ycb_ids','fps'])
    with pytest.raises(ValueError):hand.acquire(root,tmp_path/'code','b'*40,lease,opener=opener)


def test_grid_failure_cleanup_source_hash_and_sealed_receipt(tmp_path,monkeypatch):
    raw=tar_bytes(lambda rows:[(n,jpeg(1) if n.endswith('color_000001.jpg') else b,k) for n,b,k in rows])
    root,out,incoming,lease,opener=fixture(tmp_path,monkeypatch,raw)
    with pytest.raises(ValueError):hand.acquire(root,tmp_path/'code','b'*40,lease,opener=opener)
    report=json.loads((out/'report.json').read_bytes())
    assert report['status']=='fail' and report['source_rehashed_after']
    assert report['owned_partial_outputs_removed'] and report['disposable_archive_removed']
    assert not list((out/'inputs').iterdir()) and not list((out/'eval_private').iterdir())
    assert not list(incoming.iterdir())
    assert out.stat().st_mode&0o777==0o700  # failed namespace is not published as public


def test_source_mutation_postproof_not_pass(tmp_path,monkeypatch):
    root,out,incoming,lease,opener=fixture(tmp_path,monkeypatch)
    calls=[]
    def source(*args):
        calls.append(1);return {'closure_sha256':'a'*64 if len(calls)==1 else 'c'*64}
    monkeypatch.setattr(hand,'source_binding',source)
    with pytest.raises(ValueError):hand.acquire(root,tmp_path/'code','b'*40,lease,opener=opener)
    report=json.loads((out/'report.json').read_bytes())
    assert report['status']=='fail' and not report['source_rehashed_after']


def test_fresh_namespace_lease_occupied_is_fail_before_download(tmp_path,monkeypatch):
    root,out,incoming,lease,opener=fixture(tmp_path,monkeypatch)
    (out/'unknown').write_bytes(b'userwork')
    with pytest.raises(ValueError):hand.acquire(root,tmp_path/'code','b'*40,lease,opener=opener)
    assert (out/'unknown').read_bytes()==b'userwork' and not list(incoming.iterdir())


def test_wrapper_real_static_closure_and_cpu_only():
    from azure_job import runtime_bundle_paths
    repo=Path(__file__).parents[1]
    wrapper=(repo/'infra/run_dexycb_hand_acquire.sh').read_text()
    paths=runtime_bundle_paths({str(p.relative_to(repo)):p.read_bytes() for p in (repo/'infra').glob('*') if p.is_file()},
                              'infra/run_dexycb_hand_acquire.sh')
    for name in ('dexycb_hand_acquire.py','dexycb_acquire.py','dexycb_download.py','mediapipe_hands_acquire.py'):
        assert 'infra/'+name in paths
    assert 'runuser -u scenesmith' in wrapper and '9140s' in wrapper
    assert 'docker run' not in wrapper and 'nvidia' not in wrapper and 'eval_private' not in wrapper


def test_no_labels_values_numpy_models_or_external_url_cli():
    source=Path(hand.__file__).read_text()
    assert all(s not in source for s in ('import numpy','import torch','np.load','pickle.load','yaml.load','gdown'))
    assert 'dex.members(' in source and 'dex.extract_archive(' in source and 'download.fetch(' in source
    assert 'parser.parse_args(argv)' in source and 'signal.alarm(SCAN_BUDGET)' in source


@pytest.mark.parametrize('protocol_path',[hand.PROTOCOL_V1,hand.PROTOCOL_V2])
def test_real_source_binding_actual_paths_helper_pins_and_dispatch_markers(tmp_path,monkeypatch,protocol_path):
    root=tmp_path/'root';root.mkdir();rev='b'*40;code=root/'jobs'/rev/hand.JOB/'code'
    (code/'infra').mkdir(parents=True);(code/'configs').mkdir()
    for module in (hand,hand.dex,hand.download,hand.lease):
        original=Path(module.__file__);dest=code/'infra'/original.name
        dest.write_bytes(original.read_bytes());dest.chmod(0o444);monkeypatch.setattr(module,'__file__',str(dest))
    wrapper=code/'infra/run_dexycb_hand_acquire.sh';wrapper.write_bytes(b'fixture wrapper only');wrapper.chmod(0o444)
    protocol=copy.deepcopy(hand.expected_protocol(protocol_path));evidence=root/hand.dex.EVIDENCE;evidence.mkdir(parents=True)
    for name,data in [('publisher.html',b'DexYCB is licensed under by-nc/4.0'),
                      ('dex_ycb.py',b"color_{:06d}.jpg np.arange(meta['num_frames'])")]:
        path=evidence/name;path.write_bytes(data);path.chmod(0o444)
        protocol['primary_sources'][name].update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    if protocol_path==hand.PROTOCOL_V1:monkeypatch.setattr(hand,'EXPECTED_PROTOCOL',protocol)
    else:monkeypatch.setattr(hand,'expected_protocol',lambda selected=hand.PROTOCOL_V1:protocol if selected==protocol_path else hand.EXPECTED_PROTOCOL)
    config=code/protocol_path;config.write_text(json.dumps(protocol));config.chmod(0o444)
    for name,data in [('revision',(rev+'\n').encode()),('source-sha256',('c'*64+'\n').encode())]:
        path=code.parent/name;path.write_bytes(data);path.chmod(0o444)
    for path in (code/'infra',code/'configs',code):path.chmod(0o555)
    before=hand.source_binding(root,code,rev,protocol_path)
    assert before['producer_revision']==rev and before['helpers']==hand.HELPER_PINS
    assert before['protocol_file']==protocol_path and before['profile']==protocol['base']
    assert before==hand.source_binding(root,code,rev,protocol_path)
    config.chmod(0o644);config.write_text(json.dumps({**protocol,'camera':'different'}));config.chmod(0o444)
    with pytest.raises(ValueError):hand.source_binding(root,code,rev,protocol_path)
    config.chmod(0o644);config.write_text(json.dumps(protocol));config.chmod(0o444)
    wrapper.chmod(0o644)
    with pytest.raises(ValueError,match='Readonly'):hand.source_binding(root,code,rev,protocol_path)


def test_download_partial_pruned_without_fabricating_archive_or_private(tmp_path,monkeypatch):
    root,out,incoming,lease,opener=fixture(tmp_path,monkeypatch)
    def broken(row,directory,opener,deadline,stop,owned,progress):
        part=directory/(row['file']+'.part');part.write_bytes(b'ownedpartial')
        owned.append((part,hand.dex.state(part)[:2]));raise TimeoutError('predeclared timeout')
    monkeypatch.setattr(hand.download,'fetch',broken)
    with pytest.raises(ValueError):hand.acquire(root,tmp_path/'code','b'*40,lease,opener=opener)
    report=json.loads((out/'report.json').read_bytes())
    assert report['phase']=='download' and report['error_type']=='TimeoutError'
    assert report['status']=='fail' and not list(incoming.iterdir())
    assert set(p.name for p in out.iterdir())=={'report.json'}


def test_two_exact_profiles_and_no_mutation_of_legacy():
    legacy=copy.deepcopy(hand.EXPECTED_PROTOCOL);v2=hand.expected_protocol(hand.PROTOCOL_V2)
    config=Path(__file__).parents[1]/hand.PROTOCOL_V2
    hand.dex.exact(json.loads(config.read_bytes()),v2)
    assert v2['subject']=='20200903-subject-04' and v2['base']=='validation/dexycb_hand_v2'
    assert v2['archive']==dict(file='20200903-subject-04.tar.gz',bytes=12792618020,
        url='https://drive.google.com/file/d/14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA')
    for key in set(legacy)-{'subject','base','schema','archive'}: assert v2[key]==legacy[key]
    assert hand.EXPECTED_PROTOCOL==legacy and hand.expected_protocol() is hand.EXPECTED_PROTOCOL
    assert hand.profile_paths(Path('/srv/scenesmith/world-reward'),hand.PROTOCOL_V2)==[
        Path('/srv/scenesmith/world-reward/validation/dexycb_hand_v2'),Path('/srv/world-reward-data/dexycb_hand_v2')]
    for invalid in ('../configs/dexycb_hand_protocol_v2.json','configs/other.json',str(config.resolve())):
        with pytest.raises(ValueError):hand.expected_protocol(invalid)


def test_v2_whole_header_selection_and_subject_mismatch_fail_closed(tmp_path):
    protocol=hand.expected_protocol(hand.PROTOCOL_V2);path=tmp_path/'subject04.gz'
    path.write_bytes(tar_bytes(subject=protocol['subject']))
    selected,wanted,audit=hand.inspect_archive(path,time.monotonic()+10,protocol)
    assert [r['sequence_lex_index'] for r in selected]==[4,39,74]
    assert all(r['subject']==protocol['subject'] and r['frames']==2 for r in selected)
    assert len(wanted)==12 and audit['gzip_crc_verified']
    with pytest.raises(ValueError):hand.inspect_archive(path,time.monotonic()+10)
    with pytest.raises(TimeoutError):hand.inspect_archive(path,0,protocol)


def test_v2_real_reused_download_and_extraction_keeps_v1_untouched(tmp_path,monkeypatch):
    root,out,incoming,lease,opener=fixture(tmp_path,monkeypatch,protocol_path=hand.PROTOCOL_V2)
    historical=root/hand.BASE;historical.mkdir();(historical/'original').write_bytes(b'closed v1 unchanged')
    report=hand.acquire(root,tmp_path/'code','b'*40,lease,opener=opener,protocol_path=hand.PROTOCOL_V2)
    assert report['status']=='pass' and report['frames']==6 and report['annotation_values_parsed'] is False
    assert report['protocol_file']==hand.PROTOCOL_V2 and report['acquisition_profile']=='validation/dexycb_hand_v2'
    assert (historical/'original').read_bytes()==b'closed v1 unchanged'
    manifest=json.loads((out/'inputs/manifest.json').read_bytes())
    assert manifest['schema']=='world-reward-dexycb-hand-rgb-v1' and manifest['subject']=='20200903-subject-04'
    assert len(manifest['images'])==6 and all(r['source_frame_id']==r['frame_position'] for r in manifest['images'])
    assert len(list((out/'eval_private').rglob('*.npz')))==6 and not list((out/'eval_private').rglob('meta.yml'))
    assert out.stat().st_mode&0o777==0o755 and (out/'eval_private').stat().st_mode&0o777==0o700
    with pytest.raises(ValueError):hand.acquire(root,tmp_path/'code','b'*40,lease,opener=opener,protocol_path=hand.PROTOCOL_V2)


@pytest.mark.parametrize('arguments',[['--protocol','configs/no.json'],['--protocol',hand.PROTOCOL_V2,'--protocol',hand.PROTOCOL_V1],['--base','validation/other']])
def test_driver_rejects_unknown_or_duplicate_profile_before_host_io(monkeypatch,arguments):
    monkeypatch.setattr(hand.platform,'system',lambda:pytest.fail('must reject arguments before runtime I/O'))
    with pytest.raises(SystemExit):hand.main(arguments)


@pytest.mark.parametrize('arguments',[[],['--protocol',hand.PROTOCOL_V2]])
def test_wrapper_forwards_exact_profile_to_bootstrap_and_native_driver_without_network(tmp_path,arguments):
    import subprocess
    repo=Path(__file__).parents[1];bins=tmp_path/'bin';bins.mkdir();log=tmp_path/'calls'
    for name,body in [('uname','echo Linux'),('python3','printf "%s\\n" "$*" >> "$CALLS";cat >/dev/null;echo namespace-lease'),
                      ('timeout','printf "%s\\n" "$*" >> "$CALLS";cat >/dev/null')]:
        path=bins/name;path.write_text('#!/bin/bash\n'+body+'\n');path.chmod(0o755)
    root='/srv/scenesmith/world-reward';rev='b'*40
    environment={'PATH':str(bins)+':/usr/bin:/bin','CALLS':str(log),'WR_ROOT':root,
                 'WR_CODE':root+'/jobs/'+rev+'/run_dexycb_hand_acquire/code','WR_CODE_REVISION':rev}
    wrapper=repo/'infra/run_dexycb_hand_acquire.sh'
    # macOS has no Linux RLIMIT_AS; only its builtin is emulated, not profile logic.
    prefix='ulimit(){ :; }; wrapper="$1"; shift; source "$wrapper"'
    result=subprocess.run(['bash','-c',prefix,'fixture',str(wrapper),*arguments],env=environment,capture_output=True)
    assert result.returncode==0,result.stderr
    lines=log.read_text().splitlines();assert len(lines)==2
    protocol=arguments[-1] if arguments else hand.PROTOCOL_V1
    assert lines[0].endswith(protocol) and protocol in lines[1] and '9140s' in lines[1]
    assert 'runuser -u scenesmith' in lines[1] and 'WR_NAMESPACE_LEASE=namespace-lease' in lines[1]
    # Argument errors precede canonical/runtime bootstrap, with no output reservation.
    result=subprocess.run(['bash','-c',prefix,'fixture',str(wrapper),'--protocol','configs/unknown.json'],env=environment,capture_output=True)
    assert result.returncode==2 and log.read_text().splitlines()==lines


@pytest.mark.parametrize('arguments,protocol',[([],hand.PROTOCOL_V1),(['--protocol',hand.PROTOCOL_V2],hand.PROTOCOL_V2)])
def test_main_selects_profile_before_actual_acquisition_only(monkeypatch,arguments,protocol):
    import types
    captured=[]
    monkeypatch.setattr(hand.platform,'system',lambda:'Linux')
    monkeypatch.setattr(hand.pwd,'getpwnam',lambda name:types.SimpleNamespace(pw_uid=os.getuid()))
    monkeypatch.setenv('WR_ROOT',str(hand.ROOT));monkeypatch.setenv('WR_CODE','/immutable/code')
    monkeypatch.setenv('WR_CODE_REVISION','b'*40);monkeypatch.setenv('WR_NAMESPACE_LEASE','{}')
    monkeypatch.setattr(hand.signal,'signal',lambda *args:None)
    def acquire(*args,**kwargs):
        captured.append((args,kwargs));return dict(stage='fixture',status='pass',frames=6,elapsed_seconds=0)
    monkeypatch.setattr(hand,'acquire',acquire)
    hand.main(arguments)
    assert len(captured)==1 and captured[0][1]==dict(watchdog=True,protocol_path=protocol)
