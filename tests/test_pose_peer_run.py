"""Tiny peer/pose control tests; no GPU, SSH, network, models or real media."""
import hashlib
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'infra'))
import pose_peer_inputs as inputs
import pose_peer_receive as receiver
import pose_peer_run as runner
import pose_peer_send as sender
from test_pose_peer_inputs import cohort


def write(path,raw,mode=0o444):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(mode);return inputs.identity(path)


def dispatch(root,revision,name,helpers):
    code=root/'jobs'/revision/name/'code';code.mkdir(parents=True)
    write(code.parent/'revision',(revision+'\n').encode());write(code.parent/'source-sha256',('e'*64+'\n').encode())
    for name in helpers:write(code/name,('immutable fixture '+name).encode())
    return code


def seal(code):
    for path in(code,*code.rglob('*')):path.chmod(0o555 if path.is_dir()else 0o444)


@pytest.fixture
def transport(cohort,tmp_path,monkeypatch):
    root=cohort['root'];revision='c'*40;code=dispatch(root,revision,'run_pose_peer_receive',receiver.HELPERS)
    archive=tmp_path/'archive.tar';archivepin=inputs.archive(root,cohort['manifest'],archive)
    manifest=cohort['manifest']
    pins=dict(schema='world_reward.pose_peer.pins.v1',episode_index=8,clip_spec=manifest['clip_spec'],mesh_source=manifest['mesh_source'],
        tracker_helpers=manifest['tracker_helpers'],source_pins=manifest['source_pins_identity'],archive=archivepin,
        inventory_report=dict(bytes=10,sha256='a'*64,producer_revision='b'*40,script_sha256='c'*64),
        manifest=dict(bytes=len(inputs.digest_json(manifest)),sha256=hashlib.sha256(inputs.digest_json(manifest)).hexdigest()))
    write(code/'configs/pose_peer_000008_pins.json',inputs.digest_json(pins));seal(code)
    dest=tmp_path/'data';dest.mkdir();monkeypatch.setattr(inputs,'DEST',dest)
    # Existing stream primitive is tested with actual bytes below; this test uses
    # an explicit CPU test stub because macOS test uid is not production root.
    def receive(stream,destination,size,sha):
        raw=stream.read();assert len(raw)==size and hashlib.sha256(raw).hexdigest()==sha
        inputs.write(destination/'archive.tar',raw);inputs.write(destination/'report.json',b'{"status":"pass"}\n')
        return dict(status='pass',receipt_written=True)
    monkeypatch.setattr(receiver.frontend_peer_receive,'receive',receive)
    return dict(root=root,code=code,revision=revision,pins=pins,archive=archive,cohort=cohort,dest=dest)


def test_fresh_receiver_complete_retained_inputs(transport):
    report=receiver.receive(io.BytesIO(transport['archive'].read_bytes()),transport['root'],transport['code'],transport['revision'],8)
    assert report['status']=='pass'and report['original_frame_coverage_verified']is True
    assert report['retained']['files']==22
    with pytest.raises(ValueError):receiver.receive(io.BytesIO(transport['archive'].read_bytes()),transport['root'],transport['code'],transport['revision'],8)


@pytest.mark.parametrize('connection,command',[('10.0.0.5 123 10.0.0.9 2222','world-reward-pose-inputs-000008'),
    ('10.0.0.4 123 10.0.0.9 22','world-reward-pose-inputs-000008'),('10.0.0.4 123 10.0.0.9 2222','world-reward-pose-inputs-000007'),
    ('10.0.0.4 0 10.0.0.9 2222','world-reward-pose-inputs-000008')])
def test_narrow_forced_peer(connection,command):
    with pytest.raises(ValueError):receiver.validate_peer(connection,command,8)
    receiver.validate_peer('10.0.0.4 123 10.0.0.9 2222','world-reward-pose-inputs-000008',8)


def test_existing_stream_contract_rejects_truncated_and_extra(tmp_path):
    for number,raw in enumerate((b'abc',b'abcde')):
        destination=tmp_path/str(number);destination.mkdir(mode=0o700)
        report=receiver.frontend_peer_receive.receive(io.BytesIO(raw),destination,4,hashlib.sha256(b'abcd').hexdigest())
        assert report['status']=='fail'and report['bytes_received']<=4


@pytest.fixture
def pose_runtime(transport,monkeypatch,tmp_path):
    transport['received']=receiver.receive(io.BytesIO(transport['archive'].read_bytes()),transport['root'],transport['code'],transport['revision'],8)
    root=transport['root'];revision='d'*40;code=dispatch(root,revision,'run_pose_peer_run',runner.HELPERS)
    for name in inputs.TRACKER_HELPERS:
        write(code/name,(transport['cohort']['code']/name).read_bytes())
    write(code/'configs/pose_peer_000008_pins.json',inputs.digest_json(transport['pins']))
    receipt=inputs.DEST/f'pose-peer-000008-{transport["revision"]}'/'pose-receipt.json'
    received=dict(schema='world_reward.pose_peer.received_pins.v1',episode_index=8,producer_revision=transport['revision'],receipt=inputs.identity(receipt,True))
    write(code/'configs/pose_peer_000008_received_pins.json',inputs.digest_json(received));seal(code)
    lock=root/'jobs/.world-reward-h100.lock';lock.write_bytes(b'original lock')
    stream=lock.open('rb');fd=stream.fileno();fcntl.flock(fd,fcntl.LOCK_EX);calls=[];fault={}
    # NativeUID ownership and Linux-only atomic promotion are tested as explicit
    # metadata/control doubles here, not faked real model/container execution.
    monkeypatch.setattr(runner.os,'chown',lambda *args:None)
    def promote(source,target):
        assert not target.exists();source.rename(target)
    monkeypatch.setattr(runner,'promote',promote)
    monkeypatch.setattr(runner,'image',lambda root:dict(image_id=inputs.IMAGE,ordered_rootfs_sha256='f'*64))
    def run(args,**kwargs):
        calls.append(args)
        if args[0]=='nvidia-smi':return subprocess.CompletedProcess(args,0,b'123\n'if fault.get('gpu')else b'',b'')
        if '/usr/bin/timeout'in args:
            mounts=[args[i+1]for i,a in enumerate(args)if a=='--mount']
            work=root/'results'/f'pose-peer-run-000008-{revision}'/'prediction-parent'
            # Docker's exact RO binds leave these harmless owned placeholders.
            (work/'automatic_masks/masks/0').mkdir(parents=True)
            output=work/'object_pose_full';output.mkdir()
            write(output/'geometry_and_poses.npz',b'no decode tiny poses');write(output/'object_fixed_canonical.glb',b'no decode tiny geometry')
            native=dict(stage='fixed_scale_full_object_pose_initializer',status='pass',episode_index=8,original_frame_coverage_verified=True,
                input_track='track_1',ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],fixed_shape=True,mesh_source='default',
                script_sha256=transport['cohort']['manifest']['tracker_helpers']['infra/object_pose_smoke.py']['sha256'],
                geometry_and_poses_sha256=inputs.identity(output/'geometry_and_poses.npz')['sha256'],
                fixed_canonical_mesh_sha256=inputs.identity(output/'object_fixed_canonical.glb')['sha256'],
                budget_source_sha256=transport['cohort']['manifest']['files'][inputs.KIT]['sha256'],
                object_report_sha256=transport['cohort']['manifest']['files']['outputs/episode_000008/object_grounded/report.json']['sha256'],
                alignment_report_sha256=transport['cohort']['manifest']['files']['outputs/episode_000008/scale_smoke/report.json']['sha256'],
                full_depth_report_sha256=transport['cohort']['manifest']['files']['outputs/episode_000008/depth_full/report.json']['sha256'],
                input_sha256=transport['cohort']['manifest']['clip_spec']['video_sha256'],frames=[dict(frame_index=i)for i in range(3)])
            write(output/'report.json',inputs.digest_json(native))
            write(Path(args[args.index('--cidfile')+1]),('e'*64+'\n').encode(),0o400)
            kwargs['stdout'].write(b'private procedural native diagnostic\n')
            assert len([m for m in mounts if 'readonly'not in m])==1
            return subprocess.CompletedProcess(args,7 if fault.get('child')else 0)
        return subprocess.CompletedProcess(args,0,b'',b'')
    monkeypatch.setattr(runner.subprocess,'run',run)
    # Only native output ownership gate is substituted; all hashes, lock and
    # genuine archive/input provenance remain real tiny filesystem checks.
    original_stat=Path.stat
    def native_stat(path,*args,**kwargs):
        value=original_stat(path,*args,**kwargs)
        if 'object_pose_full'in path.parts and 'prediction-parent'in path.parts:
            fields=list(value);fields[4:6]=[1000,1000];return os.stat_result(fields)
        return value
    monkeypatch.setattr(Path,'stat',native_stat)
    original_lstat=Path.lstat
    def native_lstat(path,*args,**kwargs):
        value=original_lstat(path,*args,**kwargs)
        if 'object_pose_full'in path.parts and 'prediction-parent'in path.parts:
            fields=list(value);fields[4:6]=[1000,1000];return os.stat_result(fields)
        return value
    monkeypatch.setattr(Path,'lstat',native_lstat)
    yield dict(root=root,code=code,revision=revision,fd=fd,lock=lock,calls=calls,fault=fault,transport=transport)
    stream.close()


def test_pose_only_original_inputs_source_lock_before_gpu(pose_runtime):
    value=pose_runtime;before=value['lock'].read_bytes()
    report=runner.run(value['root'],value['code'],value['revision'],8,value['fd'])
    assert report['status']=='pass'and report['frames']==3 and report['source_rehashed_after']is True
    assert value['lock'].read_bytes()==before
    args=next(args for args in value['calls']if '/usr/bin/timeout'in args)
    assert args[-5:]==['--episode','8','--full-video','--mesh-source','default']
    assert args[args.index('--network')+1]=='none'and args[args.index('--user')+1]=='1000:1000'
    assert args[args.index('--entrypoint')+1]=='python'
    assert args[args.index(inputs.IMAGE)+1:args.index(inputs.IMAGE)+3]==['-B',str(value['code']/'infra/object_pose_smoke.py')]
    assert '--gpus'in args and '--read-only'in args and '--cap-drop'in args
    assert not any('weights' in a or 'body_full/predictions' in a for a in args)
    mounts=[args[i+1]for i,a in enumerate(args)if a=='--mount']
    assert mounts[1].endswith('/outputs/episode_000008')and 'readonly'not in mounts[1]
    assert all('readonly'in m for m in mounts[2:])


@pytest.mark.parametrize('fault',['gpu','child'])
def test_pose_fail_no_algorithm_rescue(pose_runtime,fault):
    value=pose_runtime;value['fault'][fault]=True
    report=runner.run(value['root'],value['code'],value['revision'],8,value['fd']);assert report['status']=='fail'
    if fault=='gpu':assert not any('/usr/bin/timeout'in a for a in value['calls'])


@pytest.mark.parametrize('fault',['source','input','lock','output'])
def test_pose_preflight_fail_before_native(pose_runtime,fault,tmp_path):
    value=pose_runtime
    if fault=='source':(value['code']/'infra/object_pose_smoke.py').chmod(0o644)
    elif fault=='input':
        p=inputs.DEST/f'pose-peer-000008-{value["transport"]["revision"]}'/'inputs/outputs/episode_000008/depth_full/000001.npz';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o400)
    elif fault=='lock':value['lock'].rename(tmp_path/'retained');value['lock'].write_bytes(b'foreign')
    else:(value['root']/'outputs/episode_000008/object_pose_full').mkdir()
    with pytest.raises(ValueError):runner.run(value['root'],value['code'],value['revision'],8,value['fd'])
    assert not value['calls']


def test_public_hostkey_no_secret_output():
    import base64,struct
    value='ssh-ed25519 '+base64.b64encode(struct.pack('>I',11)+b'ssh-ed25519'+struct.pack('>I',32)+b'x'*32).decode()
    assert sender.public_key(value+' harmless')==value
    for value in('ssh-rsa aaa','ssh-ed25519 aaa','ssh-ed25519 aa\nsecret'):
        with pytest.raises((ValueError,Exception)):sender.public_key(value)


def test_bash_syntax_closure_stream_not_consumed_by_heredoc(monkeypatch):
    for name in('run_pose_peer_inventory','run_pose_peer_receive','run_pose_peer_send','run_pose_peer_run'):
        subprocess.run(['rtk','proxy','bash','-n',str(ROOT/'infra'/f'{name}.sh')],check=True)
    source=(ROOT/'infra/run_pose_peer_receive.sh').read_text();assert '<<'not in source and 'python3 -I -B "$CODE/infra/pose_peer_receive.py"'in source
    source=(ROOT/'infra/run_pose_peer_run.sh').read_text();assert 'exec 9<"$LOCK";flock --timeout 43200 9'in source
    monkeypatch.syspath_prepend(str(ROOT/'infra'));import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes()for folder in('infra','src','configs')for p in(ROOT/folder).rglob('*')if p.is_file()and '__pycache__'not in p.parts};files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    for name in('run_pose_peer_inventory','run_pose_peer_receive','run_pose_peer_send','run_pose_peer_run'):
        selected=azure_job.runtime_bundle_paths(files,f'infra/{name}.sh');assert 'infra/pose_peer_inputs.py'in selected
    assert 'infra/object_pose_smoke.py'in azure_job.runtime_bundle_paths(files,'infra/run_pose_peer_run.sh')


def test_tiny_pins_no_frame_file_list_and_pinned_manifest_before_json(transport,monkeypatch):
    pins=transport['pins'];assert len(inputs.digest_json(pins))<=inputs.MAX_PINS
    assert set(pins['manifest'])=={'bytes','sha256'}and 'files'not in pins
    assert inputs.manifest_from_archive(transport['archive'],pins,8)==transport['cohort']['manifest']
    calls=[];original=inputs.strict_json
    monkeypatch.setattr(inputs,'strict_json',lambda raw:(calls.append(raw),original(raw))[1])
    with pytest.raises(ValueError,match='before parsing'):inputs.load_manifest(b'{untrusted',pins,8)
    assert not calls


@pytest.mark.parametrize('fault',['manifest_sha','manifest_size','clip','source','helpers','largepins'])
def test_frozen_tiny_contract_rejects_mismatch(transport,fault):
    import copy
    pins=copy.deepcopy(transport['pins'])
    if fault=='manifest_sha':pins['manifest']['sha256']='f'*64
    elif fault=='manifest_size':pins['manifest']['bytes']+=1
    elif fault=='clip':pins['clip_spec']['total_frames']=4
    elif fault=='source':pins['source_pins']['sha256']='f'*64
    elif fault=='helpers':pins['tracker_helpers']['infra/object_pose_smoke.py']['sha256']='f'*64
    else:pins['manifest']=transport['cohort']['manifest']
    with pytest.raises(ValueError):inputs.manifest_from_archive(transport['archive'],pins,8)


def test_lock_must_be_held_not_only_matching_descriptor(pose_runtime):
    value=pose_runtime;fcntl.flock(value['fd'],fcntl.LOCK_UN)
    with pytest.raises(ValueError,match='already be held'):runner.prepare(value['root'],value['code'],value['revision'],8,value['fd'])
    assert not value['calls']


@pytest.mark.parametrize('value',['08','+8','-1','30','0.0'])
def test_episode_parser_canonical(value):
    with pytest.raises((ValueError,SystemExit)):inputs.parser().parse_args(['--episode',value])


def test_argument_duplicate_fail_closed():
    with pytest.raises(ValueError,match='exactly once'):inputs.parser().parse_args(['--episode','8','--episode','9'])


@pytest.mark.parametrize('fault',['foreign','layers','architecture','error'])
def test_image_must_match_original_id_layers_and_platform(tmp_path,monkeypatch,fault):
    raw=[inputs.IMAGE,'amd64','linux',dict(Type='layers',Layers=['sha256:'+'a'*64]*44)]
    if fault=='foreign':raw[0]='sha256:'+'f'*64
    elif fault=='layers':raw[3]['Layers'].pop()
    elif fault=='architecture':raw[1]='arm64'
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,1 if fault=='error'else 0,' '.join(json.dumps(x)for x in raw).encode(),b''))
    with pytest.raises(ValueError):runner.image(tmp_path)


def test_sourcebinding_helpers_argument_never_changes_full_closure_fingerprint(pose_runtime):
    value=pose_runtime;helpers={name:inputs.identity(value['code']/name,True)for name in runner.HELPERS}
    assert inputs.code_binding(value['code'],value['revision'],helpers)==inputs.code_binding(value['code'],value['revision'],value['transport']['pins']['tracker_helpers'])


@pytest.mark.skipif(sys.platform!='linux',reason='Linux atomic renameat2 runtime only')
def test_atomic_promotion_does_not_replace_existing_directory(tmp_path):
    source=tmp_path/'source';source.mkdir();target=tmp_path/'target';target.mkdir()
    with pytest.raises(OSError):runner.promote(source,target)
    assert source.is_dir()and target.is_dir()
    target.rmdir();runner.promote(source,target);assert target.is_dir()and not source.exists()


def test_inventory_seals_full_metadata_only_on_azure(cohort,tmp_path,monkeypatch):
    import pose_peer_inventory as inventory
    code=cohort['code'];code.chmod(0o755);(code/'infra').chmod(0o755)
    for name in inventory.HELPERS:write(code/name,('immutable inventory '+name).encode())
    write(code/'configs/pose_peer_000008_source_pins.json',inputs.digest_json(cohort['source']));seal(code)
    manifest,source,old,binding=inventory.inventory(cohort['root'],code,8,'b'*40,'a'*40,
        'run_track1_initializers_only',cohort['source']['producer_helpers']['infra/run_track1_initializers_only.sh']['sha256'])
    assert manifest['source_pins_identity']==source and len(manifest['files'])==22
    assert manifest['tracker_helpers']==cohort['source']['tracker_helpers']
    assert inputs.code_binding(code,'b'*40,manifest['tracker_helpers'])==binding
    assert inputs.validate_manifest(manifest,8)==manifest


def test_sender_fixed_ssh_stream_and_independent_manifest_before_network(transport,monkeypatch):
    import base64,struct
    root=transport['root'];revision='f'*40;code=dispatch(root,revision,'run_pose_peer_send',sender.HELPERS)
    manifest=transport['cohort']['manifest'];original=root/'results'/('pose-peer-inventory-000008-'+'b'*40)
    original.mkdir(parents=True)
    write(original/'archive.tar',transport['archive'].read_bytes(),0o400)
    write(original/inputs.MANIFEST_NAME,inputs.digest_json(manifest),0o400)
    receipt=dict(stage='pose_peer_input_inventory',status='pass',episode_index=8,producer_revision='b'*40,
        script_sha256='c'*64,archive=transport['pins']['archive'],manifest=transport['pins']['manifest'])
    actual=write(original/'report.json',inputs.digest_json(receipt),0o400)
    pins=dict(transport['pins']);pins['inventory_report']=dict(**actual,producer_revision='b'*40,script_sha256='c'*64)
    write(code/'configs/pose_peer_000008_pins.json',inputs.digest_json(pins));seal(code)
    key=root/'transfer/pose-peer-client-000008/client_ed25519';write(key,b'never read private test sentinel',0o600);key.parent.chmod(0o700)
    hostkey='ssh-ed25519 '+base64.b64encode(struct.pack('>I',11)+b'ssh-ed25519'+struct.pack('>I',32)+b'x'*32).decode()
    write(key.with_suffix('.pub'),(hostkey+'\n').encode(),0o400)
    monkeypatch.setattr(sender,'key_state',lambda path:(path.lstat().st_ino,path.lstat().st_mode,path.lstat().st_size))
    original_open=Path.open
    def never_open_private(path,*args,**kwargs):
        assert path!=key,'Private bytes must not be read by the Python sender';return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',never_open_private)
    calls=[]
    def ssh(args,**kwargs):
        calls.append(args);assert kwargs['stdin'].read()==transport['archive'].read_bytes()
        assert set(kwargs['env'])=={'PATH','HOME','LANG'}and args[-2:]==['root@10.0.0.9','world-reward-pose-inputs-000008']
        summary=dict(stage='pose_peer_receive',status='pass',episode_index=8,extraction_performed=True,
            original_frame_coverage_verified=True,receipt=dict(bytes=40,sha256='a'*64))
        kwargs['stdout'].write(inputs.digest_json(summary));return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(sender.subprocess,'run',ssh)
    report=sender.send(root,code,revision,8,hostkey)
    assert report['status']=='pass'and report['private_key_bytes_read_or_recorded']is False and len(calls)==1
    assert '-T'in calls[0]and 'StrictHostKeyChecking=yes'in calls[0]


def test_other_descriptor_holding_lock_is_not_owned_by_caller(pose_runtime):
    value=pose_runtime;fcntl.flock(value['fd'],fcntl.LOCK_UN)
    with value['lock'].open('rb')as unrelated:
        fcntl.flock(unrelated.fileno(),fcntl.LOCK_EX)
        with pytest.raises(ValueError,match='another process'):runner.lock_state(value['root'],value['fd'])


def test_receiver_manifest_mismatch_abstains_before_extract(transport):
    pins=transport['code']/'configs/pose_peer_000008_pins.json';value=inputs.strict_json(pins.read_bytes());value['manifest']['sha256']='f'*64
    pins.chmod(0o644);pins.write_bytes(inputs.digest_json(value));pins.chmod(0o444)
    report=receiver.receive(io.BytesIO(transport['archive'].read_bytes()),transport['root'],transport['code'],transport['revision'],8)
    assert report['status']=='fail'and report['extraction_performed']is False
    assert not(transport['dest']/f'pose-peer-000008-{transport["revision"]}'/'inputs').exists()
