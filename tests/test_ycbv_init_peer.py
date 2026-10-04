import base64
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import tarfile

import pytest

spec = importlib.util.spec_from_file_location('ycbv_peer_test', Path(__file__).resolve().parents[1] / 'infra/ycbv_init_peer.py')
peer = importlib.util.module_from_spec(spec); spec.loader.exec_module(peer)
REV = 'a' * 40
SCRIPT = 'b' * 64


def digest(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def fixture(tmp_path, direction='init'):
    root = tmp_path / 'origin'; root.mkdir(mode=0o700)
    pins = dict(schema='world-reward-ycbv-init-peer-pins-v1', direction=direction, files={},
        producer_reports={}, source_replica=None, archive=None, inventory_report=None)
    contents = {name: ('own tiny ' + name).encode() for name in peer.payload_names(direction)}
    if direction == 'init':
        contents[peer.BASE + '/inputs/manifest.json'] = peer.encoded(dict(schema='world-reward-ycbv-point-rgb-v1',
            revision='e' * 40, license='MIT', selection='own fixed fixture', attribution='own tiny metadata', images=[{}] * 288))
    roles = ('acquisition', 'masks', 'depth') if direction == 'init' else ('objects',)
    for role in roles:
        value = dict(stage=peer.STAGES[role], status='pass', phase='complete', producer_revision=REV,
            script_sha256=SCRIPT, challenge_inputs_used=False, source_rehashed_after=True,
            all_inputs_sources_assets_outputs_rehashed=True, sources_after_reverified=True,
            private_annotations_exported_as_inference_inputs=False, ground_truth_used=False,
            private_truth_read=False, private_annotations_read=False)
        if role == 'objects':
            files = {name: digest(b'' if name.endswith('/__init__.py') else ('tiny original code ' + name).encode()) for name in peer.OBJECT_HELPERS}
            value['script_sha256'] = files['infra/ycbv_point_objects.py']['sha256']
            value['source_helpers'] = files
            prefix = f'jobs/{REV}/run_ycbv_point_objects'
            for name in files: contents[prefix + '/code/' + name] = b'' if name.endswith('/__init__.py') else ('tiny original code ' + name).encode()
            contents[prefix + '/revision'] = (REV + '\n').encode(); contents[prefix + '/source-sha256'] = ('c' * 64 + '\n').encode()
            pins['source_replica'] = dict(origin_host='scenesmith-ncc-h100-01', replica_host='world-reward-ncc-h100-02',
                producer_revision=REV, files=files, markers={name: digest(contents[prefix + '/' + name]) for name in ('revision', 'source-sha256')})
        raw = peer.encoded(value); contents[peer.REPORT_PATHS[role]] = raw
        pins['producer_reports'][role] = dict(**digest(raw), producer_revision=REV, script_sha256=value['script_sha256'])
    for name, raw in contents.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True, mode=0o700); peer.write(path, raw)
        if name in peer.payload_names(direction): pins['files'][name] = digest(raw)
    return root, pins


def archived(tmp_path, direction='init'):
    root, pins = fixture(tmp_path, direction); out = root / 'archive-result'
    report = peer.archive(root, pins, out, REV, SCRIPT, lambda: None)
    assert report['status'] == 'pass'
    pins['archive'] = report['archive']; pins['inventory_report'] = dict(**peer.identity(out / 'report.json'), producer_revision=REV, script_sha256=SCRIPT)
    return root, pins, out / 'archive.tar'


@pytest.mark.parametrize('direction', ['init', 'return'])
def test_exact_thirteen_payloads_and_explicit_source_only_on_return(tmp_path, direction):
    root, pins = fixture(tmp_path, direction)
    assert len(pins['files']) == 13
    peer.validate_pins(pins, direction)
    assert len(peer.verify_files(root, pins)) == (13 if direction == 'init' else 21)
    assert b'acquisition_receipt_host_only' in peer.manifest(pins)


@pytest.mark.parametrize('mutation', ['extra', 'drop', 'private', 'boolbytes', 'negative', 'badsha', 'unknownschema', 'duplicate'])
def test_closed_schema_fails_before_network(tmp_path, mutation, monkeypatch):
    _, pins = fixture(tmp_path); changed = copy.deepcopy(pins)
    name = next(iter(changed['files']))
    if mutation == 'extra': changed['unexpected'] = True
    elif mutation == 'drop': del changed['files'][name]
    elif mutation == 'private': changed['files'][peer.BASE + '/eval_private/scene_gt.json'] = digest(b'private')
    elif mutation == 'boolbytes': changed['files'][name]['bytes'] = True
    elif mutation == 'negative': changed['files'][name]['bytes'] = -1
    elif mutation == 'badsha': changed['files'][name]['sha256'] = 'x' * 64
    elif mutation == 'unknownschema': changed['schema'] = 'other'
    else:
        with pytest.raises(ValueError, match='Duplicate'): peer.strict(b'{"key":1,"key":2}')
        return
    monkeypatch.setattr(peer.subprocess, 'Popen', lambda *a, **k: pytest.fail('network reached'))
    with pytest.raises(ValueError): peer.validate_pins(changed, 'init', True)


def test_network_requires_independent_archive_and_inventory_not_sender_label(tmp_path):
    _, pins = fixture(tmp_path)
    with pytest.raises(ValueError, match='before networking'): peer.validate_pins(pins, 'init', True)
    pins['archive'] = digest(b'own'); pins['inventory_report'] = dict(**digest(b'own report'), producer_revision=REV, script_sha256=SCRIPT)
    assert peer.validate_pins(pins, 'init', True) is pins


@pytest.mark.parametrize('direction', ['init', 'return'])
def test_archive_publish_preserves_all_original_bytes_and_declares_replica(tmp_path, direction):
    origin, pins, path = archived(tmp_path, direction)
    replica = tmp_path / 'replica'; replica.mkdir(mode=0o700)
    (replica / 'validation').mkdir(mode=0o700)
    if direction == 'return':
        (replica / peer.BASE).mkdir(mode=0o700); (replica / 'jobs' / REV).mkdir(parents=True, mode=0o700)
    assert peer.inspect_archive(path, pins) == peer.all_files(pins)
    result = peer.publish(path, replica, pins)
    assert result['original_bytes_preserved'] and result['producer_source_replica_not_execution'] == (direction == 'return')
    for name in peer.all_files(pins):
        assert (replica / name).read_bytes() == (origin / name).read_bytes()
        assert (replica / name).stat().st_mode & 0o777 == 0o400
    if direction == 'return':
        code = replica / f'jobs/{REV}/run_ycbv_point_objects/code'
        assert code.stat().st_mode & 0o777 == 0o555
    with pytest.raises(ValueError, match='no merging'): peer.publish(path, replica, pins)


def test_receive_exact_eof_then_publication_and_failure_retains_partial(tmp_path):
    _, pins, path = archived(tmp_path)
    replica = tmp_path / 'replica'; replica.mkdir(mode=0o700); (replica / 'validation').mkdir(mode=0o700)
    result = peer.receive(io.BytesIO(path.read_bytes()), replica, pins, replica / 'received')
    assert result['status'] == 'pass' and result['leaf_count'] == 13
    assert (replica / 'received/archive.tar').exists()
    bad = tmp_path / 'bad'; bad.mkdir(mode=0o700)
    with pytest.raises(ValueError, match='Byte receiver'): peer.receive(io.BytesIO(path.read_bytes() + b'extra'), bad, pins, bad / 'received')
    assert (bad / 'received/archive.tar').exists()
    assert not (bad / peer.BASE).exists()


@pytest.mark.parametrize('kind', ['hash', 'link', 'extra', 'missing', 'order', 'manifest', 'traversal', 'memberhash'])
def test_archive_firewall_before_publication(tmp_path, kind):
    _, pins, original = archived(tmp_path)
    if kind == 'hash':
        path = tmp_path / 'bad.tar'; peer.write(path, original.read_bytes() + b'!')
    else:
        path = tmp_path / 'bad.tar'
        with tarfile.open(original, 'r:') as source:
            members = [(member, source.extractfile(member).read()) for member in source]
        if kind == 'missing': members.pop()
        if kind == 'order': members[1], members[2] = members[2], members[1]
        if kind == 'extra': members.append((tarfile.TarInfo('extra'), b''))
        if kind == 'manifest': members[0] = (members[0][0], b'x' * len(members[0][1]))
        if kind == 'memberhash': members[-1] = (members[-1][0], b'x' * len(members[-1][1]))
        if kind == 'traversal': members[1][0].name = '../private'
        if kind == 'link': members[1][0].type = tarfile.SYMTYPE; members[1][0].linkname = '/etc/passwd'
        with tarfile.open(path, 'w', format=tarfile.GNU_FORMAT) as output:
            for member, raw in members: output.addfile(member, io.BytesIO(raw) if member.type == tarfile.REGTYPE else None)
        path.chmod(0o400); pins['archive'] = peer.identity(path)
    replica = tmp_path / 'replica'; replica.mkdir(mode=0o700); (replica / 'validation').mkdir(mode=0o700)
    with pytest.raises(ValueError): peer.publish(path, replica, pins)
    assert not (replica / peer.BASE).exists()


def test_source_markers_are_copied_not_reconstructed_and_full_helper_set_bound(tmp_path):
    root, pins = fixture(tmp_path, 'return')
    marker = root / f'jobs/{REV}/run_ycbv_point_objects/revision'; marker.chmod(0o600); marker.write_text('d' * 40 + '\n'); marker.chmod(0o400)
    pins['source_replica']['markers']['revision'] = peer.identity(marker)
    with pytest.raises(ValueError, match='never reconstruct'): peer.verify_files(root, pins)
    changed = copy.deepcopy(pins); changed['source_replica']['files']['infra/extra.py'] = digest(b'x')
    with pytest.raises(ValueError): peer.validate_pins(changed, 'return')


def test_no_writable_symlink_or_hardlink_source(tmp_path):
    path = tmp_path / 'source'; path.write_bytes(b'own')
    with pytest.raises(ValueError, match='immutable'): peer.identity(path)
    path.chmod(0o400); os.link(path, tmp_path / 'hard')
    with pytest.raises(ValueError, match='Unaliased'): peer.identity(path)
    link = tmp_path / 'link'; link.symlink_to(path)
    with pytest.raises(ValueError, match='Canonical'): peer.identity(link)


def test_atomic_private_mode_even_with_umask_zero(tmp_path):
    old = os.umask(0)
    try:
        path = tmp_path / 'private'; peer.write(path, b'own')
        assert path.stat().st_mode & 0o777 == 0o400
    finally: os.umask(old)


@pytest.mark.parametrize('connection,command', [('10.0.0.4 100 10.0.0.9 2222', peer.COMMANDS['init']),
    ('10.0.0.3 100 10.0.0.9 2222', peer.COMMANDS['init']), ('10.0.0.4 0 10.0.0.9 2222', peer.COMMANDS['init']),
    ('10.0.0.4 100 10.0.0.9 22', peer.COMMANDS['init']), ('10.0.0.4 100 10.0.0.9 2222', 'sh')])
def test_exact_forced_peer(connection, command):
    if connection == '10.0.0.4 100 10.0.0.9 2222' and command == peer.COMMANDS['init']: peer.peer(connection, command, 'init')
    else:
        with pytest.raises(ValueError): peer.peer(connection, command, 'init')


def test_server_configuration_no_shell_forwarding_or_password_and_owned_stop(monkeypatch):
    text = peer.configuration(Path('/run/own'))
    for literal in ('ListenAddress 10.0.0.9', 'Port 2222', 'PermitRootLogin forced-commands-only', 'PermitTTY no',
        'AllowTcpForwarding no', 'AllowAgentForwarding no', 'PasswordAuthentication no', 'PermitUserEnvironment no', 'PermitUserRC no'):
        assert literal in text
    assert 'Subsystem' not in text and 'ForceCommand' not in text  # Individual authorized key fixes exactly one command.
    calls = []
    monkeypatch.setattr(peer, 'control_command', lambda args, **kw: calls.append(args) or 'FragmentPath=/other\nExecStart={ path=/usr/sbin/sshd /run/own/sshd_config }\n')
    with pytest.raises(ValueError, match='Only original owned'): peer.stop_owned('own.service', Path('/run/own'))
    assert len(calls) == 1


def test_actual_objects_helper_closure_and_driver_receipt_sha_not_wrapper():
    path = Path(__file__).resolve().parents[1] / 'infra/ycbv_point_objects.py'
    tree = ast.parse(path.read_text()); constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('HELPERS', 'PIN_FILE'):
                    constants[target.id] = ast.literal_eval(node.value)
    assert set(peer.OBJECT_HELPERS) == set(constants['HELPERS']) | {constants['PIN_FILE']}
    assert "script_sha256=source_before['helpers'][HELPERS[0]]['sha256']" in path.read_text()


def test_final_source_failure_seals_fail_not_misleading_pass(tmp_path):
    def failed(): raise ValueError('source mutated')
    report = {'stage': 'own', 'status': 'pass'}
    peer.seal(tmp_path / 'report.json', report, failed)
    saved = json.loads((tmp_path / 'report.json').read_bytes())
    assert saved['status'] == 'fail' and saved['transport_source_rehashed_after'] is False


@pytest.mark.parametrize('value', [{'token': 'never'}, {'nested': {'cam_R_m2c': [1]}}, {'url': 'https://host/path?sig=never'},
    {'key': '-----BEGIN OPENSSH PRIVATE KEY-----'}, {'url': 'https://person:secret@host/path'}])
def test_metadata_credentials_or_annotation_values_rejected(value):
    with pytest.raises(ValueError): peer.metadata_only(value)


def test_metadata_identity_paths_allowed_without_claiming_labels_unread():
    peer.metadata_only({'retained_files': {'scene_gt.json': {'bytes': 100, 'sha256': 'a' * 64}},
        'private_annotations_read': True, 'url': 'https://host/data'})


def test_public_key_only_canonical_wire():
    raw = struct.pack('>I', 11) + b'ssh-ed25519' + struct.pack('>I', 32) + b'\x01' * 32
    key = 'ssh-ed25519 ' + base64.b64encode(raw).decode()
    assert peer.public_key(key + ' own comment') == key
    for bad in ('ssh-rsa abc', key + '\ncommand=sh', 'ssh-ed25519 ' + base64.b64encode(b'x').decode()):
        with pytest.raises(ValueError): peer.public_key(bad)


def test_runtime_bundle_has_driver_and_generic_receiver_and_no_pose_specific_dependency():
    spec = importlib.util.spec_from_file_location('peer_bundle_test', Path(__file__).resolve().parents[1] / 'infra/azure_job.py')
    launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
    root = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(root)): p.read_bytes() for prefix in ('infra', 'src', 'configs') for p in (root / prefix).rglob('*') if p.is_file() and p.suffix in ('.py', '.sh', '.json')}
    source['pyproject.toml'] = (root / 'pyproject.toml').read_bytes()
    files = launcher.runtime_bundle_paths(source, 'infra/run_ycbv_init_peer.sh')
    assert set(peer.HELPERS) <= set(files)
    assert not any(name.startswith('infra/pose_peer') for name in files)
