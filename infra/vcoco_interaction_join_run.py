"""Blind saved16 interaction join: numerical observations, never selection/FIT."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import zipfile

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
import mediapipe_cpu_runtime_verify as rt
import vcoco_interaction_observations as numerical

ROOT = rt.ROOT
ENTRY = 'run_vcoco_interaction_join'
OUTPUT = ROOT/'results/vcoco-interaction-join-v1'
SCHEMA = 'world_reward.vcoco_interaction_join.v1'
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
BUDGET, MAXIMUM = 600, 32 << 20
POSE_REV = 'df7db36998df454d022ff8f593b8d59036c6d771'
POSE_CLOSURE = 'ae1b70b2fb2caa62432382ae001de65bf74ea25356bea0286c26f9351fc52de1'
POSE_XZ = 'fa9fb811b359963e159f0fd30e6acfe34b6fa73d7be8518c50b4f1560c630613'
EXECUTABLES = frozenset(('infra/run_coco_endpoint_prepare.sh', 'infra/run_keypoint_rgb_dwpose.sh', 'infra/run_rgb_endpoint_bank.sh'))
POSE_PINS = {'report.json': dict(bytes=46297, sha256='a0b15782e968f0a43f52d6b94ff66464ef6f5074c415838615db7b2fa2722492'),
    'native.json': dict(bytes=98052, sha256='a316d77b2183f3c356ccedd1fc4bf536f0fd86f7d11959ec5fb895c2d2d7e7b1'),
    'proof.json': dict(bytes=87620, sha256='27f95644b5975659f7c263eea9e855016b6e8a78adabb9ca78e3e39b308f1f84')}
NATIVE_FILES = ('infra/vcoco_interaction_join_run.py', 'infra/mediapipe_cpu_runtime_verify.py',
    'infra/vcoco_interaction_observations.py', 'src/world_reward/__init__.py',
    'src/world_reward/person_pose_observations.py', 'src/world_reward/hoi_detr_observations.py',
    'src/world_reward/owlv2_object_observations.py', 'src/world_reward/owlv2_candidate_bridge.py',
    'src/world_reward/interaction_tuple_evidence.py', 'src/world_reward/interaction_candidate_evidence.py')
NUMERICAL_PIN = dict(bytes=10476, sha256='e8333e3180fe009f08bb14201b20919a6bec823164d19c7f21d2124eae56a2a1')


def encode(v):
    import json
    return (json.dumps(v, sort_keys=True, allow_nan=False)+'\n').encode()


def check(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline: raise TimeoutError('Inclusive saved join deadline')


def host_modules():
    import vcoco_person_pose_observations as pose
    import vcoco_hoi_saved_replica as receiver
    return pose, receiver


def helpers():
    pose, receiver = host_modules()
    return tuple(dict.fromkeys((*NATIVE_FILES, 'infra/run_vcoco_interaction_join.sh',
        'infra/sealed_callback_publication.py', *pose.HELPERS, *receiver.HELPERS)))


def source(code, revision):
    binding = rt.source(ROOT, code, revision, ENTRY, helpers())
    pose, receiver = host_modules()
    for name in helpers():
        if name.startswith('infra/') and name.endswith('.py'):
            module = sys.modules.get(Path(name).stem)
            if module is not None: rt.require(Path(module.__file__).resolve() == code/name, 'Actual imported helper origin required')
    rt.require(Path(rt.__file__).resolve() == code/NATIVE_FILES[1]
        and Path(numerical.__file__).resolve() == code/NATIVE_FILES[2], 'Actual numerical/runtime imports required')
    rt.require(Path(__file__).resolve() == code/NATIVE_FILES[0] and binding['helpers'][NATIVE_FILES[2]] == NUMERICAL_PIN,
               'Unchanged pure numerical join required')
    rt.require(EXECUTABLES <= set(binding['helpers']), 'Complete exact executable source leaves required')
    for path in (code, *code.rglob('*'), code.parent/'revision', code.parent/'source-sha256'):
        st = path.lstat(); mode = 0o555 if path.is_dir() or (path.is_relative_to(code) and str(path.relative_to(code)) in EXECUTABLES) else 0o444
        rt.require(st.st_uid == st.st_gid == 0 and stat.S_IMODE(st.st_mode) == mode, 'Exact current root readonly source modes required')
    return dict(binding=binding, states=pose.replica.bank.source_state(code),
                markers={n: pose.replica.snapshot(code.parent/n) for n in ('revision', 'source-sha256')})


def pose_inputs(code, deadline):
    """Reconstruct old proof explicitly; never call old source()/run()/inference."""
    pose, receiver = host_modules(); old = ROOT/'jobs'/POSE_REV/pose.ENTRY/'code'
    binding = rt.source(ROOT, old, POSE_REV, pose.ENTRY, pose.HELPERS)
    rt.require(binding['entries'] == 352 and binding['closure_sha256'] == POSE_CLOSURE
        and sum(p.is_file() for p in old.rglob('*')) == 347
        and (old.parent/'source-sha256').read_bytes() == (POSE_XZ+'\n').encode()
        and all(rt.identity(code/n, 2_000_000, empty=True) == p for n, p in binding['helpers'].items()), 'Whole original pose source/helpers required')
    for p in (old, *old.rglob('*'), old.parent/'revision', old.parent/'source-sha256'):
        s = p.lstat(); expected = 0o555 if p.is_dir() or (p.is_relative_to(old) and str(p.relative_to(old)) in EXECUTABLES) else 0o444
        rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == expected, 'Original immutable Git mode/owner required')
    inputs = pose.completed_replica_inputs(code, *receiver.COMPLETION); models = pose.assets()
    values = {n: rt.pinned(pose.OUTPUT/n, pin, 2 << 20) for n, pin in POSE_PINS.items()}
    host, native, proof = (values[n] for n in POSE_PINS)
    expected = dict(source=binding, replica_pins={}, original_replica_source=inputs['original_replica_source'],
        images=inputs['images'], banks=inputs['banks'], files={**inputs['files'], **models['files']}, image_id=IMAGE,
        input_mode='completed_replica', completion_revision=receiver.COMPLETION[0], completion_identity=receiver.COMPLETION[1],
        completion_source=inputs['completion_source'])
    rt.require(proof == rt.strict(encode(expected)) and host['schema'] == pose.SCHEMA and host['status'] == 'pass'
        and host['stage'] == 'vcoco_person_pose_observations_host' and host['producer_revision'] == POSE_REV
        and host['source_binding'] == binding and host['source_stat_identity'] == pose.replica.bank.source_state(old)
        and host['native_identity'] == POSE_PINS['native.json'] and host['native_exit_status'] == 0
        and host['owned_cleanup_verified'] is host['source_inputs_assets_rehashed_after'] is host['outputs_sealed'] is True
        and all(host[k] is False for k in ('GPU_used', 'reference_metadata_read', 'FIT_performed', 'ownership_verified', 'quality_verified', 'adoption')),
        'Actual completed original pose proof/PASS required')
    pose.validate_native(native, proof, POSE_REV, POSE_PINS['proof.json'])
    folder = pose.OUTPUT; rt.canonical(folder); s = folder.lstat()
    names = set(POSE_PINS)|{'.container.cid'}|{r['file'] for r in native['images']}
    rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == 0o500 and {p.name for p in folder.iterdir()} == names,
               'Exact sealed original pose outputs required')
    files = {str(folder/n): rt.identity(folder/n, MAXIMUM) for n in names}
    for p in folder.iterdir():
        s = p.lstat(); rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == 0o400, 'Original pose400 required')
    cid = (folder/'.container.cid').read_bytes(); rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', cid), 'Original saved CID required')
    rt.require(not pose.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid.decode().strip()], deadline)
        and not pose.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/world-reward-vcoco-person-pose-'+POSE_REV[:12]+'$'], deadline),
        'Original pose container must remain absent')
    return dict(images=inputs['images'], banks=inputs['banks'], pose_rows=native['images'],
        source=binding, source_states=pose.replica.bank.source_state(old), inputs=inputs, assets=models, files=files,
        states={str(p): pose.replica.snapshot(p) for p in (folder, *folder.iterdir(), old.parent/'revision', old.parent/'source-sha256')})


def authenticate(code, revision, replica_revision, receipt_pin, deadline):
    before = source(code, revision); pose, receiver = host_modules()
    p = pose_inputs(code, deadline); h = receiver.authenticate_receiver(code, replica_revision, receipt_pin)
    rt.require(h['images'] == p['images'] and h['banks'] == p['banks'] and len(h['hoi_rows']) == 16, 'Same complete16 original populations required')
    records = []; files = {}
    for i, (e, v, q) in enumerate(zip(p['banks'], p['pose_rows'], h['hoi_rows'])):
        paths = (pose.replica.DEST/'banks'/e['file'], pose.OUTPUT/v['file'], receiver.DEST/'banks'/q['file'])
        rt.require(all(r['original_slot'] == r['acquired_ordinal'] == i for r in (e, v, q)), 'Original16 slot order required')
        for path, row in zip(paths, (e, v, q)):
            rt.require(rt.identity(path, MAXIMUM) == row['identity'], 'Original whole bank differs'); files[str(path)] = row['identity']
        records.append(dict(endpoint=e, pose=v, hoi=q, paths=[str(x) for x in paths]))
    rt.require(len(files) == 48 and len({r['endpoint']['image_id'] for r in records}) == 16, 'All48 distinct genuine NPZ/16images required')
    native_source_proof = {k: before['binding'][k] for k in ('producer_revision', 'markers', 'entries', 'closure_sha256')}
    native_source_proof['helpers'] = {n: before['binding']['helpers'][n] for n in NATIVE_FILES}
    return dict(current_source=before, pose=p, hoi=h), dict(schema=SCHEMA, source=native_source_proof, image_id=IMAGE,
        records=records, files=files, native_files=native_source_proof['helpers'])


def native_source(code, revision, proof):
    rt.require(Path(__file__).resolve() == code/NATIVE_FILES[0] and proof['source']['producer_revision'] == revision
        and proof['native_files'] == {n: proof['source']['helpers'][n] for n in NATIVE_FILES}
        and {str(p.relative_to(code)) for p in code.rglob('*') if p.is_file()} == set(NATIVE_FILES), 'Exact narrow native source whitelist required')
    for n, p in proof['source']['markers'].items(): rt.require(rt.identity(code.parent/n, 100) == p, 'Original source marker differs')
    rt.require((code.parent/'revision').read_bytes() == (revision+'\n').encode(), 'Original producer revision required')
    rt.require(rt.identity(code/NATIVE_FILES[2], 2_000_000) == NUMERICAL_PIN, 'Frozen numerical constructor bytes required')
    for name in NATIVE_FILES:
        if name.endswith('.py'):
            module_name = Path(name).stem if name.startswith('infra/') else '.'.join(Path(name).with_suffix('').parts[1:])
            module = sys.modules.get(module_name)
            if module is not None: rt.require(Path(module.__file__).resolve() == code/name, 'Actual native imported module origin required')
    return {str(path): dict(pin=rt.identity(path, 2_000_000, empty=True), state=snapshot(path))
        for path in [*(code/n for n in NATIVE_FILES), *(code.parent/n for n in ('revision', 'source-sha256'))]}


def load_bank(path, row, fields):
    """ZIP bounds + all actual plain array bytes; no numerical reconstruction yet."""
    import numpy as np
    rt.require(rt.identity(path, MAXIMUM) == row['identity'] and set(row['arrays']) == set(fields), 'Complete saved bank SHA required')
    rt.require(all(type(pin) is dict and set(pin) == {'shape', 'dtype', 'sha256'} and type(pin['shape']) is list
        and all(type(n) is int and n >= 0 for n in pin['shape']) and type(pin['dtype']) is str
        and re.fullmatch('[0-9a-f]{64}', str(pin['sha256'])) for pin in row['arrays'].values()), 'Plain original array metadata required')
    with zipfile.ZipFile(path) as archive:
        rows = archive.infolist(); rt.require(len(rows) == len(fields) and {r.filename for r in rows} == {n+'.npy' for n in fields}
            and all(not r.is_dir() and not r.flag_bits & 1 and r.file_size <= MAXIMUM for r in rows)
            and sum(r.file_size for r in rows) <= MAXIMUM, 'Bounded exact native NPZ members required')
        for info in rows:
            pin = row['arrays'][info.filename[:-4]]
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream); rt.require(version in ((1, 0), (2, 0)), 'Original NPY header version required')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream, max_header_size=4096)
                size = math.prod(shape)*dtype.itemsize
                rt.require(not dtype.hasobject and list(shape) == pin['shape'] and dtype.str == pin['dtype']
                    and size <= MAXIMUM and info.file_size == stream.tell()+size, 'Bounded exact NPY shape/dtype/payload required')
    with np.load(path, allow_pickle=False) as archive: result = {n: archive[n] for n in fields}
    rt.require(all(numerical._identity(a) == row['arrays'][n] for n, a in result.items()), 'Every saved raw array identity differs')
    return result


def summary(value):
    import numpy as np
    e = value.evidence; h = e.hoi_evidence; rows = (e.features, h.features, e.route_features)
    supports = (e.feature_supported, h.feature_supported, e.route_supported); digest = hashlib.sha256()
    for arrays in (e.arrays, h.arrays, e.hoi_routes):
        for name, a in arrays.items(): digest.update(encode([name, numerical._identity(a)]))
    for a, supported in zip(rows, supports):
        rt.require(np.isfinite(a[supported]).all() and not np.isinf(a).any(), 'Supported numerical features must be finite')
        digest.update(encode([numerical._identity(a), numerical._identity(supported)]))
    p, k, o = len(value.person.person_ids), len(value.hoi.hand_object_pairs), len(value.owl.patch_ids)
    rt.require(tuple(len(a) for a in rows) == (p*2*o, p*2*k, k*o) and o == 3600, 'Complete factorized Cartesian populations required')
    rt.require(all(np.array_equal(a, np.isfinite(v)) for a, v in zip(supports, rows)), 'Numerical support must match finite availability')
    return dict(image_id=value.image_id, original_slot=value.original_slot, acquired_ordinal=value.acquired_ordinal,
        persons=p, objects=o, native_pairs=k, person_side_object_rows=p*2*o, person_side_pair_rows=p*2*k, pair_object_rows=k*o,
        supported_counts=[int(a.sum()) for a in supports], nan_counts=[int(np.isnan(a).sum()) for a in rows],
        evidence_fingerprint=digest.hexdigest(), source_observation_references=e.source_observation_references,
        raw_bank_fingerprint=hashlib.sha256(encode([{n: numerical._identity(a) for n, a in x.items()}
            for x in (value.endpoint_arrays, value.pose_arrays, value.hoi_arrays)])).hexdigest())


def join_all(proof, deadline):
    rt.require(proof['schema'] == SCHEMA and proof['image_id'] == IMAGE and len(proof['records']) == 16 and len(proof['files']) == 48,
               'Frozen complete16/48 proof required')
    fields = (numerical.ENDPOINT_FIELDS, numerical.POSE_FIELDS, numerical.HOI_FIELDS)
    rt.require([r['endpoint']['original_slot'] for r in proof['records']] == list(range(16))
        and len({r['endpoint']['image_id'] for r in proof['records']}) == 16
        and all(len(r['paths']) == 3 and all(r[k]['original_slot'] == r[k]['acquired_ordinal'] == i
            and r[k]['image_id'] == r['endpoint']['image_id'] and r[k]['original_frame_index'] == 0
            and r[k]['image_size'] == r['endpoint']['image_size'] and r[k]['file'] == f'image_{i:06d}.npz'
            for k in ('endpoint', 'pose', 'hoi'))
            for i, r in enumerate(proof['records']))
        and len({p for r in proof['records'] for p in r['paths']}) == 48, 'Complete original slot/image/file census required')
    for record in proof['records']:
        for key, path, names in zip(('endpoint', 'pose', 'hoi'), record['paths'], fields):
            rt.require(proof['files'][path] == record[key]['identity'], 'Proof file association differs')
            load_bank(Path(path), record[key], names); check(deadline)
    result = []
    for record in proof['records']:
        args = []
        for key, path, names in zip(('endpoint', 'pose', 'hoi'), record['paths'], fields):
            args.extend((record[key], load_bank(Path(path), record[key], names)))
        value = numerical.reconstruct_interaction(*args); result.append(summary(value)); del value, args; check(deadline)
    return result


def cpu(code, revision, proof_pin, deadline):
    def expired(*_): raise TimeoutError('Inclusive native saved join')
    handlers = {v: signal.signal(v, expired) for v in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.alarm(max(1, math.ceil(deadline-time.monotonic())))
    try: return cpu_work(code, revision, proof_pin, deadline)
    finally:
        signal.alarm(0)
        for v, h in handlers.items(): signal.signal(v, h)


def cpu_work(code, revision, proof_pin, deadline):
    import numpy as np
    proof = rt.pinned(OUTPUT/'proof.json', proof_pin, 2 << 20); before = native_source(code, revision, proof)
    rt.require(os.environ.get('WR_IMAGE_ID') == IMAGE and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'}
        and sys.version_info[:2] == (3, 11) and np.__version__ == '1.26.3', 'Original offlineB47 CPU NumPy ABI required')
    files = {n: rt.identity(n, MAXIMUM) for n in proof['files']}
    rt.require(files == proof['files'], 'All raw banks authenticated before numerical joins')
    states = {n: snapshot(n) for n in files}; proof_state = snapshot(OUTPUT/'proof.json'); owner = OUTPUT.lstat(); report = dict(schema=SCHEMA, stage='native_vcoco_interaction_join', status='fail',
        phase='preflight', producer_revision=revision, image_id=IMAGE, proof_identity=proof_pin, images=[], GPU_used=False,
        models_loaded=0, reference_metadata_read=False, RGB_decoded=False, FIT_performed=False, selection_performed=False,
        ownership_verified=False, quality_verified=False, adoption=False, source_inputs_rehashed_after=False)
    try:
        report['phase'] = 'all47_array_prevalidation_and_join'; report['images'] = join_all(proof, deadline)
        report.update(status='pass', phase='complete', arrays_per_image=47)
    except BaseException as exc: report['error_type'] = error(exc)
    finally:
        try:
            rt.require(native_source(code, revision, proof) == before and {n: rt.identity(n, MAXIMUM) for n in files} == files
                and {n: snapshot(n) for n in files} == states and rt.identity(OUTPUT/'proof.json', 2 << 20) == proof_pin
                and snapshot(OUTPUT/'proof.json') == proof_state, 'Native source/raw bank/proof changed')
            report['source_inputs_rehashed_after'] = True; check(deadline)
        except BaseException as exc: report.update(status='fail', post_error_type=error(exc))
        publish_native(report, deadline, owner)
    return report


def publish_native(report, deadline, owner):
    allowed = {'proof.json', 'native.json'}|({'.container.cid'} if (OUTPUT/'.container.cid').exists() else set())
    now = OUTPUT.lstat()
    rt.require(stat.S_ISDIR(now.st_mode) and stat.S_IMODE(now.st_mode) == 0o700
        and (now.st_dev, now.st_ino, now.st_uid, now.st_gid) == (owner.st_dev, owner.st_ino, owner.st_uid, owner.st_gid)
        and {p.name for p in OUTPUT.iterdir()} == allowed-{'native.json'}, 'Foreign native parent rejected before open')
    with (OUTPUT/'native.json').open('x+b') as f:
        os.fchmod(f.fileno(), 0o400); opened = os.fstat(f.fileno())
        def update():
            now = OUTPUT.lstat(); leaf = (OUTPUT/'native.json').lstat()
            rt.require((now.st_dev, now.st_ino, now.st_uid, now.st_gid) == (owner.st_dev, owner.st_ino, owner.st_uid, owner.st_gid)
                and stat.S_IMODE(now.st_mode) == 0o700 and stat.S_ISREG(leaf.st_mode) and leaf.st_nlink == 1
                and (leaf.st_dev, leaf.st_ino, leaf.st_uid, leaf.st_gid) == (opened.st_dev, opened.st_ino, opened.st_uid, opened.st_gid)
                and {p.name for p in OUTPUT.iterdir()} == allowed, 'Foreign native publication rejected')
            f.seek(0); f.write(encode(report)); f.truncate(); f.flush(); os.fsync(f.fileno())
        try:
            update(); rt.require(rt.identity(OUTPUT/'native.json', 2 << 20) == dict(bytes=len(encode(report)),
                sha256=hashlib.sha256(encode(report)).hexdigest()), 'Native receipt publication differs'); check(deadline)
        except BaseException as exc: report.update(status='fail', publication_failed=True, publication_error_type=error(exc)); update()


def snapshot(path):
    s = rt.canonical(path).lstat()
    return tuple(getattr(s, k) for k in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_nlink', 'st_uid', 'st_gid', 'st_mtime_ns', 'st_ctime_ns'))


def error(exc): return type(exc).__name__ if type(exc).__name__ in ('ValueError', 'RuntimeError', 'TimeoutError', 'OSError', 'KeyError', 'ImportError', 'ModuleNotFoundError') else 'other'


def command(args, deadline):
    check(deadline); v = subprocess.run(args, env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent', DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
        capture_output=True, timeout=min(10, max(.001, deadline-time.monotonic())), check=False)
    rt.require(v.returncode == 0 and len(v.stdout) <= 1 << 20, 'Bounded private lifecycle failed'); return v.stdout.decode().strip()


def image(deadline):
    v = rt.strict(command(['docker', 'image', 'inspect', IMAGE, '--format', '{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}'], deadline))
    rt.require(v['Id'] == IMAGE and v['Architecture'] == 'amd64' and v['Os'] == 'linux' and v['RootFS']['Type'] == 'layers', 'Exact originalB47 required'); return v


def cleanup(name, revision, deadline):
    path = OUTPUT/'.container.cid'
    if path.exists():
        rt.identity(path, 65, readonly=False); raw = path.read_bytes(); rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', raw), 'Owned saved CID required'); cid = raw.decode().strip()
        found = command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline); rt.require(found in ('', cid), 'Ambiguous saved CID')
        if found:
            v = command(['docker', 'inspect', cid, '--format', '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'], deadline)
            rt.require(v == IMAGE+'|/'+name+'|'+ENTRY+'|'+revision, 'Refuse foreign container removal'); command(['docker', 'rm', '-f', cid], deadline)
        rt.require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline), 'Exact owned CID survives'); path.chmod(0o400)
    rt.require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Exact owned name survives')


def validate_native(v, proof, revision, proof_pin):
    rt.require(v['schema'] == SCHEMA and v['stage'] == 'native_vcoco_interaction_join' and v['status'] == 'pass' and v['phase'] == 'complete'
        and v['producer_revision'] == revision and v['image_id'] == IMAGE and v['proof_identity'] == proof_pin
        and v['source_inputs_rehashed_after'] is True and v['arrays_per_image'] == 47 and len(v['images']) == 16 and v['models_loaded'] == 0
        and all(v[k] is False for k in ('GPU_used', 'reference_metadata_read', 'RGB_decoded', 'FIT_performed', 'selection_performed', 'ownership_verified', 'quality_verified', 'adoption')),
        'Complete blind saved numerical join required')
    for i, (r, original) in enumerate(zip(v['images'], proof['records'])):
        p, k = original['pose']['persons'], original['hoi']['hand_object_pairs']
        sizes = (p*2*3600*10, p*2*k*15, k*3600*2)
        rt.require(type(r.get('supported_counts')) is list and type(r.get('nan_counts')) is list
            and len(r['supported_counts']) == len(r['nan_counts']) == 3
            and all(type(a) is int and type(b) is int and 0 <= a <= size and b == size-a
                for a, b, size in zip(r['supported_counts'], r['nan_counts'], sizes)), 'Exact availability counts required')
        rt.require(r['image_id'] == original['endpoint']['image_id'] and r['original_slot'] == r['acquired_ordinal'] == i
            and (r['persons'], r['objects'], r['native_pairs']) == (p, 3600, k)
            and (r['person_side_object_rows'], r['person_side_pair_rows'], r['pair_object_rows']) == (p*2*3600, p*2*k, k*3600)
            and all(re.fullmatch('[0-9a-f]{64}', str(r[n])) for n in ('raw_bank_fingerprint', 'evidence_fingerprint')),
            'All original Cartesian counts/order required')


def dispatch(code, revision, replica_revision, receipt_pin):
    started = time.monotonic(); deadline = started+BUDGET
    def expired(*_): raise TimeoutError('Inclusive600s saved join')
    handlers = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}; signal.alarm(BUDGET)
    try: return dispatch_work(code, revision, replica_revision, receipt_pin, started, deadline)
    finally:
        signal.alarm(0)
        for s, h in handlers.items(): signal.signal(s, h)


def dispatch_work(code, revision, replica_revision, receipt_pin, started, deadline):
    import sealed_callback_publication as publication
    before = proof = initial_image = artifacts = None; name = 'world-reward-vcoco-interaction-join-'+revision[:12]
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01', 'OriginalB47 VM01 root required')
    rt.canonical(OUTPUT); rt.require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'Fresh join namespace required')
    OUTPUT.mkdir(mode=0o700); s = OUTPUT.lstat(); owner = (s.st_dev, s.st_ino, s.st_uid)
    report = dict(schema=SCHEMA, stage='vcoco_interaction_join_host', status='fail', producer_revision=revision,
        image_id=IMAGE, source_inputs_assets_image_rehashed_after=False, owned_cleanup_verified=False,
        GPU_used=False, models_loaded=0, RGB_decoded=False, reference_metadata_read=False, FIT_performed=False,
        selection_performed=False, ownership_verified=False, quality_verified=False, adoption=False)
    try:
        before, proof = authenticate(code, revision, replica_revision, receipt_pin, deadline); initial_image = image(deadline)
        report.update(input_proof=before, source_binding=before['current_source']['binding'], image_identity=initial_image)
        rt.require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Occupied new CPU name')
        rt.write(OUTPUT/'proof.json', encode(proof)); proof_pin = rt.identity(OUTPUT/'proof.json', 2 << 20)
        argv = ['docker', 'run', '--rm', '--name', name, '--cidfile', str(OUTPUT/'.container.cid'), '--label', 'world-reward.job='+ENTRY,
            '--label', 'world-reward.revision='+revision, '--network', 'none', '--user', '0:0', '--memory', '6g', '--cpus', '4',
            '--pids-limit', '256', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--tmpfs', '/tmp:rw,nosuid,size=256m', '--entrypoint', '/usr/bin/env']
        for p in [*(code/n for n in NATIVE_FILES), *(code.parent/n for n in ('revision', 'source-sha256')), *(Path(n) for n in proof['files'])]:
            rt.canonical(p); rt.require(p.is_file() and not any(c in str(p) for c in ',\n'), 'Individual readonly file mounts required')
            argv += ['--mount', f'type=bind,src={p},dst={p},readonly']
        argv += ['--mount', f'type=bind,src={OUTPUT},dst={OUTPUT}', IMAGE, '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin', 'HOME=/tmp',
            'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'WR_IMAGE_ID='+IMAGE,
            '/opt/conda/bin/python', '-I', '-B', str(code/NATIVE_FILES[0]), '--native', '--code', str(code), '--revision', revision,
            '--proof-bytes', str(proof_pin['bytes']), '--proof-sha256', proof_pin['sha256'], '--deadline', format(deadline-20, '.17g')]
        v = subprocess.run(argv, env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent', DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=max(.001, deadline-time.monotonic()-20), check=False)
        report['native_exit_status'] = v.returncode; report['native_identity'] = rt.identity(OUTPUT/'native.json', 2 << 20)
        native = rt.pinned(OUTPUT/'native.json', report['native_identity'], 2 << 20)
        artifacts = {n: dict(pin=rt.identity(OUTPUT/n, 2 << 20), state=snapshot(OUTPUT/n)) for n in ('proof.json', 'native.json')}
        rt.require(v.returncode == 0, 'Native CPU join failed'); validate_native(native, proof, revision, proof_pin)
        report.update(status='pass', images=native['images'])
    except BaseException as exc: report['error_type'] = error(exc)
    finally:
        try: cleanup(name, revision, min(deadline+10, time.monotonic()+10)); report['owned_cleanup_verified'] = True
        except BaseException as exc: report.update(status='fail', cleanup_error_type=error(exc))
        try:
            after, again = authenticate(code, revision, replica_revision, receipt_pin, deadline)
            rt.require(before is not None and after == before and again == proof and image(deadline) == initial_image, 'All original source/inputs/assets/image changed')
            report['source_inputs_assets_image_rehashed_after'] = True
            if artifacts is not None:
                rt.require({n: dict(pin=rt.identity(OUTPUT/n, 2 << 20), state=snapshot(OUTPUT/n)) for n in artifacts} == artifacts, 'Original native/proof artifacts changed')
            if report['status'] == 'pass': validate_native(native, proof, revision, proof_pin)
        except BaseException as exc: report.update(status='fail', post_error_type=error(exc))
        allowed = {p.name for p in OUTPUT.iterdir()}; rt.require(allowed <= {'proof.json', 'native.json', '.container.cid'}, 'Foreign join output')
        for p in OUTPUT.iterdir(): rt.identity(p, 2 << 20)
        publication.publish(OUTPUT, report, deadline, started, owner, allowed, encode=encode, identity=rt.identity,
            snapshot=snapshot, require=rt.require, check=check, sync=sync, error=error, report_maximum=2 << 20)
    return report


def sync(path):
    fd = os.open(path, os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def arguments(argv):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--native', action='store_true'); p.add_argument('--code'); p.add_argument('--revision'); p.add_argument('--deadline', type=float)
    p.add_argument('--proof-bytes', type=int); p.add_argument('--proof-sha256')
    p.add_argument('--hoi-replica-revision'); p.add_argument('--hoi-receipt-bytes', type=int); p.add_argument('--hoi-receipt-sha256')
    a = p.parse_args(argv)
    if a.native:
        rt.require(all(getattr(a, n) is None for n in ('hoi_replica_revision', 'hoi_receipt_bytes', 'hoi_receipt_sha256'))
            and re.fullmatch('[0-9a-f]{40}', str(a.revision)) and type(a.code) is str and type(a.deadline) is float and math.isfinite(a.deadline), 'Only explicit native proof/origin permitted')
        a.pin = dict(bytes=a.proof_bytes, sha256=a.proof_sha256)
    else:
        rt.require(all(getattr(a, n) is None for n in ('code', 'revision', 'deadline', 'proof_bytes', 'proof_sha256'))
            and re.fullmatch('[0-9a-f]{40}', str(a.hoi_replica_revision)), 'Only actual HOI replica pins permitted')
        a.pin = dict(bytes=a.hoi_receipt_bytes, sha256=a.hoi_receipt_sha256)
    rt.require(type(a.pin['bytes']) is int and 0 < a.pin['bytes'] <= 2 << 20 and re.fullmatch('[0-9a-f]{64}', str(a.pin['sha256'])), 'Independent actual receipt/proof byte pins required')
    return a


if __name__ == '__main__':
    a = arguments(sys.argv[1:]); result = cpu(Path(a.code), a.revision, a.pin, a.deadline) if a.native else dispatch(
        Path(os.environ.get('WR_CODE', '/invalid')), os.environ.get('WR_CODE_REVISION', ''), a.hoi_replica_revision, a.pin)
    print(encode(dict(stage=result['stage'], status=result['status'])).decode(), end=''); raise SystemExit(0 if result['status'] == 'pass' else 1)
