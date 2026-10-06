"""Blind full48 saved observations: complete unselected evidence, never FIT.

System-host imports are stdlib only. Host lineage/replica authentication is
separate from a narrow offline NumPy worker; no RGB/models/private references
are mounted. The frozen numerical constructor is reused without modification.
"""
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
import vcoco_full_interaction_core as numerical

ROOT = rt.ROOT
ENTRY = 'run_vcoco_full_interaction_join'
OUTPUT = ROOT/'results/vcoco-full-interaction-join-v1'
SCHEMA = 'world_reward.vcoco_full_interaction_join.v1'
IMAGE = rt.BASE  # Existing VM02 CPU runtime, not VM01's different Docker ID.
BUDGET, MAXIMUM, MAX_EVIDENCE, MAX_TOTAL = 600, 32 << 20, 512 << 20, 8 << 30
MAX_CONTROL_TOTAL = 4 << 20
NUMERICAL_PIN = dict(bytes=11109, sha256='31a20673ec6a862535d7caa2250ed9fa8b8d9bbd2352b160ae3924d9782be961')
HOI_REV = 'a451aa1d28338d5144398a1b6ca1e2877414ce48'
HOI_DECLARATION = dict(files=371, entries=376,
    closure_sha256='4484f35add27e2cec6b1fbd888559524caf31e791ac8406c3f3eca6a65751b13',
    source_XZ_sha256='6e779c9180ea709aacf3f11cc47023e097bfd217dd59b509cae5b04037e5b2b5')
HOI_PINS = {'report.json': dict(bytes=190702, sha256='b7acf4153fd33ec43e9b460aaa83f1f457b5995af56bebe3c45375dc0df7f078'),
    'model.json': dict(bytes=169647, sha256='d358a40231a98b587f287870eb5011efd8ffeb692cfcea6b0affc5f718b37cdc'),
    'model_proof.json': dict(bytes=328169, sha256='53a3e84d0b6a273ec5586dcf9e52a388d343a16132161aa9185a072bf1a15a2f')}
HOI_EXECUTABLES = frozenset(('infra/run_coco_endpoint_prepare.sh', 'infra/run_rgb_endpoint_bank.sh', 'infra/run_vcoco_full_hoi_observations.sh'))
EXECUTABLES = HOI_EXECUTABLES | {'infra/run_keypoint_rgb_dwpose.sh'}
NATIVE_FILES = ('infra/vcoco_full_interaction_join.py', 'infra/mediapipe_cpu_runtime_verify.py',
    'infra/vcoco_full_interaction_core.py', 'src/world_reward/__init__.py',
    'src/world_reward/person_pose_observations.py', 'src/world_reward/hoi_detr_observations.py',
    'src/world_reward/owlv2_object_observations.py', 'src/world_reward/owlv2_candidate_bridge.py',
    'src/world_reward/interaction_tuple_evidence.py', 'src/world_reward/interaction_candidate_evidence.py')
FLAGS = ('GPU_used', 'RGB_decoded', 'reference_metadata_read', 'FIT_performed',
         'selection_performed', 'ownership_verified', 'quality_verified', 'adoption')


def encode(value):
    import json
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def check(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('Inclusive600s full saved join')


def snapshot(path):
    s = rt.canonical(path).lstat()
    return tuple(getattr(s, k) for k in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_nlink', 'st_uid', 'st_gid', 'st_mtime_ns', 'st_ctime_ns'))


def sync(path):
    fd = os.open(path, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


def error(exc):
    return type(exc).__name__ if type(exc).__name__ in ('ValueError', 'RuntimeError', 'TimeoutError', 'OSError', 'KeyError', 'ImportError', 'ModuleNotFoundError') else 'other'


def host_modules():
    import vcoco_full_public_replica as endpoint
    import vcoco_full_hoi_run as hoi
    import vcoco_full_pose_replica as pose
    return endpoint, hoi, pose


def helpers():
    endpoint, hoi, pose = host_modules()
    return tuple(dict.fromkeys((*NATIVE_FILES, 'infra/run_vcoco_full_interaction_join.sh',
        'infra/sealed_callback_publication.py', *endpoint.HELPERS, *hoi.HELPERS, *pose.HELPERS)))


def modes(code, executable):
    paths = (code, *code.rglob('*'), code.parent/'revision', code.parent/'source-sha256')
    for p in paths:
        s = rt.canonical(p).lstat()
        mode = 0o555 if p.is_dir() or (p.is_relative_to(code) and str(p.relative_to(code)) in executable) else 0o444
        rt.require(s.st_uid == s.st_gid == 0 and stat.S_IMODE(s.st_mode) == mode, 'Exact immutable source owner/Git modes required')
    return {str(p): snapshot(p) for p in paths}


def source(code, revision):
    value = rt.source(ROOT, code, revision, ENTRY, helpers())
    for n in helpers():
        if n.endswith('.py'):
            name = Path(n).stem if n.startswith('infra/') else '.'.join(Path(n).with_suffix('').parts[1:])
            module = sys.modules.get(name)
            if module is not None: rt.require(Path(module.__file__).resolve() == code/n, 'Current imported helper origin required')
    rt.require(Path(__file__).resolve() == code/NATIVE_FILES[0] and value['helpers'][NATIVE_FILES[2]] == NUMERICAL_PIN,
               'Frozen full numerical source required')
    return dict(binding=value, states=modes(code, EXECUTABLES))


def hoi_inputs(code, current, projection, deadline):
    _, hoi, _ = host_modules()
    values = {n: rt.pinned(hoi.OUTPUT/n, p, 2 << 20) for n, p in HOI_PINS.items()}
    host, native, proof = (values[n] for n in HOI_PINS)
    old = ROOT/'jobs'/HOI_REV/hoi.ENTRY/'code'
    binding = rt.source(ROOT, old, HOI_REV, hoi.ENTRY, tuple(host['source_binding']['helpers']))
    rt.require(binding == host['source_binding'] and binding['entries'] == HOI_DECLARATION['entries']
        and binding['closure_sha256'] == HOI_DECLARATION['closure_sha256']
        and sum(p.is_file() for p in old.rglob('*')) == HOI_DECLARATION['files']
        and (old.parent/'source-sha256').read_bytes() == (HOI_DECLARATION['source_XZ_sha256']+'\n').encode()
        and all(current['helpers'][n] == pin for n, pin in binding['helpers'].items()),
        'Whole original HOI source/unchanged metadata validator required')
    source_states = modes(old, HOI_EXECUTABLES)
    rt.require(host['schema'] == hoi.SCHEMA and host['stage'] == 'full48_hoi_observations_host'
        and host['producer_revision'] == HOI_REV and host['status'] == 'pass' and host['phase'] == 'complete'
        and host['native_report_identity'] == HOI_PINS['model.json'] and host['native_proof_identity'] == HOI_PINS['model_proof.json']
        and host['images'] == native['images'] and host['endpoint_receipt_pins'] == hoi.ENDPOINT_PINS
        and host['outputs_sealed'] is host['owned_cleanup_verified'] is host['source_inputs_runtime_assets_rehashed_after'] is True
        and all(host[k] is False for k in hoi.FLAGS), 'Pinned original48 HOI PASS required')
    rt.require(host['saved_output_identities']['model_proof.json'] == HOI_PINS['model_proof.json']
        and proof['markers'] == binding['markers'] and set(proof['native_files']) == set(hoi.NATIVE_FILES)
        and all(proof['native_files'][n] == binding['helpers'][n] for n in hoi.NATIVE_FILES)
        and proof['image_id'] == hoi.IMAGE and proof['projection_identity'] == host['saved_output_identities']['public_bank.json']
        and rt.pinned(hoi.OUTPUT/'public_bank.json', proof['projection_identity'], 2 << 20) == projection, 'Original HOI native source/public projection binding required')
    hoi.validate_native(native, proof, HOI_REV, 'model', HOI_PINS['model_proof.json'], projection)
    expected = set(host['saved_output_identities'])|{'report.json'}
    folder = hoi.OUTPUT; s = folder.lstat()
    rt.require(stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode) == 0o500 and s.st_uid == s.st_gid == 0
        and len(expected) == 58 and {p.name for p in folder.iterdir()} == expected, 'Exact original58 HOI leaves required')
    files = {str(folder/n): rt.identity(folder/n, 16 << 20) for n in expected}
    rt.require(all(files[str(folder/n)] == p for n, p in host['saved_output_identities'].items())
        and all(Path(n).lstat().st_uid == Path(n).lstat().st_gid == 0 and stat.S_IMODE(Path(n).lstat().st_mode) == 0o400 for n in files), 'Original HOI bytes/modes required')
    for phase in ('overlay', 'model'):
        cid = (folder/(phase+'.cid')).read_bytes()
        absent(cid, 'world-reward-vcoco-full-hoi-'+phase+'-'+HOI_REV[:12], deadline)
    return dict(rows=native['images'], files=files, states={str(p): snapshot(p) for p in (folder, *folder.iterdir())},
                source=binding, source_states=source_states)


def authenticate(code, revision, replica_revision, receipt_pin, pose_pins, deadline):
    before = source(code, revision); endpoint, hoi, pose = host_modules()
    projection, _, _, e = endpoint.sender_inputs(code, before['binding'], deadline)
    h = hoi_inputs(code, before['binding'], projection, deadline)
    p = pose.authenticate_receiver(code, replica_revision, receipt_pin)
    rt.require(p['pose_receipt_pins'] == pose_pins and re.fullmatch('[0-9a-f]{40}', p['original_pose_revision'])
        and p['sender_source_live_verified'] is p['sender_runtime_live_verified'] is False and len(p['rows']) == 48,
        'Independent actual pose/import pins required; sender not falsely live')
    projection_path = Path(p['projection_path'])
    v = rt.pinned(projection_path, p['projection_identity'], 2 << 20)
    rt.require(set(v) == {'schema', 'rows'} and v['schema'] == 'world_reward.public_person_pose_reference.v1' and v['rows'] == p['rows'], 'Public pose-only projection required')
    rows = []; files = {}
    for i, (er, pr, hr) in enumerate(zip(projection['banks'], p['rows'], h['rows'])):
        paths = (endpoint.endpoint.OUTPUT/er['file'], pose.DEST/'banks'/pr['file'], hoi.OUTPUT/hr['file'])
        for path, row in zip(paths, (er, pr, hr)):
            rt.require(rt.identity(path, MAXIMUM) == row['identity'], 'Complete original NPZ identity differs'); files[str(path)] = row['identity']
        rows.append(dict(endpoint=er, pose=pr, hoi=hr, paths=[str(x) for x in paths]))
    proof = dict(schema=SCHEMA, producer_revision=revision, image_id=IMAGE, source={
        'helpers': {n: before['binding']['helpers'][n] for n in NATIVE_FILES}, 'markers': before['binding']['markers']}, records=rows, files=files)
    validate_population(proof)
    return dict(current_source=before, endpoint=e, hoi=h, pose=p), proof


def native_source(code, revision, proof):
    rt.require(Path(__file__).resolve() == code/NATIVE_FILES[0] and proof['producer_revision'] == revision
        and set(proof['source']['helpers']) == set(NATIVE_FILES)
        and {str(p.relative_to(code)) for p in code.rglob('*') if p.is_file()} == set(NATIVE_FILES), 'Exact public numerical whitelist required')
    for n, pin in proof['source']['markers'].items(): rt.require(rt.identity(code.parent/n, 100) == pin, 'Current marker differs')
    rt.require((code.parent/'revision').read_bytes() == (revision+'\n').encode(), 'Current producer revision required')
    rows = {str(p): dict(pin=rt.identity(p, 2 << 20, empty=True), state=snapshot(p))
            for p in [*(code/n for n in NATIVE_FILES), *(code.parent/n for n in ('revision', 'source-sha256'))]}
    rt.require({n: rows[str(code/n)]['pin'] for n in NATIVE_FILES} == proof['source']['helpers']
        and proof['source']['helpers'][NATIVE_FILES[2]] == NUMERICAL_PIN, 'Frozen numerical/helper byte pins required')
    for n in NATIVE_FILES:
        module = sys.modules.get(Path(n).stem if n.startswith('infra/') else '.'.join(Path(n).with_suffix('').parts[1:]))
        if module is not None: rt.require(Path(module.__file__).resolve() == code/n, 'Native imported helper origin differs')
    return rows


def load_bank(path, row, fields, *, maximum=MAXIMUM):
    """ZIP bounds + all actual plain array bytes; no numerical reconstruction yet."""
    import numpy as np
    rt.require(rt.identity(path, maximum) == row['identity'] and set(row['arrays']) == set(fields), 'Complete saved bank SHA required')
    rt.require(all(type(pin) is dict and set(pin) == {'shape', 'dtype', 'sha256'} and type(pin['shape']) is list
        and all(type(n) is int and n >= 0 for n in pin['shape']) and type(pin['dtype']) is str
        and re.fullmatch('[0-9a-f]{64}', str(pin['sha256'])) for pin in row['arrays'].values()), 'Plain original array metadata required')
    with zipfile.ZipFile(path) as archive:
        rows = archive.infolist(); rt.require(len(rows) == len(fields) and {r.filename for r in rows} == {n+'.npy' for n in fields}
            and all(not r.is_dir() and not r.flag_bits & 1 and r.file_size <= maximum for r in rows)
            and sum(r.file_size for r in rows) <= maximum, 'Bounded exact native NPZ members required')
        for info in rows:
            pin = row['arrays'][info.filename[:-4]]
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream); rt.require(version in ((1, 0), (2, 0)), 'Original NPY header version required')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream, max_header_size=4096)
                size = math.prod(shape)*dtype.itemsize
                rt.require(not dtype.hasobject and list(shape) == pin['shape'] and dtype.str == pin['dtype']
                    and size <= maximum and info.file_size == stream.tell()+size, 'Bounded exact NPY shape/dtype/payload required')
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


def validate_population(proof):
    rt.require(set(proof) == {'schema', 'producer_revision', 'image_id', 'source', 'records', 'files'}
        and proof['schema'] == SCHEMA and proof['image_id'] == IMAGE and len(proof['records']) == 48 and len(proof['files']) == 144,
        'Exact public full48/144-file proof required')
    rt.require(type(proof['records']) is list and type(proof['files']) is dict
        and all(type(r) is dict and set(r) == {'endpoint', 'pose', 'hoi', 'paths'} for r in proof['records']), 'Strict record schema required')
    rt.require(len({r['endpoint']['image_id'] for r in proof['records']}) == 48
        and len({p for r in proof['records'] for p in r['paths']}) == 144, 'Complete injective original population required')
    for i, r in enumerate(proof['records']):
        rt.require(set(r) == {'endpoint', 'pose', 'hoi', 'paths'} and len(r['paths']) == 3
            and all(x['original_slot'] == x['acquired_ordinal'] == i and x['original_frame_index'] == 0
                and x['image_id'] == r['endpoint']['image_id'] and x['image_size'] == r['endpoint']['image_size']
                and x['file'] == f'image_{i:06d}.npz' for x in (r['endpoint'], r['pose'], r['hoi']))
            and r['pose']['endpoint_bank_identity'] == r['hoi']['endpoint_bank_identity'] == r['endpoint']['identity']
            and r['pose']['person_ids'] == r['hoi']['source_person_ids'] == r['endpoint']['person_ids']
            and r['pose']['persons'] == r['endpoint']['person_retained_rows']
            and all(proof['files'][path] == row['identity'] for path, row in zip(r['paths'], (r['endpoint'], r['pose'], r['hoi']))),
            'Original IDs/grid/slots/input bank associations required')


def evidence_arrays(value):
    import numpy as np
    e, h = value.evidence, value.evidence.hoi_evidence
    result = {prefix+name: a for prefix, arrays in (('base__', e.arrays), ('local__', h.arrays), ('bridge__', e.hoi_routes))
              for name, a in arrays.items()}
    result.update(base_features=e.features, base_supported=e.feature_supported, local_features=h.features,
                  local_supported=h.feature_supported, bridge_features=e.route_features, bridge_supported=e.route_supported,
                  object_ids=np.asarray(e.objects.object_ids))
    return result


def save_evidence(value, deadline):
    import numpy as np
    arrays = evidence_arrays(value)
    rt.require(sum(a.nbytes for a in arrays.values()) <= MAX_EVIDENCE-(1 << 20), 'Full evidence image exceeds fixed512MiB bound; never truncate')
    row = summary(value); row.update(file=f'image_{value.original_slot:06d}.npz',
        arrays={n: numerical._identity(a) for n, a in arrays.items()}, person_ids=list(value.person.person_ids),
        object_id_fingerprint=numerical._identity(arrays['object_ids']), feature_names=[list(value.evidence.feature_names),
            list(value.evidence.hoi_evidence.feature_names), list(value.evidence.route_feature_names)])
    path = OUTPUT/row['file']; owner = None
    try:
        with path.open('xb') as stream:
            s = os.fstat(stream.fileno()); owner = (s.st_dev, s.st_ino, s.st_uid)
            os.fchmod(stream.fileno(), 0o400); np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        sync(OUTPUT); check(deadline); row['identity'] = rt.identity(path, MAX_EVIDENCE)
        reopened = load_bank(path, row, tuple(arrays), maximum=MAX_EVIDENCE)
        rt.require(all(reopened[n].tobytes() == a.tobytes() for n, a in arrays.items()), 'Saved full evidence changed on reopen')
        check(deadline); return row
    except BaseException:
        if owner is not None and path.exists() and snapshot(path)[:2]+(path.stat().st_uid,) == owner:
            path.unlink(); sync(OUTPUT)
        raise


def join_all(proof, deadline, *, records_out=None):
    validate_population(proof)
    fields = (numerical.ENDPOINT_FIELDS, numerical.POSE_FIELDS, numerical.HOI_FIELDS)
    for record in proof['records']:
        for key, path, names in zip(('endpoint', 'pose', 'hoi'), record['paths'], fields):
            load_bank(Path(path), record[key], names); check(deadline)
    result = [] if records_out is None else records_out; total = 0
    for record in proof['records']:
        args = []
        for key, path, names in zip(('endpoint', 'pose', 'hoi'), record['paths'], fields):
            args.extend((record[key], load_bank(Path(path), record[key], names)))
        value = numerical.reconstruct_interaction(*args, population=48)
        row = save_evidence(value, deadline); total += row['identity']['bytes']
        result.append(row)
        rt.require(total <= MAX_TOTAL, 'Full48 evidence exceeds fixed8GiB output bound; no subset rescue'); del value, args; check(deadline)
    return result


def command(args, deadline):
    check(deadline); v = subprocess.run(args, env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent', DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
        capture_output=True, timeout=min(10, max(.001, deadline-time.monotonic())), check=False)
    rt.require(v.returncode == 0 and len(v.stdout) <= 1 << 20, 'Bounded private CPU lifecycle failed')
    return v.stdout.decode().strip()


def absent(raw, name, deadline):
    rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', raw) and not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+raw.decode().strip()], deadline)
        and not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Original producer CID/name must be absent')


def image(deadline):
    v = rt.strict(command(['docker', 'image', 'inspect', IMAGE, '--format', '{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}'], deadline))
    rt.require(v['Id'] == IMAGE and v['Architecture'] == 'amd64' and v['Os'] == 'linux'
        and v['RootFS']['Type'] == 'layers' and type(v['RootFS']['Layers']) is list and len(v['RootFS']['Layers']) == 44,
        'Exact existing VM02 CPU image/layers required')
    return v


def cleanup(name, revision, deadline):
    path = OUTPUT/'.container.cid'
    if path.exists():
        rt.identity(path, 65, readonly=False); raw = path.read_bytes()
        rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', raw), 'Owned saved CID required'); cid = raw.decode().strip()
        found = command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline)
        rt.require(found in ('', cid), 'Ambiguous saved CID')
        if found:
            v = command(['docker', 'inspect', cid, '--format', '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'], deadline)
            rt.require(v == IMAGE+'|/'+name+'|'+ENTRY+'|'+revision, 'Never remove foreign or renamed container')
            command(['docker', 'rm', '-f', cid], deadline)
        absent(raw, name, deadline); path.chmod(0o400)
    rt.require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Owned name survives')


def expected_evidence_metadata(original):
    p, k = original['pose']['persons'], original['hoi']['hand_object_pairs']; result = {}
    base = dict(person_slots=(), side_indices=(), generic_object_slots=(), pose_keypoint_indices=(2,),
        pose_original_xy=(2, 2), pose_raw_scores=(2,), pose_native_valid=(2,), pose_in_original_image=(2,),
        person_boxes_original_xyxy=(4,), person_detector_scores=(), object_boxes_original_xyxy=(4,),
        object_raw_scores=(), object_box_positive_area=(), object_box_in_original_image=())
    local = dict(person_slots=(), side_indices=(), native_pair_slots=(), detection_slots=(2,), query_ids=(2,),
        retained_nms_positions=(2,), native_flat_keep=(2,), pose_keypoint_indices=(2,), pose_original_xy=(2, 2),
        pose_raw_scores=(2,), pose_native_valid=(2,), pose_in_original_image=(2,), person_boxes_original_xyxy=(4,),
        person_detector_scores=(), hoi_boxes_original_xyxy=(2, 4), hoi_raw_scores=(2,), hoi_decayed_scores=(2,),
        hoi_logits=(2,), hoi_box_positive_area=(2,), hoi_box_in_original_image=(2,))
    bridge = dict(native_pair_slots=(), generic_object_slots=(), detection_slots=(2,), query_ids=(2,), raw_logits=(2,))
    for prefix, rows, count in (('base__', base, p*7200), ('local__', local, p*2*k), ('bridge__', bridge, k*3600)):
        for name, tail in rows.items():
            dtype = '|b1' if name in ('pose_native_valid', 'pose_in_original_image', 'object_box_positive_area', 'object_box_in_original_image', 'hoi_box_positive_area', 'hoi_box_in_original_image') else (
                '<f8' if name in ('pose_original_xy', 'person_boxes_original_xyxy', 'person_detector_scores') else (
                '<f4' if name in ('pose_raw_scores', 'object_boxes_original_xyxy', 'object_raw_scores', 'hoi_boxes_original_xyxy',
                    'hoi_raw_scores', 'hoi_decayed_scores', 'hoi_logits', 'raw_logits') else '<i8'))
            result[prefix+name] = ([count, *tail], dtype)
    for name, count, width in (('base', p*7200, 10), ('local', p*2*k, 15), ('bridge', k*3600, 2)):
        result[name+'_features'] = ([count, width], '<f8'); result[name+'_supported'] = ([count, width], '|b1')
    result['object_ids'] = ([3600], '<U17')
    return result


def validate_native(v, proof, revision, proof_pin):
    rt.require(v['schema'] == SCHEMA and v['stage'] == 'native_full48_interaction_join' and v['status'] == 'pass' and v['phase'] == 'complete'
        and v['producer_revision'] == revision and v['image_id'] == IMAGE and v['proof_identity'] == proof_pin
        and v['source_inputs_rehashed_after'] is True and v['all144_banks_prevalidated'] is True
        and v['input_arrays_per_image'] == 47 and v['output_arrays_per_image'] == 46
        and len(v['images']) == 48 and v['models_loaded'] == 0 and all(v[k] is False for k in FLAGS), 'Complete blind saved48 native receipt required')
    for i, (r, original) in enumerate(zip(v['images'], proof['records'])):
        p, k = original['pose']['persons'], original['hoi']['hand_object_pairs']; sizes = (p*2*3600*10, p*2*k*15, k*3600*2)
        rt.require(r['image_id'] == original['endpoint']['image_id'] and r['original_slot'] == r['acquired_ordinal'] == i
            and (r['persons'], r['objects'], r['native_pairs']) == (p, 3600, k)
            and (r['person_side_object_rows'], r['person_side_pair_rows'], r['pair_object_rows']) == (p*7200, p*2*k, k*3600)
            and len(r['supported_counts']) == len(r['nan_counts']) == 3
            and all(type(a) is int and type(b) is int and 0 <= a <= n and b == n-a for a, b, n in zip(r['supported_counts'], r['nan_counts'], sizes))
            and all(re.fullmatch('[0-9a-f]{64}', str(r[n])) for n in ('raw_bank_fingerprint', 'evidence_fingerprint'))
            and r['file'] == f'image_{i:06d}.npz' and r['person_ids'] == original['endpoint']['person_ids']
            and r['object_id_fingerprint'] == r['arrays']['object_ids']
            and rt.identity(OUTPUT/r['file'], MAX_EVIDENCE) == r['identity'] and type(r['arrays']) is dict and set(r['arrays']) == set(expected_evidence_metadata(original))
            and all(type(pin) is dict and set(pin) == {'shape', 'dtype', 'sha256'} and pin['shape'] == shape and pin['dtype'] == dtype
                and re.fullmatch('[0-9a-f]{64}', str(pin['sha256'])) for name, (shape, dtype) in expected_evidence_metadata(original).items()
                for pin in (r['arrays'][name],)),
            'Complete unselected evidence population/bytes required')


def native(code, revision, proof_pin, deadline):
    import numpy as np
    proof = rt.pinned(OUTPUT/'proof.json', proof_pin, 2 << 20); before = None; files = {}; states = {}
    owner = OUTPUT.lstat(); report = dict(schema=SCHEMA, stage='native_full48_interaction_join', status='fail', phase='preflight',
        producer_revision=revision, image_id=IMAGE, proof_identity=proof_pin, images=[],
        input_arrays_per_image=47, output_arrays_per_image=46, models_loaded=0,
        source_inputs_rehashed_after=False, all144_banks_prevalidated=False, **{n: False for n in FLAGS})
    try:
        before = native_source(code, revision, proof)
        rt.require(os.environ.get('WR_IMAGE_ID') == IMAGE and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'} and sys.version_info[:2] == (3, 11) and np.__version__ == '1.26.3',
            'Offline existing7eb CPU NumPy ABI required')
        files = {n: rt.identity(n, MAXIMUM) for n in proof['files']}; states = {n: snapshot(n) for n in files}
        rt.require(files == proof['files'], 'Full144 original raw bank bytes required')
        proof_state = snapshot(OUTPUT/'proof.json')
        report['phase'] = 'all144_array_prevalidation_and_join'
        join_all(proof, deadline, records_out=report['images']); report['all144_banks_prevalidated'] = True
        report.update(status='pass', phase='complete')
    except BaseException as exc: report['error_type'] = error(exc)
    finally:
        try:
            rt.require(before is not None and native_source(code, revision, proof) == before
                and {n: rt.identity(n, MAXIMUM) for n in files} == files and {n: snapshot(n) for n in files} == states
                and rt.identity(OUTPUT/'proof.json', 2 << 20) == proof_pin and snapshot(OUTPUT/'proof.json') == proof_state,
                'Native source/raw bank/proof drift')
            report['source_inputs_rehashed_after'] = True; check(deadline)
        except BaseException as exc: report.update(status='fail', post_error_type=error(exc))
        publish_native(report, deadline, owner)
    return report


def publish_native(report, deadline, owner):
    completed = {p.name for p in OUTPUT.iterdir() if re.fullmatch(r'image_[0-9]{6}\.npz', p.name)}
    rt.require(completed == {r['file'] for r in report['images']}, 'Every completed evidence leaf has an original row')
    allowed = {'proof.json', *completed}|({'.container.cid'} if (OUTPUT/'.container.cid').exists() else set())
    now = OUTPUT.lstat()
    rt.require(stat.S_ISDIR(now.st_mode) and stat.S_IMODE(now.st_mode) == 0o700
        and (now.st_dev, now.st_ino, now.st_uid, now.st_gid) == (owner.st_dev, owner.st_ino, owner.st_uid, owner.st_gid)
        and {p.name for p in OUTPUT.iterdir()} == allowed, 'Owned native parent/all completed evidence leaves required')
    with (OUTPUT/'native.json').open('x+b') as stream:
        os.fchmod(stream.fileno(), 0o400); opened = os.fstat(stream.fileno())
        def update():
            directory = OUTPUT.lstat(); leaf = (OUTPUT/'native.json').lstat()
            rt.require((directory.st_dev, directory.st_ino, directory.st_uid, directory.st_gid)
                == (owner.st_dev, owner.st_ino, owner.st_uid, owner.st_gid) and stat.S_IMODE(directory.st_mode) == 0o700
                and snapshot(OUTPUT/'native.json')[:2] == (opened.st_dev, opened.st_ino) and leaf.st_nlink == 1
                and stat.S_ISREG(leaf.st_mode) and stat.S_IMODE(leaf.st_mode) == 0o400
                and leaf.st_uid == opened.st_uid and leaf.st_gid == opened.st_gid
                and {p.name for p in OUTPUT.iterdir()} == allowed|{'native.json'}, 'Foreign native receipt rejected')
            stream.seek(0); stream.write(encode(report)); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        try:
            update(); sync(OUTPUT); check(deadline)
        except BaseException as exc:
            report.update(status='fail', publication_failed=True, publication_error_type=error(exc)); update()


def dispatch(code, revision, replica_revision, receipt_pin, pose_pins):
    started = time.monotonic(); deadline = started+BUDGET
    handlers = {s: signal.signal(s, lambda *_: (_ for _ in ()).throw(TimeoutError('Inclusive600s host saved join')))
                for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL, BUDGET)
    try: return dispatch_work(code, revision, replica_revision, receipt_pin, pose_pins, started, deadline)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, handler in handlers.items(): signal.signal(s, handler)


def dispatch_work(code, revision, replica_revision, receipt_pin, pose_pins, started, deadline):
    import sealed_callback_publication as publication
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02', 'Actual VM02 root CPU host required')
    rt.canonical(OUTPUT); rt.require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'Fresh saved48 join namespace required')
    OUTPUT.mkdir(mode=0o700); s = OUTPUT.lstat(); owner = (s.st_dev, s.st_ino, s.st_uid)
    before = proof = initial_image = native_value = None; name = 'world-reward-vcoco-full-interaction-join-'+revision[:12]
    report = dict(schema=SCHEMA, stage='full48_interaction_join_host', status='fail', phase='source', producer_revision=revision,
        image_id=IMAGE, owned_cleanup_verified=False, source_inputs_image_rehashed_after=False, models_loaded=0, **{n: False for n in FLAGS})
    try:
        before, proof = authenticate(code, revision, replica_revision, receipt_pin, pose_pins, deadline); initial_image = image(deadline)
        report.update(source_binding=before['current_source']['binding'], input_proof=before, pose_receipt_pins=pose_pins, pose_replica_identity=receipt_pin)
        rt.require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Occupied owned CPU name')
        rt.write(OUTPUT/'proof.json', encode(proof), 0o400); proof_pin = rt.identity(OUTPUT/'proof.json', 2 << 20)
        argv = ['docker', 'run', '--rm', '--name', name, '--cidfile', str(OUTPUT/'.container.cid'), '--label', 'world-reward.job='+ENTRY,
            '--label', 'world-reward.revision='+revision, '--network', 'none', '--user', '0:0', '--memory', '8g', '--cpus', '4',
            '--pids-limit', '256', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--tmpfs', '/tmp:rw,nosuid,size=256m', '--entrypoint', '/usr/bin/env']
        for path in [*(code/n for n in NATIVE_FILES), *(code.parent/n for n in ('revision', 'source-sha256')), *(Path(n) for n in proof['files'])]:
            rt.canonical(path); rt.require(path.is_file() and all(c not in str(path) for c in (',', '\n')), 'Individual public readonly mounts required')
            argv += ['--mount', f'type=bind,src={path},dst={path},readonly']
        argv += ['--mount', f'type=bind,src={OUTPUT},dst={OUTPUT}', IMAGE, '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin', 'HOME=/tmp',
            'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'WR_IMAGE_ID='+IMAGE,
            '/opt/conda/bin/python', '-I', '-B', str(code/NATIVE_FILES[0]), '--native', '--code', str(code), '--revision', revision,
            '--proof-bytes', str(proof_pin['bytes']), '--proof-sha256', proof_pin['sha256'], '--deadline', format(deadline-20, '.17g')]
        report['phase'] = 'native'
        result = subprocess.run(argv, env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent', DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=max(.001, deadline-time.monotonic()-20), check=False)
        report['native_exit_status'] = result.returncode; report['native_identity'] = rt.identity(OUTPUT/'native.json', 2 << 20)
        native_value = rt.pinned(OUTPUT/'native.json', report['native_identity'], 2 << 20)
        rt.require(result.returncode == 0, 'Saved native join failed'); validate_native(native_value, proof, revision, proof_pin)
        report.update(status='pass', phase='complete', images=native_value['images'])
    except BaseException as exc: report.update(error_type=error(exc), failure_stage=report['phase'])
    finally:
        try: cleanup(name, revision, min(deadline+10, time.monotonic()+10)); report['owned_cleanup_verified'] = True
        except BaseException as exc: report.update(status='fail', cleanup_error_type=error(exc))
        try:
            after, again = authenticate(code, revision, replica_revision, receipt_pin, pose_pins, deadline)
            rt.require(before is not None and after == before and again == proof and image(deadline) == initial_image, 'Current/original source/all raw banks/image drift')
            report['source_inputs_image_rehashed_after'] = True
            if native_value is not None and report['status'] == 'pass': validate_native(native_value, proof, revision, proof_pin)
        except BaseException as exc: report.update(status='fail', post_error_type=error(exc))
        allowed = {p.name for p in OUTPUT.iterdir()}
        rt.require(allowed <= {'proof.json', 'native.json', '.container.cid', *(f'image_{i:06d}.npz' for i in range(48))}, 'Foreign saved evidence outputs')
        parent = OUTPUT.lstat()
        rt.require(stat.S_ISDIR(parent.st_mode) and stat.S_IMODE(parent.st_mode) == 0o700
            and (parent.st_dev, parent.st_ino, parent.st_uid, parent.st_gid) == (*owner, s.st_gid), 'Original output parent required')
        leaves = {n: rt.canonical(OUTPUT/n).lstat() for n in allowed}
        rt.require(all(stat.S_ISREG(v.st_mode) and v.st_nlink == 1 and stat.S_IMODE(v.st_mode) == 0o400
            and (v.st_uid, v.st_gid) == (s.st_uid, s.st_gid) and 0 < v.st_size <= MAX_EVIDENCE for v in leaves.values()),
            'Only bounded originally owned readonly output leaves may be sealed')
        report['output_capacity_gate_passed'] = (sum(v.st_size for n, v in leaves.items() if n.startswith('image_')) <= MAX_TOTAL
            and sum(v.st_size for v in leaves.values()) <= MAX_TOTAL+MAX_CONTROL_TOTAL)
        if not report['output_capacity_gate_passed']:
            report.update(status='fail', output_capacity_error_type='ValueError', failure_stage='output_capacity')
        report['saved_outputs'] = {n: rt.identity(OUTPUT/n, MAX_EVIDENCE) for n in allowed}
        publication.publish(OUTPUT, report, deadline, started, owner, allowed, encode=encode, identity=rt.identity,
            snapshot=snapshot, require=rt.require, check=check, sync=sync, error=error, maximum=MAX_EVIDENCE, report_maximum=2 << 20)
    return report


def arguments(argv):
    rt.require(len(argv) == len(set(x for x in argv if x.startswith('--'))) + len([x for x in argv if not x.startswith('--')]), 'Duplicate CLI controls forbidden')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--native', action='store_true'); parser.add_argument('--code'); parser.add_argument('--revision'); parser.add_argument('--deadline', type=float)
    parser.add_argument('--proof-bytes', type=int); parser.add_argument('--proof-sha256')
    parser.add_argument('--pose-replica-revision'); parser.add_argument('--pose-receipt-bytes', type=int); parser.add_argument('--pose-receipt-sha256')
    for name in ('host', 'native', 'proof'):
        parser.add_argument('--pose-'+name+'-bytes', type=int); parser.add_argument('--pose-'+name+'-sha256')
    a = parser.parse_args(argv)
    if a.native:
        rt.require(all(getattr(a, n) is None for n in ('pose_replica_revision', 'pose_receipt_bytes', 'pose_receipt_sha256',
            'pose_host_bytes', 'pose_host_sha256', 'pose_native_bytes', 'pose_native_sha256', 'pose_proof_bytes', 'pose_proof_sha256'))
            and re.fullmatch('[0-9a-f]{40}', str(a.revision)) and type(a.code) is str and type(a.deadline) is float and math.isfinite(a.deadline), 'Explicit native source/proof only')
        a.pin = dict(bytes=a.proof_bytes, sha256=a.proof_sha256)
    else:
        rt.require(a.code is a.revision is a.deadline is a.proof_bytes is a.proof_sha256 is None
            and re.fullmatch('[0-9a-f]{40}', str(a.pose_replica_revision)), 'Independent actual pose/import pins only')
        a.pin = dict(bytes=a.pose_receipt_bytes, sha256=a.pose_receipt_sha256)
        a.pose_pins = {filename: dict(bytes=getattr(a, 'pose_'+label+'_bytes'), sha256=getattr(a, 'pose_'+label+'_sha256'))
            for filename, label in (('report.json', 'host'), ('native.json', 'native'), ('proof.json', 'proof'))}
    for pin in [a.pin, *getattr(a, 'pose_pins', {}).values()]:
        rt.require(type(pin['bytes']) is int and 0 < pin['bytes'] <= 2 << 20 and re.fullmatch('[0-9a-f]{64}', str(pin['sha256'])), 'Exact independently supplied bytes/SHA required')
    return a


if __name__ == '__main__':
    a = arguments(sys.argv[1:])
    if a.native:
        handlers = {s: signal.signal(s, lambda *_: (_ for _ in ()).throw(TimeoutError())) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
        signal.setitimer(signal.ITIMER_REAL, max(.001, a.deadline-time.monotonic()))
        result = native(Path(a.code), a.revision, a.pin, a.deadline)
    else: result = dispatch(Path(os.environ.get('WR_CODE', '/invalid')), os.environ.get('WR_CODE_REVISION', ''), a.pose_replica_revision, a.pin, a.pose_pins)
    print(encode(dict(stage=result['stage'], status=result['status'])).decode(), end='')
    raise SystemExit(0 if result['status'] == 'pass' else 1)
