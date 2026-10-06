"""Host-only, read-only authentication of the original complete48 saved join.

No arrays, RGB, roles or models are decoded. The new caller owns its deadline
and authenticates this seam again after its operation. Only the returned public
proof/rows/files are suitable for a numerical worker; lineage stays on the host.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import stat
import sys

import vcoco_full_interaction_join as join

rt, ROOT = join.rt, join.ROOT
JOIN_REV = '0907f3e3042d7a7c2ccafc9b4799f114226ca059'
DECLARATION = dict(files=396, entries=401,
    closure_sha256='0cde8fc590e1d2ae9974da034e89f6b34f967bc10d1ba71c0b94bcd4a2a0724a',
    source_XZ_sha256='1fdc65a28c3265dd2d3a643b8406d23e7b5104670aa7aefb47191de2a596106c')
POSE_REPLICA_REV = '1f1453daae8343dc8a077b918e93f7fe7c36e467'
POSE_IMPORT_PIN = dict(bytes=12515, sha256='d871126667422258066cb461b274d29fa36a513ee71c04b1c23852ab09b6b303')
POSE_PINS = {
    'report.json': dict(bytes=160633, sha256='f4fd034ade378261cc227539dd0f3e331b136a8762ad8f8f67d03d96e8ce3c26'),
    'native.json': dict(bytes=341744, sha256='d81a5d2db927609daedc1d2f9c97d88ab0fa1a5287ac3901086cb8735296a137'),
    'proof.json': dict(bytes=4221, sha256='45090e658b7980a8056022c3c472453a39e4f7d6cbca86a8530e971b6ab9d023')}
HELPER = 'infra/vcoco_full_interaction_inputs.py'
PUBLIC_SCHEMA = 'world_reward.vcoco_full_interaction_public_inputs.v1'


def normalized(value):
    return rt.strict(join.encode(value))


def _current(code, revision, entry, helpers):
    rt.require(type(helpers) is tuple and HELPER in helpers and set(join.helpers()) <= set(helpers),
               'Explicit complete new caller helper closure required')
    binding = rt.source(ROOT, code, revision, entry, helpers)
    for name in helpers:
        if name.endswith('.py'):
            key = Path(name).stem if name.startswith('infra/') else '.'.join(Path(name).with_suffix('').parts[1:])
            module = sys.modules.get(key)
            if module is not None:
                rt.require(Path(module.__file__).resolve() == code/name, 'Current imported helper origin differs')
    rt.require(Path(__file__).resolve() == code/HELPER
        and Path(join.__file__).resolve() == code/join.NATIVE_FILES[0], 'Actual new input/join helper origin required')
    states = {}
    for path in (code, *code.rglob('*'), code.parent/'revision', code.parent/'source-sha256'):
        s = rt.canonical(path).lstat()
        rt.require(s.st_uid == s.st_gid == 0 and (stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode) == 0o555
            or stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and stat.S_IMODE(s.st_mode) in (0o444, 0o555)),
            'Root-owned immutable complete caller source required')
        states[str(path)] = join.snapshot(path)
    return dict(binding=binding, states=states)


def _original(code, current):
    old = ROOT/'jobs'/JOIN_REV/join.ENTRY/'code'
    binding = rt.source(ROOT, old, JOIN_REV, join.ENTRY, join.helpers())
    rt.require(binding['entries'] == DECLARATION['entries'] and binding['closure_sha256'] == DECLARATION['closure_sha256']
        and sum(p.is_file() for p in old.rglob('*')) == DECLARATION['files']
        and (old.parent/'source-sha256').read_bytes() == (DECLARATION['source_XZ_sha256']+'\n').encode()
        and all(current['helpers'][n] == pin for n, pin in binding['helpers'].items())
        and binding['helpers'][join.NATIVE_FILES[2]] == join.NUMERICAL_PIN,
        'Whole original join Git/XZ and unchanged helper bytes required')
    return dict(binding=binding, states=join.modes(old, join.EXECUTABLES))


def _inputs(code, current, original, deadline, checkpoint):
    endpoint, hoi, pose = join.host_modules()
    projection, _, _, e = endpoint.sender_inputs(code, current['binding'], deadline); checkpoint()
    h = join.hoi_inputs(code, current['binding'], projection, deadline); checkpoint()
    p = pose.authenticate_receiver(code, POSE_REPLICA_REV, POSE_IMPORT_PIN); checkpoint()
    rt.require(p['pose_receipt_pins'] == POSE_PINS and p['import_identity'] == POSE_IMPORT_PIN
        and p['import_source']['producer_revision'] == POSE_REPLICA_REV
        and p['original_pose_revision'] == pose.POSE_REV and len(p['rows']) == 48
        and p['sender_source_live_verified'] is p['sender_runtime_live_verified'] is False,
        'Actual original pose/import pins required; sender not falsely live')
    v = rt.pinned(Path(p['projection_path']), p['projection_identity'], 2 << 20)
    rt.require(set(v) == {'schema', 'rows'} and v['schema'] == pose.PUBLIC_SCHEMA and v['rows'] == p['rows'],
               'Only original public pose rows required')
    rows, files = [], {}
    for er, pr, hr in zip(projection['banks'], p['rows'], h['rows']):
        paths = (endpoint.endpoint.OUTPUT/er['file'], pose.DEST/'banks'/pr['file'], hoi.OUTPUT/hr['file'])
        for path, row in zip(paths, (er, pr, hr)):
            rt.require(rt.identity(path, join.MAXIMUM) == row['identity'], 'Every original raw NPZ SHA required')
            files[str(path)] = row['identity']
        rows.append(dict(endpoint=er, pose=pr, hoi=hr, paths=list(map(str, paths)))); checkpoint()
    proof = dict(schema=join.SCHEMA, producer_revision=JOIN_REV, image_id=join.IMAGE, source={
        'helpers': {n: original['binding']['helpers'][n] for n in join.NATIVE_FILES},
        'markers': original['binding']['markers']}, records=rows, files=files)
    join.validate_population(proof)
    return dict(current_source=original, endpoint=e, hoi=h, pose=p), proof


def _summaries(rows, proof):
    fields = (join.numerical.ENDPOINT_FIELDS, join.numerical.POSE_FIELDS, join.numerical.HOI_FIELDS)
    for row, record in zip(rows, proof['records']):
        raw = []
        for key, names in zip(('endpoint', 'pose', 'hoi'), fields):
            arrays = record[key]['arrays']
            rt.require(set(arrays) == set(names), 'Full original17/11/19 metadata required')
            raw.append({n: arrays[n] for n in names})
        rt.require(hashlib.sha256(join.encode(raw)).hexdigest() == row['raw_bank_fingerprint'],
                   'Original47 metadata fingerprint differs')
        digest = hashlib.sha256(); names = join.expected_evidence_metadata(record)
        for prefix in ('base__', 'local__', 'bridge__'):
            for name in names:
                if name.startswith(prefix): digest.update(join.encode([name[len(prefix):], row['arrays'][name]]))
        for prefix in ('base', 'local', 'bridge'):
            digest.update(join.encode([row['arrays'][prefix+'_features'], row['arrays'][prefix+'_supported']]))
        rt.require(digest.hexdigest() == row['evidence_fingerprint'], 'Saved46 derived evidence fingerprint differs')
        refs = row['source_observation_references']
        rt.require(type(refs) is list and len(refs) == 3
            and [r[0] for r in refs] == ['world_reward.person_pose_observations.v1',
                'world_reward.generic_object_observations.v1', 'world_reward.hoi_detr_observations.v1']
            and all(type(r) is list and len(r) == 2 and re.fullmatch('[0-9a-f]{64}', str(r[1])) for r in refs),
            'Original numerical observation references required; not source authentication')


def _outputs(host, native, pins, checkpoint):
    expected = {'report.json', 'native.json', 'proof.json', '.container.cid', *(f'image_{i:06d}.npz' for i in range(48))}
    directory = rt.canonical(join.OUTPUT); s = directory.lstat()
    rt.require(stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode) == 0o500 and s.st_uid == s.st_gid == 0
        and {p.name for p in directory.iterdir()} == expected, 'Exact sealed original52-leaf namespace required')
    files, states = {}, {str(directory): join.snapshot(directory)}
    for name in sorted(expected):
        path = directory/name; s = rt.canonical(path).lstat()
        maximum = join.MAX_EVIDENCE if name.startswith('image_') else (65 if name == '.container.cid' else 2 << 20)
        rt.require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and stat.S_IMODE(s.st_mode) == 0o400
            and s.st_uid == s.st_gid == 0, 'Original root400 single-link leaf required')
        files[str(path)] = rt.identity(path, maximum); states[str(path)] = join.snapshot(path); checkpoint()
    rt.require(set(host['saved_outputs']) == expected-{'report.json'}
        and all(files[str(directory/n)] == pin for n, pin in host['saved_outputs'].items())
        and all(files[str(directory/n)] == pin for n, pin in pins.items())
        and all(files[str(directory/r['file'])] == r['identity'] for r in native['images'])
        and sum(files[str(directory/n)]['bytes'] for n in expected if n.startswith('image_')) <= join.MAX_TOTAL
        and sum(p['bytes'] for n, p in files.items() if Path(n).name != 'report.json') <= join.MAX_TOTAL+join.MAX_CONTROL_TOTAL,
        'Complete original output ledger/capacity differs')
    return files, states


def authenticate_saved_join(code, new_revision, new_entry, new_helpers, pins, checkpoint, *, deadline):
    """Return public full47 metadata/48 evidence rows, plus HOST-ONLY lineage.

    ``pins`` is the independently supplied report.json/native.json/proof.json
    map, never learned from the saved producer alone. No old producer executes.
    This authenticates bytes and saved completion, not numerical correctness,
    positive references, FIT membership, licenses, ownership or model quality.
    """
    def check(): join.check(deadline); checkpoint(); join.check(deadline)
    check()
    rt.require(type(pins) is dict and set(pins) == {'report.json', 'native.json', 'proof.json'}, 'All three caller pins required')
    pins = deepcopy(pins)
    for pin in pins.values():
        rt.require(type(pin) is dict and set(pin) == {'bytes', 'sha256'} and type(pin['bytes']) is int
            and 0 < pin['bytes'] <= 2 << 20 and type(pin['sha256']) is str
            and re.fullmatch('[0-9a-f]{64}', pin['sha256']), 'Independent bounded receipt byte/SHA required')

    def capture():
        check(); current = _current(code, new_revision, new_entry, new_helpers)
        original = _original(code, current['binding']); check()
        values = {n: rt.pinned(join.OUTPUT/n, pin, 2 << 20) for n, pin in pins.items()}
        host, native, saved = (values[n] for n in ('report.json', 'native.json', 'proof.json'))
        rt.require(host['schema'] == join.SCHEMA and host['stage'] == 'full48_interaction_join_host'
            and host['producer_revision'] == JOIN_REV and host['status'] == 'pass' and host['phase'] == 'complete'
            and host['source_binding'] == original['binding'] and host['native_identity'] == pins['native.json']
            and host['native_exit_status'] == 0 and host['image_id'] == join.IMAGE and host['models_loaded'] == 0
            and host['images'] == native['images'] and host['pose_receipt_pins'] == POSE_PINS
            and host['pose_replica_identity'] == POSE_IMPORT_PIN
            and host['outputs_sealed'] is host['owned_cleanup_verified'] is host['source_inputs_image_rehashed_after']
                is host['output_capacity_gate_passed'] is True and all(host[n] is False for n in join.FLAGS)
            and not any(n in host for n in ('error_type', 'post_error_type', 'cleanup_error_type', 'publication_failed',
                'failure_stage', 'publication_error_type', 'output_capacity_error_type')),
            'Actual complete sealed original join PASS required')
        before, proof = _inputs(code, current, original, deadline, check)
        rt.require(normalized(before) == host['input_proof'] and normalized(proof) == saved,
                   'Exact original input proof after JSON roundtrip required')
        join.validate_native(native, saved, JOIN_REV, pins['proof.json']); _summaries(native['images'], saved)
        rt.require(not any(n in native for n in ('error_type', 'post_error_type', 'publication_failed', 'publication_error_type')),
                   'Native failure cannot qualify saved completion')
        files, states = _outputs(host, native, pins, check)
        join.absent((join.OUTPUT/'.container.cid').read_bytes(), 'world-reward-vcoco-full-interaction-join-'+JOIN_REV[:12], deadline)
        runtime = join.image(deadline); check()
        return dict(current_source=current, original_input_proof=before, proof=saved, rows=native['images'],
                    files=files, states=states, runtime=runtime)

    before = capture(); after = capture()
    rt.require(normalized(before) == normalized(after), 'Caller/original sources, inputs, outputs or image changed')
    check()
    return dict(schema=PUBLIC_SCHEMA, public_proof=deepcopy(before['proof']), rows=deepcopy(before['rows']),
        evidence_files={str(join.OUTPUT/r['file']): r['identity'] for r in before['rows']},
        output_files=deepcopy(before['files']), output_states=deepcopy(before['states']),
        host_lineage=dict(current_source=before['current_source'], original_input_proof=before['original_input_proof'],
            join_receipt_pins=pins, join_declaration=deepcopy(DECLARATION), runtime=before['runtime']),
        source_inputs_outputs_image_rehashed_after=True, NumPy_imported=False, RGB_decoded=False,
        reference_values_read=False, models_loaded=0, selection_performed=False, FIT_performed=False,
        sender_source_live_verified=False, sender_runtime_live_verified=False, ownership_verified=False, quality_verified=False)
