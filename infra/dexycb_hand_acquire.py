"""Explicit fresh DexYCB byte profiles; no annotation values or models decoded."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import pwd
import re
import signal
import stat
import threading
import time
import urllib.request

import dexycb_acquire as dex
import dexycb_download as download
import mediapipe_hands_acquire as lease

ROOT = dex.ROOT
BASE = 'validation/dexycb_hand_v1'
INCOMING = Path('/srv/world-reward-data/dexycb_hand_v1')
JOB = 'run_dexycb_hand_acquire'
SUBJECT, CAMERA, INDICES = '20200820-subject-03', '836212060125', [4, 39, 74]
DOWNLOAD_BUDGET, SCAN_BUDGET, CLEANUP_GRACE = 7200, 1800, 120
PROTOCOL_V1 = 'configs/dexycb_hand_protocol_v1.json'
PROTOCOL_V2 = 'configs/dexycb_hand_protocol_v2.json'
PROTOCOL_V3 = 'configs/dexycb_hand_protocol_v3.json'
HELPER_PINS = {
    'dexycb_acquire.py': {'bytes': 21916, 'sha256': '6078c54a862f98ee4aa9790a39387e2e89922789fce5ca4ee63edbc075834554'},
    'dexycb_download.py': {'bytes': 10680, 'sha256': '8ead4771dd49d6a0f0332c840bd0fa68aef00d6588fc0d41a3c12e433fe47591'},
    'mediapipe_hands_acquire.py': {'bytes': 21381, 'sha256': '8293510c02777cd5b844865285736ff1a653631a8803849a9047ee9030196570'},
}
EXPECTED_PROTOCOL = {
    'schema': 'world-reward-dexycb-hand-acquisition-v1', 'base': BASE,
    'subject': SUBJECT, 'camera': CAMERA, 'sequence_count': 100, 'sequence_lex_indices': INDICES,
    'archive': {'file': SUBJECT+'.tar.gz', 'bytes': 12197037343,
                'url': 'https://drive.google.com/file/d/1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq'},
    'primary_sources': dex.EXPECTED_PROTOCOL['primary_sources'], 'helper_pins': HELPER_PINS,
    'archive_root': 'direct_subject_directory_only', 'license': 'CC-BY-NC-4.0',
    'archive_hash_basis': 'caller_measured_not_publisher_checksum', 'all_original_frames': True,
    'original_rgb_size': [640, 480], 'private_retention': 'matching_opaque_labels_npz_only',
    'timestamps_available': False, 'challenge_overlap_verified': False, 'training_overlap_verified': False,
    'download_budget_seconds': DOWNLOAD_BUDGET, 'scan_extract_budget_seconds': SCAN_BUDGET,
    'cleanup_grace_seconds': CLEANUP_GRACE, 'member_cap_bytes': dex.MEMBER_CAP,
    'expanded_cap_bytes': dex.EXPANDED_CAP, 'retained_cap_bytes': dex.RETAINED_CAP,
}


PROFILE_ARCHIVES = {
    PROTOCOL_V2: ('v2', '20200903-subject-04', 12792618020, '14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA'),
    PROTOCOL_V3: ('v3', '20200908-subject-05', 12815420651, '1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS'),
}


def expected_protocol(protocol_path=PROTOCOL_V1):
    """Frozen whole-cohort opt-ins, never arbitrary paths or source overrides."""
    if protocol_path == PROTOCOL_V1: return EXPECTED_PROTOCOL
    dex.require(protocol_path in PROFILE_ARCHIVES, 'Unknown immutable acquisition profile')
    version, subject, size, file_id = PROFILE_ARCHIVES[protocol_path]
    return {**EXPECTED_PROTOCOL, 'schema': 'world-reward-dexycb-hand-acquisition-'+version,
        'base': 'validation/dexycb_hand_'+version, 'subject': subject,
        'archive': {'file': subject+'.tar.gz', 'bytes': size,
                    'url': 'https://drive.google.com/file/d/'+file_id}}


def profile_paths(root, protocol_path=PROTOCOL_V1):
    protocol = expected_protocol(protocol_path)
    return [root/protocol['base'], INCOMING if protocol_path == PROTOCOL_V1
            else Path('/srv/world-reward-data')/Path(protocol['base']).name]


def source_binding(root, code, revision, protocol_path=PROTOCOL_V1):
    expected = expected_protocol(protocol_path)
    code = dex.canonical(code)
    dex.require(re.fullmatch('[0-9a-f]{40}', revision) and code == root/'jobs'/revision/JOB/'code'
                and Path(__file__) == code/'infra/dexycb_hand_acquire.py', 'Canonical source-bound producer required')
    config = code/protocol_path
    protocol = dex.strict_json(config.read_bytes()); dex.exact(protocol, expected)
    closure = {}
    for path in (code, *sorted(code.rglob('*'))):
        dex.canonical(path); dex.require(not path.lstat().st_mode & 0o222, 'Readonly full source closure required')
        closure[str(path.relative_to(code))] = {'directory': True} if path.is_dir() else dex.identity(path, readonly=True, empty=True)
    required = ['infra/dexycb_hand_acquire.py', 'infra/run_dexycb_hand_acquire.sh',
                protocol_path, *('infra/'+name for name in HELPER_PINS)]
    dex.require(all(name in closure and 'sha256' in closure[name] for name in required), 'Incomplete own source closure')
    for module in (dex, download, lease):
        path = Path(module.__file__)
        dex.require(path == code/'infra'/path.name and dex.identity(path, readonly=True) == HELPER_PINS[path.name],
                    'Frozen reusable helper differs')
    markers = {name: dex.identity(code.parent/name) for name in ('revision', 'source-sha256')}
    dex.require((code.parent/'revision').read_bytes() == (revision+'\n').encode()
                and re.fullmatch(b'[0-9a-f]{64}\n', (code.parent/'source-sha256').read_bytes()), 'Dispatch markers differ')
    primary = {}
    for name, pin in protocol['primary_sources'].items():
        path = root/dex.EVIDENCE/name; actual = dex.identity(path, readonly=True)
        dex.require(actual == {k: pin[k] for k in ('bytes', 'sha256')}, 'Primary publisher/format differs')
        text = path.read_text()
        dex.require(('DexYCB is licensed under' in text and 'by-nc/4.0' in text) if name == 'publisher.html'
                    else all(x in text for x in ('color_{:06d}.jpg', "np.arange(meta['num_frames'])")),
                    'Primary license/format assertion absent')
        primary[name] = actual
    return {'producer_revision': revision, 'protocol_file': protocol_path, 'profile': protocol['base'],
            'protocol': dex.identity(config, readonly=True),
            'markers': markers, 'helpers': HELPER_PINS, 'primary_sources': primary,
            'closure_sha256': hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest()}


def check(deadline):
    if time.monotonic() > deadline: raise TimeoutError('Acquisition phase budget exceeded')


def inspect_archive(path, deadline, protocol=None):
    """Whole header inventory before selection; label member bytes stay opaque."""
    protocol = protocol if protocol is not None else EXPECTED_PROTOCOL
    subject, camera = protocol['subject'], protocol['camera']
    names, parents, sequences, audit = {}, set(), set(), {}
    for name, entry, _ in dex.members(path, subject, audit):
        check(deadline); pure = PurePosixPath(name)
        dex.require(name not in names and len(names) < 1000000, 'Duplicate/inventory cap')
        dex.require(not any(str(p) in names and names[str(p)][1] for p in pure.parents)
                    and (not entry.isfile() or name not in parents), 'Ancestor file collision')
        parents.update(str(p) for p in pure.parents); names[name] = (entry.size, entry.isfile())
        if len(pure.parts) >= 2:
            dex.require(re.fullmatch('[0-9]{8}_[0-9]{6}', pure.parts[1]), 'Publisher sequence format differs')
            sequences.add(pure.parts[1])
    ordered = sorted(sequences); dex.require(len(ordered) == 100, 'Exactly100 lexical publisher sequences required')
    selected, wanted = [], {}
    for index in protocol['sequence_lex_indices']:
        sequence = ordered[index]; prefix = f'{subject}/{sequence}/{camera}/'
        matching = lambda regex: {int(PurePosixPath(n).stem.split('_')[-1]): n for n, (_, regular) in names.items()
            if regular and n.startswith(prefix) and re.fullmatch(regex, n[len(prefix):])}
        colors, labels = matching(r'color_[0-9]{6}\.jpg'), matching(r'labels_[0-9]{6}\.npz')
        ids = sorted(colors)
        dex.require(ids and ids == list(range(len(ids))) and sorted(labels) == ids, 'Full matching original RGB/label IDs required')
        selected.append(dict(subject=subject, sequence=sequence, sequence_lex_index=index, camera=camera, frames=len(ids)))
        for frame in ids:
            wanted[colors[frame]] = ('rgb', f'sequence_{index:03d}_frame_{frame:06d}.jpg')
            wanted[labels[frame]] = ('private', labels[frame])
    audit['member_count'] = len(names); check(deadline)
    return selected, wanted, audit


def write_report(path, report):
    with path.open('x') as output:
        json.dump(report, output, sort_keys=True, allow_nan=False); output.write('\n')
        output.flush(); os.fsync(output.fileno()); os.fchmod(output.fileno(), 0o444)


def acquire(root, code, revision, namespace_lease, *, opener=None, watchdog=False, protocol_path=PROTOCOL_V1):
    protocol = expected_protocol(protocol_path)
    subject, camera = protocol['subject'], protocol['camera']
    started = time.monotonic(); before = source_binding(root, code, revision, protocol_path)
    folders = profile_paths(root, protocol_path)
    lease.validate_namespace_lease(namespace_lease, folders, before['closure_sha256'])
    output, incoming = folders
    report = dict(stage='external_dexycb_hand_rgb_private_byte_acquisition', status='fail', phase='download',
        producer_revision=revision, source_before=before, protocol_file=protocol_path,
        acquisition_profile=protocol['base'], archive_sha_publisher_verified=False,
        annotation_values_parsed=False, inference_performed=False, gpu_used=False,
        training_overlap_verified=False, challenge_overlap_verified=False,
        download_budget_seconds=DOWNLOAD_BUDGET, scan_extract_budget_seconds=SCAN_BUDGET)
    owned, directories, partials, proof, archive_state, failure = {}, [], [], None, None, None
    path = incoming/protocol['archive']['file']; scan_started = None
    try:
        if watchdog: signal.alarm(DOWNLOAD_BUDGET)
        opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), download.PublicRedirect())
        row = protocol['archive']; progress = {row['file']: {'bytes_received': 0}}
        downloaded = download.fetch(row, incoming, opener, started+DOWNLOAD_BUDGET,
                                    threading.Event(), partials, progress)
        check(started+DOWNLOAD_BUDGET); report['download'] = downloaded
        proof = {k: downloaded[k] for k in ('bytes', 'sha256')}; archive_state = dex.state(path)
        report['download_elapsed_seconds'] = time.monotonic()-started
        scan_started = time.monotonic(); deadline = scan_started+SCAN_BUDGET
        if watchdog: signal.alarm(SCAN_BUDGET)
        report['phase'] = 'inventory'; selected, wanted, audit = inspect_archive(path, deadline, protocol)
        report['selected_sequences'] = selected; report['archive_inventory'] = audit
        report['phase'] = 'extract'
        (output/'inputs').mkdir(mode=0o755); (output/'eval_private').mkdir(mode=0o700)
        identities, retained = dex.extract_archive(path, subject, wanted, output, owned, directories)
        check(deadline)
        records = [dict(file=f"sequence_{s['sequence_lex_index']:03d}_frame_{i:06d}.jpg",
            **identities[f"sequence_{s['sequence_lex_index']:03d}_frame_{i:06d}.jpg"], width=640, height=480,
            sequence=s['sequence'], sequence_lex_index=s['sequence_lex_index'], camera=camera,
            frame_position=i, source_frame_id=i) for s in selected for i in range(s['frames'])]
        manifest = dict(schema='world-reward-dexycb-hand-rgb-v1', subject=subject, sequences=selected,
            images=records, source_archive=proof, license='CC-BY-NC-4.0', timestamps_available=False,
            training_overlap_verified=False, challenge_overlap_verified=False)
        manifest_path = output/'inputs/manifest.json'; write_report(manifest_path, manifest)
        owned[manifest_path] = dex.state(manifest_path)
        report['phase'] = 'postverify'
        for dest, original in owned.items(): dex.require(dex.state(dest) == original, 'Owned output changed')
        for name, (_, relative) in wanted.items():
            dest = output/('inputs' if relative.endswith('.jpg') else 'eval_private')/relative
            dex.require(dex.identity(dest, readonly=True) == identities[relative], 'Retained original bytes differ')
        dex.require(dex.identity(path, readonly=True) == proof and dex.state(path) == archive_state, 'Original archive changed')
        dex.require(source_binding(root, code, revision, protocol_path) == before, 'Source changed before archive cleanup')
        check(deadline)
        path.unlink(); report['disposable_archive_removed'] = True
        report.update(status='pass', phase='complete', public_manifest=dex.identity(manifest_path, readonly=True),
                      retained_files=identities, retained_bytes=retained, frames=len(records), sequences=3)
    except BaseException as error:
        failure = error; report.update(status='fail', error_type=type(error).__name__)
    finally:
        if watchdog: signal.alarm(CLEANUP_GRACE)
        cleanup_started = time.monotonic()
        try:
            for part, inode in partials:
                if part.exists():
                    dex.require(not part.is_symlink() and dex.state(part)[:2] == inode and part.lstat().st_uid == os.getuid(),
                                'Refuse cleanup of changed download partial')
                    part.unlink()
            if failure:
                for dest, original in reversed(list(owned.items())):
                    dex.require(dex.state(dest) == original, 'Refuse cleanup of changed output'); dest.unlink()
                for directory in reversed(directories): directory.rmdir()
                report['owned_partial_outputs_removed'] = True
                if proof is not None and path.exists():
                    dex.require(dex.state(path) == archive_state and dex.identity(path, readonly=True) == proof,
                                'Refuse cleanup of changed owned archive')
                    path.unlink(); report['disposable_archive_removed'] = True
        except BaseException as cleanup_error:
            failure = failure or cleanup_error; report['status'] = 'fail'
            report['cleanup_error_type'] = type(cleanup_error).__name__
        try: report['source_rehashed_after'] = source_binding(root, code, revision, protocol_path) == before
        except Exception: report['source_rehashed_after'] = False
        report['elapsed_seconds'] = time.monotonic()-started
        if scan_started is not None: report['scan_extract_elapsed_seconds'] = time.monotonic()-scan_started
        if not report['source_rehashed_after'] or (scan_started is not None and not failure
                and time.monotonic()-scan_started > SCAN_BUDGET):
            report['status'] = 'fail'; failure = failure or ValueError('Source/budget postcheck failed')
        report['cleanup_elapsed_seconds'] = time.monotonic()-cleanup_started
        write_report(output/'report.json', report)
        if report['status'] == 'pass': output.chmod(0o755)
        if watchdog: signal.alarm(0)
    if failure: raise ValueError('DexYCB acquisition failed; inspect sealed receipt') from None
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--protocol', choices=(PROTOCOL_V1, PROTOCOL_V2, PROTOCOL_V3), default=PROTOCOL_V1)
    import sys
    supplied = list(sys.argv[1:] if argv is None else argv)
    if sum(value == '--protocol' or value.startswith('--protocol=') for value in supplied) > 1:
        parser.error('Protocol may be supplied only once')
    args = parser.parse_args(argv)
    dex.require(platform.system() == 'Linux' and os.environ.get('WR_ROOT') == str(ROOT)
                and os.getuid() == pwd.getpwnam('scenesmith').pw_uid, 'Azure CPU scenesmith required')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Acquisition phase timeout')))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('Acquisition terminated')))
    result = acquire(ROOT, Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'],
                     dex.strict_json(os.environ['WR_NAMESPACE_LEASE'].encode()), watchdog=True, protocol_path=args.protocol)
    print(json.dumps({k: result[k] for k in ('stage', 'status', 'frames', 'elapsed_seconds')}))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(json.dumps(dict(stage='external_dexycb_hand_rgb_private_byte_acquisition',
                              status='fail', error_type=type(error).__name__)))
        raise SystemExit(1) from None
