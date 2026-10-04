"""CPU-only DexYCB RGB/private-byte extraction; no network, models or label decoding."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import stat
import struct
import tarfile
import time

ROOT = Path('/srv/scenesmith/world-reward')
BASE = 'validation/dexycb_identity_v1'
INCOMING = Path('/srv/world-reward-data/dexycb_identity_download_v1')
EVIDENCE = 'vendor/research/dexycb_identity_v1'
JOB = 'run_dexycb_acquire'
BUDGET, CLEANUP_GRACE = 1800, 120
MEMBER_CAP, EXPANDED_CAP, RETAINED_CAP = 64 << 20, 120 << 30, 8 << 30
INDICES = [0, 16, 32, 48, 64, 80]
CAMERA = '836212060125'
SUBJECTS = ('20200709-subject-01', '20200813-subject-02')
EXPECTED_PROTOCOL = {
    'schema': 'world-reward-dexycb-identity-acquisition-v1', 'base': BASE,
    'scope': 'acquisition_plan_only_not_frozen_model_or_quality_protocol',
    'camera': CAMERA, 'sequence_count_per_subject': 100, 'sequence_lex_indices': INDICES,
    'subjects': {
        SUBJECTS[0]: {'role': 'blind_evaluation', 'fit_indices': [], 'decision_indices': []},
        SUBJECTS[1]: {'role': 'calibration', 'fit_indices': [0, 32, 64], 'decision_indices': [16, 48, 80]},
    },
    'archives': {
        SUBJECTS[0]: {'file': SUBJECTS[0] + '.tar.gz', 'bytes': 12412314463,
                     'url': 'https://drive.google.com/file/d/1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk'},
        SUBJECTS[1]: {'file': SUBJECTS[1] + '.tar.gz', 'bytes': 12004145048,
                     'url': 'https://drive.google.com/file/d/1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62'},
    },
    'primary_sources': {
        'publisher.html': {'url': 'https://dex-ycb.github.io/', 'bytes': 14026,
            'sha256': '09b12e4a37ccd0142e849101456a7389727b13a4d798434d0c35b686ccc9d008'},
        'dex_ycb.py': {'url': 'https://raw.githubusercontent.com/NVlabs/dex-ycb-toolkit/64551b001d360ad83bc383157a559ec248fb9100/dex_ycb_toolkit/dex_ycb.py',
            'bytes': 8713, 'sha256': 'f73074505bb822b01178dc7aae9778f5107efea479d43423f4fe2dc37224d8ad'},
    },
    'license': 'CC-BY-NC-4.0', 'archive_hash_basis': 'caller_measured_not_publisher_checksum',
    'archive_root': 'direct_subject_directory_only', 'all_original_frames': True,
    'timestamps_available': False, 'training_overlap_verified': False, 'challenge_overlap_verified': False,
    'budget_seconds': BUDGET, 'cleanup_grace_seconds': CLEANUP_GRACE,
    'member_cap_bytes': MEMBER_CAP, 'expanded_cap_bytes': EXPANDED_CAP,
    'retained_cap_bytes': RETAINED_CAP, 'download_performed': False,
}


def require(value, message):
    if not value: raise ValueError(message)


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON keys'); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def exact(value, expected):
    require(type(value) is type(expected), 'Protocol field type differs')
    if type(expected) is dict:
        require(set(value) == set(expected), 'Exact protocol keys required')
        for key in expected: exact(value[key], expected[key])
    elif type(expected) is list:
        require(len(value) == len(expected), 'Protocol list length differs')
        for a, b in zip(value, expected): exact(a, b)
    else: require(value == expected, 'Frozen acquisition protocol differs')


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and
            not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical nonsymlink path required')
    return path


def state(path):
    s = path.lstat()
    return (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_uid)


def identity(path, *, readonly=False, empty=False):
    path = canonical(path); before = state(path)
    require(stat.S_ISREG(before[2]) and before[3] == 1 and before[4] >= (0 if empty else 1)
            and (not readonly or not before[2] & 0o222), 'Nonempty regular source without aliases required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    require(state(path) == before, 'Artifact changed while hashing')
    return {'bytes': before[4], 'sha256': digest.hexdigest()}


def source_bindings(protocol_path, evidence):
    protocol = strict_json(protocol_path.read_bytes()); exact(protocol, EXPECTED_PROTOCOL)
    sources = {str(p): identity(p, readonly=True) for p in (Path(__file__), protocol_path)}
    revision = os.environ.get('WR_CODE_REVISION')
    if revision:
        code = canonical(protocol_path.parent.parent)
        require(code == ROOT/'jobs'/revision/JOB/'code' and Path(__file__) == code/'infra/dexycb_acquire.py',
                'Actual producer source path differs')
        require((code.parent/'revision').read_bytes() == (revision+'\n').encode()
                and re.fullmatch(b'[0-9a-f]{64}\n', (code.parent/'source-sha256').read_bytes()), 'Dispatch markers differ')
        for path in sorted(code.rglob('*')):
            canonical(path); require(not path.lstat().st_mode & 0o222, 'Readonly source closure required')
            if path.is_file(): sources[str(path)] = identity(path, readonly=True, empty=True)
        for name in ('revision', 'source-sha256'): sources[str(code.parent/name)] = identity(code.parent/name)
    for name, pin in protocol['primary_sources'].items():
        path = evidence / name; actual = identity(path, readonly=True)
        require(actual == {k: pin[k] for k in ('bytes', 'sha256')}, 'Primary source text differs')
        text = path.read_text()
        require(('DexYCB is licensed under' in text and 'by-nc/4.0' in text) if name == 'publisher.html'
                else all(x in text for x in ('ycb_grasp_ind', 'color_{:06d}.jpg', 'np.arange(meta[\'num_frames\'])')),
                'Primary license/format assertion missing')
        sources[str(path)] = actual
    return protocol, sources


def members(path, subject, audit):
    """Full gzip CRC verification; only metadata is interpreted in this iterator."""
    expanded = 0
    with gzip.open(path, 'rb') as compressed:
        class Limited:
            def read(self, count=-1):
                nonlocal expanded
                require(count >= 0, 'Unbounded decompression forbidden')
                block = compressed.read(count); expanded += len(block)
                require(expanded <= EXPANDED_CAP, 'Expanded archive cap exceeded')
                return block
        stream = Limited()
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            for entry in archive:
                raw = entry.name[:-1] if entry.isdir() and entry.name.endswith('/') else entry.name
                path_name = PurePosixPath(raw)
                require(raw and len(raw) <= 4096 and path_name.as_posix() == raw and not path_name.is_absolute()
                        and '..' not in path_name.parts and '\\' not in raw and not any(ord(c) < 32 or ord(c) == 127 for c in raw)
                        and path_name.parts[0] == subject, 'Unsafe path or unsupported archive root')
                require((entry.isfile() or entry.isdir()) and not entry.issym() and not entry.islnk()
                        and not entry.sparse and not any(k.startswith('GNU.sparse') for k in entry.pax_headers),
                        'Only ordinary files/directories, no links or sparse members')
                require(0 <= entry.size <= MEMBER_CAP and (not entry.isdir() or entry.size == 0), 'Tar member size cap/type')
                yield raw, entry, archive
            trailing = 0
            for block in iter(lambda: archive.fileobj.read(1 << 20), b''):
                require(not any(block), 'Nonzero bytes after tar terminator'); trailing += len(block)
            require(trailing >= 512, 'Two complete tar termination blocks required')
    audit['decompressed_bytes'] = expanded; audit['gzip_crc_verified'] = True


def inspect_archive(path, subject):
    names, parents_seen, sequences, audit = {}, set(), set(), {}
    total = 0
    for name, entry, _ in members(path, subject, audit):
        require(name not in names and len(names) < 1000000, 'Duplicate member or inventory cap')
        require(not any(str(parent) in names and names[str(parent)][1] for parent in PurePosixPath(name).parents),
                'File/directory ancestor collision')
        if entry.isfile(): require(name not in parents_seen, 'File/directory ancestor collision')
        parents_seen.update(str(parent) for parent in PurePosixPath(name).parents)
        names[name] = (entry.size, entry.isfile()); total += entry.size
        require(total <= EXPANDED_CAP, 'Expanded member-byte cap exceeded')
        parts = PurePosixPath(name).parts
        if len(parts) >= 2:
            require(re.fullmatch(r'[0-9]{8}_[0-9]{6}', parts[1]), 'Unexpected sequence name')
            sequences.add(parts[1])
    ordered = sorted(sequences); require(len(ordered) == 100, 'Exactly100 publisher sequences required')
    selected, wanted = [], {}
    for index in INDICES:
        sequence = ordered[index]; prefix = f'{subject}/{sequence}/{CAMERA}/'
        colors = {int(PurePosixPath(name).name[6:12]): name for name, (_, regular) in names.items()
                  if regular and name.startswith(prefix) and re.fullmatch(r'color_[0-9]{6}\.jpg', name[len(prefix):])}
        labels = {int(PurePosixPath(name).name[7:13]): name for name, (_, regular) in names.items()
                  if regular and name.startswith(prefix) and re.fullmatch(r'labels_[0-9]{6}\.npz', name[len(prefix):])}
        ids = sorted(colors); require(ids and ids == list(range(len(ids))) and sorted(labels) == ids,
                                      'Full matching contiguous original RGB/label member IDs required')
        meta = f'{subject}/{sequence}/meta.yml'
        require(meta in names and names[meta][1] and names[meta][0] > 0, 'Original meta.yml bytes required')
        role = 'blind_evaluation' if subject == SUBJECTS[0] else ('fit' if index in (0, 32, 64) else 'decision')
        selected.append({'subject': subject, 'sequence': sequence, 'sequence_lex_index': index,
                         'camera': CAMERA, 'frames': len(ids), 'partition': role})
        wanted[meta] = ('private', meta)
        for frame in ids:
            wanted[colors[frame]] = ('rgb', f'subject_{SUBJECTS.index(subject)+1:02d}_sequence_{index:03d}_frame_{frame:06d}.jpg')
            wanted[labels[frame]] = ('private', labels[frame])
    audit.update(member_count=len(names), expanded_member_bytes=total)
    return selected, wanted, audit


def jpeg_size(path):
    """Validate original JPEG grid from headers, without reencoding or pixel decoding."""
    with path.open('rb') as stream:
        require(stream.read(2) == b'\xff\xd8', 'Original RGB is not JPEG')
        for _ in range(4096):
            require(stream.read(1) == b'\xff', 'Malformed JPEG marker')
            marker = stream.read(1)
            while marker == b'\xff': marker = stream.read(1)
            require(marker and marker not in (b'\xda', b'\xd9', b'\x00'), 'Missing JPEG size header')
            if marker == b'\x01' or 0xd0 <= marker[0] <= 0xd7: continue
            raw = stream.read(2); require(len(raw) == 2, 'Truncated JPEG length')
            size = struct.unpack('>H', raw)[0]; require(size >= 2, 'Invalid JPEG length')
            if marker[0] in (0xc0, 0xc1, 0xc2):
                require(size == 17, 'Three-channel JPEG SOF segment length differs')
                header = stream.read(6); require(len(header) == 6, 'Truncated JPEG size')
                precision, height, width, channels = struct.unpack('>BHHB', header)
                require(precision == 8 and channels == 3, 'Original8-bit three-channel RGB JPEG required')
                return width, height
            stream.seek(size - 2, 1)
    raise ValueError('JPEG marker count exceeded')


def extract_archive(path, subject, wanted, output, owned, directories, remaining=RETAINED_CAP):
    identities, audit, retained = {}, {}, 0
    for name, entry, archive in members(path, subject, audit):
        if name not in wanted: continue
        kind, relative = wanted[name]; dest = output / ('inputs' if kind == 'rgb' else 'eval_private') / relative
        missing = [p for p in (dest.parent, *dest.parent.parents) if p != output and output in p.parents and not p.exists()]
        for parent in reversed(missing):
            parent.mkdir(mode=0o700 if kind == 'private' else 0o755); directories.append(parent)
        retained += entry.size; require(entry.size > 0 and retained <= remaining, 'Retained-byte cap exceeded')
        digest, count = hashlib.sha256(), 0
        try:
            with dest.open('xb') as target, archive.extractfile(entry) as source:
                for block in iter(lambda: source.read(1 << 20), b''):
                    target.write(block); digest.update(block); count += len(block)
                require(count == entry.size, 'Extracted byte count differs'); target.flush(); os.fsync(target.fileno())
            dest.chmod(0o444 if kind == 'rgb' else 0o400)
        finally:
            if dest.exists(): owned[dest] = state(dest)
        if kind == 'rgb': require(jpeg_size(dest) == (640, 480), 'Original camera RGB grid differs')
        identities[relative] = {'bytes': count, 'sha256': digest.hexdigest()}
    require(set(identities) == {v[1] for v in wanted.values()}, 'All selected original bytes required')
    return identities, retained


def acquire(root, measured_hashes, protocol_path, *, remove_owned_archives=False):
    started = time.perf_counter()
    root = canonical(root); protocol_path = canonical(protocol_path)
    protocol, before_sources = source_bindings(protocol_path, root / EVIDENCE)
    require(set(measured_hashes) == set(SUBJECTS) and all(type(v) is str and re.fullmatch(r'[0-9a-f]{64}', v)
            for v in measured_hashes.values()), 'Both caller-measured archive SHA256 values required')
    incoming = canonical(INCOMING)
    require({p.name for p in incoming.iterdir()} == {s+'.tar.gz' for s in SUBJECTS}, 'Exactly two original incoming archives')
    output = root / BASE; output.mkdir(mode=0o755, parents=True, exist_ok=False)
    output.chmod(0o755)
    (output / 'inputs').mkdir(mode=0o755); (output / 'inputs').chmod(0o755)
    (output / 'eval_private').mkdir(mode=0o700); (output / 'eval_private').chmod(0o700)
    report = {'stage': 'external_dexycb_identity_rgb_private_byte_acquisition', 'status': 'running', 'phase': 'archive_hash',
              'archive_sha_publisher_verified': False, 'archive_hash_basis': protocol['archive_hash_basis'],
              'annotation_values_parsed': False, 'inference_performed': False, 'gpu_used': False,
              'source_before': before_sources, 'disposable_archives_removed': False}
    owned, directories, proofs, archive_states = {}, [], {}, {}
    try:
        archives = {s: incoming / (s+'.tar.gz') for s in SUBJECTS}
        for subject, path in archives.items():
            require(state(path)[7] == os.getuid(), 'Incoming archives must be runtime-owned')
            archive_states[subject] = state(path)
            proofs[subject] = identity(path)
            require(proofs[subject] == {'bytes': protocol['archives'][subject]['bytes'], 'sha256': measured_hashes[subject]},
                    'Original supplied archive size/SHA differs')
        report['archive_proofs'] = proofs; report['phase'] = 'inventory'
        inspected = {s: inspect_archive(path, s) for s, path in archives.items()}
        require(sum(row[2]['decompressed_bytes'] for row in inspected.values()) <= EXPANDED_CAP, 'Combined expanded cap')
        report['phase'] = 'extract'; all_ids, sequences, retained = {}, [], 0
        for subject, path in archives.items():
            selected, wanted, audit = inspected[subject]
            ids, size = extract_archive(path, subject, wanted, output, owned, directories, RETAINED_CAP-retained); retained += size
            all_ids.update(ids); sequences.extend(selected)
        records = []
        for sequence in sequences:
            for frame in range(sequence['frames']):
                file = f"subject_{SUBJECTS.index(sequence['subject'])+1:02d}_sequence_{sequence['sequence_lex_index']:03d}_frame_{frame:06d}.jpg"
                records.append(dict(file=file, **all_ids[file], width=640, height=480, subject=sequence['subject'],
                                    sequence=sequence['sequence'], camera=CAMERA, frame_position=frame, source_frame_id=frame))
        manifest = {'schema': 'world-reward-dexycb-identity-rgb-v1', 'license': protocol['license'],
                    'sequences': sequences, 'images': records, 'timestamps_available': False,
                    'frame_count_evidence': 'all_RGB_and_label_member_headers_no_meta_values', 'source_archives': proofs,
                    'training_overlap_verified': False, 'challenge_overlap_verified': False}
        path = output / 'inputs/manifest.json'; path.write_text(json.dumps(manifest, sort_keys=True)+'\n'); path.chmod(0o444)
        owned[path] = state(path)
        report['phase'] = 'postverify'
        for dest, pin_state in owned.items(): require(state(dest) == pin_state, 'Owned extraction changed')
        for relative, expected in all_ids.items():
            dest = output / ('inputs' if relative.endswith('.jpg') else 'eval_private') / relative
            require(identity(dest, readonly=True) == expected, 'Retained-byte proof differs')
        require({s: identity(p) for s, p in archives.items()} == proofs and
                {s: state(p) for s, p in archives.items()} == archive_states, 'Original archives changed')
        after = source_bindings(protocol_path, root / EVIDENCE)[1]
        require(after == before_sources, 'Source/protocol evidence changed')
        if remove_owned_archives:
            for subject, path in archives.items():
                require(identity(path) == proofs[subject] and state(path) == archive_states[subject], 'Original archive changed before removal')
                path.unlink()
            report['disposable_archives_removed'] = True
        report.update(status='pass', phase='complete', source_after=after, source_rehashed_after=True,
                      public_manifest=identity(output / 'inputs/manifest.json', readonly=True),
                      retained_files=all_ids, frames=len(records), sequences=12, retained_bytes=retained,
                      archive_inventory={s: v[2] for s, v in inspected.items()}, private_values_interpreted=False)
    except BaseException as error:
        report.update(status='fail', error_type=type(error).__name__, error=str(error)[:1000])
        if signal.getsignal(signal.SIGALRM) != signal.SIG_DFL:
            signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Cleanup120s exceeded')))
            signal.alarm(CLEANUP_GRACE)
        for path, original in reversed(list(owned.items())):
            require(state(path) == original, 'Refuse cleanup of changed owned extraction'); path.unlink()
        for parent in reversed(directories): parent.rmdir()
        report['owned_partial_outputs_removed'] = True
        raise
    finally:
        try:
            after = source_bindings(protocol_path, root / EVIDENCE)[1]
            report['source_after'] = after; report['source_rehashed_after'] = after == before_sources
        except Exception as error:
            after = None; report['source_rehashed_after'] = False
            report['source_postcheck_error'] = type(error).__name__
        if after != before_sources: report.update(status='fail', error_type='ValueError', error='Source/protocol evidence changed')
        report['elapsed_seconds'] = time.perf_counter()-started
        report['producer_revision'] = os.environ.get('WR_CODE_REVISION')
        report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path = output/'report.json'
        with path.open('x') as stream:
            stream.write(json.dumps(report, sort_keys=True, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
        path.chmod(0o444)
        require(after == before_sources, 'Source/protocol evidence changed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject01-sha256', required=True); parser.add_argument('--subject02-sha256', required=True)
    parser.add_argument('--remove-owned-archives', action='store_true'); args = parser.parse_args()
    require(platform.system() == 'Linux', 'Heavy source archives stay on Azure Linux')
    code = canonical(Path(os.environ['WR_CODE'])); revision = os.environ['WR_CODE_REVISION']
    require(re.fullmatch(r'[0-9a-f]{40}', revision) and code == ROOT/'jobs'/revision/JOB/'code'
            and Path(__file__) == code/'infra/dexycb_acquire.py', 'Canonical immutable producer required')
    require((code.parent/'revision').read_bytes() == (revision+'\n').encode(), 'Producer marker differs')
    require(all(not p.lstat().st_mode & 0o222 for p in (code, *code.rglob('*'))), 'Readonly source closure required')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Acquisition1800s exceeded')))
    signal.alarm(BUDGET)
    try:
        report = acquire(ROOT, dict(zip(SUBJECTS, (args.subject01_sha256, args.subject02_sha256))),
                         code/'configs/dexycb_identity_protocol.json', remove_owned_archives=args.remove_owned_archives)
        print(json.dumps({k: report[k] for k in ('stage','status','frames','sequences','elapsed_seconds')}))
    finally: signal.alarm(0)


if __name__ == '__main__': main()
