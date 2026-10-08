"""Bounded stdlib publication of fresh full4D producer receipts.

No model/data decoding or historical adoption is performed. Native producers
validate their numeric payloads; this host layer binds their PASS receipts and
source ledgers, seals only new experiment files, and publishes independent pins.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

import cari_clip_inputs as inputs
from world_reward.artifact_paths import episode_output, output_prefix, pin_path

ROOT = Path('/srv/scenesmith/world-reward')
MAX_BYTES = 2 << 30
REPORT_BYTES = 4 << 20
SURFACE_FILES = {'report.json', 'native.json', 'object_fixed_canonical.glb',
                 'geometry.npz', 'candidate_geometry.npz', 'mapping.json'}
SURFACE_AUXILIARY = {'.container.cid', 'native.log'}
SHARED = {
    'prepare': ('cari_shared_prepare.py', 'world_reward_native_cari_shared_initializer_full_video',
                {'report.json', 'shared_initializer.pkl', 'direct_parameters.npz', 'target.npy'}),
    'forward': ('cari_full_forward.py', 'world_reward_native_cari_shared_full_video_forward',
                {'report.json', 'coconet.pth'}),
    'refined': ('cari_full_refine.py', 'world_reward_native_cari_shared_full_video_refinement',
                {'report.json', 'refined.pth'}),
    'export': ('cari_full_export.py', 'world_reward_native_cari_shared_full_video_direct_export',
               {'report.json', 'trajectory.npz', 'native_parameters.npz', 'target.npy', 'object_aligned.glb'}),
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and
            not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical path without aliases required')
    return path


def _same(before, after):
    return all(getattr(before, name) == getattr(after, name) for name in
               ('st_dev', 'st_ino', 'st_size', 'st_mode', 'st_nlink', 'st_uid',
                'st_gid', 'st_mtime_ns', 'st_ctime_ns'))


def identity(path, *, readonly=True, maximum=MAX_BYTES):
    path = canonical(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= maximum
            and (not readonly or not before.st_mode & 0o222), 'Bounded positive single-link readonly file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        require(_same(before, os.fstat(stream.fileno())), 'File inode changed before hashing')
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
        require(_same(before, os.fstat(stream.fileno())), 'File changed while hashing')
    require(_same(before, path.lstat()), 'File changed after hashing')
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def _strict(raw):
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, 'Duplicate JSON key forbidden')
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON forbidden')))


def _json(path, *, readonly=True):
    before = identity(path, readonly=readonly, maximum=REPORT_BYTES)
    value = _strict(Path(path).read_bytes())
    require(identity(path, readonly=readonly, maximum=REPORT_BYTES) == before, 'Receipt changed during interpretation')
    require(type(value) is dict, 'Object receipt required')
    return value, before


def _context(root, code, revision, episode):
    root, code = canonical(root), canonical(code)
    require(root == ROOT and type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision),
            'Fixed Azure runtime and immutable revision required')
    require(code == root / 'jobs' / revision / 'run_full4d_sample' / 'code', 'Actual full4D dispatcher source required')
    require(output_prefix() == f'experiments/full4d-v1-{revision}/outputs', 'Exact fresh revision-bound output namespace required')
    base = episode_output(root, episode)
    require(stat.S_ISDIR(code.lstat().st_mode) and not code.stat().st_mode & 0o222,
            'Readonly original source required')
    require((code.parent / 'revision').read_bytes() == (revision + '\n').encode() and
            re.fullmatch(b'[0-9a-f]{64}\n', (code.parent / 'source-sha256').read_bytes()),
            'Actual dispatcher markers required')
    return root, code, base


def _no_oracle(report):
    require(report.get('ground_truth_used') is False and report.get('hand_labeled_test') is False
            and report.get('oracle_modes') == [], 'Explicit no-GT/no-hand-label/no-oracle receipt required')
    require(all(report.get(key, False) is False for key in ('ground_truth_read', 'private_truth_read')),
            'Private truth access forbidden')


def _producer(report, code, revision, episode, stage, script, *, total=None):
    require(report.get('status') == 'pass' and report.get('stage') == stage
            and type(report.get('episode_index')) is int and report['episode_index'] == episode
            and report.get('producer_revision') == revision
            and report.get('script_sha256') == identity(code / 'infra' / script)['sha256'],
            'Actual new passing producer/source receipt required')
    require(report.get('input_track') == 'track_1', 'Track1-only producer required')
    _no_oracle(report)
    if total is not None:
        require(type(report.get('frames')) is int and report['frames'] == total,
                'Full original timeline producer required')


def _helpers(report, code):
    rows = report.get('source_helpers')
    require(type(rows) is dict and 0 < len(rows) <= 128, 'Bounded source helper ledger required')
    for name, expected in rows.items():
        require(type(name) is str and re.fullmatch(r'(infra|src)/[A-Za-z0-9_./-]+', name)
                and '..' not in Path(name).parts, 'Explicit own source helper path required')
        require(identity(code / name, maximum=2 << 20) == expected, 'Actual readonly source helper differs')


def _inventory(directory, expected):
    directory = canonical(directory)
    require(directory.is_dir() and {p.name for p in directory.iterdir()} == set(expected),
            'Exclusive exact producer inventory required')
    return {name: identity(directory / name, readonly=False) for name in sorted(expected)}


def _seal(path, base, expected):
    path, base = canonical(path), canonical(base)
    require(path.is_relative_to(base) and path != base, 'Only fresh experiment output files may be sealed')
    require(identity(path, readonly=False) == expected, 'Output changed before sealing')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = path.lstat()
        require(_same(before, os.fstat(descriptor)) and stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                'Owned original output inode required')
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(identity(path) == expected, 'Output changed while sealing')


def _write(code, episode, role, value):
    path = canonical(pin_path(code, episode, role))
    require(path.parent.is_dir(), 'Dispatcher must exclusively create its separate pin directory')
    raw = (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode()
    require(0 < len(raw) <= REPORT_BYTES, 'Bounded pin document required')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        with os.fdopen(descriptor, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(identity(path) == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
            'Published independent pin bytes differ')
    return value


def _literal_tuple(code, relative, name):
    source = canonical(code / relative)
    before = identity(source, maximum=2 << 20)
    tree = ast.parse(source.read_bytes(), filename=str(source))
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    require(len(values) == 1, 'Exactly one source-bound literal helper ledger required')
    value = ast.literal_eval(values[0])
    require(type(value) is tuple and all(type(row) is str for row in value)
            and identity(source, maximum=2 << 20) == before, 'Literal readonly helper ledger differs')
    return value


def surface_pin(root, code, revision, episode):
    root, code, base = _context(root, code, revision, episode)
    directory = base / ('object_budget_surface_' + revision)
    local = _inventory(directory, SURFACE_FILES | SURFACE_AUXILIARY)
    require(stat.S_IMODE(directory.stat().st_mode) == 0o555,
            'Original surface host must seal its complete proposal namespace')
    host, host_id = _json(directory / 'report.json')
    native, native_id = _json(directory / 'native.json')
    require(host.get('stage') == 'world_reward_object_budget_surface_host_v1' and host.get('status') == 'pass'
            and host.get('phase') == 'complete' and host.get('episode_index') == episode
            and host.get('producer_revision') == revision and host.get('native') == native
            and host.get('native_identity') == native_id
            and host.get('owned_container_removed') is True and host.get('owned_scratch_removed') is True,
            'Complete fresh sealed surface host/native receipt required')
    bound = native.get('source_binding', {})
    script_sha = identity(code / 'infra/object_budget_solid.py')['sha256']
    require(bound.get('source_entry') == 'run_full4d_sample'
            and bound.get('producer_revision') == revision
            and bound.get('helpers', {}).get('infra/object_budget_solid.py', {}).get('sha256') == script_sha
            and host.get('source_binding') == host.get('source_binding_after') == native.get('source_binding_after') == bound,
            'Actual namespace/source-bound surface producer required')
    require(native.get('status') == 'pass' and native.get('phase') == 'complete'
            and native.get('stage') == 'world_reward_object_budget_surface_native_v1'
            and native.get('episode_index') == episode and native.get('producer_revision') == revision
            and native.get('input_track') == 'track_1', 'Complete native surface proposal required')
    _no_oracle(native)
    require(all(row.get(key) is True for row in (host, native) for key in
                ('source_rehashed_after', 'inputs_qualification_rehashed_after', 'runtime_rehashed_after')),
            'Complete source/input/runtime postchecks required')
    scale = native.get('metric_scale_baked_once')
    require(type(scale) is float and math.isfinite(scale) and scale > 0
            and native.get('object_scale') == 1., 'Original once-baked positive clip scale required')
    output_ids = {name: local[name] for name in SURFACE_FILES - {'report.json', 'native.json'}}
    require(host.get('outputs') == native.get('outputs') == output_ids, 'Frozen surface output manifest differs')
    first, _ = _json(code / 'configs/surface_identity_qualification_pins.json')
    qualification, _ = _json(code / 'configs/surface_qslim_qualification_pins.json')
    proposal = directory.relative_to(root).as_posix()
    relative_base = base.relative_to(root).as_posix()
    qbase = 'results/surface-qslim-qualify-' + qualification['producer_revision']
    names = [proposal + '/' + name for name in sorted(SURFACE_FILES)] + [
        relative_base + '/object_grounded/' + name for name in ('report.json', 'object.glb', 'transform.json', 'intrinsics.json')
    ] + [relative_base + '/scale_smoke/report.json',
         'results/surface-identity-qualify-' + first['producer_revision'] + '/native.json',
         qbase + '/native.json', qbase + '/report.json', 'results/surface-qslim-independent-v2/report.json']
    require(len(names) == len(set(names)) == 15, 'Exact fifteen surface ancestry files required')
    readonly_results = {qbase + '/native.json', 'results/surface-qslim-independent-v2/report.json',
                        'results/surface-identity-qualify-' + first['producer_revision'] + '/native.json'}
    rows = {name: identity(root / name, readonly=name in readonly_results or name.startswith(proposal + '/'))
            for name in names}
    require(rows[proposal + '/report.json'] == host_id, 'Original host report identity differs')
    helpers = _literal_tuple(code, 'infra/surface_geometry_loader.py', 'SOURCE_HELPERS') + ('src/world_reward/artifact_paths.py',)
    helper_ids = {name: identity(code / name, maximum=2 << 20) for name in helpers}
    pin = dict(schema='world_reward.surface_mesh_pins.v1', episode_index=episode,
               input_sha256=native['input_sha256'], metric_scale_baked_once=scale,
               report=host_id | dict(producer_revision=revision, script_sha256=script_sha),
               files=rows, source_helpers=helper_ids)
    return _write(code, episode, 'surface_mesh', pin)


def input_pin(root, code, revision, episode, total):
    root, code, base = _context(root, code, revision, episode)
    spec = inputs.PublicClipSpec(episode, total, 'front_stereo_camera_left', 1152, 1536)
    paths = inputs.relative_paths(spec)
    report, report_id = _json(root / paths['input_report'], readonly=False)
    _producer(report, code, revision, episode, 'world_reward_native_cari_inputs', 'cari_prepare.py', total=total)
    require(report.get('object_source') == 'surface' and report.get('original_frame_coverage_verified') is True,
            'Full original generic surface input preparation required')
    names = inputs.source_paths(spec, object_source='surface')
    require(len(names) == 15, 'Exact fifteen public source files required')
    rows = {name: identity(root / name, readonly=False) for name in sorted(names)}
    require(rows[paths['input_report']] == report_id, 'Input producer receipt changed')
    pin = dict(schema='world-reward-cari-clip-input-pins-v3', object_source='surface', clip_spec=asdict(spec),
               input_report=report_id | dict(producer_revision=revision, script_sha256=report['script_sha256']), source_files=rows)
    inputs.validate_pins(spec, pin)
    inputs.validate_reports(root, spec, pin)
    for name, row in rows.items():
        _seal(root / name, base, row)
    return _write(code, episode, 'input', pin)


def shared_pin(role, root, code, revision, episode, total):
    root, code, base = _context(root, code, revision, episode)
    require(role in SHARED, 'Explicit prepare/forward/refined/export role required')
    spec = inputs.PublicClipSpec(episode, total, 'front_stereo_camera_left', 1152, 1536)
    script, stage, expected = SHARED[role]
    directory = base / ('cari_shared_' + role + '_v1')
    rows = _inventory(directory, expected)
    report, report_id = _json(directory / 'report.json', readonly=False)
    _producer(report, code, revision, episode, stage, script, total=total)
    require(report.get('phase') == 'complete' and report.get('clip_spec') == asdict(spec)
            and report.get('original_frame_indices') == list(range(total))
            and report.get('source_inputs_assets_rehashed') is True
            and report.get('source_helpers_rehashed') is True, 'Complete full-timeline native lifecycle required')
    _helpers(report, code)
    require(report.get('source_helpers', {}).get('infra/' + script, {}).get('sha256') == report['script_sha256'],
            'Producer script/source helper binding differs')
    require(report.get('output_files') == {name: row for name, row in rows.items() if name != 'report.json'}
            and rows['report.json'] == report_id, 'Exclusive frozen output manifest differs')
    for name, row in rows.items():
        _seal(directory / name, base, row)
    directory.chmod(0o555)
    require(_inventory(directory, expected) == rows, 'Sealed output inventory changed')
    value = dict(schema=f'world-reward-cari-shared-{role}-pins-v1', clip_spec=asdict(spec))
    value[role] = report_id | dict(producer_revision=revision, script_sha256=report['script_sha256'])
    value[role + '_files'] = rows
    return _write(code, episode, 'shared_' + role, value)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument('--role', choices=('surface', 'input', *SHARED), required=True)
    parser.add_argument('--episode', type=int, choices=range(30), required=True)
    parser.add_argument('--total', type=int)
    args = parser.parse_args()
    root, code, revision = Path(os.environ['WR_ROOT']), Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION']
    if args.role == 'surface':
        surface_pin(root, code, revision, args.episode)
    else:
        require(args.total is not None, 'Original total frames required')
        if args.role == 'input':
            input_pin(root, code, revision, args.episode, args.total)
        else:
            shared_pin(args.role, root, code, revision, args.episode, args.total)
    print(json.dumps(dict(stage='full4d_producer_pin', status='pass', episode_index=args.episode, role=args.role)))


if __name__ == '__main__':
    main()
