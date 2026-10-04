"""Independent original export Git-byte binding; no historical code executes.

The fixed per-clip pin authenticates the complete original dispatch, not a
report's self-selected helper map. Current numerical consumer code is separately
compatible; new source-profile enumeration is not an old producer identity.
"""
from __future__ import annotations
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import stat

NUMERICAL_SOURCES = (
    'infra/cari_full_export.py', 'infra/cari_full_refine.py', 'infra/cari_full_forward.py',
    'infra/cari_shared_prepare.py', 'infra/cari96_prepare.py', 'infra/cari96_inputs.py', 'infra/cari96_native.py',
    'infra/cari_runner.py', 'infra/cari96_forward.py', 'infra/cari_refine.py',
    'infra/cari_converter.py', 'infra/body_smoke.py', 'src/world_reward/shared_identity.py',
    'src/world_reward/timeline.py', 'src/world_reward/data.py', 'src/world_reward/contracts.py',
    'src/world_reward/submission.py',
)
INPUT_NUMERICS = ('PublicClipSpec', '_spec', 'relative_paths', 'inferred_camera', '_float_array', '_rigid', 'validate_wild')


def _require(value, message):
    if not value: raise ValueError(message)


def _hex(value, length):
    _require(type(value) is str and re.fullmatch(f'[0-9a-f]{{{length}}}', value), 'Exact source digest required')


def _read(path, *, empty=False, readonly=True):
    path = Path(path)
    _require(path.is_absolute() and path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical unaliased source required')
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and (0 if empty else 1) <= before.st_size <= 2_000_000
             and (not readonly or not before.st_mode & 0o222), 'Bounded immutable regular source required')
    raw = path.read_bytes(); after = path.lstat()
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink')
    _require(all(getattr(before, f) == getattr(after, f) for f in fields), 'Source changed while read')
    return raw, dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), mode=stat.S_IMODE(after.st_mode))


def _json(raw):
    def unique(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, 'Duplicate source pin key'); result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite source pins')))


def _input_numeric_ast(raw):
    tree = ast.parse(raw.decode('utf-8'))
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in INPUT_NUMERICS}
    _require(set(nodes) == set(INPUT_NUMERICS), 'Complete unchanged input numerical definitions required')
    return {name: ast.dump(nodes[name], include_attributes=False) for name in INPUT_NUMERICS}


def verify_export_source(root, code, spec, export_pins):
    """Return original source/proof after full immutable ledger+compatibility.

    Source identities are independently frozen from Git. This is provenance,
    not re-execution of producer geometry or a reconstruction-quality claim.
    """
    root, code = Path(root), Path(code)
    for p in (root, code):
        _require(p.is_absolute() and p.resolve() == p and p.is_dir() and not any(q.is_symlink() for q in (p, *p.parents)), 'Canonical source root/code required')
    pin_path = code / f'configs/cari_clip_{spec.episode_index:06d}_export_source_pins.json'
    raw, pin_id = _read(pin_path); pins = _json(raw)
    keys = {'schema', 'clip_spec', 'export', 'producer_revision', 'job', 'source_archive_sha256', 'files'}
    _require(type(pins) is dict and set(pins) == keys and pins['schema'] == 'world_reward.cari_export_source_pins.v1', 'Exact original export source pin schema required')
    _require(pins['clip_spec'] == asdict(spec) and type(pins['clip_spec']) is dict
             and all(type(pins['clip_spec'][k]) is type(v) for k, v in asdict(spec).items())
             and pins['export'] == export_pins['export'] and pins['job'] == 'run_cari_shared_stage_queued', 'Independent original export clip/report binding differs')
    revision = pins['producer_revision']; _hex(revision, 40); _hex(pins['source_archive_sha256'], 64)
    _require(revision == export_pins['export']['producer_revision'], 'Original source revision differs from frozen export')
    original = root / 'jobs' / revision / pins['job'] / 'code'
    _require(original.parent.is_dir() and {p.name for p in original.parent.iterdir()} == {'code', 'revision', 'source-sha256'}, 'Exact original dispatch parent inventory required')
    marker_ids = {}
    for name, value in (('revision', revision), ('source-sha256', pins['source_archive_sha256'])):
        contents, marker_ids[name] = _read(original.parent / name)
        _require(contents == (value + '\n').encode(), 'Original export source/archive marker differs')
    files = pins['files']; _require(type(files) is dict and 1 <= len(files) <= 512, 'Complete bounded source ledger required')
    for name, row in files.items():
        _require(type(name) is str and re.fullmatch(r'(?:infra|src|configs)/[A-Za-z0-9_./-]+|pyproject\.toml', name)
                 and all(p not in ('', '.', '..', '__pycache__', '.git') for p in name.split('/')), 'Source-only original filename required')
        _require(type(row) is dict and set(row) == {'bytes', 'sha256', 'mode'} and type(row['bytes']) is int and 0 <= row['bytes'] <= 2_000_000
                 and type(row['mode']) is int and row['mode'] in (0o444, 0o555), 'Exact original source byte/mode pin required')
        _hex(row['sha256'], 64)
    observed = {}; directories = set()
    expected_directories = {'.', *(str(parent) for name in files for parent in Path(name).parents)}
    for path in (original, *sorted(original.rglob('*'))):
        s = path.lstat()
        _require(path.resolve() == path and not path.is_symlink(), 'Original source alias forbidden')
        if stat.S_ISDIR(s.st_mode):
            _require(stat.S_IMODE(s.st_mode) == 0o555, 'Original source directories must be readonly/traversable')
            directories.add(str(path.relative_to(original)))
        else: observed[str(path.relative_to(original))] = _read(path, empty=True)[1]
    _require(observed == files and directories == expected_directories
             and sum(row['bytes'] for row in observed.values()) <= 16_000_000, 'Full original Git-byte ledger differs')
    _require(files['infra/cari_full_export.py']['sha256'] == export_pins['export']['script_sha256'], 'Original exporter SHA differs')
    compatibility = {}
    for name in NUMERICAL_SOURCES:
        _require(name in files, 'Required numerical source absent from original ledger')
        current = _read(code / name)[1]
        _require(current['bytes'] == files[name]['bytes'] and current['sha256'] == files[name]['sha256'], 'Current consumer numerical source differs: ' + name)
        compatibility[name] = current
    input_name = 'infra/cari_clip_inputs.py'
    old_input, _ = _read(original / input_name); current_input, current_input_id = _read(code / input_name)
    _require(_input_numeric_ast(old_input) == _input_numeric_ast(current_input), 'Current input geometry/gauge numerics differ')
    consumer_sources = {name: _read(code / name)[1] for name in ('infra/cari_export_source.py', 'infra/cari_shared_episode_loader.py')}
    _require(_read(pin_path)[1] == pin_id, 'Independent export source pins changed')
    proof = dict(producer_revision=revision, job=pins['job'], source_archive_sha256=pins['source_archive_sha256'],
        pins_identity=pin_id, files_verified=len(files), original_marker_identities=marker_ids,
        current_numeric_sources=compatibility, current_input_helper=current_input_id, current_consumer_sources=consumer_sources,
        input_numeric_definitions=list(INPUT_NUMERICS), historical_code_executed=False, quality_verified=False)
    return original, proof
