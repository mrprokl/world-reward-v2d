"""Metadata-only capacity adapter for a complete fixed-surface pose receipt.

The original surface reader, model/geometry equations and all 32MiB roles stay
unchanged. Only the exact revision-bound, readonly full-pose report may be up to
64MiB. Hashing preserves every byte, inode and provenance gate; no rewriting,
numeric decoding, prediction, simplification or historical adoption occurs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import stat

from world_reward.artifact_paths import output_prefix

ROOT = Path('/srv/scenesmith/world-reward')
MAX_POSE_REPORT_BYTES = 64 << 20


def require(value, reason):
    if not value:
        raise ValueError(reason)


def _loader():
    import surface_geometry_loader
    return surface_geometry_loader


def _pose_report_role(path):
    path = Path(path)
    prefix = output_prefix()
    if re.fullmatch(r'experiments/full4d-v1-[0-9a-f]{40}/outputs', prefix) is None:
        return False
    return (path.name == 'report.json' and path.parent.name == 'object_pose_full_surface'
            and re.fullmatch(r'episode_0000(?:0[0-9]|1[0-9]|2[0-9])', path.parent.parent.name) is not None
            and path.parent.parent.parent == ROOT / prefix)


def identity(path, *, readonly=True):
    path = Path(path)
    if not _pose_report_role(path):
        return _loader().identity(path, readonly=readonly)
    require(path.is_absolute() and path.resolve() == path
            and not any(parent.is_symlink() for parent in (path, *path.parents)),
            'Canonical original full-pose report required')
    before = path.lstat()
    # A larger report is never permission to consume an unsealed prediction,
    # even during a readonly=False ancestry recheck.
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
            and 0 < before.st_size <= MAX_POSE_REPORT_BYTES and not before.st_mode & 0o222,
            'Bounded single-link readonly full-pose report required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    after = path.lstat()
    require(all(getattr(before, key) == getattr(after, key) for key in
                ('st_dev', 'st_ino', 'st_size', 'st_mode', 'st_nlink', 'st_uid', 'st_gid',
                 'st_mtime_ns', 'st_ctime_ns')), 'Original full-pose report changed during hashing')
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def recheck(ledger):
    """Same original ledger, with the named readonly report capacity exception."""
    require(all((identity(path, readonly=False) if _pose_report_role(path)
                 else _loader()._source_identity(path)) == pin for path, pin in ledger.items()),
            'Original immutable inputs or source changed during surface consumption')


def verify_capacity_source(original, candidate):
    """Exactly three source-only substitutions; no numeric statement may differ."""
    require(type(original) is bytes and type(candidate) is bytes, 'Explicit immutable source bytes required')
    replacements = (
        (b'    from surface_geometry_loader import load, identity, strict_json, recheck, preflight_geometry_and_poses, SOURCE_HELPERS\n',
         b'    from surface_geometry_loader import load, strict_json, preflight_geometry_and_poses, SOURCE_HELPERS\n'
         b'    from surface_pose_report_capacity import identity, recheck\n'),
        (b'        from surface_geometry_loader import identity as surface_identity, recheck as surface_recheck\n',
         b'        from surface_pose_report_capacity import identity as surface_identity, recheck as surface_recheck\n'),
        (b"    helpers={*SOURCE_HELPERS,'infra/cari_prepare.py','infra/cari_wrapper_common.sh','infra/run_cari_prepare.sh','src/world_reward/artifact_paths.py'}\n",
         b"    helpers={*SOURCE_HELPERS,'infra/cari_prepare.py','infra/cari_wrapper_common.sh','infra/run_cari_prepare.sh','src/world_reward/artifact_paths.py','infra/surface_pose_report_capacity.py'}\n"),
    )
    expected = original
    for old, new in replacements:
        require(expected.count(old) == 1, 'Exact original source seam required')
        expected = expected.replace(old, new, 1)
    require(candidate == expected, 'Only the explicit metadata capacity imports/ledger may change')
    return dict(numeric_source_unchanged=True, exact_source_substitutions=3,
                original_sha256=hashlib.sha256(original).hexdigest(),
                candidate_sha256=hashlib.sha256(candidate).hexdigest(),
                maximum_pose_report_bytes=MAX_POSE_REPORT_BYTES)


def verify_preparation_source(original_path, candidate_path):
    """Verify immutable paths using the stronger exact-byte substitution gate."""
    return verify_capacity_source(Path(original_path).read_bytes(), Path(candidate_path).read_bytes())
