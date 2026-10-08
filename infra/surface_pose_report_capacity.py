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
    """Exact capacity/batch scheduling seams; no numeric statement may differ."""
    require(type(original) is bytes and type(candidate) is bytes, 'Explicit immutable source bytes required')
    replacements = (
        (b'    from surface_geometry_loader import load, identity, strict_json, recheck, preflight_geometry_and_poses, SOURCE_HELPERS\n',
         b'    from surface_geometry_loader import load, strict_json, preflight_geometry_and_poses, SOURCE_HELPERS\n'
         b'    from surface_pose_report_capacity import identity, recheck\n'),
        (b'        from surface_geometry_loader import identity as surface_identity, recheck as surface_recheck\n',
         b'        from surface_pose_report_capacity import identity as surface_identity, recheck as surface_recheck\n'),
        (b"    helpers={*SOURCE_HELPERS,'infra/cari_prepare.py','infra/cari_wrapper_common.sh','infra/run_cari_prepare.sh','src/world_reward/artifact_paths.py'}\n",
         b"    helpers={*SOURCE_HELPERS,'infra/cari_prepare.py','infra/cari_wrapper_common.sh','infra/run_cari_prepare.sh','src/world_reward/artifact_paths.py','infra/surface_pose_report_capacity.py'}\n"),
        (b'    from prep.mhr_depth_h5 import MHRDepthH5Writer, validate_depth_h5, read_metric_depth\n',
         b'    from prep.mhr_depth_h5 import DepthFrameRecord, MHRDepthH5Writer, validate_depth_h5, read_metric_depth\n'),
        (b'    sys.path.insert(0, str(native_root))\n',
         b'    if sha256(native_root / "prep/mhr_depth_h5.py") != "1429760952205d35c87157c05941defa20dc450f2b014441ca7cd5d39b45b0c5":\n        raise RuntimeError("Exact qualified native depth-writer source required")\n    sys.path.insert(0, str(native_root))\n'),
        (b'                          alignment_input_identity=identity, encoding_workers=8) as writer:\n        for index, name in enumerate(names):\n',
         b'                          alignment_input_identity=identity, encoding_workers=8) as writer:\n        pending = []\n        for index, name in enumerate(names):\n'),
        (b'            writer.write_frame(camera_name, index, raw, aligned, scale=scale, shift=0., valid_count=int(valid.sum()))\n',
         b'            pending.append(DepthFrameRecord(index, raw, aligned, scale, 0., int(valid.sum())))\n            if len(pending) == 8:\n                writer.write_frames(camera_name, pending)\n                pending = []\n'),
        (b'        writer.mark_complete()\n',
         b'        if pending:\n            writer.write_frames(camera_name, pending)\n        writer.mark_complete()\n'),
        (b"    if args.mesh_source == 'solid':\n        _solid_recheck(solid_ledger)\n",
         b'    result["depth_encoding"] = {"batch_size": 8, "encoding_workers": 8,\n        "native_source_sha256": "1429760952205d35c87157c05941defa20dc450f2b014441ca7cd5d39b45b0c5",\n        "compression_or_quantization_changed": False, "validation_changed": False}\n    if args.mesh_source == \'solid\':\n        _solid_recheck(solid_ledger)\n'),
        (b'                print(json.dumps({"stage": "cari_prepare_depth", "frames_complete": index + 1}), flush=True)\n',
         b'                print(json.dumps({"stage": "cari_prepare_depth", "frames_submitted": index + 1, "frames_complete": index + 1 - len(pending)}), flush=True)\n'),
    )
    expected = original
    for old, new in replacements:
        require(expected.count(old) == 1, 'Exact original source seam required')
        expected = expected.replace(old, new, 1)
    require(candidate == expected, 'Only explicit capacity, native source receipt and batch scheduling seams may change')
    return dict(numeric_source_unchanged=True, exact_source_substitutions=len(replacements), depth_batch_size=8,
                depth_native_source_sha256="1429760952205d35c87157c05941defa20dc450f2b014441ca7cd5d39b45b0c5",
                original_sha256=hashlib.sha256(original).hexdigest(),
                candidate_sha256=hashlib.sha256(candidate).hexdigest(),
                maximum_pose_report_bytes=MAX_POSE_REPORT_BYTES)


def verify_preparation_source(original_path, candidate_path):
    """Verify immutable paths using the stronger exact-byte substitution gate."""
    return verify_capacity_source(Path(original_path).read_bytes(), Path(candidate_path).read_bytes())
