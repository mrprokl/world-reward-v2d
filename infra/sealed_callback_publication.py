"""Owned sealed receipt + callback, with fixed safe failure-stage diagnostics.

Injected byte/state/deadline functions belong to the authenticated caller. This
helper neither authenticates their provenance nor performs the callback itself.
It never serializes exception text; old failed publications remain unchanged.
"""
import hashlib
import os
from pathlib import Path
import stat
import time

STAGES = ('namespace', 'sealing', 'sealed_verify', 'sealed_deadline',
          'after_seal', 'cleanup_write', 'cleanup_verify', 'cleanup_deadline')
ERROR_TYPES = frozenset(('ValueError', 'RuntimeError', 'TimeoutError', 'OSError',
                        'KeyError', 'ImportError', 'ModuleNotFoundError', 'other'))


def publish(out, report, deadline, started, owner, allowed, after_seal=None, *,
            encode, identity, snapshot, require, check, sync, error,
            maximum=32 << 20, report_maximum=256 << 10):
    """Preserve original 700→500/400-leaf checks, pins and one original report FD.

    A failed namespace check never mutates a foreign directory. A replaced report
    path is never overwritten; if the owned receipt cannot be safely updated the
    exception propagates. The caller must not infer PASS from an absent receipt.
    """
    out = Path(out)
    s = out.lstat()
    require((s.st_dev, s.st_ino, s.st_uid) == owner and s.st_nlink >= 2
            and stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode) == 0o700
            and {p.name for p in out.iterdir()} == set(allowed), 'Foreign receipt namespace')
    directory_gid = s.st_gid
    stage = 'namespace'
    with (out/'report.json').open('x+b') as stream:
        os.fchmod(stream.fileno(), 0o400)
        opened = os.fstat(stream.fileno())

        def original_file():
            directory = out.lstat()
            require(stat.S_ISDIR(directory.st_mode)
                    and (directory.st_dev, directory.st_ino, directory.st_uid) == owner
                    and directory.st_gid == directory_gid, 'Foreign receipt namespace')
            now = (out/'report.json').lstat()
            require(stat.S_ISREG(now.st_mode) and now.st_nlink == 1
                    and (now.st_dev, now.st_ino, now.st_uid, now.st_gid)
                    == (opened.st_dev, opened.st_ino, opened.st_uid, opened.st_gid),
                    'Foreign report inode rejected')

        def update():
            original_file()
            stream.seek(0); stream.write(encode(report)); stream.truncate()
            stream.flush(); os.fsync(stream.fileno())

        def verify(leaves):
            directory = out.lstat()
            require(stat.S_ISDIR(directory.st_mode) and directory.st_nlink >= 2
                    and (directory.st_dev, directory.st_ino, directory.st_uid) == owner
                    and directory.st_gid == directory_gid
                    and stat.S_IMODE(directory.st_mode) == 0o500,
                    'Sealed publication namespace changed')
            original_file()
            for name, before in leaf_stats.items():
                now = (out/name).lstat()
                require(stat.S_ISREG(now.st_mode) and now.st_nlink == 1
                        and stat.S_IMODE(now.st_mode) == 0o400
                        and now.st_uid == opened.st_uid
                        and (now.st_dev, now.st_ino, now.st_uid, now.st_gid)
                        == (before.st_dev, before.st_ino, before.st_uid, before.st_gid),
                        'Owned immutable receipt leaf changed')
            raw = encode(report)
            expected = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            require(identity(out/'report.json', report_maximum) == expected
                    and snapshot(out/'report.json')[:2] == (opened.st_dev, opened.st_ino)
                    and {p.name for p in out.iterdir()} == set(allowed)|{'report.json'}
                    and all(identity(out/n, maximum) == pin for n, pin in leaves.items()),
                    'Sealed publication changed')

        try:
            s = out.lstat()
            require((s.st_dev, s.st_ino, s.st_uid) == owner and s.st_nlink >= 2
                    and stat.S_ISDIR(s.st_mode) and s.st_gid == directory_gid
                    and stat.S_IMODE(s.st_mode) == 0o700
                    and {p.name for p in out.iterdir()} == set(allowed)|{'report.json'}, 'Foreign receipt namespace')
            leaf_stats = {p.name: p.lstat() for p in out.iterdir()}
            leaves = {p.name: identity(p, maximum) for p in out.iterdir() if p.name != 'report.json'}
            require(all(stat.S_ISREG(s.st_mode) and s.st_nlink == 1
                        and stat.S_IMODE(s.st_mode) == 0o400 and s.st_uid == opened.st_uid
                        for s in leaf_stats.values()), 'Owned immutable receipt leaves')
            stage = 'sealing'
            out.chmod(0o500); sync(out); sync(out.parent)
            report.update(outputs_sealed=True, elapsed_seconds=time.monotonic()-started); update()
            stage = 'sealed_verify'; verify(leaves)
            stage = 'sealed_deadline'; check(deadline)
            if after_seal is not None and report['status'] == 'pass':
                stage = 'after_seal'; after_seal()
                report['blob_cleanup_verified'] = True
                report['elapsed_seconds'] = time.monotonic()-started
                stage = 'cleanup_write'; update()
                stage = 'cleanup_verify'; verify(leaves)
                stage = 'cleanup_deadline'; check(deadline)
        except BaseException as exc:
            category = error(exc)
            report.update(status='fail', publication_failed=True,
                          publication_error_type=category if type(category) is str and category in ERROR_TYPES else 'other',
                          publication_failure_stage=stage)
            update()
    return report
