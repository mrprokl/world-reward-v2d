"""Small stdlib immutable metadata and Linux NOREPLACE directory primitives."""
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat


def canonical(path):
    path=Path(path)
    if not path.is_absolute()or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):
        raise ValueError('Canonical absolute nonsymlink path required')
    return path


def _state(path):
    s=path.lstat();return(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)


def _pin(value):
    if(type(value)is not dict or set(value)!={'bytes','sha256'}or type(value['bytes'])is not int or value['bytes']<=0
            or type(value['sha256'])is not str or re.fullmatch('[0-9a-f]{64}',value['sha256'])is None):
        raise ValueError('Exact positive-byte SHA256 pin required')
    return value


def identity(path,maximum=200000):
    path=canonical(path);s=_state(path)
    if not stat.S_ISREG(s[2])or s[6]!=1 or not 0<s[3]<=maximum:raise ValueError('Bounded unaliased metadata required')
    raw=path.read_bytes()
    if _state(path)!=s:raise ValueError('Metadata changed during read')
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def _json(raw):
    def pairs(rows):
        result={}
        for k,v in rows:
            if k in result:raise ValueError('Duplicate JSON field')
            result[k]=v
        return result
    def invalid(_):raise ValueError('Nonfinite JSON constant')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)


def _markers(code,revision):
    canonical(code)
    if not code.is_dir()or re.fullmatch('[0-9a-f]{40}',revision)is None:raise ValueError('Immutable full-revision source required')
    for name in('revision','source-sha256'):
        path=canonical(code.parent/name);before=identity(path,128);raw=path.read_bytes()
        if((name=='revision'and raw!=(revision+'\n').encode())or
                (name=='source-sha256'and re.fullmatch(b'[0-9a-f]{64}\n',raw)is None)or identity(path,128)!=before):
            raise ValueError('Exact unchanged dispatch marker required')


def rename_noreplace(source,destination):
    """No overwrite or non-atomic fallback, including occupied empty dirs."""
    if platform.system()!='Linux':raise ValueError('Linux renameat2 NOREPLACE required')
    libc=ctypes.CDLL(None,use_errno=True)
    try:call=libc.renameat2
    except AttributeError:raise ValueError('Atomic renameat2 unavailable')from None
    call.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint);call.restype=ctypes.c_int
    if call(-100,os.fsencode(source),-100,os.fsencode(destination),1):
        code=ctypes.get_errno()
        if code in(errno.ENOSYS,errno.EINVAL):raise ValueError('Atomic no-overwrite rename unsupported; no fallback')
        raise OSError(code,'Atomic no-overwrite archive rename failed')


def _sync_directory(path):
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)
