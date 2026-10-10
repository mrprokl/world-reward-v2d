"""Pinned pure-Python dependency overlay on Azure; no pip or GPU/runtime change."""
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path('/srv/scenesmith/world-reward')
DIRECTORY = 'vendor/research/vda_dependencies_v1'
RECEIPT = 'results/video-depth-dependencies-v1.json'
URL = ('https://files.pythonhosted.org/packages/05/ec/'
       'fa6963f1198172c2b75c9ab6ecefb3045991f92f75f5eb41b6621b198123/'
       'easydict-1.13-py3-none-any.whl')
WHEEL_BYTES = 6804
WHEEL_SHA256 = '6b787daf4dcaf6377b4ad9403a5cee5a86adbc0ca9a5bcf5410e9902002aeac2'
MODULE_SHA256 = '7d561cb13a5f3d12d8b9fdf5fce82e7a15ec42f3bf1361574c7310d83614513a'
LICENSE_SHA256 = 'da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768'


def extract(raw, directory):
    if len(raw) != WHEEL_BYTES or hashlib.sha256(raw).hexdigest() != WHEEL_SHA256:
        raise ValueError('Official wheel byte identity differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
        names = wheel.namelist()
        if len(names) != 6 or len(set(names)) != 6:
            raise ValueError('Exact six-file pure-Python wheel required')
        for name in names:
            p = Path(name)
            if (p.is_absolute() or '..' in p.parts or
                    not (name == 'easydict/__init__.py' or name.startswith('easydict-1.13.dist-info/'))):
                raise ValueError('Unexpected wheel path')
        if (hashlib.sha256(wheel.read('easydict/__init__.py')).hexdigest() != MODULE_SHA256 or
                hashlib.sha256(wheel.read('easydict-1.13.dist-info/LICENSE')).hexdigest() != LICENSE_SHA256):
            raise ValueError('Pinned module and LGPL notice required')
        rows = []
        for name in names:
            data = wheel.read(name); path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream: stream.write(data)
            path.chmod(0o444)
            rows.append(dict(file=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return rows


def main():
    code = Path(os.environ['WR_CODE']); rev = os.environ['WR_CODE_REVISION']
    if (os.environ['WR_ROOT'] != str(ROOT) or
            code != ROOT/'jobs'/rev/'run_video_depth_dependencies/code' or
            Path(__file__).resolve() != code/'infra/video_depth_dependencies.py'):
        raise ValueError('Immutable Azure acquisition only')
    directory = ROOT/DIRECTORY; receipt = ROOT/RECEIPT
    if directory.exists() or receipt.exists() or directory.resolve() != directory:
        raise ValueError('Fresh dependency namespace required')
    directory.mkdir(); started = time.monotonic()
    report = dict(status='fail', producer_revision=rev, pip_used=False, GPU_used=False,
                  package='easydict', version='1.13', license='LGPL-3.0',
                  torch_runtime_changed=False, native_VDA_source_changed=False,
                  wheel_sha256=WHEEL_SHA256)
    try:
        with urllib.request.urlopen(URL, timeout=30) as response:
            if response.geturl() != URL: raise ValueError('Unexpected publisher redirect')
            raw = response.read(WHEEL_BYTES+1)
        report.update(status='pass', files=extract(raw, directory))
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error='Pinned dependency acquisition failed')
        for path in sorted(directory.rglob('*'), key=lambda p:len(p.parts), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        directory.rmdir()
    report['elapsed_seconds'] = time.monotonic()-started
    with receipt.open('x') as stream: json.dump(report, stream, allow_nan=False); stream.write('\n')
    receipt.chmod(0o444)
    print(json.dumps({k:report.get(k) for k in ['status','error','elapsed_seconds']}))
    if report['status'] != 'pass': raise SystemExit(1)


if __name__ == '__main__': main()
