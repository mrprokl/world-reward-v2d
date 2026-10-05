"""Isolated Azure MMCV1.7.2 CUDA operator gate, before any HOI checkpoint/RGB.

Original complete CUDA extension is built once against the selected base ABI.
MSDeformAttn/NMS/RoIAlign must execute on CUDA; original soft-NMS is CPU-only.
No upstream numerical patch, dataset, model, existing image or Torch mutation.
"""
import argparse
import email.parser
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/hoi_detr_runtime_v4')
ENTRY = 'run_hoi_detr_runtime_verify'
PROTOCOL = 'configs/hoi_detr_runtime_v1.json'
PROTOCOL_PIN = dict(bytes=7695, sha256='a5bd9a4953615db706986ca5e8a339f87bbfa63a30105a71cc335a5c114620f8')
ACQUIRE_PIN = dict(bytes=29349, sha256='130ca5bb5c4925c0069b5a3181c54848b78cb6c043dc688f29aa1f8bb66c3845')
HELPERS = ('infra/hoi_detr_runtime_verify.py', 'infra/run_hoi_detr_runtime_verify.sh', PROTOCOL,
           'infra/hoi_detr_acquire.py', 'infra/mediapipe_cpu_runtime_verify.py', 'infra/mediapipe_hands_acquire.py')
BLOCK = 1 << 20
SAFE_ENV = dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin', HOME='/nonexistent', LANG='C.UTF-8', DOCKER_HOST='unix://'+str(ROOT/'docker.sock'))
FLAGS = ('quality_verified', 'hoi_model_inference_qualified', 'model_weights_read', 'rgb_read', 'dataset_instantiated', 'ground_truth_used', 'challenge_inputs_used', 'training_overlap_verified', 'license_eligibility_verified')
# Only fixed public requirement labels may enter a native failure receipt. Never
# serialize upstream exception text, paths, traceback or environment values.
NATIVE_REQUIREMENTS = frozenset((
    'Original acquisition helper path required',
    'Pinned readonly acquisition helper required',
    'Independent acquisition helper differs',
    'Actual isolated source/markers required',
    'Canonical immutable/control path required',
    'Bounded single-link regular artifact required',
    'Artifact changed while hashing',
    'Duplicate receipt/config field',
    'Nonfinite metadata',
    'Exact independent SHA/byte pin required',
    'Independent artifact identity differs',
    'Original immutable source namespace required',
    'Actual dispatch markers required',
    'Readonly complete source closure required',
    'Required immutable source/config closure missing',
    'Inclusive native runtime deadline',
    'Unprivileged offline native container required',
    'Parent source/base proof differs',
    'Parent owned CID must precede native imports',
    'Complete source-only acquisition manifest required',
    'Closed numerical source/dependency inventory required',
    'Readonly source-only inputs required',
    'No foreign source/model/RGB artifact permitted',
    'Original source/dependency posthash differs',
    'Original full source proof differs',
    'Original CP311/Torch/NumPy ABI required',
    'Frozen unchanged base dependencies/headers required',
    'CPU build must not execute CUDA',
    'Build copy differs from original source',
    'Frozen native host compiler differs',
    'Exact CUDA12.4 compiler required',
    'Original full native extension build failed; no retry/patch',
    'Native build modified original numerical/build source',
    'Exactly one full native MMCV extension required',
    'No generated build/runtime symlinks',
    'CPU build never initialized CUDA',
    'Native compile receipt late/failure',
    'Original built runtime differs before ops',
    'Exact added dependency versions required',
    'Exactly built native MMCV extension required',
    'No HOI/model/dataset imports in operator gate',
    'Actual single H100 SM90 required',
    'Native CUDA MSDeformAttn disagrees with original PyTorch reference',
    'Native CUDA NMS indices differ',
    'Original CPU soft-NMS decay/removal/indices differ',
    'Native CUDA RoIAlign analytic grid differs',
    'Built runtime changed after native operators',
    'Source changed after operators',
    'Native operators late/failure',
))


def require(value, message):
    if not value:
        raise ValueError(message)


def native_failure_requirement(error):
    if type(error) is ValueError and len(error.args) == 1 and type(error.args[0]) is str and error.args[0] in NATIVE_REQUIREMENTS:
        return error.args[0]
    return 'redacted_non_allowlisted_error'


def helpers(code):
    path = Path(code)/'infra/hoi_detr_acquire.py'
    require(path.is_absolute() and path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)), 'Original acquisition helper path required')
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and not before.st_mode & 0o222 and before.st_size == ACQUIRE_PIN['bytes'], 'Pinned readonly acquisition helper required')
    raw = path.read_bytes(); after = path.lstat()
    require(hashlib.sha256(raw).hexdigest() == ACQUIRE_PIN['sha256'] and all(getattr(before, k) == getattr(after, k) for k in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_uid', 'st_gid', 'st_nlink')), 'Independent acquisition helper differs')
    spec = importlib.util.spec_from_file_location('wr_hoi_runtime_acquisition', path)
    acq = importlib.util.module_from_spec(spec); spec.loader.exec_module(acq)
    rt, mp = acq.helpers(code)
    return rt, mp, acq


def source(rt, code, revision):
    require(Path(__file__).resolve() == code/HELPERS[0] and {p.name for p in code.parent.iterdir()} == {'code', 'revision', 'source-sha256'}, 'Actual isolated source/markers required')
    return rt.source(ROOT, code, revision, ENTRY, HELPERS)


def protocol(rt, code):
    p = rt.pinned(code/PROTOCOL, PROTOCOL_PIN, 32 << 10)
    require(p['schema'] == 'world_reward.hoi_detr_runtime.v4' and p['scope'] == 'native_mmcv_operator_qualification_only'
            and p['root'] == str(ROOT) and p['data_root'] == str(DATA) and p['output'] == 'results/hoi-detr-runtime-v4'
            and (p['budget_seconds'], p['cleanup_grace_seconds'], p['outer_seconds']) == (1800, 60, 1860)
            and p['minimum_free_bytes'] == 16 << 30 and all(p[k] is False for k in FLAGS), 'Frozen operator-only scope required')
    s = p['source']
    require(s['repository'] == 'open-mmlab/mmcv' and s['revision'] == '4c01b026f0afa5a91a5f54aea313788da1e40f95'
            and s['git_root_tree_sha1'] == 'e5a9980219f78ae70ec004c18d7be58773925870'
            and s['commit_date'][:10] <= p['publication_cutoff'] and s['url'] == 'https://codeload.github.com/'+s['repository']+'/tar.gz/'+s['revision']
            and s['archive_prefix'] == 'mmcv-'+s['revision'] and 0 < s['blob_count'] <= s['maximum_blobs'] <= 2048
            and 0 < s['expanded_bytes'] <= s['maximum_expanded_bytes'] == 64 << 20 and s['maximum_archive_bytes'] == 64 << 20, 'Original complete MMCV tree required')
    require([r['name'] for r in p['wheels']] == ['addict', 'yapf', 'tomli'] and [r['version'] for r in p['wheels']] == ['2.4.0', '0.40.1', '2.0.1'], 'Minimal original MMCV import dependencies required')
    require(p['build'] == dict(MMCV_WITH_OPS='1', FORCE_CUDA='1', MMCV_WITH_TRT='0', MMCV_WITH_ORT='0', MAX_JOBS='32', native_numerical_source_modified=False, extension='mmcv._ext', cpp_standard='native_setup_cxx17'), 'No native operator rewrite/backend substitution')
    require(p['build_cpus'] == p['maximum_jobs'] == 32 and p['gpu_cpus'] == 4 and p['cpu_memory'] == '128g' and p['gpu_memory'] == '16g', 'Prospective measured VM02 resource budget required')
    require(p['compiler']['torch_cuda_arch_list'] == '9.0' and p['operators']['amp'] is False and p['operators']['tf32'] is False
            and p['operators']['softNMS']['native_cpu_only'] is True, 'Original FP32 SM90 and CPU soft-NMS required')
    for r in p['wheels']:
        require(r['filename'].endswith('-py3-none-any.whl') and r['date'] <= p['publication_cutoff'] and r['url'].startswith('https://files.pythonhosted.org/packages/') and r['url'].endswith('/'+r['filename'])
                and 0 < r['bytes'] < 500000 and re.fullmatch('[0-9a-f]{64}', r['sha256']), 'Frozen pure-Python publisher wheel required')
    return p


def check(deadline):
    require(time.monotonic() < deadline, 'Inclusive native runtime deadline')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_):
        raise ValueError('Original public source redirects rejected')


def fetch(mp, acq, data, row, deadline, owned, directories, opener):
    url = urllib.parse.urlsplit(row['url'])
    require(url.scheme == 'https' and url.hostname in ('codeload.github.com', 'raw.githubusercontent.com', 'files.pythonhosted.org') and not url.username and not url.password and not url.query and not url.fragment, 'Exact unsigned public source endpoint required')
    check(deadline); target = data/row['file']; acq.parents(data, target, directories); part = target.with_name(target.name+'.part')
    maximum = row.get('maximum_archive_bytes', row.get('bytes')); sha = hashlib.sha256(); count = 0
    with opener.open(urllib.request.Request(row['url'], headers={'Accept-Encoding': 'identity'}), timeout=min(30, max(.001, deadline-time.monotonic()))) as response:
        require(response.status == 200 and response.geturl() == row['url'] and response.headers.get('Content-Encoding', 'identity') == 'identity'
                and response.headers.get_content_type() != 'text/html', 'Immutable public transport differs')
        length = response.headers.get('Content-Length')
        require(length is None or length.isdecimal() and (int(length) == row['bytes'] if 'sha256' in row else 0 < int(length) <= maximum), 'Publisher byte length differs')
        with part.open('xb') as stream:
            os.fchmod(stream.fileno(), 0o600); s = part.lstat(); owned.append((part, (s.st_dev, s.st_ino, s.st_uid)))
            while True:
                check(deadline); b = response.read(min(BLOCK, maximum-count+1)); require(len(b) <= BLOCK and count+len(b) <= maximum, 'Publisher overflow')
                if not b:
                    break
                if count == 0:
                    require(not b.lstrip().lower().startswith((b'<html', b'<!doctype html')) and not b.startswith(b'version https://git-lfs.github.com/spec/'), 'HTML/LFS pointer rejected')
                count += len(b); sha.update(b); stream.write(b)
            require(count > 0 and ('sha256' not in row or count == row['bytes'] and sha.hexdigest() == row['sha256']), 'Independent publisher bytes/SHA differ')
            stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
        mp.publish(part, target)
    return dict(file=row['file'], bytes=count, sha256=sha.hexdigest())


def keep_source(acq, name):
    p = acq.relative(name)
    if p.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING')) and p.suffix.lower() in ('', '.txt', '.md', '.rst'):
        return True
    if '__pycache__' in p.parts:
        return False
    if p.parts[0] == 'mmcv' and p.suffix.lower() in ('.py', '.cpp', '.cc', '.c', '.cu', '.cuh', '.h', '.hpp', '.m', '.mm', '.mlu'):
        return True
    if p.parts[0] == 'requirements' and p.suffix == '.txt':
        return True
    return name in ('setup.py', 'setup.cfg', 'MANIFEST.in', 'requirements.txt', 'README.md', '.gitattributes')


def unpack(rt, mp, acq, data, archive_path, spec, deadline, owned, directories, artifacts):
    blobs, dirs, links = {}, set(), set(); total = entries = 0; prefix = spec['archive_prefix']
    with tarfile.open(archive_path, 'r:gz') as archive:
        for m in archive:
            check(deadline); entries += 1; require(entries <= spec['maximum_tar_entries'], 'Native archive entry cap')
            full = m.name.rstrip('/') if m.isdir() else m.name; acq.relative(full)
            require(full == prefix or full.startswith(prefix+'/') and m.size >= 0, 'Original MMCV archive prefix required')
            n = full[len(prefix)+1:] if full != prefix else ''
            require(m.isdir() or m.isfile() or m.issym(), 'Source hardlinks/special entries rejected')
            if m.isdir():
                require(full not in dirs and m.size == 0, 'Duplicate/nonempty archive directory'); dirs.add(full); continue
            require(n and n not in blobs and 0 <= m.size <= spec['maximum_member_bytes'], 'Unique bounded original Git member required')
            if m.issym():
                r = spec['symlinks'].get(n); require(r and m.size == 0 and m.linkname == r['target'], 'Only original symlink text; no following')
                b = m.linkname.encode('utf-8'); oid = hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
                require(len(b) == r['bytes'] and oid == r['git_blob_sha1'], 'Exact original Git link differs')
                blobs[n] = dict(mode='120000', bytes=len(b), git_blob_sha1=oid); total += len(b); links.add(n)
            else:
                count = 0; raw = bytearray(); sha = hashlib.sha256(); git = hashlib.sha1(b'blob '+str(m.size).encode()+b'\0'); keep = keep_source(acq, n)
                require(not keep or m.size <= spec['maximum_retained_file_bytes'], 'Original retained source file cap')
                with archive.extractfile(m) as stream:
                    while True:
                        check(deadline); b = stream.read(BLOCK)
                        if not b: break
                        count += len(b); require(count <= m.size, 'Source member overflow'); sha.update(b); git.update(b)
                        if keep: raw.extend(b)
                require(count == m.size, 'Source member truncated'); total += count
                mode = '100755' if m.mode & 0o111 else '100644'; blobs[n] = dict(mode=mode, bytes=count, git_blob_sha1=git.hexdigest())
                if keep:
                    raw.decode('utf-8'); require(b'\0' not in raw, 'Original UTF-8 numerical/build/notice source only')
                    target = data/'source/mmcv'/n; acq.publish_bytes(mp, data, target, raw, owned, directories)
                    artifacts.append(dict(file=str(target.relative_to(data)), bytes=count, sha256=sha.hexdigest(), git_blob_sha1=git.hexdigest(), original_git_mode=mode))
            require(total <= spec['maximum_expanded_bytes'] and len(blobs) <= spec['maximum_blobs'], 'Original source total cap')
    allowed = {prefix}
    for n in blobs:
        allowed.update(prefix+'/'+str(p) for p in PurePosixPath(n).parents if str(p) != '.')
    require(dirs <= allowed and links == set(spec['symlinks']) and len(blobs) == spec['blob_count'] and total == spec['expanded_bytes']
            and acq.git_tree(blobs) == spec['git_root_tree_sha1'], 'Independent complete MMCV root tree differs')
    for n, pin in spec['required_source_pins'].items():
        require(rt.identity(data/'source/mmcv'/n, 500000) == pin, 'Independent original native source/notice differs')
    return artifacts, dict(git_root_tree_sha1=spec['git_root_tree_sha1'], blobs=len(blobs), expanded_bytes=total, symlink_texts_verified=len(links), links_materialized=False,
                           native_executable_git_modes_preserved=True, all_original_blobs_authenticated=True, original_inventory_sha256=hashlib.sha256(json.dumps(blobs, sort_keys=True).encode()).hexdigest())


def wheel_notice(path, row, publisher):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist(); names = [r.filename for r in entries]
        require(len(names) == len(set(names)) and len(names) <= 4096 and sum(r.file_size for r in entries) <= 8 << 20, 'Bounded exact wheel inventory required')
        for r in entries:
            p = PurePosixPath(r.filename.rstrip('/'))
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in r.filename and not stat.S_ISLNK(r.external_attr >> 16) and not r.flag_bits & 1, 'Wheel links/traversal/encryption rejected')
        metadata = [n for n in names if n.endswith('.dist-info/METADATA')]; require(len(metadata) == 1 and archive.getinfo(metadata[0]).file_size <= 128 << 10, 'Unique bounded wheel metadata required')
        m = email.parser.BytesParser().parsebytes(archive.read(metadata[0]))
        require(m['Name'].lower().replace('_', '-') == row['name'] and m['Version'] == row['version'], 'Exact original dependency metadata required')
        notices = [n for n in names if '.dist-info/' in n and PurePosixPath(n).name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING'))]
        require(notices and all(0 < archive.getinfo(n).file_size <= 64 << 10 for n in notices), 'Mandatory original embedded license/notices required')
        normalized = lambda b: b.decode('utf-8').replace('\r\n', '\n').strip()
        require(any(normalized(archive.read(n)) == normalized(publisher) for n in notices), 'Pinned publisher grant differs from embedded wheel')
        return dict(license=row['license'], requires_dist=m.get_all('Requires-Dist', []), notices={n: dict(bytes=archive.getinfo(n).file_size, sha256=hashlib.sha256(archive.read(n)).hexdigest()) for n in notices})


def data_proof(rt, data, manifest_pin):
    m = rt.pinned(data/'manifest.json', manifest_pin, 2 << 20)
    require(m['schema'] == 'world_reward.hoi_detr_mmcv_inputs.v1' and m['source']['all_original_blobs_authenticated'] is True, 'Complete source-only acquisition manifest required')
    rows = m['artifacts']; expected = {data/r['file'] for r in rows} | {data/'manifest.json'}
    require(len(rows) == len({r['file'] for r in rows}) and all(r['file'].startswith(('source/mmcv/', 'wheels/', 'notices/')) for r in rows), 'Closed numerical source/dependency inventory required')
    actual = set()
    for p in (data, *data.rglob('*')):
        rt.canonical(p); s = p.lstat(); require(not s.st_mode & 0o222 and (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)), 'Readonly source-only inputs required')
        if p.is_file(): actual.add(p)
    require(actual == expected, 'No foreign source/model/RGB artifact permitted')
    for r in rows:
        require(rt.identity(data/r['file'], 16 << 20, empty=r['bytes'] == 0) == {k: r[k] for k in ('bytes', 'sha256')}, 'Original source/dependency posthash differs')
    return m


def command(args, deadline, *, log=None):
    check(deadline)
    if log is None:
        r = subprocess.run(args, env=SAFE_ENV, capture_output=True, timeout=min(15, max(.01, deadline-time.monotonic())), check=False)
        require(r.returncode == 0 and len(r.stdout) <= 32768 and len(r.stderr) <= 32768, 'Bounded native lifecycle control failed')
        return r.stdout.decode().rstrip('\n')
    # BuildKit treats a local full image ID in FROM as a registry tag. The
    # measured classic builder resolves that exact existing ID without aliases
    # or remote pulls; this changes image packaging only, never native math.
    build_env = dict(SAFE_ENV, DOCKER_BUILDKIT='0') if args[:2] == ['docker', 'build'] else SAFE_ENV
    with log.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o400)
        r = subprocess.run(args, env=build_env, stdout=stream, stderr=subprocess.STDOUT, timeout=max(.01, deadline-time.monotonic()), check=False)
    require(log.stat().st_size <= 16 << 20 and r.returncode == 0, 'Native build/operator process failed')


def image(name, deadline):
    raw = command(['docker', 'image', 'inspect', name, '--format', '{{json .Id}} {{json .Architecture}} {{json .Os}} {{json .RootFS}} {{json .Config.Labels}}'], deadline)
    decoder = json.JSONDecoder(); values = []
    while raw.strip():
        v, end = decoder.raw_decode(raw.lstrip()); values.append(v); raw = raw.lstrip()[end:]
    require(len(values) == 5 and re.fullmatch('sha256:[0-9a-f]{64}', values[0]) and values[1:3] == ['amd64', 'linux'] and values[3]['Type'] == 'layers'
            and all(re.fullmatch('sha256:[0-9a-f]{64}', v) for v in values[3]['Layers']), 'Actual Linux AMD64 image identity required')
    return dict(Id=values[0], Architecture=values[1], Os=values[2], RootFS=values[3], Labels=values[4] or {})


def absent(name, deadline, *, image_name=False):
    require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Container namespace occupied') if not image_name else None
    if image_name:
        r = subprocess.run(['docker', 'image', 'inspect', name, '--format', '{{.Id}}'], env=SAFE_ENV, capture_output=True, timeout=min(10, max(.01, deadline-time.monotonic())), check=False)
        qualified = name if ':' in name.rsplit('/', 1)[-1] else name+':latest'
        errors = (f'Error: No such image: {name}\n'.encode(), f'\nError: No such image: {name}\n'.encode(),
                  f'Error response from daemon: No such image: {qualified}\n'.encode())
        require(r.returncode == 1 and r.stdout in (b'', b'\n') and r.stderr in errors, 'Target must genuinely be absent; never retagged')


def container_plan(code, data, out, image_id, revision, *, native, p, proof_pin, deadline):
    name = 'world-reward-hoi-mmcv-'+('ops-' if native else 'build-')+revision[:12]
    cidfile = out/('ops.cid' if native else 'build.cid')
    mounts = [(code.parent, code.parent, True), (data, data, True), (out, out, False)]
    args = ['docker', 'create', '--name', name, '--cidfile', str(cidfile), '--label', 'world-reward.job='+ENTRY, '--label', 'world-reward.revision='+revision,
            '--network', 'none', '--user', '1000:1000', '--memory', p['gpu_memory'] if native else p['cpu_memory'], '--cpus', str(p['gpu_cpus'] if native else p['build_cpus']), '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--tmpfs', '/tmp:rw,nosuid,size=2g', '--entrypoint', '/usr/bin/env']
    if native: args += ['--gpus', 'driver=nvidia,count=all']
    for src, dst, ro in mounts:
        args += ['--mount', f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    args += [image_id, '-i', 'PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/bin:/bin', 'HOME=/tmp', 'PYTHONDONTWRITEBYTECODE=1',
             'WR_ROOT='+str(ROOT), 'WR_CODE='+str(code), 'WR_CODE_REVISION='+revision, 'WR_RUNTIME_DEADLINE='+format(deadline, '.17g'),
             'CUDA_VISIBLE_DEVICES=0' if native else 'CUDA_VISIBLE_DEVICES=-1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
             '/opt/conda/bin/python', '-I', '-B', str(code/HELPERS[0]), '--native' if native else '--compile', '--proof-bytes', str(proof_pin['bytes']), '--proof-sha256', proof_pin['sha256']]
    return name, cidfile, mounts, args


def validate_container(value, *, image_id, name, revision, mounts, native):
    require(value['Image'] == image_id and value['Name'] == '/'+name and value['Config']['User'] == '1000:1000'
            and value['Config']['Labels'].get('world-reward.job') == ENTRY and value['Config']['Labels'].get('world-reward.revision') == revision
            and value['HostConfig']['NetworkMode'] == 'none' and value['HostConfig']['ReadonlyRootfs'] is True and not value['HostConfig']['Privileged'] and value['HostConfig'].get('CapDrop') == ['ALL']
            and 'no-new-privileges' in (value['HostConfig'].get('SecurityOpt') or []), 'Exact isolated owner/image/sandbox required')
    actual = {(m['Source'], m['Destination'], not m['RW']) for m in value['Mounts'] if m['Type'] == 'bind'}
    require(actual == {(str(a), str(b), c) for a, b, c in mounts} and not any(m['Type'] == 'volume' for m in value['Mounts']), 'Only public source/code/fresh output mounts allowed')
    devices = value['HostConfig'].get('DeviceRequests') or []
    require(bool(devices) is native and not value['HostConfig'].get('Devices') and not value['HostConfig'].get('Binds')
            and not value['HostConfig'].get('VolumesFrom'), 'GPU request only for genuine native operators')
    if native:
        require(len(devices) == 1 and devices[0].get('Driver') == 'nvidia' and devices[0].get('Count') == -1
                and devices[0].get('Capabilities') == [['gpu']], 'Only explicit native GPU request required')


def cleanup(cidfile, name, revision, image_id, deadline):
    require(cidfile.exists(), 'Owned CID receipt missing')
    s = cidfile.lstat(); require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == os.getuid() and s.st_size <= 65, 'Owned regular CID required')
    raw = cidfile.read_bytes(); require(re.fullmatch(b'[0-9a-f]{64}\n?', raw), 'Exact container ID required'); cid = raw.decode().strip()
    ids = command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline)
    require(ids in ('', cid), 'Ambiguous container identity')
    if ids:
        actual = command(['docker', 'inspect', cid, '--format', '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'], deadline)
        require(actual == image_id+'|/'+name+'|'+ENTRY+'|'+revision, 'Foreign container never removed')
        require(command(['docker', 'rm', '-f', cid], deadline) == cid, 'Owned container removal failed')
    require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline)
            and not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Independent owned container absence required')
    cidfile.chmod(0o444)


def native_proof(code, revision, p, proof_pin, *, native):
    rt, mp, acq = helpers(code); deadline = float(os.environ['WR_RUNTIME_DEADLINE']); check(deadline)
    require(os.geteuid() == 1000 and os.environ.get('WR_ROOT') == str(ROOT) and {x.name for x in Path('/sys/class/net').iterdir()} == {'lo'}, 'Unprivileged offline native container required')
    own = source(rt, code, revision); out = rt.canonical(ROOT/p['output']); proof = rt.pinned(out/('operators_proof.json' if native else 'compile_proof.json'), proof_pin, 2 << 20)
    require(proof['source_binding'] == own and proof['protocol_identity'] == PROTOCOL_PIN and proof['base_image']['Id'] == p['base_image_id'], 'Parent source/base proof differs')
    cidfile = out/('ops.cid' if native else 'build.cid'); t = cidfile.lstat()
    require(stat.S_ISREG(t.st_mode) and t.st_nlink == 1 and t.st_uid == 0 and re.fullmatch(b'[0-9a-f]{64}\n?', cidfile.read_bytes()), 'Parent owned CID must precede native imports')
    m = data_proof(rt, DATA, proof['inputs_manifest_identity'])
    require(proof['inputs_source'] == m['source'] and proof['inputs_source']['git_root_tree_sha1'] == p['source']['git_root_tree_sha1'], 'Original full source proof differs')
    return rt, mp, acq, out, proof, m, deadline


def versions(p):
    import importlib.metadata as metadata
    import sysconfig
    import numpy as np
    import torch
    require(sys.version.split()[0] == p['platform']['python'] and sysconfig.get_config_var('SOABI') == p['platform']['SOABI'] and torch.__version__ == p['platform']['torch'] and np.__version__ == p['platform']['numpy'], 'Original CP311/Torch/NumPy ABI required')
    found = {n: metadata.version(n) for n in p['base_distributions']}
    require(found == p['base_distributions'] and Path(sysconfig.get_path('include'), 'Python.h').is_file(), 'Frozen unchanged base dependencies/headers required')
    return torch, np, dict(python=sys.version.split()[0], SOABI=sysconfig.get_config_var('SOABI'), torch=torch.__version__, numpy=np.__version__, base_distributions=found, cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))


def compile_native(code, revision, p, proof_pin):
    started = time.monotonic(); rt, mp, acq, out, proof, m, deadline = native_proof(code, revision, p, proof_pin, native=False)
    torch, _, v = versions(p); require(not torch.cuda.is_initialized() and not torch.cuda.is_available(), 'CPU build must not execute CUDA')
    source_dir = out/'.scratch/source'; site = out/'.scratch/site'; site.mkdir(mode=0o700)
    original = {r['file'].removeprefix('source/mmcv/'): r for r in m['artifacts'] if r['file'].startswith('source/mmcv/')}
    source_dir.mkdir(mode=0o700)
    for n, r in original.items():
        check(deadline); target = source_dir/n; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(DATA/r['file'], target)
        require(rt.identity(target, 500000, readonly=False, empty=r['bytes'] == 0) == {k: r[k] for k in ('bytes', 'sha256')}, 'Build copy differs from original source')
    compiler = {}
    for name, key in (('gcc', 'gcc_path'), ('g++', 'gxx_path')):
        r = subprocess.run([p['compiler'][key], '-dumpfullversion', '-dumpversion'], capture_output=True, timeout=5, check=True)
        require(r.stdout.decode().strip() == p['compiler'][name], 'Frozen native host compiler differs'); compiler[name] = r.stdout.decode().strip()
    r = subprocess.run([p['compiler']['nvcc_path'], '--version'], capture_output=True, timeout=5, check=True)
    require(('V'+p['compiler']['nvcc']) in r.stdout.decode() and p['compiler']['cuda_home'] == '/usr/local/cuda', 'Exact CUDA12.4 compiler required'); compiler['nvcc'] = p['compiler']['nvcc']
    env = dict(PATH='/opt/conda/bin:/usr/local/cuda/bin:/usr/bin:/bin', HOME='/tmp', CUDA_VISIBLE_DEVICES='-1', CUDA_HOME=p['compiler']['cuda_home'], CC=p['compiler']['gcc_path'], CXX=p['compiler']['gxx_path'], TORCH_CUDA_ARCH_LIST=p['compiler']['torch_cuda_arch_list'],
               MMCV_WITH_OPS='1', FORCE_CUDA='1', MMCV_WITH_TRT='0', MMCV_WITH_ORT='0', MAX_JOBS=str(p['maximum_jobs']), PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(site), PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    wheels = [str(DATA/'wheels'/r['filename']) for r in p['wheels']]
    for args in ([sys.executable, '-I', '-B', '-m', 'pip', '--isolated', 'install', '--no-index', '--no-deps', '--no-cache-dir', '--no-compile', '--target', str(site), *wheels],
                 [sys.executable, '-B', 'setup.py', 'build_ext', '--inplace']):
        check(deadline); r = subprocess.run(args, cwd=source_dir, env=env, timeout=max(.01, deadline-time.monotonic()), check=False)
        require(r.returncode == 0, 'Original full native extension build failed; no retry/patch')
    for n, r in original.items():
        require(rt.identity(source_dir/n, 500000, readonly=False, empty=r['bytes'] == 0) == {k: r[k] for k in ('bytes', 'sha256')}, 'Native build modified original numerical/build source')
    ext = list((source_dir/'mmcv').glob('_ext*.so')); require(len(ext) == 1, 'Exactly one full native MMCV extension required')
    for path in (source_dir, site):
        for q in sorted(path.rglob('*'), reverse=True):
            require(not q.is_symlink() and (q.is_dir() or q.is_file()), 'No generated build/runtime symlinks')
    require(not torch.cuda.is_initialized(), 'CPU build never initialized CUDA')
    value = dict(stage='hoi_detr_mmcv_native_compile', status='pass', producer_revision=revision, source_binding=proof['source_binding'], inputs_manifest_identity=proof['inputs_manifest_identity'], versions=v, compiler=compiler,
                 native_source_unchanged=True, cuda_initialized=False, full_native_extension=True, extension=rt.identity(ext[0], 1 << 30, readonly=False), extension_filename=ext[0].name, elapsed_seconds=time.monotonic()-started,
                 **{k: False for k in FLAGS})
    acq.write_receipt(out/'compile.json', value, proof['started_monotonic'], deadline); require(value['status'] == 'pass', 'Native compile receipt late/failure')
    print(json.dumps(dict(stage=value['stage'], status=value['status'])))


def operator_cases(torch, ops, p):
    """Procedural math controls; no learned model/data or semantic prediction."""
    c = p['operators']; torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and list(torch.cuda.get_device_capability()) == p['platform']['gpu_capability'], 'Actual single H100 SM90 required')
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
    a = c['MSDeformAttn']; shapes = torch.tensor(a['spatial_shapes'], dtype=torch.int64, device='cuda'); starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
    n, heads, dims, q, levels, points = (a[k] for k in ('batch', 'heads', 'head_dimension', 'queries', 'levels', 'points'))
    value = torch.rand(n, int(shapes.prod(1).sum()), heads, dims, device='cuda', dtype=torch.float32)*.01
    sampling = torch.rand(n, q, heads, levels, points, 2, device='cuda', dtype=torch.float32); weights = torch.rand(n, q, heads, levels, points, device='cuda', dtype=torch.float32)+1e-5
    weights /= weights.sum((-1, -2), keepdim=True)
    expected = multi_scale_deformable_attn_pytorch(value, shapes, sampling, weights)
    result = MultiScaleDeformableAttnFunction.apply(value, shapes, starts, sampling, weights, a['im2col_step']); torch.cuda.synchronize()
    require(result.is_cuda and result.dtype == torch.float32 and torch.isfinite(result).all().item() and torch.allclose(result, expected, rtol=a['rtol'], atol=a['atol']), 'Native CUDA MSDeformAttn disagrees with original PyTorch reference')
    records = dict(MSDeformAttn=dict(device='cuda', shape=list(result.shape), max_abs_error=float((result-expected).abs().max()), original_reference=True, native_cuda_function=True))
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.], [4., 4., 6., 6.]], device='cuda'); scores = torch.tensor([.9, .8, .7], device='cuda')
    _, idx = ops.nms(boxes, scores, c['NMS']['iou_threshold'], offset=c['NMS']['offset'])
    require(idx.is_cuda and idx.tolist() == c['NMS']['expected_indices'], 'Native CUDA NMS indices differ')
    records['NMS'] = dict(device='cuda', indices=idx.tolist(), exact=True)
    dets, idx = ops.soft_nms(boxes, scores, iou_threshold=c['softNMS']['iou_threshold'], method=c['softNMS']['method'], min_score=c['softNMS']['min_score'], offset=c['softNMS']['offset'])
    # Original soft_nms executes its extension on CPU then returns to the
    # input device. CUDA output is native transfer semantics, not a CUDA backend.
    require(dets.is_cuda and idx.is_cuda and idx.tolist() == c['softNMS']['expected_indices'] and torch.equal(dets[:, :4], boxes[idx]) and torch.allclose(dets[:, 4], scores[idx]), 'Original CPU soft-NMS decay/removal/indices differ')
    records['softNMS'] = dict(device='cuda', native_backend_device='cpu', native_cpu_only=True, input_device='cuda', output_device='cuda', indices=idx.tolist(), scores=dets[:, 4].tolist(), identical_overlap_score_decayed_to_zero=True)
    r = c['RoIAlign']; x = torch.arange(16, dtype=torch.float32, device='cuda').reshape(1, 1, 4, 4); rois = torch.tensor([[0., 0., 0., 4., 4.]], device='cuda')
    y = ops.roi_align(x, rois, r['output_size'], r['spatial_scale'], r['sampling_ratio'], r['pool_mode'], r['aligned']); target = torch.tensor([[[[2.5, 4.5], [10.5, 12.5]]]], device='cuda')
    require(y.is_cuda and y.dtype == torch.float32 and torch.allclose(y, target, rtol=0, atol=r['atol']), 'Native CUDA RoIAlign analytic grid differs')
    records['RoIAlign'] = dict(device='cuda', shape=list(y.shape), max_abs_error=float((y-target).abs().max()), analytic_reference=True)
    return records


def native_ops(code, revision, p, proof_pin):
    started = time.monotonic(); rt, _, acq, out, proof, _, deadline = native_proof(code, revision, p, proof_pin, native=True)
    root = Path('/opt/world-reward-hoi-mmcv'); runtime_pin = rt.pinned(out/'runtime_manifest.json', proof['runtime_manifest_identity'], 2 << 20)
    for r in runtime_pin['artifacts']:
        require(rt.identity(root/r['file'], 1 << 30, empty=r['bytes'] == 0) == {k: r[k] for k in ('bytes', 'sha256')}, 'Original built runtime differs before ops')
    sys.path[:0] = [str(root/'site'), str(root/'source')]
    torch, np, v = versions(p)
    import importlib.metadata as metadata
    require({r['name']: metadata.version(r['name']) for r in p['wheels']} == {r['name']: r['version'] for r in p['wheels']}, 'Exact added dependency versions required')
    import mmcv
    from mmcv import ops
    import mmcv._ext as ext
    require(mmcv.__version__ == '1.7.2' and Path(mmcv.__file__).is_relative_to(root/'source') and Path(ext.__file__).name == proof['extension_filename']
            and rt.identity(Path(ext.__file__), 1 << 30) == proof['extension_identity'], 'Exactly built native MMCV extension required')
    require(not any(n == 'mmdet' or n.startswith(('mmdet.', 'projects.')) for n in sys.modules), 'No HOI/model/dataset imports in operator gate')
    records = operator_cases(torch, ops, p); check(deadline)
    for r in runtime_pin['artifacts']:
        require(rt.identity(root/r['file'], 1 << 30, empty=r['bytes'] == 0) == {k: r[k] for k in ('bytes', 'sha256')}, 'Built runtime changed after native operators')
    data_proof(rt, DATA, proof['inputs_manifest_identity']); require(source(rt, code, revision) == proof['source_binding'], 'Source changed after operators')
    value = dict(stage='hoi_detr_mmcv_native_operators', status='pass', producer_revision=revision, protocol_identity=PROTOCOL_PIN, source_binding=proof['source_binding'], inputs_manifest_identity=proof['inputs_manifest_identity'], runtime_manifest_identity=proof['runtime_manifest_identity'],
                 versions=v, extension_identity=proof['extension_identity'], native_source_unchanged=True, operators=records, fp32=True, amp=False, tf32=False, elapsed_seconds=time.monotonic()-started, **{k: False for k in FLAGS})
    acq.write_receipt(out/'operators.json', value, proof['started_monotonic'], deadline); require(value['status'] == 'pass', 'Native operators late/failure')
    print(json.dumps(dict(stage=value['stage'], status=value['status'])))


def remove_owned_folder(rt, folder, inode, out):
    t = folder.lstat()
    require(rt.canonical(folder) == folder and folder.parent == out and (t.st_dev, t.st_ino, t.st_uid) == inode, 'Only original owned scratch removed')
    for q in folder.rglob('*'):
        require(not q.is_symlink(), 'Scratch symlink replacement rejected')
        if q.is_dir(): q.chmod(0o700)
    folder.chmod(0o700); shutil.rmtree(folder)


def seal_inputs(data, directories, artifacts):
    expected = {data/r['file'] for r in artifacts}
    for p in sorted(data.rglob('*'), reverse=True):
        s = p.lstat(); require(not p.is_symlink() and (p in directories if p.is_dir() else stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and p in expected), 'Foreign acquisition namespace')
        p.chmod(0o555 if p.is_dir() else 0o444)
    data.chmod(0o555)


def make_context(rt, out, compile_report, deadline, owned_folders):
    context = out/'.image_context'; context.mkdir(mode=0o700); t = context.lstat(); owned_folders.append((context, (t.st_dev, t.st_ino, t.st_uid)));  runtime = context/'runtime'; runtime.mkdir(mode=0o700)
    for src, dst in ((out/'.scratch/source/mmcv', runtime/'source/mmcv'), (out/'.scratch/site', runtime/'site')):
        shutil.copytree(src, dst)
    # Original native notices remain with the reusable numerical runtime.
    notices = runtime/'notices'; notices.mkdir(mode=0o700)
    for p in sorted((DATA/'source/mmcv').rglob('*')):
        if p.is_file() and p.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING')):
            dst = notices/'source'/p.relative_to(DATA/'source/mmcv'); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dst)
    shutil.copytree(DATA/'notices', notices/'dependencies')
    records = []
    for p in sorted(runtime.rglob('*'), reverse=True):
        check(deadline); require(not p.is_symlink() and (p.is_dir() or p.is_file()), 'Original runtime contains generated link/special')
        p.chmod(0o555 if p.is_dir() else 0o444)
        if p.is_file(): records.append(dict(file=str(p.relative_to(runtime)), **rt.identity(p, 1 << 30, empty=True)))
    runtime.chmod(0o555)
    value = dict(schema='world_reward.hoi_detr_built_runtime.v1', extension=compile_report['extension'], artifacts=records)
    return context, value


def run(code, revision, *, opener=None):
    import fcntl
    started = time.monotonic(); rt, mp, acq = helpers(code); code = rt.canonical(code); own = source(rt, code, revision); p = protocol(rt, code); deadline = started+p['budget_seconds']
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02' and os.environ.get('WR_ROOT') == str(ROOT), 'Exact Azure root lifecycle required')
    data = rt.canonical(DATA); out = rt.canonical(ROOT/p['output'])
    require(data.parent.is_dir() and out.parent.is_dir() and not data.exists() and not out.exists() and shutil.disk_usage(data.parent).free >= p['minimum_free_bytes'], 'Fresh namespaces/16GiB free required')
    base = image(p['base_image_id'], deadline)
    require(base['Id'] == p['base_image_id'] and len(base['RootFS']['Layers']) == p['base_layers'] and hashlib.sha256(json.dumps(base['RootFS']['Layers'], separators=(',', ':')).encode()).hexdigest() == p['base_ordered_rootfs_sha256'], 'Original complete base rootfs required')
    absent(p['target_image'], deadline, image_name=True)
    data.mkdir(mode=0o700); out.mkdir(mode=0o755); os.chown(out, 1000, 1000)
    data_stat, out_stat = data.lstat(), out.lstat(); namespaces = [(data, (data_stat.st_dev, data_stat.st_ino, data_stat.st_uid)), (out, (out_stat.st_dev, out_stat.st_ino, out_stat.st_uid))]
    scratch = out/'.scratch'; scratch.mkdir(mode=0o700); os.chown(scratch, 1000, 1000); t = scratch.lstat(); owned_folders = [(scratch, (t.st_dev, t.st_ino, t.st_uid))]
    owned, artifacts, dirs, containers = [], [], set(), []; failure = None; child = None; lock_fd = None; archive_pin = None
    r = dict(stage='hoi_detr_runtime_operator_qualification', status='fail', phase='preflight', producer_revision=revision, source_binding=own, protocol_identity=PROTOCOL_PIN, base_image=base, budget_seconds=p['budget_seconds'], cleanup_grace_seconds=p['cleanup_grace_seconds'],
             source_rehashed_after=False, inputs_rehashed_after=False, base_unchanged=False, owned_containers_removed=False, owned_scratch_removed=False, native_extension_qualified=False, **{k: False for k in FLAGS})
    try:
        opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        r['phase'] = 'original_mmcv_source'
        archive_pin = fetch(mp, acq, data, dict(p['source'], file='.archive/mmcv.tar.gz'), deadline, owned, dirs, opener)
        r['source_archive_identity'] = dict(bytes=archive_pin['bytes'], sha256=archive_pin['sha256'], hash_basis='measured_archive_sha256_not_independent')
        artifacts, original = unpack(rt, mp, acq, data, data/archive_pin['file'], p['source'], deadline, owned, dirs, artifacts)
        require(rt.identity(data/archive_pin['file'], 64 << 20) == {k: archive_pin[k] for k in ('bytes', 'sha256')}, 'Owned source archive changed')
        (data/archive_pin['file']).unlink(); archive_pin = None
        r['phase'] = 'minimal_public_dependencies'; notices = {}
        for row in p['wheels']:
            license_row = dict(row['publisher_license'], file='notices/'+row['name']+'.LICENSE')
            artifacts.append(fetch(mp, acq, data, license_row, deadline, owned, dirs, opener))
            wheel = fetch(mp, acq, data, dict(row, file='wheels/'+row['filename']), deadline, owned, dirs, opener)
            artifacts.append(wheel); notices[row['name']] = wheel_notice(data/wheel['file'], row, (data/license_row['file']).read_bytes())
        m = dict(schema='world_reward.hoi_detr_mmcv_inputs.v1', source=original, artifacts=artifacts, wheel_notices=notices)
        raw = (json.dumps(m, sort_keys=True)+'\n').encode(); require(len(raw) < 2 << 20, 'Source/dependency manifest cap')
        acq.publish_bytes(mp, data, data/'manifest.json', raw, owned, dirs); manifest_pin = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        seal_inputs(data, dirs, artifacts+[dict(file='manifest.json')]); data_proof(rt, data, manifest_pin)
        proof = dict(source_binding=own, protocol_identity=PROTOCOL_PIN, base_image=base, inputs_manifest_identity=manifest_pin, inputs_source=original, started_monotonic=started)
        rt.write(out/'compile_proof.json', (json.dumps(proof, sort_keys=True)+'\n').encode(), 0o444); proof_pin = rt.identity(out/'compile_proof.json', 2 << 20)
        r.update(inputs_manifest_identity=manifest_pin, source=original, dependency_notices=notices, phase='cpu_native_full_extension_build')
        name, cidfile, mounts, args = container_plan(code, data, out, base['Id'], revision, native=False, p=p, proof_pin=proof_pin, deadline=deadline)
        absent(name, deadline); containers.append((cidfile, name, base['Id'])); command(args, deadline)
        value = json.loads(command(['docker', 'inspect', name, '--format', '{{json .}}'], deadline)); validate_container(value, image_id=base['Id'], name=name, revision=revision, mounts=mounts, native=False)
        command(['docker', 'start', '-a', name], deadline, log=out/'compile.log'); cleanup(cidfile, name, revision, base['Id'], deadline); containers.pop()
        require(rt.identity(out/'compile_proof.json', 2 << 20) == proof_pin, 'Published original compile proof changed')
        compiled = rt.strict((out/'compile.json').read_bytes())
        require(compiled['status'] == 'pass' and compiled['source_binding'] == own and compiled['native_source_unchanged'] is True and compiled['cuda_initialized'] is False and all(compiled[k] is False for k in FLAGS), 'Actual original compile PASS required')
        context, runtime = make_context(rt, out, compiled, deadline, owned_folders)
        rt.write(out/'runtime_manifest.json', (json.dumps(runtime, sort_keys=True)+'\n').encode(), 0o444); runtime_pin = rt.identity(out/'runtime_manifest.json', 2 << 20)
        dockerfile = ('FROM '+base['Id']+'\nCOPY runtime /opt/world-reward-hoi-mmcv\n').encode(); rt.write(context/'Dockerfile', dockerfile, 0o444)
        command(['docker', 'build', '--pull=false', '--network', 'none', '--label', 'world-reward.job='+ENTRY, '--label', 'world-reward.revision='+revision, '--label', 'world-reward.source='+own['closure_sha256'], '--label', 'world-reward.protocol='+PROTOCOL_PIN['sha256'], '--tag', p['target_image'], '--file', str(context/'Dockerfile'), str(context)], deadline, log=out/'image.log')
        child = image(p['target_image'], deadline)
        require(child['Id'] != base['Id'] and child['RootFS']['Layers'][:p['base_layers']] == base['RootFS']['Layers'] and len(child['RootFS']['Layers']) > p['base_layers']
                and child['Labels'].get('world-reward.job') == ENTRY and child['Labels'].get('world-reward.revision') == revision and child['Labels'].get('world-reward.source') == own['closure_sha256'], 'New owned child must preserve all original base layers')
        proof.update(runtime_manifest_identity=runtime_pin, extension_identity=compiled['extension'], extension_filename=compiled['extension_filename'])
        # New immutable proof is separately named, never overwrite published bytes.
        rt.write(out/'operators_proof.json', (json.dumps(proof, sort_keys=True)+'\n').encode(), 0o444)
        proof_pin = rt.identity(out/'operators_proof.json', 2 << 20)
        lock = rt.canonical(ROOT/'jobs/.world-reward-h100.lock'); s = lock.lstat(); require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Existing cooperative H100 lock required')
        lock_fd = os.open(lock, os.O_RDONLY|os.O_NOFOLLOW); require((os.fstat(lock_fd).st_dev, os.fstat(lock_fd).st_ino) == (s.st_dev, s.st_ino), 'Original lock inode required'); fcntl.flock(lock_fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], deadline), 'No duplicate GPU jobs permitted')
        r['phase'] = 'native_cuda_operators'
        name, cidfile, mounts, args = container_plan(code, data, out, child['Id'], revision, native=True, p=p, proof_pin=proof_pin, deadline=deadline)
        absent(name, deadline); containers.append((cidfile, name, child['Id'])); command(args, deadline)
        value = json.loads(command(['docker', 'inspect', name, '--format', '{{json .}}'], deadline)); validate_container(value, image_id=child['Id'], name=name, revision=revision, mounts=mounts, native=True)
        command(['docker', 'start', '-a', name], deadline, log=out/'operators.log'); cleanup(cidfile, name, revision, child['Id'], deadline); containers.pop()
        require(rt.identity(out/'operators_proof.json', 2 << 20) == proof_pin and rt.identity(out/'runtime_manifest.json', 2 << 20) == runtime_pin, 'Published operator/runtime proof changed')
        native = rt.strict((out/'operators.json').read_bytes())
        require(native['status'] == 'pass' and native['source_binding'] == own and native['runtime_manifest_identity'] == runtime_pin and native['extension_identity'] == compiled['extension'] and native['native_source_unchanged'] is True
                and set(native['operators']) == {'MSDeformAttn', 'NMS', 'softNMS', 'RoIAlign'} and all(native[k] is False for k in FLAGS), 'Complete genuine native operator receipt required')
        require(not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], deadline), 'GPU process survived owned cleanup')
        r.update(phase='complete', status='pass', child_image=child, compile_report_identity=rt.identity(out/'compile.json', 32768), operator_report_identity=rt.identity(out/'operators.json', 32768), runtime_manifest_identity=runtime_pin,
                 native_extension_identity=compiled['extension'], native_extension_qualified=True, operators=native['operators'], versions=native['versions'])
    except Exception as error:
        failure = error; r['error_type'] = type(error).__name__
    finally:
        signal.alarm(p['cleanup_grace_seconds'])
        try:
            grace = time.monotonic()+p['cleanup_grace_seconds']
            for cidfile, name, image_id in containers:
                if cidfile.exists():
                    cleanup(cidfile, name, revision, image_id, grace)
                else:
                    require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], grace), 'Ambiguous created container without CID; never delete foreign')
            r['owned_containers_removed'] = True
            for part, inode in owned:
                if part.exists() or part.is_symlink():
                    s = part.lstat(); require(not part.is_symlink() and s.st_nlink == 1 and (s.st_dev, s.st_ino, s.st_uid) == inode, 'Owned partial replaced'); part.unlink()
            if archive_pin is not None:
                require(rt.identity(data/archive_pin['file'], 64 << 20) == {k: archive_pin[k] for k in ('bytes', 'sha256')}, 'Owned archive replacement'); (data/archive_pin['file']).unlink()
            for folder, inode in namespaces:
                t = folder.lstat(); require(rt.canonical(folder) == folder and (t.st_dev, t.st_ino, t.st_uid) == inode, 'Owned namespace replaced; never clean foreign')
            for folder, inode in owned_folders:
                if folder.exists():
                    remove_owned_folder(rt, folder, inode, out)
            r['owned_scratch_removed'] = True
            if 'manifest_pin' in locals():
                data_proof(rt, data, manifest_pin); r['inputs_rehashed_after'] = True
            else:
                seal_inputs(data, dirs, artifacts)
            if child is not None:
                require(image(p['target_image'], grace) == child, 'Owned runtime image changed after operators')
            r['base_unchanged'] = image(p['base_image_id'], grace) == base
            r['source_rehashed_after'] = source(rt, code, revision) == own
            if failure and child is not None:
                require(image(child['Id'], grace) == child, 'Owned failed child image changed'); command(['docker', 'image', 'rm', '--no-prune', child['Id']], grace)
                r['failed_owned_image_removed'] = True
        except Exception as error:
            failure = failure or error; r['post_error_type'] = type(error).__name__
        finally:
            if lock_fd is not None: os.close(lock_fd)
            signal.alarm(0)
        r['elapsed_seconds'] = time.monotonic()-started
        if failure or r['elapsed_seconds'] > p['budget_seconds'] or not all(r[k] for k in ('source_rehashed_after', 'inputs_rehashed_after', 'base_unchanged', 'owned_containers_removed', 'owned_scratch_removed', 'native_extension_qualified')):
            r['status'] = 'fail'
        try:
            if r['status'] == 'pass':
                for n in ('compile.log', 'image.log', 'operators.log'):
                    log = out/n; t = log.lstat(); require(stat.S_ISREG(t.st_mode) and t.st_nlink == 1 and t.st_uid == 0, 'Only owned build logs removed'); log.unlink()
            # Seal namespace BEFORE fail-safe publication. Root can publish into
            # its own readonly directory; no post-PASS cleanup remains.
            out.chmod(0o555)
        except Exception as exc:
            r.update(status='fail', phase='final_sealing', final_error_type=type(exc).__name__)
        acq.write_receipt(out/'report.json', r, started, deadline)
    require(r['status'] == 'pass', 'Native MMCV qualification failed; sealed concise receipt retained')
    return r


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False); mode = parser.add_mutually_exclusive_group(); mode.add_argument('--compile', action='store_true'); mode.add_argument('--native', action='store_true')
    parser.add_argument('--proof-bytes', type=int); parser.add_argument('--proof-sha256'); args = parser.parse_args()
    code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']; rt, _, _ = helpers(code); p = protocol(rt, code)
    if args.compile or args.native:
        require(type(args.proof_bytes) is int and args.proof_bytes > 0 and re.fullmatch('[0-9a-f]{64}', args.proof_sha256 or ''), 'Independent parent proof pin required')
        pin = dict(bytes=args.proof_bytes, sha256=args.proof_sha256)
        function = compile_native if args.compile else native_ops
        try:
            function(code, revision, p, pin)
        except Exception as exc:
            out = ROOT/p['output']; name = 'compile.json' if args.compile else 'operators.json'
            if not (out/name).exists():
                failure = dict(stage='hoi_detr_mmcv_native_compile' if args.compile else 'hoi_detr_mmcv_native_operators', status='fail', phase='native_execution', error_type=type(exc).__name__, requirement=native_failure_requirement(exc), producer_revision=revision, **{k: False for k in FLAGS})
                _, _, acq = helpers(code); acq.write_receipt(out/name, failure, time.monotonic(), float(os.environ['WR_RUNTIME_DEADLINE']))
            raise
    else:
        value = run(code, revision); print(json.dumps({k: value[k] for k in ('stage', 'status', 'phase', 'elapsed_seconds')}))


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print(json.dumps(dict(stage='hoi_detr_runtime_operator_qualification', status='fail', error_type=type(error).__name__)))
        raise SystemExit(1) from None
