"""Build an isolated one-line exact raster optimization; qualify, never adopt.

With non-overflow coarse bins, atomicAdd reserves contiguous disjoint intervals
whose union is [0,hits). Same CUDA stream completes all writes before fine reads.
Therefore first -1 is the entire empty suffix: replacing continue by break cannot
change any valid face visitation/order/arithmetic. Overflow invalidates this proof
and is forbidden by the independently gated conservative AABB bound.

Acquire only 152 pinned source/license files on Azure, compile only _C into an
isolated child path, leave every original package/binary/layer unchanged. The
qualification swaps only the mesh-raster module's _C reference, not camera code,
geometry, projected vertices, resolution, evidence or model outputs. Not adopted.
"""
from __future__ import annotations

import argparse
import errno
import ssl
import hashlib
import importlib
import importlib.util
import concurrent.futures
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import socket
import time
import types
import urllib.error
import urllib.request

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_raster_prefix_runtime'
REVISION = '33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba'
BASE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
SITE = '/opt/world-reward/raster-prefix'
LABEL = 'world_reward.raster_prefix.owner'
HELPERS = ('infra/raster_prefix_runtime.py', 'infra/run_raster_prefix_runtime.sh',
    'infra/mediapipe_cpu_runtime_verify.py', 'infra/camera_render.py',
    'src/world_reward/raster_capacity.py', 'src/world_reward/__init__.py')
MANIFEST_SHA256 = 'e1ec10ebcf63594f7fc91a9e7722a34185b897fc5e04f9d634405d36fd20e085'
PATCH_PATH = 'pytorch3d/csrc/rasterize_meshes/rasterize_meshes.cu'
PATCH_SHA256 = '643212e2f4fda5adf9c0b97272e16e718cf66f04e753a9837dff5b7ffc50f39d'
OLD = b'        continue; // bin_faces uses -1 as a sentinal value.'
NEW = b'        break; // bin_faces uses -1 as a sentinal value.'
FIRST = ROOT/'results/form-hoi-external-predict-125aab2fbbe3a9b1d0f88bea8cb9f2fe17a719ed/2026-06-03_17-02-20_beige_bin_ground_desk_03'


def patch_source(raw):
    require(hashlib.sha256(raw).hexdigest() == PATCH_SHA256 and raw.count(OLD) == 1,
            'Exact pinned fine kernel and unique one-line patch required')
    changed = raw.replace(OLD, NEW)
    require(changed.replace(NEW, OLD) == raw, 'No other CUDA source change allowed')
    return changed


def source_manifest(tree):
    require(tree['sha'] == REVISION and tree.get('truncated') is False, 'Complete exact primary tree required')
    rows = [dict(path=x['path'], bytes=x['size'], git_blob_sha=x['sha']) for x in tree['tree']
        if x['type'] == 'blob' and (x['path'].startswith('pytorch3d/csrc/') or
                                   x['path'] in ('setup.py', 'LICENSE', 'pytorch3d/__init__.py'))]
    require(len(rows) == 152 and sum(x['bytes'] for x in rows) == 919588 and
        hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == MANIFEST_SHA256,
        'Frozen exact source-only build closure differs')
    return rows


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_a, **_kw): raise ValueError('Primary source redirect forbidden')


def acquire(url, limit, *, deadline=None):
    """Bounded primary-source GET; retry only transient transport/server errors."""
    tree_url = f'https://api.github.com/repos/facebookresearch/pytorch3d/git/trees/{REVISION}?recursive=1'
    prefix = f'https://raw.githubusercontent.com/facebookresearch/pytorch3d/{REVISION}/'
    suffix = url[len(prefix):] if url.startswith(prefix) else ''
    require(url == tree_url or (re.fullmatch(r'[A-Za-z0-9_./-]+', suffix) and
        '..' not in PurePosixPath(suffix).parts and not PurePosixPath(suffix).is_absolute() and
        (suffix.startswith('pytorch3d/csrc/') or suffix in ('setup.py', 'LICENSE', 'pytorch3d/__init__.py'))),
        'Exact primary pinned source-only endpoints required; no archives/data/redirects')
    require(type(limit) is int and 0 < limit <= 2 << 20, 'Per-file source bound required')
    deadline = time.monotonic()+60 if deadline is None else deadline
    for attempt in range(3):
        remaining = deadline-time.monotonic()
        require(remaining > 0, 'Inclusive pinned source acquisition deadline exceeded')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(urllib.request.Request(url, headers={'User-Agent': 'WorldReward-source-audit'}),
                             timeout=min(20., remaining)) as f:
                raw = f.read(limit+1)
            require(len(raw) <= limit, 'Pinned primary source exceeds exact per-file bound')
            return raw
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504): raise
        except urllib.error.URLError as error:
            reason = error.reason
            transient = isinstance(reason, (TimeoutError, socket.timeout, ConnectionError, ssl.SSLEOFError)) or (
                isinstance(reason, OSError) and reason.errno in (errno.ECONNRESET, errno.ECONNABORTED,
                    errno.ETIMEDOUT, errno.EPIPE, errno.EHOSTUNREACH, socket.EAI_AGAIN))
            if not transient: raise
        except (TimeoutError, socket.timeout, ConnectionError, http.client.IncompleteRead):
            pass
        if attempt == 2: raise ValueError('Transient primary source acquisition exhausted three bounded attempts')
        delay = min(float(attempt+1), max(0., deadline-time.monotonic()))
        time.sleep(delay)
    raise AssertionError('unreachable')


def acquire_selected(rows, destination, *, seconds=300):
    """Eight parallel exact tiny blobs; verify git SHA before exclusive write."""
    expected = {row['path']: row for row in rows}
    require(len(expected) == len(rows) and sum(x['bytes'] for x in rows) <= 919588,
        'Unique bounded source-only closure required')
    for name, row in expected.items():
        require(re.fullmatch(r'[A-Za-z0-9_./-]+', name) and '..' not in PurePosixPath(name).parts
            and not PurePosixPath(name).is_absolute() and
            (name.startswith('pytorch3d/csrc/') or name in ('setup.py', 'LICENSE', 'pytorch3d/__init__.py'))
            and type(row['bytes']) is int and 0 < row['bytes'] <= 2 << 20
            and re.fullmatch('[0-9a-f]{40}', row['git_blob_sha']), 'Safe exact publisher source blob required')
    deadline = time.monotonic()+seconds
    def fetch(row):
        raw = acquire(f'https://raw.githubusercontent.com/facebookresearch/pytorch3d/{REVISION}/'+row['path'],
                      row['bytes'], deadline=deadline)
        git_sha = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        require(len(raw) == row['bytes'] and git_sha == row['git_blob_sha'], 'Exact publisher git blob differs; never retry changed bytes')
        return row['path'], raw
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending = [pool.submit(fetch, row) for row in rows]
        try:
            for future in concurrent.futures.as_completed(pending, timeout=seconds):
                name, raw = future.result()
                require(time.monotonic() < deadline, 'Inclusive source acquisition deadline exceeded')
                path = destination/name; path.parent.mkdir(parents=True, exist_ok=True)
                require(canonical(path) == path, 'No symlink in owned source destination')
                with path.open('xb') as f:
                    f.write(raw); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o400)
        except BaseException:
            for future in pending: future.cancel()
            raise
    return dict(files=len(rows), bytes=sum(x['bytes'] for x in rows), concurrent_requests=8,
        per_request_max_attempts=3, retry_scope='transient_transport_HTTP408_429_5xx_only',
        source_blob_git_sha1_verified=True, publisher_archive_downloaded=False,
        elapsed_seconds=time.monotonic()-(deadline-seconds))


def seal(path, value):
    require(not path.exists(), 'Never overwrite immutable runtime result')
    with path.open('xb') as f:
        f.write((json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode())
        f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)


def docker(arguments, *, seconds, log=None):
    env = dict(PATH='/usr/bin:/bin', HOME='/nonexistent', LANG='C.UTF-8',
        DOCKER_HOST='unix://'+str(ROOT/'docker.sock'), DOCKER_BUILDKIT='0')
    cmd = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=15s', str(seconds)+'s', 'docker', *arguments]
    if log is not None:
        with log.open('xb') as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=seconds+20)
            f.flush(); os.fchmod(f.fileno(), 0o444)
        require(result.returncode == 0, 'Runtime command failed; bounded technical log stays Azure')
        return ''
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=seconds+20)
    require(result.returncode == 0 and len(result.stdout) < 8 << 20, 'Bounded Docker control failed')
    return result.stdout


def recipe():
    return (f'FROM {BASE}\nCOPY source /tmp/pytorch3d-source\n'
        'RUN cd /tmp/pytorch3d-source && CUDA_VISIBLE_DEVICES=-1 FORCE_CUDA=1 '
        'CUDA_HOME=/usr/local/cuda CUB_HOME=/usr/local/cuda/include '
        'TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 '
        f'/opt/conda/bin/python setup.py build_ext --build-lib {SITE} --build-temp /tmp/p3d-build '
        f'&& find {SITE} -type f -name "_C*.so" -exec chmod 444 {{}} \\; '
        '&& rm -rf /tmp/pytorch3d-source /tmp/p3d-build\n')


def preflight():
    """Base compiler/dependency/binary check; no CUDA initialization/inference."""
    import platform
    import numpy
    import torch
    import pytorch3d
    import pytorch3d._C
    import setuptools
    import ninja
    require(platform.machine() == 'x86_64' and sys.version_info[:2] == (3, 11)
        and torch.__version__ == '2.5.1+cu124' and pytorch3d.__version__ == '0.7.9'
        and not torch.cuda.is_initialized(), 'Existing qualified exact CPU/compiler runtime required')
    nvcc = subprocess.run(['/usr/local/cuda/bin/nvcc', '--version'], capture_output=True, text=True, timeout=10)
    require(nvcc.returncode == 0 and 'release 12.4' in nvcc.stdout, 'Existing native CUDA12.4 compiler required')
    for name in ('c++',): require(shutil.which(name) is not None, 'Existing native C++ compiler required')
    result = dict(torch=torch.__version__, numpy=numpy.__version__, pytorch3d=pytorch3d.__version__,
        original_binary=identity(Path(pytorch3d._C.__file__), 200 << 20, readonly=False),
        original_python=identity(Path(pytorch3d.__file__), 20000, readonly=False),
        preflight_seconds_budget=60, CUDA_initialized=False,
        native_compiler_available=True, extra_packages_installed=False)
    print(json.dumps(result), flush=True)


def _candidate_extension():
    """Import isolated extension without changing any original package file."""
    files = list((Path(SITE)/'pytorch3d').glob('_C*.so'))
    require(len(files) == 1, 'Exactly one isolated native raster binary required')
    namespace = types.ModuleType('wr_raster_prefix'); namespace.__path__ = [str(files[0].parent)]
    sys.modules['wr_raster_prefix'] = namespace
    spec = importlib.util.spec_from_file_location('wr_raster_prefix._C', files[0])
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, files[0]


def prefix_invariant(torch, original):
    """Actual native coarse bins must be nonnegative prefix + all -1 suffix."""
    f = torch.tensor([[[0, 0, 2], [.2, 0, 2], [0, .2, 2]],
        [[-.4, -.3, 3], [.3, -.2, 3], [.1, .4, 3]]], device='cuda', dtype=torch.float32)
    starts = torch.tensor([0], device='cuda', dtype=torch.int64)
    sizes = torch.tensor([2], device='cuda', dtype=torch.int64)
    bins = original._rasterize_meshes_coarse(f, starts, sizes, (96, 128), 0., 16, 2)
    negative = bins < 0
    require(not torch.any((negative.cumsum(-1) > 0) & ~negative), 'Actual coarse valid bins must have no prefix holes')
    return dict(actual_coarse_valid_prefix=True, capacity_nonoverflow=True, original_native_coarse_used=True)



def fixed_bin_kernel_gate(torch, original, candidate):
    """Same actual coarse prefix/order, fine outputs and one-pixel backward ABI.

    A one-pixel cotangent avoids unrelated atomic summation-order nondeterminism.
    This is a limited kernel/gradient compatibility probe, not training validation.
    """
    f = torch.tensor([[[0, 0, 2], [.25, 0, 2.1], [0, .25, 2.2]],
        [[-.2, -.15, 3], [.3, -.15, 3.1], [0, .3, 3.2]]], device='cuda', dtype=torch.float32)
    starts = torch.tensor([0], device='cuda', dtype=torch.int64)
    sizes = torch.tensor([2], device='cuda', dtype=torch.int64)
    neighbors = torch.full((2,), -1, device='cuda', dtype=torch.int64)
    bins = original._rasterize_meshes_coarse(f, starts, sizes, (96, 128), 0., 16, 2)
    args = (f, bins, neighbors, (96, 128), 0., 16, 1, True, False, False)
    a = original._rasterize_meshes_fine(*args); b = candidate._rasterize_meshes_fine(*args)
    require(all(torch.equal(x, y) for x, y in zip(a, b)), 'Same-bin fine pix/Z/bary/dist bit parity required')
    pix, z, bary, dist = a
    valid = torch.nonzero(pix >= 0, as_tuple=False)
    require(len(valid) > 0, 'Manufactured kernel probe must cover a pixel')
    selected = tuple(valid[len(valid)//2].tolist())
    gz = torch.zeros_like(z); gb = torch.zeros_like(bary); gd = torch.zeros_like(dist)
    gz[selected] = .7; gb[selected] = torch.tensor([.2, -.3, .4], device='cuda'); gd[selected] = .1
    gradients = [m.rasterize_meshes_backward(f, pix, gz, gb, gd, True, False) for m in (original, candidate)]
    require(torch.isfinite(gradients[0]).all() and torch.equal(*gradients),
        'Same-bin one-pixel native backward bit parity required')
    return dict(coarse_inputs_shared=True, fine_all_outputs_bit_equal=True,
        one_pixel_cotangent_backward_bit_equal=True, broad_training_validation=False)

def pair(torch, renderer, raster_module, original, candidate, name, vertices, faces, K, batch):
    outputs = {}; timings = {}; diagnostics = {}
    try:
        for label, module in (('original_capacity', original), ('prefix_break_capacity', candidate)):
            raster_module._C = module; info = {}
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
            if batch == 1:
                value = renderer.raster_camera_mesh(vertices[0], faces, K, full_face_capacity=False, capacity_diagnostics=info)
            else:
                value = renderer.raster_camera_mesh_batch(vertices[:batch], faces, K,
                    full_face_capacity=False, capacity_diagnostics=info)
            torch.cuda.synchronize()
            timings[label] = dict(seconds=time.perf_counter()-start, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
            outputs[label] = value; diagnostics[label] = info
        am, az = outputs['original_capacity']; bm, bz = outputs['prefix_break_capacity']
        equal = torch.equal(am, bm); close = torch.allclose(az, bz, atol=1e-6, rtol=0., equal_nan=True)
        require(equal and close and diagnostics['original_capacity'] == diagnostics['prefix_break_capacity'],
            'Exact geometry/capacity masks and camera-Z parity required')
        delta = float(torch.abs(az[am]-bz[am]).max()) if am.any() else 0.
        return dict(case=name, batch_size=batch, mesh_vertices=vertices.shape[1], mesh_faces=len(faces),
            exact_mask_parity=True, camera_z_max_error_m=delta, depth_values_bit_equal=bool(torch.equal(az[am], bz[am])),
            depth_atol_m=1e-6, depth_rtol=0.,
            image_size_hw=[1152, 1536], native_capacity=diagnostics['original_capacity'],
            synchronized_single_trial=timings,
            single_trial_speed_ratio=timings['original_capacity']['seconds']/timings['prefix_break_capacity']['seconds'])
    finally:
        raster_module._C = original


def qualify(code, revision, out):
    import numpy as np
    import torch
    import camera_render as renderer
    require({x.name for x in Path('/sys/class/net').iterdir()} == {'lo'} and torch.cuda.is_available(),
            'Offline Azure-only CUDA qualification required')
    raster_module = importlib.import_module('pytorch3d.renderer.mesh.rasterize_meshes')
    original = raster_module._C; candidate, binary = _candidate_extension()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    args = types.SimpleNamespace(root=ROOT, external_expected_faces=743576,
        external_object_report=FIRST/'object/report.json', external_object_report_bytes=9165,
        external_object_report_sha256='caf4c261e4370e0534c192aec93c3964929c11954079659098c09cadd4465904',
        external_body_report=FIRST/'body_depth/report.json', external_body_report_bytes=29676,
        external_body_report_sha256='0b864574dfec12553cf02d7d3981f3e0b624c8c62f901f259c4f63f6b382442a')
    report = dict(schema='world_reward.raster_prefix_qualification.v1', status='fail', producer_revision=revision,
        scope='inference_raster_allocation_only_not_HOI_gain_not_adopted', source_revision=REVISION,
        candidate_binary=identity(binary, 200 << 20), original_binary=identity(Path(original.__file__), 200 << 20, readonly=False),
        source_pins=renderer._raster_source_pins(), image_id=os.environ['WR_IMAGE_ID'], rows=[],
        geometry_deleted=False, grid_resized=False, model_or_RGB_loaded=False, truth_read=False, adopted=False)
    started = time.monotonic()
    try:
        report['prefix_invariant'] = prefix_invariant(torch, original)
        report['fixed_bin_kernel_gate'] = fixed_bin_kernel_gate(torch, original, candidate)
        posed, faces, K, provenance = renderer._load_external_capacity_geometry(args)
        report['external_geometry'] = provenance
        proc, pf, pk = renderer._procedural_capacity_cases()
        for name, vertices, topology, matrix in (('procedural_random_multiobject_occlusion', proc, pf, pk),
                                                ('actual_first_external_object', posed, faces, K)):
            for batch in (1, 4):
                row = pair(torch, renderer, raster_module, original, candidate, name, vertices, topology, matrix, batch)
                report['rows'].append(row)
                print(json.dumps(dict(case=name, batch_size=batch, parity=True)), flush=True)
        require(renderer._load_external_capacity_geometry(args)[3] == provenance, 'Exact inferred assets changed during gate')
        report['status'] = 'pass'; report['decision'] = 'INFERENCE_PREFIX_QUALIFIED_NOT_ADOPTED'
    except Exception as error:
        report['error_type'] = type(error).__name__
        raise
    finally:
        raster_module._C = original
        report['elapsed_seconds'] = time.monotonic()-started
        seal(out/'qualification.json', report)


def build(code, revision):
    binding = source(ROOT, code, revision, ENTRY, HELPERS); started = time.monotonic()
    out = ROOT/'results'/('raster-prefix-runtime-'+revision)
    require(not out.exists(), 'Fresh owned runtime output required'); out.mkdir(mode=0o755)
    tag = 'world-reward/raster-prefix:'+revision
    report = dict(schema='world_reward.raster_prefix_runtime.v1', status='fail', producer_revision=revision,
        base_image_id=BASE, target_image=tag, source_before=binding, one_line_only_patch=True,
        base_modified=False, packages_installed=False, models_loaded=False, reference_inputs_read=False,
        scientific_adoption=False, GPU_build=False)
    context = out/'context'
    try:
        require(not docker(['image', 'ls', '-q', '--no-trunc', tag], seconds=20).strip(), 'Never overwrite existing child')
        base = strict(docker(['image', 'inspect', BASE], seconds=20))[0]
        require(base['Id'] == BASE, 'Exact qualified base required')
        probe = docker(['run', '--rm', '--network=none', '--read-only', '--user', '0:0',
            '--tmpfs', '/tmp:rw,nosuid,size=256m', '--mount', f'type=bind,src={code.parent},dst={code.parent},readonly',
            '--entrypoint', '/usr/bin/env', BASE, '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin', 'HOME=/tmp',
            'CUDA_VISIBLE_DEVICES=-1', 'PYTHONDONTWRITEBYTECODE=1', f'PYTHONPATH={code}/infra:{code}/src',
            '/opt/conda/bin/python', '-B', str(code/'infra/raster_prefix_runtime.py'), 'preflight'], seconds=60)
        report['preflight'] = strict(probe)
        context.mkdir(mode=0o700); src = context/'source'; src.mkdir(mode=0o700)
        rows = source_manifest(strict(acquire(f'https://api.github.com/repos/facebookresearch/pytorch3d/git/trees/{REVISION}?recursive=1', 2 << 20)))
        report['source_acquisition'] = acquire_selected(rows, src)
        patch = src/PATCH_PATH; changed = patch_source(patch.read_bytes())
        patch.chmod(0o600); patch.write_bytes(changed); patch.chmod(0o400)
        report['publisher_source'] = dict(revision=REVISION, manifest_sha256=MANIFEST_SHA256,
            files=152, bytes=919588, license_pin=identity(src/'LICENSE', 20000),
            original_fine_sha256=PATCH_SHA256, patched_fine_sha256=hashlib.sha256(changed).hexdigest())
        (context/'Dockerfile').write_text(recipe())
        docker(['build', '--network=none', '--pull=false', '--label', LABEL+'='+revision, '-t', tag, str(context)],
            seconds=1500, log=out/'build.log')
        child = strict(docker(['image', 'inspect', tag], seconds=20))[0]
        require(child['Id'] != BASE and child['Config']['Labels'][LABEL] == revision and
            child['RootFS']['Layers'][:len(base['RootFS']['Layers'])] == base['RootFS']['Layers'],
            'New child must retain every original base layer unchanged')
        report['qualified_image_id'] = child['Id']
        name = 'wr-raster-prefix-qualify-'+revision; cid = out/'.container.cid'
        paths = [code.parent, FIRST/'object/report.json', FIRST/'object/object.glb',
            FIRST/'object/transform.json', FIRST/'body_depth/report.json', FIRST/'body_depth/gauge.json']
        mounts = []
        for path in paths:
            canonical(path); mounts += ['--mount', f'type=bind,src={path},dst={path},readonly']
        try:
            docker(['run', '--rm', '--name', name, '--cidfile', str(cid), '--label', LABEL+'='+revision,
                '--gpus', 'all', '--network=none', '--read-only', '--user', '0:0', '--cap-drop', 'ALL',
                '--security-opt', 'no-new-privileges', '--cpus', '4', '--memory', '64g', '--shm-size', '2g',
                '--tmpfs', '/tmp:rw,nosuid,exec,size=1g', *mounts, '--mount', f'type=bind,src={out},dst={out}',
                '--entrypoint', '/usr/bin/env', child['Id'], '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin', 'HOME=/tmp',
                'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=4', 'PYTHONDONTWRITEBYTECODE=1',
                f'WR_ROOT={ROOT}', f'WR_CODE={code}', f'WR_CODE_REVISION={revision}', f'WR_IMAGE_ID={child["Id"]}',
                f'PYTHONPATH={code}/src:{code}/infra', '/opt/conda/bin/python', '-B',
                str(code/'infra/raster_prefix_runtime.py'), 'qualify'], seconds=300, log=out/'qualification.log')
        finally:
            if cid.exists():
                actual = cid.read_text().strip(); require(re.fullmatch('[0-9a-f]{64}', actual), 'Exact owned CID required')
                active = docker(['ps', '-aq', '--no-trunc', '--filter', 'id='+actual], seconds=20).strip()
                if active:
                    record = strict(docker(['inspect', actual], seconds=20))[0]
                    require(record['Image'] == child['Id'] and record['Name'] == '/'+name and
                        record['Config']['Labels'][LABEL] == revision, 'Remove only owned runtime container')
                    docker(['rm', '-f', actual], seconds=20)
                require(not docker(['ps', '-aq', '--no-trunc', '--filter', 'id='+actual], seconds=20).strip(), 'Owned GPU container remains')
                cid.chmod(0o444)
            else:
                require(not docker(['ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], seconds=20).strip(),
                    'No owned GPU container may remain after failure without CID')
        qa = strict((out/'qualification.json').read_bytes())
        require(qa['status'] == 'pass' and qa['producer_revision'] == revision, 'Real prefix parity must pass')
        require(qa['original_binary'] == report['preflight']['original_binary'],
            'Isolated child must retain actual original _C binary unchanged')
        report['qualification_pin'] = identity(out/'qualification.json', 131072)
        require(strict(docker(['image', 'inspect', BASE], seconds=20))[0] == base, 'Base image changed')
        report['source_after'] = source(ROOT, code, revision, ENTRY, HELPERS)
        require(report['source_after'] == binding, 'Runtime source changed')
        report['status'] = 'pass'; report['decision'] = 'ISOLATED_PREFIX_RUNTIME_QUALIFIED_NOT_ADOPTED'
    except Exception as error:
        report['error_type'] = type(error).__name__
        if isinstance(error, ValueError): report['error_context'] = str(error)[:300]
        raise
    finally:
        # Retain concise source/license identities and Azure-only failure logs.
        # No publisher archive is fetched; remove the owned build workspace.
        if context.exists(): shutil.rmtree(context)
        report['elapsed_seconds'] = time.monotonic()-started
        seal(out/'report.json', report)
        print(json.dumps(dict(status=report['status'], stage='raster_prefix_runtime')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=('build', 'preflight', 'qualify'))
    args = parser.parse_args()
    if args.mode == 'preflight': preflight(); return
    code = canonical(Path(os.environ['WR_CODE'])); revision = os.environ['WR_CODE_REVISION']
    require(re.fullmatch('[0-9a-f]{40}', revision) and Path(os.environ['WR_ROOT']) == ROOT, 'Exact Azure runtime source required')
    if args.mode == 'build': build(code, revision)
    else: qualify(code, revision, ROOT/'results'/('raster-prefix-runtime-'+revision))


if __name__ == '__main__': main()
