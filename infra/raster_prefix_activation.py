"""Export one qualified binary on Azure CPU; opt-in scoped raster forward only.

Base packages/Python/geometry remain unchanged. Other operators and backward
remain original. Inference-only; caller requires nonoverflow AABB capacity.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import types

from mediapipe_cpu_runtime_verify import canonical, identity, pinned, require, strict

ROOT = Path('/srv/scenesmith/world-reward')
BASE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
FILENAME = '_C.cpython-311-x86_64-linux-gnu.so'
CHILD_FILE = '/opt/world-reward/raster-prefix/pytorch3d/'+FILENAME
LOCK = threading.RLock()


def qualification(report_path, pin, *, current_image):
    require(current_image == BASE, 'Unchanged base image required')
    path = canonical(report_path)
    require(path.name == 'report.json' and path.parent.parent == ROOT/'results', 'Runtime namespace required')
    runtime = pinned(path, pin, 131072)
    rev = runtime.get('producer_revision')
    require(re.fullmatch('[0-9a-f]{40}', str(rev)) and path.parent.name == 'raster-prefix-runtime-'+rev,
        'Exact runtime producer required')
    require(runtime.get('schema') == 'world_reward.raster_prefix_runtime.v1' and runtime.get('status') == 'pass'
        and runtime.get('decision') == 'ISOLATED_PREFIX_RUNTIME_QUALIFIED_NOT_ADOPTED'
        and runtime.get('base_image_id') == BASE and runtime.get('one_line_only_patch') is True
        and all(runtime.get(k) is False for k in ('base_modified', 'packages_installed', 'models_loaded',
            'reference_inputs_read', 'scientific_adoption', 'GPU_build'))
        and runtime.get('source_before') == runtime.get('source_after'), 'Complete isolated runtime PASS required')
    qa_path = path.parent/'qualification.json'; qa_pin = runtime['qualification_pin']
    qa = pinned(qa_path, qa_pin, 131072)
    require(qa.get('schema') == 'world_reward.raster_prefix_qualification.v1' and qa.get('status') == 'pass'
        and qa.get('decision') == 'INFERENCE_PREFIX_QUALIFIED_NOT_ADOPTED' and qa.get('producer_revision') == rev
        and qa.get('image_id') == runtime.get('qualified_image_id')
        and qa.get('source_revision') == '33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba'
        and qa.get('original_binary') == runtime['preflight']['original_binary']
        and all(qa.get(k) is False for k in ('geometry_deleted', 'grid_resized', 'model_or_RGB_loaded', 'truth_read', 'adopted')),
        'Original ABI/no-data qualification required')
    require(qa['prefix_invariant'] == dict(actual_coarse_valid_prefix=True, capacity_nonoverflow=True, original_native_coarse_used=True)
        and qa['fixed_bin_kernel_gate'] == dict(coarse_inputs_shared=True, fine_all_outputs_bit_equal=True,
            one_pixel_cotangent_backward_bit_equal=True, broad_training_validation=False), 'Kernel/prefix gates required')
    wanted = {(name, b) for name in ('procedural_random_multiobject_occlusion', 'actual_first_external_object') for b in (1, 4)}
    require(len(qa['rows']) == 4 and {(r['case'], r['batch_size']) for r in qa['rows']} == wanted and
        all(r['exact_mask_parity'] is True and r['depth_atol_m'] == 1e-6 and r['depth_rtol'] == 0.
            and 0 <= r['camera_z_max_error_m'] <= 1e-6 and r['image_size_hw'] == [1152, 1536]
            and r['native_capacity']['native_NDC_transform'] is True for r in qa['rows']), 'Full-grid B1/B4 parity required')
    import camera_render
    require(qa['source_pins'] == {k: runtime['source_before']['helpers'][k] for k in qa['source_pins']},
        'Qualified source/closure mismatch')
    require(qa['source_pins'] == camera_render._raster_source_pins(), 'Camera/capacity source mismatch')
    return runtime, qa, dict(runtime_report=pin, qualification_report=qa_pin), (path, qa_path)


def activation(report_path, pin, *, current_image):
    path = canonical(report_path); value = pinned(path, pin, 131072)
    require(value.get('schema') == 'world_reward.raster_prefix_activation.v1' and value.get('status') == 'pass'
        and value.get('base_image_id') == current_image == BASE and value.get('GPU_requested') is False
        and value.get('container_started') is False and value.get('container_absence_verified') is True,
        'Sealed CPU-only export PASS required')
    runtime_path = canonical(Path(value['runtime_report_path']))
    runtime, qa, receipt_pins, receipts = qualification(runtime_path, value['receipt_pins']['runtime_report'], current_image=current_image)
    require(receipt_pins == value['receipt_pins'] and runtime['producer_revision'] == value['runtime_revision']
        and value['candidate_binary'] == qa['candidate_binary'], 'Export qualification mismatch')
    binary = canonical(path.parent/FILENAME)
    require(path.name == 'report.json' and path.parent == ROOT/'results'/('raster-prefix-activation-'+value['runtime_revision']),
        'Exact activation namespace required')
    require(identity(binary, 200 << 20) == value['candidate_binary'], 'Exported binary mismatch')
    return dict(report=value, qualification=qa, binary=binary,
        readonly_mounts=[str(p) for p in (path, binary, *receipts)],
        environment=dict(WR_RASTER_PREFIX_ACTIVATION=str(path), WR_RASTER_PREFIX_ACTIVATION_BYTES=str(pin['bytes']),
                         WR_RASTER_PREFIX_ACTIVATION_SHA256=pin['sha256']))


def _docker(args):
    result = subprocess.run(['docker', *args], env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent',
        DOCKER_HOST='unix://'+str(ROOT/'docker.sock')), capture_output=True, text=True, timeout=30)
    require(result.returncode == 0 and len(result.stdout) <= 131072, 'CPU export failed')
    return result.stdout.strip()


def export(report_path, pin):
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01',
        'Azure VM01 CPU export required')
    runtime, qa, receipt_pins, _ = qualification(report_path, pin, current_image=BASE)
    rev = runtime['producer_revision']; out = ROOT/'results'/('raster-prefix-activation-'+rev)
    require(not out.exists(), 'Fresh export namespace required')
    out.mkdir(mode=0o755); name = 'wr-raster-prefix-export-'+rev; cid = None
    report = dict(schema='world_reward.raster_prefix_activation.v1', status='fail', runtime_revision=rev,
        runtime_report_path=str(report_path), receipt_pins=receipt_pins, base_image_id=BASE,
        candidate_binary=qa['candidate_binary'], GPU_requested=False, container_started=False,
        container_absence_verified=False)
    try:
        require(not _docker(['ps', '-aq', '--filter', 'name=^/'+name+'$']), 'No duplicate container')
        cid = _docker(['create', '--name', name, '--label', 'world_reward.raster_prefix.export='+rev,
            '--network=none', '--read-only', '--user', '0:0', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--entrypoint', '/bin/true', runtime['qualified_image_id']])
        require(re.fullmatch('[0-9a-f]{64}', cid), 'Exact export CID required')
        state = strict(_docker(['inspect', cid]))[0]
        require(state['Image'] == runtime['qualified_image_id'] and state['State']['Status'] == 'created'
            and state['State']['Running'] is False, 'Export must never run')
        _docker(['cp', cid+':'+CHILD_FILE, str(out/FILENAME)])
        binary = canonical(out/FILENAME); binary.chmod(0o444)
        require(identity(binary, 200 << 20) == qa['candidate_binary'], 'Copied binary/qualification mismatch')
        report['status'] = 'pass'
    finally:
        if cid and re.fullmatch('[0-9a-f]{64}', cid):
            state = strict(_docker(['inspect', cid]))[0]
            require(state['Image'] == runtime['qualified_image_id'] and state['Name'] == '/'+name
                and state['Config']['Labels']['world_reward.raster_prefix.export'] == rev, 'Owned export container required')
            _docker(['rm', cid]); report['container_absence_verified'] = not _docker(['ps', '-aq', '--no-trunc', '--filter', 'id='+cid])
        if cid is None:
            report['container_absence_verified'] = not _docker(['ps', '-aq', '--filter', 'name=^/'+name+'$'])
        if not report['container_absence_verified']: report['status'] = 'fail'
        with (out/'report.json').open('x') as f:
            f.write(json.dumps(report, sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)
    return out/'report.json'


class RasterOnlyProxy:
    def __init__(self, original, candidate): self.original, self.candidate = original, candidate
    def rasterize_meshes(self, *args, **kwargs): return self.candidate.rasterize_meshes(*args, **kwargs)
    def __getattr__(self, name): return getattr(self.original, name)


@contextmanager
def scoped_raster(report_path=None, pin=None, *, current_image, full_face_capacity=False):
    require(type(full_face_capacity) is bool, 'Explicit boolean reference mode required')
    if full_face_capacity or report_path is None:
        yield False; return
    proof = activation(report_path, pin, current_image=current_image)
    module = importlib.import_module('pytorch3d.renderer.mesh.rasterize_meshes')
    import pytorch3d._C as original
    require(identity(Path(original.__file__), 200 << 20, readonly=False) == proof['qualification']['original_binary'],
        'Original ABI mismatch')
    with LOCK:
        namespace = 'wr_raster_prefix_'+proof['report']['runtime_revision']
        if namespace not in sys.modules:
            package = types.ModuleType(namespace); package.__path__ = [str(proof['binary'].parent)]; sys.modules[namespace] = package
        key = namespace+'._C'
        candidate = sys.modules.get(key)
        if candidate is None:
            spec = importlib.util.spec_from_file_location(key, proof['binary'])
            candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)
            sys.modules[key] = candidate
        previous = module._C; module._C = RasterOnlyProxy(original, candidate)
        try: yield True
        finally: module._C = previous

