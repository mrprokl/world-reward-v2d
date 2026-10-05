"""One real, original HOI-DETR forward on authored RGB; not semantic validation.

A qualified MMCV image is reused by exact ID. FairScale is an offline pure-wheel
RO overlay, never installed into that image. Original acquired source/checkpoint
remain untouched. Only the authenticated author's absolute sys.path append is
removed in a derived source copy; load_from/train_cfg are explicit inference
configuration overrides, not numerical source edits. No dataset/API train import.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path('/srv/scenesmith/world-reward')
DATA = Path('/srv/world-reward-data/hoi_detr_v1')
ENTRY = 'run_hoi_detr_model_qualify'
PROTOCOL = 'configs/hoi_detr_model_qualify_v1.json'
PROTOCOL_PIN = dict(bytes=5005, sha256='8d6184588be0d4be5d8feec600b74b6e7806e953cc3b81c7c87c3e7579e937ac')
HELPERS = ('infra/hoi_detr_model_qualify.py', 'infra/run_hoi_detr_model_qualify.sh', PROTOCOL,
           'infra/hoi_detr_runtime_verify.py', 'infra/hoi_detr_acquire.py',
           'infra/mediapipe_cpu_runtime_verify.py', 'infra/mediapipe_hands_acquire.py',
           'src/world_reward/hoi_detr_observations.py')
FLAGS = ('quality_verified', 'actor_identity_verified', 'contact_verified', 'training_overlap_verified',
         'license_eligibility_verified', 'ground_truth_used', 'challenge_inputs_used', 'adoption')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_helpers(code):
    # Independently pinned stdlib bootstrap precedes any current driver import.
    path = Path(code)/'infra/hoi_detr_acquire.py'
    require(path.resolve() == path and not path.is_symlink(), 'Canonical original acquisition helper required')
    before = path.lstat(); raw = path.read_bytes(); after = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and not before.st_mode & 0o222
            and len(raw) == 29349 and hashlib.sha256(raw).hexdigest() == '130ca5bb5c4925c0069b5a3181c54848b78cb6c043dc688f29aa1f8bb66c3845'
            and all(getattr(before, k) == getattr(after, k) for k in ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_uid', 'st_nlink')), 'Independent acquisition bootstrap differs')
    spec = importlib.util.spec_from_file_location('wr_hoi_model_acquisition', path)
    acq = importlib.util.module_from_spec(spec); spec.loader.exec_module(acq); rt, mp = acq.helpers(code)
    code = rt.canonical(code); source(rt, code, os.environ['WR_CODE_REVISION'])
    path = code/'infra/hoi_detr_runtime_verify.py'
    spec = importlib.util.spec_from_file_location('wr_hoi_model_runtime', path)
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    return rt, mp, acq, runtime


def source(rt, code, revision):
    require(Path(__file__).resolve() == code/HELPERS[0] and {x.name for x in code.parent.iterdir()} == {'code', 'revision', 'source-sha256'}, 'Actual complete model-gate source required')
    return rt.source(ROOT, code, revision, ENTRY, HELPERS)


def protocol(rt, code):
    p = rt.pinned(code/PROTOCOL, PROTOCOL_PIN, 16 << 10)
    require(p['schema'] == 'world_reward.hoi_detr_model_qualification.v4' and p['scope'] == 'one_procedural_RGB_full_native_model_runtime_only'
            and p['root'] == str(ROOT) and p['output'] == 'results/hoi-detr-model-qualify-v4'
            and (p['budget_seconds'], p['cleanup_grace_seconds'], p['outer_seconds']) == (1800, 60, 1860)
            and (p['cpu_cpus'], p['cpu_memory'], p['gpu_cpus'], p['gpu_memory']) == (4, '16g', 4, '64g')
            and p['amp'] is False and p['tf32'] is False and p['seed'] == 0 and all(p[k] is False for k in FLAGS), 'Frozen one-forward scope required')
    require(p['runtime']['path'] == 'results/hoi-detr-runtime-v4/report.json' and p['runtime']['unresolved_pin_is_fail_before_checkpoint_decode'] is True,
            'Independent native operator qualification required')
    require(len(p['import_wheels']) == 1 and p['import_wheels'][0]['name'] == 'terminaltables'
            and p['import_wheels'][0]['version'] == '3.1.10' and p['import_wheels'][0]['bytes'] == 15155
            and p['import_wheels'][0]['publication_date'] <= '2026-09-30', 'Minimal pre-cutoff import wheel required')
    return p


MODEL_REQUIREMENTS = frozenset({
    'Actual independently pinned MMCV PASS is not yet available',
    'Readonly full overlay inventory changed',
    'CPU original import/config closure required',
    'Complete original native operator PASS required',
    'Offline original FairScale wheel build failed',
    'Offline wheel install/source preservation failed',
    'Exact original FairScale LICENSE required',
    'One pure original FairScale wheel required',
    'Unchanged original model import dependencies required',
    'Actual one H100 required',
    'Original training locator/approved inference override differs',
    'Active source/config weight loader forbidden',
    'Full original active model configuration required',
    'Original safe state_dict checkpoint required',
    'Strict all model state keys/shapes/dtypes required',
    'Original EMA checkpoint schema policy required',
    'Unambiguous native EMA buffer names required',
    'Registration must not resume or skip native buffers',
    'Exact complete native EMA backup schema required',
    'No state remapping/partial checkpoint load allowed',
    'Exactly qualified full native MMCV extension required',
    'No author train API/foreign absolute source imports',
    'Original RGB-only native test loader required',
    'Native image-only preprocessing required',
    'CPU overlay receipt late/failure',
    'Model receipt late/failure',
})


def failure_requirement(error, runtime):
    if type(error) is ValueError and len(error.args) == 1 and type(error.args[0]) is str and error.args[0] in MODEL_REQUIREMENTS:
        return error.args[0]
    return runtime.native_failure_requirement(error)


def row_pin(row):
    return {k: row[k] for k in ('bytes', 'sha256')}


def exact(rt, path, pin, maximum, *, empty=False):
    require(rt.identity(path, maximum, empty=empty) == pin, 'Independent immutable bytes differ')
    return pin


def old_source(rt, record, entry, helpers):
    code = ROOT/'jobs'/record['producer_revision']/entry/'code'
    require({x.name for x in code.parent.iterdir()} == {'code', 'revision', 'source-sha256'}, 'Original source-only parent inventory differs')
    require(rt.source(ROOT, code, record['producer_revision'], entry, helpers) == record, 'Original whole source/modes differ')


def authenticate_runtime(rt, runtime, p, deadline):
    # A null pin is a deliberate unresolved dependency, never a self-hash/PASS.
    require(type(p['runtime']['report']) is dict, 'Actual independently pinned MMCV PASS is not yet available')
    path = ROOT/p['runtime']['path']; report = rt.pinned(path, p['runtime']['report'], 32 << 10)
    require(report['stage'] == 'hoi_detr_runtime_operator_qualification' and report['status'] == 'pass' and report['phase'] == 'complete'
            and all(report[k] is True for k in ('native_extension_qualified', 'source_rehashed_after', 'inputs_rehashed_after', 'base_unchanged', 'owned_containers_removed', 'owned_scratch_removed'))
            and all(report[k] is False for k in runtime.FLAGS), 'Complete original native operator PASS required')
    old_source(rt, report['source_binding'], runtime.ENTRY, runtime.HELPERS)
    old = ROOT/'jobs'/report['producer_revision']/runtime.ENTRY/'code'
    c = rt.pinned(old/runtime.PROTOCOL, report['protocol_identity'], 16 << 10)
    require(c['schema'] == 'world_reward.hoi_detr_runtime.v4' and c['platform']['torch'] == '2.5.1+cu124', 'Actual v4 unchanged base ABI required')
    frozen = {str(path): p['runtime']['report'], str(old/runtime.PROTOCOL): report['protocol_identity']}
    for name, key in (('compile.json', 'compile_report_identity'), ('operators.json', 'operator_report_identity'), ('runtime_manifest.json', 'runtime_manifest_identity')):
        f = path.parent/name; value = rt.pinned(f, report[key], 2 << 20); frozen[str(f)] = report[key]
        if name != 'runtime_manifest.json':
            require(value['status'] == 'pass' and value['source_binding'] == report['source_binding'] and value['native_source_unchanged'] is True, 'Original compile/operator receipt differs')
            if name == 'operators.json':
                require(set(value['operators']) == {'MSDeformAttn', 'NMS', 'softNMS', 'RoIAlign'} and value['extension_identity'] == report['native_extension_identity'], 'All original native operators required')
        else:
            require(value['extension'] == report['native_extension_identity'], 'Whole runtime extension binding differs'); manifest = value
    current = runtime.image(report['child_image']['Id'], deadline)
    require(current == report['child_image'] and current['RootFS']['Layers'][:47] == report['base_image']['RootFS']['Layers'], 'Qualified exact image/full rootfs differs')
    return dict(report_identity=p['runtime']['report'], report=report, configuration=c, manifest=manifest, frozen=frozen, image=current)


def authenticate_acquisition(rt, acq, p):
    pin = p['acquisition']; path = ROOT/'results/hoi-detr-acquire-v1.json'; report = rt.pinned(path, pin['report'], 32 << 10)
    require(report['status'] == 'pass' and report['producer_revision'] == pin['producer_revision']
            and all(report[k] is True for k in ('first_party_license_declarations_verified', 'source_rehashed_after', 'artifacts_rehashed_after', 'owned_partials_removed', 'owned_archive_removed')),
            'Original whole acquisition PASS required')
    old_source(rt, report['source_binding'], acq.JOB, acq.HELPERS)
    manifest = rt.pinned(DATA/'source_manifest.json', pin['source_manifest'], 2 << 20)
    require(manifest['source']['revision'] == pin['source_revision'] and manifest['source']['git_root_tree_sha1'] == pin['git_root_tree_sha1']
            and manifest['source']['all_git_blobs_authenticated'] is True and acq.git_tree(manifest['complete_original_git_blobs']) == pin['git_root_tree_sha1'], 'Original complete Git source differs')
    rows = manifest['retained_artifacts'] + report['public_assets']; frozen = {str(path): pin['report'], str(DATA/'source_manifest.json'): pin['source_manifest']}
    require(len(rows) == len({r['file'] for r in rows}), 'Unique source/notice/checkpoint inventory required')
    for row in rows:
        acq.relative(row['file']); f = DATA/row['file']; exact(rt, f, row_pin(row), 6_000_000_000, empty=row['bytes'] == 0); frozen[str(f)] = row_pin(row)
    require(frozen[str(DATA/'weights/epoch_5.pth')] == pin['checkpoint'] and (DATA/'weights/README.md').read_bytes() == b'---\r\nlicense: mit\r\n---\r\n', 'Independent opaque checkpoint/card differs')
    require(set(frozen) == {str(f) for f in DATA.rglob('*') if f.is_file()} | {str(path)}, 'No unverified acquired source/model file permitted')
    return dict(report_identity=pin['report'], source_manifest_identity=pin['source_manifest'], source_binding=report['source_binding'], frozen=frozen), manifest


def remove_source_path(raw, expected):
    """One exact nonnumeric Expr removed; all remaining AST is identical."""
    tree = ast.parse(raw); matches = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'append'
        and isinstance(n.value.func.value, ast.Attribute) and n.value.func.value.attr == 'path'
        and isinstance(n.value.func.value.value, ast.Name) and n.value.func.value.value.id == 'sys'
        and len(n.value.args) == 1 and not n.value.keywords and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == expected]
    require(len(matches) == 1, 'Exactly original nonnumeric path Expr required')
    node = matches[0]; lines = raw.decode('utf-8').splitlines(keepends=True)
    require(node.lineno == node.end_lineno and lines[node.lineno-1].strip() == "sys.path.append('"+expected+"')", 'Original single-line source Expr differs')
    lines[node.lineno-1] = '\n'; derived = ''.join(lines).encode(); tree.body.remove(node)
    require(ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(derived), include_attributes=False), 'Only approved path Expr may change')
    return derived, dict(original=row_pin(dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())), derived=dict(bytes=len(derived), sha256=hashlib.sha256(derived).hexdigest()), remaining_AST_unchanged=True)


def inventory(rt, folder, maximum=16 << 20, *, readonly=True):
    rows = []
    for path in sorted(folder.rglob('*')):
        rt.canonical(path); s = path.lstat()
        require(stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Only unaliased overlay files/directories permitted')
        if path.is_file(): rows.append(dict(file=str(path.relative_to(folder)), **rt.identity(path, maximum, readonly=readonly, empty=True)))
    require(rows, 'Empty overlay forbidden')
    # pathlib component ordering differs from relative-string ordering for
    # package versus package-version.dist-info. Bind all rows in one canonical
    # ordering; never omit files to satisfy an inventory comparison.
    return sorted(rows, key=lambda row: row['file'])


def check_inventory(rt, folder, rows):
    require(type(rows) is list and len(rows) == len({r['file'] for r in rows})
            and inventory(rt, folder) == sorted(rows, key=lambda r: r['file']), 'Readonly full overlay inventory changed')


def seal_tree(folder):
    for f in sorted(folder.rglob('*'), reverse=True):
        s = f.lstat(); require(stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'No overlay symlinks/specials')
        f.chmod(0o555 if f.is_dir() else 0o444)
    folder.chmod(0o555)


def derive_source(rt, acq, manifest, target, p):
    rows = [r for r in manifest['retained_artifacts'] if r['file'].startswith('source/hoi-detr/')]
    for row in rows:
        name = row['file'].removeprefix('source/hoi-detr/'); acq.relative(name); dst = target/name
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(DATA/row['file'], dst)
        require(rt.identity(dst, 500000, readonly=False, empty=row['bytes'] == 0) == row_pin(row), 'Derived copy differs before approved patch')
    patch = p['compatibility_patch']; path = target/patch['path']; exact(rt, DATA/'source/hoi-detr'/patch['path'], patch['identity'], 500000)
    raw, proof = remove_source_path(path.read_bytes(), patch['remove_exact_sys_path_append']); path.write_bytes(raw)
    exact(rt, DATA/'source/hoi-detr'/p['native_config']['path'], p['native_config']['identity'], 500000)
    seal_tree(target); return proof


def unpack_fairscale(rt, acq, archive, target, c, deadline, check):
    names = set(); total = 0
    with tarfile.open(archive, 'r:gz') as z:
        for m in z:
            check(deadline); name = m.name.rstrip('/') if m.isdir() else m.name; acq.relative(name)
            require(name == 'fairscale-0.4.13' or name.startswith('fairscale-0.4.13/'), 'Exact FairScale sdist prefix required')
            require(name not in names and (m.isdir() or m.isfile()) and not (set(m.pax_headers)-{'mtime', 'atime', 'ctime'}), 'Unique regular FairScale sdist only'); names.add(name)
            require(0 <= m.size <= c['maximum_expanded_bytes'] and len(names) <= c['maximum_members'], 'Bounded FairScale sdist required'); total += m.size
            require(total <= c['maximum_expanded_bytes'], 'FairScale expansion cap')
            dst = target/name.removeprefix('fairscale-0.4.13/'); require(name != 'fairscale-0.4.13' or m.isdir(), 'Source root must be directory')
            if name == 'fairscale-0.4.13': continue
            if m.isdir(): dst.mkdir(parents=True, exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.extractfile(m) as src, dst.open('xb') as out: shutil.copyfileobj(src, out, 1 << 20)
    require(rt.identity(target/'LICENSE', 64 << 10, readonly=False) == row_pin(c['publisher_license']), 'Exact original FairScale LICENSE required')


def configuration_policy(cfg, c):
    require(cfg.get('load_from') == c['original_load_from'] and c['inference_overrides'] == {'load_from': None, 'model.train_cfg': None}, 'Original training locator/approved inference override differs')
    cfg['load_from'] = None; cfg['model']['train_cfg'] = None
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ('pretrained', 'init_cfg', 'load_from', 'resume_from'):
                    require(child is None, 'Active source/config weight loader forbidden')
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value: walk(child)
    walk(cfg)
    require(cfg['model']['backbone']['use_act_checkpoint'] is True and cfg['model']['query_head']['num_query'] == 1500
            and cfg['model']['query_head']['num_classes'] == 3, 'Full original active model configuration required')
    return dict(original_load_from=c['original_load_from'], inference_overrides=c['inference_overrides'], downloads=False)


def register_original_checkpoint_buffers(model, cfg, hook_class, c):
    """Original native schema registration, no EMA update/swap or weight choice.

    Author checkpoint includes backups of every original state tensor. Register
    these with the authenticated original hook, then strict-load ALL fields.
    This preserves original inference's non-EMA fields; no checkpoint key drop.
    """
    from types import SimpleNamespace
    require(list(cfg.custom_hooks) == [c['native_hook']] and c['registration_only'] is True
            and c['skip_buffers'] is False and c['swap_or_update'] is False
            and c['forward_uses_original_non_ema_fields'] is True, 'Original EMA checkpoint schema policy required')
    before = dict(model.state_dict()); names = {key: 'ema_'+key.replace('.', '_') for key in before}
    require(len(set(names.values())) == len(names) and not (set(names.values()) & set(before)), 'Unambiguous native EMA buffer names required')
    hook = hook_class(momentum=c['native_hook']['momentum'])
    require(hook.skip_buffers is False and hook.checkpoint is None, 'Registration must not resume or skip native buffers')
    hook.before_run(SimpleNamespace(model=model))
    after = model.state_dict()
    require(set(after) == set(before) | set(names.values()) and hook.param_ema_buffer == names
            and all(after[key].shape == before[key].shape and after[key].dtype == before[key].dtype
                    and after[names[key]].shape == before[key].shape and after[names[key]].dtype == before[key].dtype for key in before),
            'Exact complete native EMA backup schema required')
    return dict(original_state_keys=len(before), registered_backup_keys=len(names), full_state_keys=len(after),
                native_registration_only=True, forward_uses_original_non_ema_fields=True, ema_swapped=False, checkpoint_keys_discarded=0)


def strict_checkpoint(model, checkpoint, torch):
    require(type(checkpoint) is dict and type(checkpoint.get('state_dict')) in (dict, __import__('collections').OrderedDict), 'Original safe state_dict checkpoint required')
    actual = checkpoint['state_dict']; expected = model.state_dict()
    require(set(actual) == set(expected) and all(isinstance(v, torch.Tensor) and v.shape == expected[k].shape and v.dtype == expected[k].dtype and bool(torch.isfinite(v).all()) for k, v in actual.items()), 'Strict all model state keys/shapes/dtypes required')
    result = model.load_state_dict(actual, strict=True)
    require(not result.missing_keys and not result.unexpected_keys, 'No state remapping/partial checkpoint load allowed')
    return dict(keys=len(actual), tensor_bytes=sum(v.numel()*v.element_size() for v in actual.values()), outer_fields=sorted(checkpoint), strict=True, weights_only=True)


def native_context(code, revision, p, proof_pin, phase):
    rt, _, acq, runtime = load_helpers(code); deadline = float(os.environ['WR_MODEL_DEADLINE']); runtime.check(deadline)
    require(os.geteuid() == 1000 and os.environ.get('WR_ROOT') == str(ROOT) and {x.name for x in Path('/sys/class/net').iterdir()} == {'lo'}, 'Exact offline unprivileged model container required')
    own = source(rt, code, revision); out = ROOT/p['output']; proof = rt.pinned(out/(phase+'_proof.json'), proof_pin, 2 << 20)
    require(proof['source_binding'] == own and proof['protocol_identity'] == PROTOCOL_PIN and os.environ.get('WR_IMAGE_ID') == proof['runtime']['image']['Id'], 'Own source/actual image proof differs')
    cid = out/(phase+'.cid'); s = cid.lstat()
    require(s.st_uid == 0 and stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and re.fullmatch(b'[0-9a-f]{64}\n?', cid.read_bytes()), 'Parent genuine container ID required before imports')
    return rt, acq, runtime, out, proof, deadline


def cpu_overlay(code, revision, p, proof_pin):
    rt, acq, runtime, out, proof, deadline = native_context(code, revision, p, proof_pin, 'overlay')
    root = out/'.overlay'; source_dir = root/'fairscale'; source_dir.mkdir(); wheels = root/'wheels'; wheels.mkdir(); site = root/'site'; site.mkdir()
    exact(rt, root/p['fairscale']['file'], row_pin(p['fairscale']), 1 << 20); exact(rt, root/'FairScale.LICENSE', row_pin(p['fairscale']['publisher_license']), 64 << 10)
    unpack_fairscale(rt, acq, root/p['fairscale']['file'], source_dir, p['fairscale'], deadline, runtime.check)
    seal_tree(source_dir); before = inventory(rt, source_dir); build = root/'build'; shutil.copytree(source_dir, build)
    for f in (build, *build.rglob('*')): f.chmod(0o755 if f.is_dir() else 0o644)
    torch, _, versions = runtime.versions(proof['runtime']['configuration'])
    require(not torch.cuda.is_initialized(), 'CPU overlay must never initialize CUDA')
    env = dict(PATH='/opt/conda/bin:/usr/bin:/bin', HOME='/tmp', CUDA_VISIBLE_DEVICES='-1', BUILD_CUDA_EXTENSIONS='0', PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONDONTWRITEBYTECODE='1')
    commands = [[sys.executable, '-I', '-B', '-m', 'pip', '--isolated', 'wheel', '--no-index', '--no-deps', '--no-build-isolation', '--no-cache-dir', '--wheel-dir', str(wheels), str(build)]]
    for args in commands:
        runtime.check(deadline); result = subprocess.run(args, env=env, timeout=max(.01, deadline-time.monotonic()), check=False); require(result.returncode == 0, 'Offline original FairScale wheel build failed')
    wheel = list(wheels.glob('*.whl')); require(len(wheel) == 1 and wheel[0].name == 'fairscale-0.4.13-py3-none-any.whl', 'One pure original FairScale wheel required')
    notices = runtime.wheel_notice(wheel[0], p['fairscale'], (root/'FairScale.LICENSE').read_bytes())
    extra_wheels = []
    for row in p['import_wheels']:
        exact(rt, root/row['file'], row_pin(row), 1 << 20); exact(rt, root/row['publisher_license']['file'], row_pin(row['publisher_license']), 64 << 10)
        notices[row['name']] = runtime.wheel_notice(root/row['file'], row, (root/row['publisher_license']['file']).read_bytes())
        extra_wheels.append(str(root/row['file']))
    result = subprocess.run([sys.executable, '-I', '-B', '-m', 'pip', '--isolated', 'install', '--no-index', '--no-deps', '--no-cache-dir', '--no-compile', '--target', str(site), str(wheel[0]), *extra_wheels], env=env, timeout=max(.01, deadline-time.monotonic()), check=False)
    require(result.returncode == 0 and all(rt.identity(source_dir/r['file'], 16 << 20, empty=r['bytes'] == 0) == row_pin(r) for r in before), 'Offline wheel install/source preservation failed')
    require(not any(f.suffix in ('.so', '.pyd', '.dll') for f in site.rglob('*')) and not torch.cuda.is_initialized(), 'FairScale CPU-only pure overlay required')
    seal_tree(site); seal_tree(wheels)
    # Qualify the full import/config closure on CPU BEFORE acquiring the GPU.
    # Original source/registries are used; no model, datasets or weights built.
    mmcv_root = Path('/opt/world-reward-hoi-mmcv')
    sys.path[:0] = [str(site), str(mmcv_root/'site'), str(mmcv_root/'source'), str(root/'source')]
    import importlib
    for name in ('fairscale.nn.checkpoint', 'mmdet.models.builder', 'projects.models', 'mmdet.datasets.pipelines'):
        importlib.import_module(name)
    from mmcv import Config
    cfg = Config.fromfile(str(root/'source'/p['native_config']['path']), import_custom_modules=False)
    configuration_policy(cfg._cfg_dict, p['native_config'])
    require(not torch.cuda.is_initialized() and not any(n == 'mmdet.apis' or n.startswith('mmdet.apis.') for n in sys.modules)
            and not any(x.startswith(('/gpfs', '/lus')) for x in sys.path), 'CPU original import/config closure required')
    rt.write(out/'overlay_site.json', (json.dumps(dict(artifacts=inventory(rt, site)), sort_keys=True)+'\n').encode(), 0o444)
    value = dict(stage='hoi_detr_fairscale_overlay', status='pass', source_binding=proof['source_binding'], versions=versions, site_manifest_identity=rt.identity(out/'overlay_site.json', 256 << 10), wheel=rt.identity(wheel[0], 16 << 20), notices=notices, cuda_initialized=False, base_installed=False, full_import_closure_qualified=True)
    acq.write_receipt(out/'overlay.json', value, proof['started_monotonic'], deadline)
    require(value['status'] == 'pass', 'CPU overlay receipt late/failure')


def gpu_model(code, revision, p, proof_pin):
    rt, acq, runtime, out, proof, deadline = native_context(code, revision, p, proof_pin, 'model')
    root = out/'.overlay'; require({f.name for f in root.iterdir()} == {'source', 'site'}, 'Only authenticated executable overlays permitted'); check_inventory(rt, root/'source', proof['overlay']['source']); check_inventory(rt, root/'site', proof['overlay']['site'])
    mmcv_root = Path('/opt/world-reward-hoi-mmcv')
    check_inventory(rt, mmcv_root, proof['runtime']['manifest']['artifacts'])
    exact(rt, DATA/'weights/epoch_5.pth', p['acquisition']['checkpoint'], 6_000_000_000)
    sys.path[:0] = [str(root/'site'), str(mmcv_root/'site'), str(mmcv_root/'source'), str(root/'source'), str(code/'src')]
    torch, np, versions = runtime.versions(proof['runtime']['configuration'])
    import importlib.metadata as metadata
    require({n: metadata.version(n) for n in p['extra_base_distributions']} == p['extra_base_distributions'] and metadata.version('fairscale') == '0.4.13' and all(metadata.version(r['name']) == r['version'] for r in p['import_wheels']), 'Unchanged original model import dependencies required')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0), 'Actual one H100 required')
    torch.manual_seed(p['seed']); torch.cuda.manual_seed_all(p['seed']); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    from mmcv import Config
    import mmcv, mmcv._ext as extension
    require(Path(mmcv.__file__).is_relative_to(mmcv_root/'source') and rt.identity(Path(extension.__file__), 1 << 30) == proof['runtime']['report']['native_extension_identity'], 'Exactly qualified full native MMCV extension required')
    from mmcv.parallel import collate, scatter
    from mmcv.ops import batched_nms
    from mmdet.models.builder import build_detector
    import projects.models
    from mmdet.datasets.pipelines import Compose
    from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
    from world_reward.hoi_detr_observations import NativeHOIOperations, infer_hoi_detr_frame
    require(not any(n == 'mmdet.apis' or n.startswith('mmdet.apis.') for n in sys.modules) and not any(x.startswith(('/gpfs', '/lus')) for x in sys.path), 'No author train API/foreign absolute source imports')
    cfg = Config.fromfile(str(root/'source'/p['native_config']['path']), import_custom_modules=False)
    overrides = configuration_policy(cfg._cfg_dict, p['native_config'])
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')); runtime.check(deadline)
    from mmdet.core.hook.ema import ExpMomentumEMAHook
    exact(rt, root/'source'/p['checkpoint_buffers']['source_path'], p['checkpoint_buffers']['source_identity'], 64 << 10)
    buffer_schema = register_original_checkpoint_buffers(model, cfg, ExpMomentumEMAHook, p['checkpoint_buffers'])
    checkpoint = torch.load(DATA/'weights/epoch_5.pth', map_location='cpu', weights_only=True)
    loaded = strict_checkpoint(model, checkpoint, torch); del checkpoint
    model.to(device='cuda', dtype=torch.float32).eval(); runtime.check(deadline)
    pipeline = [dict(x) for x in cfg.data.test.pipeline]
    require(pipeline[0] == dict(type='LoadImageFromFile'), 'Original RGB-only native test loader required')
    pipeline[0] = dict(type='LoadImageFromWebcam'); transform = Compose(pipeline)
    def prepare(rgb):
        result = transform(dict(img=np.array(rgb[:, :, ::-1], copy=True)))
        require(set(result) == {'img', 'img_metas'}, 'Native image-only preprocessing required'); return result
    c = p['procedural_RGB']; y, x = np.indices((c['height'], c['width']), dtype=np.int32)
    rgb = np.stack(((x*3+y*5)%256, (x*7+y*11)%256, (x*13+y*17)%256), axis=2).astype(np.uint8)
    operations = NativeHOIOperations(prepare, collate, scatter, bbox_cxcywh_to_xyxy, batched_nms, torch, torch.device('cuda:0'))
    result = infer_hoi_detr_frame(model, rgb, c['original_frame_index'], operations); torch.cuda.synchronize(); runtime.check(deadline)
    arrays = {k: v for k, v in vars(result).items() if isinstance(v, np.ndarray)}
    with (out/'observations.npz').open('xb') as stream: np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    exact(rt, DATA/'weights/epoch_5.pth', p['acquisition']['checkpoint'], 6_000_000_000); check_inventory(rt, root/'source', proof['overlay']['source']); check_inventory(rt, root/'site', proof['overlay']['site']); check_inventory(rt, mmcv_root, proof['runtime']['manifest']['artifacts'])
    require(source(rt, code, revision) == proof['source_binding'], 'Current full source changed after actual model')
    value = dict(stage='hoi_detr_full_native_model', status='pass', source_binding=proof['source_binding'], checkpoint_identity=p['acquisition']['checkpoint'], strict_checkpoint=loaded, checkpoint_buffer_schema=buffer_schema,
                 versions=versions, configuration=overrides, actual_full_model=True, native_forward_calls=1, procedural_rgb_identity=dict(bytes=rgb.nbytes, sha256=hashlib.sha256(rgb.tobytes()).hexdigest()),
                 preprocessing='original_LoadImageFromWebcam_BGR_input_then_native_test_pipeline', decoder_queries=1500, original_frame_index=c['original_frame_index'], native_cpu_softNMS=True,
                 hand_object_pairs=len(result.hand_object_pairs), object_target_pairs=len(result.object_target_pairs), observations=rt.identity(out/'observations.npz', 32 << 20),
                 original_source_unchanged=True, checkpoint_rehashed_after=True, overlays_rehashed_after=True, elapsed_seconds=time.monotonic()-proof['started_monotonic'], **{k: False for k in FLAGS})
    acq.write_receipt(out/'native.json', value, proof['started_monotonic'], deadline); require(value['status'] == 'pass', 'Model receipt late/failure')


def container_plan(code, out, revision, p, proof, phase, deadline):
    name = 'world-reward-hoi-model-'+phase+'-'+revision[:12]; cid = out/(phase+'.cid'); image_id = proof['runtime']['image']['Id']; overlay = out/'.overlay'
    mounts = [(code.parent, code.parent, True), (out, out, False)] if phase == 'overlay' else [(code.parent, code.parent, True), (out, out, False), (overlay, overlay, True), (DATA/'weights/epoch_5.pth', DATA/'weights/epoch_5.pth', True)]
    args = ['docker', 'create', '--name', name, '--cidfile', str(cid), '--label', 'world-reward.job='+ENTRY, '--label', 'world-reward.revision='+revision,
            '--network', 'none', '--read-only', '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', str(p['gpu_cpus'] if phase == 'model' else p['cpu_cpus']),
            '--memory', p['gpu_memory'] if phase == 'model' else p['cpu_memory'], '--tmpfs', '/tmp:rw,nosuid,size=2g', '--entrypoint', '/usr/bin/env']
    if phase == 'model': args += ['--gpus', 'driver=nvidia,count=all']
    for a, b, ro in mounts: args += ['--mount', f'type=bind,src={a},dst={b}'+(',readonly' if ro else '')]
    args += [image_id, '-i', 'PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/bin:/bin', 'HOME=/tmp', 'PYTHONDONTWRITEBYTECODE=1', 'WR_ROOT='+str(ROOT), 'WR_CODE='+str(code), 'WR_CODE_REVISION='+revision,
             'WR_IMAGE_ID='+image_id, 'WR_MODEL_DEADLINE='+format(deadline, '.17g'), 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'CUDA_VISIBLE_DEVICES=0' if phase == 'model' else 'CUDA_VISIBLE_DEVICES=-1',
             '/opt/conda/bin/python', '-I', '-B', str(code/HELPERS[0]), '--'+phase, '--proof-bytes', str(proof['pin']['bytes']), '--proof-sha256', proof['pin']['sha256']]
    return name, cid, mounts, args


def validate_container(value, mounts, name, revision, image_id, phase):
    h, c = value['HostConfig'], value['Config']
    require(value['Image'] == image_id and value['Name'] == '/'+name and c['User'] == '1000:1000' and c['Labels'].get('world-reward.job') == ENTRY and c['Labels'].get('world-reward.revision') == revision
            and h['NetworkMode'] == 'none' and h['ReadonlyRootfs'] is True and not h['Privileged'] and h['CapDrop'] == ['ALL'] and 'no-new-privileges' in h['SecurityOpt'], 'Exact owner/image/sandbox required')
    require({(m['Source'], m['Destination'], not m['RW']) for m in value['Mounts'] if m['Type'] == 'bind'} == {(str(a), str(b), ro) for a, b, ro in mounts}
            and not any(m['Type'] == 'volume' for m in value['Mounts']) and not h.get('Devices') and not h.get('Binds') and not h.get('VolumesFrom'), 'Only exact code/overlay/checkpoint/fresh output permitted')
    devices = h.get('DeviceRequests') or []
    require(bool(devices) == (phase == 'model') and (not devices or len(devices) == 1 and devices[0]['Driver'] == 'nvidia' and devices[0]['Count'] == -1 and devices[0]['Capabilities'] == [['gpu']]), 'Only one explicit native GPU request permitted')


def cleanup(runtime, cidfile, name, revision, image_id, deadline):
    if not cidfile.exists():
        runtime.absent(name, deadline); return
    s = cidfile.lstat(); require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == 0 and s.st_size <= 65, 'Own regular CID required')
    raw = cidfile.read_bytes(); require(re.fullmatch(b'[0-9a-f]{64}\n?', raw), 'Own exact CID required'); cid = raw.decode().rstrip('\n')
    ids = runtime.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline); require(ids in ('', cid), 'Exact CID inventory required')
    if ids:
        actual = runtime.command(['docker', 'inspect', cid, '--format', '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'], deadline)
        require(actual == image_id+'|/'+name+'|'+ENTRY+'|'+revision, 'Foreign container never removed')
        require(runtime.command(['docker', 'rm', '-f', cid], deadline) == cid, 'Own container removal failed')
    require(not runtime.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline), 'Own CID survived cleanup'); runtime.absent(name, deadline); cidfile.chmod(0o444)


def run(code, revision, *, opener=None):
    import fcntl
    started = time.monotonic(); rt, mp, acq, runtime = load_helpers(code); own = source(rt, code, revision); p = protocol(rt, code); deadline = started+p['budget_seconds']
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02' and os.environ.get('WR_ROOT') == str(ROOT), 'Exact Azure root host required')
    qualified = authenticate_runtime(rt, runtime, p, deadline); acquired, manifest = authenticate_acquisition(rt, acq, p)
    out = rt.canonical(ROOT/p['output']); require(not out.exists() and out.parent.is_dir(), 'Fresh model qualification namespace required'); out.mkdir(mode=0o755); os.chown(out, 1000, 1000)
    s = out.lstat(); own_inode = (s.st_dev, s.st_ino, s.st_uid); overlay = out/'.overlay'; overlay.mkdir(mode=0o755); os.chown(overlay, 1000, 1000); s = overlay.lstat(); overlay_inode = (s.st_dev, s.st_ino, s.st_uid)
    proof = dict(source_binding=own, protocol_identity=PROTOCOL_PIN, runtime=qualified, acquisition=dict(report_identity=acquired['report_identity'], source_manifest_identity=acquired['source_manifest_identity'], source_binding=acquired['source_binding']), started_monotonic=started)
    report = dict(stage='hoi_detr_model_qualification', status='fail', phase='preflight', producer_revision=revision, source_binding=own, protocol_identity=PROTOCOL_PIN,
                  runtime_report_identity=qualified['report_identity'], acquisition_report_identity=acquired['report_identity'], actual_model_qualified=False, source_rehashed_after=False, inputs_rehashed_after=False,
                  image_unchanged=False, owned_containers_removed=False, owned_overlay_removed=False, **{k: False for k in FLAGS})
    containers = []; lock_fd = None; failure = None; owned, dirs = [], set()
    try:
        opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), runtime.NoRedirect())
        report['phase'] = 'fairscale_source_and_notices'
        for row in (p['fairscale']['publisher_license'], p['fairscale']): runtime.fetch(mp, acq, overlay, row, deadline, owned, dirs, opener)
        exact(rt, overlay/'FairScale.LICENSE', row_pin(p['fairscale']['publisher_license']), 64 << 10)
        rt.write(out/'FairScale.LICENSE', (overlay/'FairScale.LICENSE').read_bytes(), 0o444)
        report['fairscale_license_identity'] = rt.identity(out/'FairScale.LICENSE', 64 << 10)
        for row in p['import_wheels']:
            for value in (row['publisher_license'], row): runtime.fetch(mp, acq, overlay, value, deadline, owned, dirs, opener)
            rt.write(out/row['publisher_license']['file'], (overlay/row['publisher_license']['file']).read_bytes(), 0o444)
        patch = derive_source(rt, acq, manifest, overlay/'source', p)
        report['phase'] = 'offline_cpu_overlay'
        for phase in ('overlay', 'model'):
            if phase == 'model':
                compiled = rt.strict((out/'overlay.json').read_bytes()); require(compiled['status'] == 'pass' and compiled['source_binding'] == own and compiled['cuda_initialized'] is False and compiled['base_installed'] is False and compiled['full_import_closure_qualified'] is True, 'Real pure CPU overlay PASS required')
                site_manifest = rt.pinned(out/'overlay_site.json', compiled['site_manifest_identity'], 256 << 10); check_inventory(rt, overlay/'site', site_manifest['artifacts'])
                # Dispose original sdist/build copies before the inference mount;
                # no unverified build debris shares the executable RO overlay.
                disposable = [p['fairscale']['publisher_license'], p['fairscale'], *[r for w in p['import_wheels'] for r in (w['publisher_license'], w)]]
                for row in disposable:
                    f=overlay/row['file']; exact(rt, f, row_pin(row), 1 << 20); f.unlink()
                for folder in (overlay/'fairscale', overlay/'build', overlay/'wheels'):
                    for f in sorted(folder.rglob('*'), reverse=True):
                        require(not f.is_symlink(), 'Own build debris links rejected')
                        if f.is_dir(): f.chmod(0o700)
                    folder.chmod(0o700); shutil.rmtree(folder)
                seal_tree(overlay); proof['overlay'] = dict(source=inventory(rt, overlay/'source'), site=site_manifest['artifacts'], patch=patch)
                raw = (json.dumps(proof['overlay'], sort_keys=True)+'\n').encode(); rt.write(out/'overlay_manifest.json', raw, 0o444); report['overlay_manifest_identity'] = rt.identity(out/'overlay_manifest.json', 2 << 20)
                lock = rt.canonical(ROOT/'jobs/.world-reward-h100.lock'); s = lock.lstat(); require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Existing cooperative GPU lock required')
                lock_fd = os.open(lock, os.O_RDONLY|os.O_NOFOLLOW); require((os.fstat(lock_fd).st_dev, os.fstat(lock_fd).st_ino) == (s.st_dev, s.st_ino), 'Exact readonly GPU lock inode required'); fcntl.flock(lock_fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
                proof['lock_inode'] = [s.st_dev, s.st_ino]; require(not runtime.command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], deadline), 'GPU must be idle after lock')
                report['phase'] = 'full_native_model'
            rt.write(out/(phase+'_proof.json'), (json.dumps(proof, sort_keys=True)+'\n').encode(), 0o444); context = dict(proof, pin=rt.identity(out/(phase+'_proof.json'), 2 << 20))
            name, cid, mounts, args = container_plan(code, out, revision, p, context, phase, deadline); runtime.absent(name, deadline); containers.append((cid, name, qualified['image']['Id']))
            runtime.command(args, deadline); value = json.loads(runtime.command(['docker', 'inspect', name, '--format', '{{json .}}'], deadline)); validate_container(value, mounts, name, revision, qualified['image']['Id'], phase)
            runtime.command(['docker', 'start', '-a', name], deadline, log=out/(phase+'.log')); cleanup(runtime, cid, name, revision, qualified['image']['Id'], deadline); containers.pop()
        native = rt.strict((out/'native.json').read_bytes()); require(native['status'] == 'pass' and native['source_binding'] == own and native['actual_full_model'] is True and native['native_forward_calls'] == 1
            and all(native[k] is False for k in FLAGS), 'Genuine one-forward model receipt required')
        report.update(status='pass', phase='complete', native_report_identity=rt.identity(out/'native.json', 32 << 10), observations_identity=native['observations'], actual_model_qualified=True)
    except Exception as exc:
        failure = exc; report['error_type'] = type(exc).__name__
        report['requirement'] = failure_requirement(exc, runtime)
    finally:
        signal.alarm(p['cleanup_grace_seconds'])
        try:
            grace = time.monotonic()+p['cleanup_grace_seconds']; t = out.lstat(); require(rt.canonical(out) == out and (t.st_dev, t.st_ino, t.st_uid) == own_inode, 'Own output namespace replaced')
            for cid, name, image_id in containers: cleanup(runtime, cid, name, revision, image_id, grace)
            report['owned_containers_removed'] = True
            for part, inode in owned:
                if part.exists():
                    t = part.lstat(); require(not part.is_symlink() and stat.S_ISREG(t.st_mode) and t.st_nlink == 1 and (t.st_dev, t.st_ino, t.st_uid) == inode, 'Only own partial removed'); part.unlink()
            for path, pin in acquired['frozen'].items(): exact(rt, Path(path), pin, 6_000_000_000, empty=pin['bytes'] == 0)
            for path, pin in qualified['frozen'].items(): exact(rt, Path(path), pin, 2 << 20)
            old_source(rt, acquired['source_binding'], acq.JOB, acq.HELPERS)
            old_source(rt, qualified['report']['source_binding'], runtime.ENTRY, runtime.HELPERS)
            if (out/'FairScale.LICENSE').exists(): exact(rt, out/'FairScale.LICENSE', row_pin(p['fairscale']['publisher_license']), 64 << 10)
            report['inputs_rehashed_after'] = True; report['source_rehashed_after'] = source(rt, code, revision) == own; report['image_unchanged'] = runtime.image(qualified['image']['Id'], grace) == qualified['image']
            runtime.remove_owned_folder(rt, overlay, overlay_inode, out); report['owned_overlay_removed'] = True
            if lock_fd is not None:
                t = (ROOT/'jobs/.world-reward-h100.lock').lstat(); require((t.st_dev, t.st_ino) == tuple(proof['lock_inode']), 'Original GPU lease inode changed')
        except Exception as exc:
            failure = failure or exc; report['post_error_type'] = type(exc).__name__
        finally:
            if lock_fd is not None: os.close(lock_fd)
            signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-started
        if failure or report['elapsed_seconds'] >= p['budget_seconds'] or not all(report[k] for k in ('source_rehashed_after', 'inputs_rehashed_after', 'image_unchanged', 'owned_containers_removed', 'owned_overlay_removed')): report['status'] = 'fail'
        try:
            if report['status'] == 'pass':
                for name in ('overlay.log', 'model.log'):
                    f = out/name; t = f.lstat(); require(stat.S_ISREG(t.st_mode) and t.st_nlink == 1 and t.st_uid == 0, 'Only original owned logs removed'); f.unlink()
            for f in sorted(out.iterdir()):
                t = f.lstat(); require(stat.S_ISREG(t.st_mode) and t.st_nlink == 1, 'Only owned concise model evidence retained'); f.chmod(0o444)
            out.chmod(0o555)
        except Exception as exc:
            report.update(status='fail', phase='final_sealing', sealing_error_type=type(exc).__name__)
        acq.write_receipt(out/'report.json', report, started, deadline)
    require(report['status'] == 'pass', 'Model gate failed; immutable bounded receipt retained')
    return report


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False); group = parser.add_mutually_exclusive_group(); group.add_argument('--overlay', action='store_true'); group.add_argument('--model', action='store_true')
    parser.add_argument('--proof-bytes', type=int); parser.add_argument('--proof-sha256'); args = parser.parse_args()
    code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']; rt, _, acq, _ = load_helpers(code); p = protocol(rt, code)
    if args.overlay or args.model:
        require(type(args.proof_bytes) is int and args.proof_bytes > 0 and re.fullmatch('[0-9a-f]{64}', args.proof_sha256 or ''), 'Independent native proof pin required')
        phase = 'overlay' if args.overlay else 'model'
        try:
            (cpu_overlay if args.overlay else gpu_model)(code, revision, p, dict(bytes=args.proof_bytes, sha256=args.proof_sha256))
        except Exception as exc:
            name = 'overlay.json' if args.overlay else 'native.json'; out = ROOT/p['output']
            if not (out/name).exists():
                _, _, _, runtime = load_helpers(code)
                acq.write_receipt(out/name, dict(stage='hoi_detr_'+phase, status='fail', error_type=type(exc).__name__, requirement=failure_requirement(exc, runtime), phase='native_execution', **{k: False for k in FLAGS}), time.monotonic(), float(os.environ['WR_MODEL_DEADLINE']))
            raise
    else:
        value = run(code, revision); print(json.dumps({k: value[k] for k in ('stage', 'status', 'phase', 'elapsed_seconds')}))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        print(json.dumps(dict(stage='hoi_detr_model_qualification', status='fail', error_type=type(exc).__name__)))
        raise SystemExit(1) from None
