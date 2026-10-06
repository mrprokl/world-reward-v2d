"""One frozen label-free FIT32 cost profile; no selection, loss or model run.

Root host authenticates historical cohort, saved join and image graph. Native
receives only opaque membership and complete public observation metadata/files.
Neither private lineage nor RGB/roles are mounted. No failed-profile rescue.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
import mediapipe_cpu_runtime_verify as rt
import vcoco_full_interaction_join as join

ROOT, IMAGE = rt.ROOT, rt.BASE
ENTRY = 'run_vcoco_selector_blind_cost'
OUTPUT = ROOT/'results/vcoco-selector-blind-cost-v1'
SCHEMA = 'world_reward.vcoco_selector_blind_cost.v1'
BUDGET, MEMORY, MAX_CONTROL = 3600, 64 << 30, 4 << 20
DECISION = 'BLIND_FULL32_PARTITION_COST_PASS_NOT_FIT_OR_QUALITY'
CV_NAMESPACE = 'world_reward.vcoco_selector_cv_v1/'
JOIN_PINS = {
    'report.json': dict(bytes=1146677, sha256='88130d1b4db4f07153360b31292b52ccbafe723248a8368a845f10ea2ac7aa52'),
    'native.json': dict(bytes=442232, sha256='dcb28399e35dcea15a638b3776286653f51bd5531d4311d5e6731a00f598988e'),
    'proof.json': dict(bytes=509236, sha256='de98fa780685234c979caee468a5aca680da4599306f5a2e4d35feb7ce08618e')}
NATIVE_FILES = (*join.NATIVE_FILES, 'infra/vcoco_selector_blind_cost.py',
    'infra/vcoco_selector_blind_cost_native.py', 'infra/coherent_pair_gpu_probe.py',
    'infra/coherent_pair_gpu_objective_probe.py',
    'src/world_reward/coherent_pair_partition.py',
    'src/world_reward/coherent_pair_marginal_objective.py',
    'src/world_reward/coherent_pair_learning.py', 'src/world_reward/coherent_route_scorer.py',
    'src/world_reward/coherent_pair_cache.py', 'src/world_reward/coherent_pair_packed.py',
    'src/world_reward/coherent_pair_packed_score.py', 'src/world_reward/coherent_pair_marginal.py',
    'src/world_reward/coherent_pair_packed_torch.py')
FALSE_FLAGS = ('RGB_decoded', 'reference_values_read', 'positive_masks_used', 'models_loaded',
    'FIT_performed', 'optimizer_performed', 'CAL_evaluated', 'selection_performed',
    'ownership_verified', 'quality_verified', 'adoption', 'full_fit_cost_qualified')


def check(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('Inclusive3600s fixed blind profile')


def manifest():
    return dict(schema=SCHEMA, images=32, source_population=48, objects_per_image=3600,
        all_native_persons_sides_pairs_retained=True, CPU_memory_bytes=MEMORY,
        GPU_memory_bytes=MEMORY, budget_seconds=BUDGET, temperature=1., alphas=[0., 1.],
        theta_controls=['zero17', 'normalized_1_through_17'], repeats=2,
        oracle_rtol=1e-12, oracle_atol=1e-12, scale_fit='original_fit_scales_tuple32',
        preparation='resident_CPU32_then_complete_image_streamed_GPU',
        statistic='supported_group_log_partition_and_analytic_VJP_without_labels',
        CV_namespace=CV_NAMESPACE, CV_folds=4, replacement_count=0, retry_count=0,
        timing_is_objective_cost_guarantee=False, **{n: False for n in FALSE_FLAGS})


def helpers():
    import vcoco_fit_cal_context as context
    return tuple(dict.fromkeys((*join.helpers(), *context.freeze.HELPERS,
        'infra/vcoco_full_interaction_inputs.py', 'infra/vcoco_fit_cal_context.py',
        'infra/run_vcoco_selector_blind_cost.sh', 'infra/research_image_identity.py', *NATIVE_FILES)))


def membership(cohort, records):
    """Pure opaque projection, only after host live cohort authentication."""
    import vcoco_fit_cal_selection as original
    rt.require(type(cohort) is dict and type(cohort.get('records')) is list
        and len(cohort['records']) == len(records) == 48, 'Complete fixed48 cohort required')
    ids = []
    for i, (row, record) in enumerate(zip(cohort['records'], records)):
        rt.require(type(row['slot']) is int and row['slot'] == i and row['split'] == ('FIT' if i < 32 else 'CAL')
            and type(row['image_id']) is int and 0 < row['image_id'] < 10**12,
            'Original32FIT/16CAL membership/order required')
        opaque = hashlib.sha256((original.NAMESPACE+f"{row['image_id']:012d}").encode()).hexdigest()[:32]
        rt.require(type(record['endpoint']['original_slot']) is int and record['endpoint']['original_slot'] == i
            and record['endpoint']['image_id'] == opaque, 'Authenticated native/opaque join mapping differs')
        ids.append(opaque)
    rt.require(len(set(ids)) == 48, 'Injective original photo identities required')
    order = sorted(ids[:32], key=lambda iid: (hashlib.sha256((CV_NAMESPACE+iid).encode()).hexdigest(), iid))
    folds = {iid: i % 4 for i, iid in enumerate(order)}
    return tuple(dict(slot=i, image_id=iid, fold=folds[iid]) for i, iid in enumerate(ids[:32]))


def authenticate(code, revision, deadline):
    import vcoco_full_interaction_inputs as inputs
    import vcoco_fit_cal_context as context
    import coherent_pair_gpu_probe as original
    closure = helpers(); checkpoint = lambda: check(deadline)
    saved = inputs.authenticate_saved_join(code, revision, ENTRY, closure, JOIN_PINS, checkpoint, deadline=deadline)
    binding = saved['host_lineage']['current_source']['binding']
    rt.require(all(binding['helpers'][n] == dict(bytes=size, sha256=digest)
        for n, (size, digest) in original.REUSED.items() if n in NATIVE_FILES)
        and binding['helpers']['src/world_reward/coherent_pair_marginal_objective.py'] ==
            dict(bytes=13862, sha256='2542534221708a9a3146aa5ff25306b3957a0a101bdb98a181bac05ab10855c1')
        and binding['helpers']['src/world_reward/coherent_pair_partition.py'] ==
            dict(bytes=7344, sha256='63527123823696a357746dfb12cd0511843c45d3204d0dd657fd6d7aa3965c9f')
        and binding['helpers']['infra/coherent_pair_gpu_objective_probe.py'] ==
            dict(bytes=38953, sha256='7049240f015220afb12c0f3718577dbf8250e5ae9a9903e07bbcd4c0fcec74e5'),
        'Unchanged original math/runtime/partition helper bytes required')
    _, _, cohort, lineage = context.authenticate(code, revision, ENTRY, closure, checkpoint)
    selection = membership(cohort, saved['public_proof']['records'])
    return dict(saved=saved, cohort_lineage=lineage, selection=selection, image=image_graph(deadline))


def image_graph(deadline):
    """Historical exact sealed-export graph plus current ordered rootfs, no TAR."""
    import research_image_identity as identity
    # Independent scalar pins are added only after the read-only image observer.
    rt.require(bool(IMAGE_PINS), 'Independently frozen historical graph pins required')
    folder = ROOT/'transfer/vm02-v1'
    values, states = {}, {}
    for n, pin in IMAGE_PINS.items():
        path = rt.canonical(folder/n); s = path.lstat()
        uid, mode = (1000, 0o600) if n == 'export.json' else (0, 0o644)
        rt.require(s.st_uid == s.st_gid == uid and stat.S_IMODE(s.st_mode) == mode
            and rt.identity(path, MAX_CONTROL, readonly=False) == pin, 'Exact legacy writable receipt identity/owner/mode required')
        states[str(path)] = join.snapshot(path); values[n] = rt.strict(path.read_bytes())
        rt.require(join.snapshot(path) == states[str(path)] and rt.identity(path, MAX_CONTROL, readonly=False) == pin,
            'Legacy graph receipt drift during read')
    imported, graph, exported = (values[n] for n in ('import.json', 'image-identity-v2.json', 'export.json'))
    rt.require(imported['stage'] == 'world_reward_research_import' and imported['status'] == 'pass'
        and imported['source_OCI_index_id'] == graph['source_OCI_index_id'] == identity.INDEX_ID
        and imported['image_id'] == graph['image_id'] == identity.CONFIG_ID == IMAGE
        and imported['sealed_image_tar_sha256'] == graph['sealed_image_tar_sha256'] == identity.EXPORT_SHA
        and imported['rootfs_layers'] == graph['rootfs_layers'] == 44
        and imported['rootfs_diff_ids'] == graph['rootfs_diff_ids']
        and imported['image_content_changed'] is imported['image_rebuilt'] is False
        and graph['image_content_changed'] is graph['image_rebuilt'] is False
        and imported['export_receipt_sha256'] == IMAGE_PINS['export.json']['sha256'],
        'Exact historical image graph/import/export receipt binding required')
    live = join.image(deadline)
    rt.require(live['RootFS']['Layers'] == graph['rootfs_diff_ids'], 'Current full ordered44 layers differ')
    rt.require(graph['platform_manifest_id'] == imported['platform_manifest_id'] ==
        'sha256:de690d04d890f7eaf301babee01e6224bf874d41d06615cba9efd20f8596aea2', 'Exact source platform graph required')
    rt.require(all(join.snapshot(Path(n)) == s and rt.identity(Path(n), MAX_CONTROL, readonly=False) ==
        IMAGE_PINS[Path(n).name] for n, s in states.items()), 'Legacy graph receipt changed after live inspect')
    return dict(pins=IMAGE_PINS, values=values, live=live, states=states,
        legacy_receipts_writable=True, sealed_archive_rehashed=False, historical_GPU_qualification_inherited=False)


IMAGE_PINS = {
    'export.json': dict(bytes=647, sha256='2bcb3c0e11ecacc6cd6051447380760eda20266a7dfe17420c47e41a2a196d8b'),
    'import.json': dict(bytes=4272, sha256='c98fb4ec995f2cbaecaeabecb0e1b35a5192e6a828efb5710f599197fd67b549'),
    'image-identity-v2.json': dict(bytes=3778, sha256='9bf4d946b2ad116c57c2fb88cdb37f92535df5b3238a069deec1a592cffb80af')}


def public_payload(before, revision):
    binding = before['saved']['host_lineage']['current_source']['binding']
    return dict(schema=SCHEMA, producer_revision=revision, image_id=IMAGE, manifest=manifest(),
        public_proof=before['saved']['public_proof'], summary_rows=before['saved']['rows'],
        selection=list(before['selection']),
        source=dict(files={n: binding['helpers'][n] for n in NATIVE_FILES}, markers=binding['markers']))


def native_source(code, revision, payload):
    rt.require(set(payload) == {'schema', 'producer_revision', 'image_id', 'manifest',
        'public_proof', 'summary_rows', 'selection', 'source'}
        and payload['schema'] == SCHEMA and payload['producer_revision'] == revision
        and payload['image_id'] == IMAGE and payload['manifest'] == manifest()
        and type(payload['source']) is dict and set(payload['source']) == {'files', 'markers'}
        and type(payload['source']['files']) is dict and set(payload['source']['files']) == set(NATIVE_FILES)
        and set(payload['source']['markers']) == {'revision', 'source-sha256'}
        and {str(p.relative_to(code)) for p in code.rglob('*') if p.is_file()} == set(NATIVE_FILES),
        'Exact thin public native source/schema whitelist required')
    join.validate_population(payload['public_proof'])
    validate_selection(payload['selection'], payload['public_proof']['records'])
    rt.require(payload['source']['files'][join.NATIVE_FILES[2]] == join.NUMERICAL_PIN,
        'Unchanged full numerical constructor required')
    paths = [*(code/n for n in NATIVE_FILES), *(code.parent/n for n in ('revision', 'source-sha256'))]
    states = {str(p): dict(identity=rt.identity(p, 2 << 20, empty=True), stat=join.snapshot(p)) for p in paths}
    rt.require(all(states[str(code/n)]['identity'] == pin for n, pin in payload['source']['files'].items())
        and all(states[str(code.parent/n)]['identity'] == pin for n, pin in payload['source']['markers'].items())
        and (code.parent/'revision').read_bytes() == (revision+'\n').encode(), 'Native source/markers differ')
    for n in NATIVE_FILES:
        key = Path(n).stem if n.startswith('infra/') else '.'.join(Path(n).with_suffix('').parts[1:])
        module = sys.modules.get(key)
        if module is not None: rt.require(Path(module.__file__).resolve() == code/n, 'Actual native helper origin differs')
    return states


def validate_selection(selection, records):
    rt.require(type(selection) is list and len(selection) == 32, 'Fixed32 public membership required')
    ids = []
    for i, row in enumerate(selection):
        rt.require(type(row) is dict and set(row) == {'slot', 'image_id', 'fold'}
            and type(row['slot']) is int and row['slot'] == i and type(row['fold']) is int and 0 <= row['fold'] < 4
            and type(row['image_id']) is str and re.fullmatch('[0-9a-f]{32}', row['image_id'])
            and row['image_id'] == records[i]['endpoint']['image_id'], 'Exact opaque32 slot/fold/order required')
        ids.append(row['image_id'])
    rt.require(len(set(ids)) == 32, 'Injective public FIT IDs required')
    order = sorted(ids, key=lambda iid: (hashlib.sha256((CV_NAMESPACE+iid).encode()).hexdigest(), iid))
    folds = {iid: i % 4 for i, iid in enumerate(order)}
    rt.require(all(r['fold'] == folds[r['image_id']] for r in selection), 'Predeclared role-independent CV fold hash differs')


def write_native(report, owner, deadline):
    allowed = {'proof.json', '.container.cid'}
    s = OUTPUT.lstat()
    rt.require(join.snapshot(OUTPUT)[:2] == owner[:2] and s.st_uid == s.st_gid == 0
        and stat.S_IMODE(s.st_mode) == 0o700 and {p.name for p in OUTPUT.iterdir()} == allowed,
        'Owned native namespace required')
    with (OUTPUT/'native.json').open('x+b') as stream:
        os.fchmod(stream.fileno(), 0o400); opened = os.fstat(stream.fileno())
        def update():
            parent, leaf = OUTPUT.lstat(), (OUTPUT/'native.json').lstat()
            rt.require(join.snapshot(OUTPUT)[:2] == owner[:2]
                and stat.S_ISDIR(parent.st_mode) and stat.S_IMODE(parent.st_mode) == 0o700
                and parent.st_uid == parent.st_gid == 0 and stat.S_ISREG(leaf.st_mode)
                and stat.S_IMODE(leaf.st_mode) == 0o400 and leaf.st_uid == leaf.st_gid == 0 and leaf.st_nlink == 1
                and join.snapshot(OUTPUT/'native.json')[:2] == (opened.st_dev, opened.st_ino)
                and {p.name for p in OUTPUT.iterdir()} == allowed|{'native.json'}, 'Foreign native output rejected')
            raw = join.encode(report); rt.require(len(raw) <= MAX_CONTROL, 'Bounded scalar native receipt required')
            stream.seek(0); stream.write(raw); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        try:
            update(); join.sync(OUTPUT); check(deadline)
        except BaseException as exc:
            report.update(status='fail', publication_failed=True, publication_error_type=join.error(exc)); update()


def native(code, revision, pin, deadline):
    report = dict(schema=SCHEMA, stage='blind32_partition_native', status='fail', phase='preflight',
        producer_revision=revision, image_id=IMAGE, manifest=manifest(), proof_identity=pin,
        source_rehashed_after=False, **{n: False for n in FALSE_FLAGS})
    owner = join.snapshot(OUTPUT); before = payload = None
    def cancelled(*_): raise TimeoutError('Shared blind native deadline')
    handlers = {s: signal.signal(s, cancelled) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    try:
        check(deadline); payload = rt.pinned(OUTPUT/'proof.json', pin, MAX_CONTROL)
        before = native_source(code, revision, payload); proof_state = join.snapshot(OUTPUT/'proof.json')
        rt.require(os.geteuid() == 0 and sys.platform == 'linux' and sys.version_info[:2] == (3, 11)
            and os.environ.get('WR_IMAGE_ID') == IMAGE and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
            and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'}, 'Actual offline sealed7eb native required')
        import numpy as np
        import torch as t
        import coherent_pair_gpu_probe as g
        import vcoco_selector_blind_cost_native as worker
        worker.measure(np, t, g, rt, payload['public_proof'], payload['summary_rows'],
            tuple(payload['selection']), lambda: check(deadline), report)
        report.update(status='pass', phase='complete')
    except BaseException as exc: report.update(error_type=join.error(exc))
    finally:
        try:
            rt.require(before is not None and native_source(code, revision, payload) == before
                and rt.identity(OUTPUT/'proof.json', MAX_CONTROL) == pin
                and join.snapshot(OUTPUT/'proof.json') == proof_state, 'Native source/public proof drift')
            report['source_rehashed_after'] = True; check(deadline)
        except BaseException as exc: report.update(status='fail', post_error_type=join.error(exc))
        try: write_native(report, owner, deadline)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in handlers.items(): signal.signal(s, handler)
    return report


def cleanup(name, revision, deadline):
    cidfile = OUTPUT/'.container.cid'
    if cidfile.exists() or cidfile.is_symlink():
        rt.canonical(cidfile); s = cidfile.lstat(); state = join.snapshot(cidfile)
        rt.require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == s.st_gid == 0
            and stat.S_IMODE(s.st_mode) in (0o400, 0o600, 0o644), 'Original root single-link CID required')
        pin = rt.identity(cidfile, 65, readonly=False)
        raw = cidfile.read_bytes(); rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', raw)
            and join.snapshot(cidfile) == state and rt.identity(cidfile, 65, readonly=False) == pin, 'Owned CID required')
        cid = raw.decode().strip(); found = join.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline)
        rt.require(found in ('', cid), 'Ambiguous CID')
        if found:
            value = join.command(['docker', 'inspect', cid, '--format',
                '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'], deadline)
            rt.require(value == IMAGE+'|/'+name+'|'+ENTRY+'|'+revision, 'Never remove foreign container')
            join.command(['docker', 'rm', '-f', cid], deadline)
        join.absent(raw, name, deadline)
        rt.require(join.snapshot(cidfile) == state and rt.identity(cidfile, 65, readonly=False) == pin, 'Owned CID drift before seal')
        fd = os.open(cidfile, os.O_RDONLY|os.O_NOFOLLOW)
        try:
            now = os.fstat(fd); rt.require((now.st_dev, now.st_ino) == state[:2], 'Original CID descriptor required')
            os.fchmod(fd, 0o400)
        finally: os.close(fd)
    rt.require(not join.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Owned name survives')


def host(code, revision):
    import sealed_callback_publication as publication
    started = time.monotonic(); deadline = started+BUDGET
    rt.canonical(OUTPUT); rt.require(not OUTPUT.exists(), 'Fresh one-shot blind profile namespace required')
    OUTPUT.mkdir(mode=0o700); owner = join.snapshot(OUTPUT)[:2]+(OUTPUT.stat().st_uid,)
    before = payload = native_value = None; name = 'world-reward-vcoco-selector-blind-cost-'+revision[:12]
    report = dict(schema=SCHEMA, stage='blind32_partition_host', status='fail', phase='source',
        producer_revision=revision, image_id=IMAGE, manifest=manifest(), owned_cleanup_verified=False,
        source_inputs_image_rehashed_after=False, decision='CLOSED_BLIND_COST', **{n: False for n in FALSE_FLAGS})
    def cancelled(*_): raise TimeoutError('Inclusive3600s blind host')
    handlers = {s: signal.signal(s, cancelled) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL, BUDGET)
    try:
        before = authenticate(code, revision, deadline); payload = public_payload(before, revision)
        report.update(input_proof=before, selection_identity=dict(bytes=len(join.encode(payload['selection'])),
            sha256=hashlib.sha256(join.encode(payload['selection'])).hexdigest()))
        rt.require(not join.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Occupied new name')
        rt.write(OUTPUT/'proof.json', join.encode(payload), 0o400); pin = rt.identity(OUTPUT/'proof.json', MAX_CONTROL)
        report['public_proof_identity'] = pin
        argv = ['docker', 'run', '--rm', '--name', name, '--cidfile', str(OUTPUT/'.container.cid'),
            '--label', 'world-reward.job='+ENTRY, '--label', 'world-reward.revision='+revision,
            '--gpus', 'all', '--network', 'none', '--user', '0:0', '--memory', '64g', '--memory-swap', '64g',
            '--cpus', '4', '--pids-limit', '256', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=256m', '--entrypoint', '/usr/bin/env']
        for path in [*(code/n for n in NATIVE_FILES), *(code.parent/n for n in ('revision', 'source-sha256')),
                     *(Path(n) for n in payload['public_proof']['files'])]:
            rt.canonical(path); rt.require(path.is_file() and not any(c in str(path) for c in (',', '\n')), 'Public individual mount required')
            argv += ['--mount', f'type=bind,src={path},dst={path},readonly']
        argv += ['--mount', f'type=bind,src={OUTPUT},dst={OUTPUT}', IMAGE, '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin', 'HOME=/tmp',
            'WR_IMAGE_ID='+IMAGE, 'CUBLAS_WORKSPACE_CONFIG=:4096:8', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
            'MKL_NUM_THREADS=4', '/opt/conda/bin/python', '-I', '-B', str(code/'infra/vcoco_selector_blind_cost.py'),
            '--native', '--code', str(code), '--revision', revision, '--proof-bytes', str(pin['bytes']),
            '--proof-sha256', pin['sha256'], '--deadline', format(deadline-30, '.17g')]
        report['phase'] = 'native'
        result = subprocess.run(argv, env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent', DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=max(.001, deadline-time.monotonic()-30), check=False)
        report['native_exit_status'] = result.returncode; report['native_identity'] = rt.identity(OUTPUT/'native.json', MAX_CONTROL)
        native_value = rt.pinned(OUTPUT/'native.json', report['native_identity'], MAX_CONTROL)
        validate_native(native_value, payload, pin); rt.require(result.returncode == 0, 'Native profile failed')
        report.update(status='pass', phase='complete', decision=DECISION)
    except BaseException as exc: report.update(error_type=join.error(exc), failure_stage=report['phase'])
    finally:
        try: cleanup(name, revision, min(deadline+10, time.monotonic()+10)); report['owned_cleanup_verified'] = True
        except BaseException as exc: report.update(status='fail', cleanup_error_type=join.error(exc))
        try:
            after = authenticate(code, revision, deadline)
            rt.require(before is not None and join.encode(before) == join.encode(after), 'Full sources/cohort/inputs/image drift')
            report['source_inputs_image_rehashed_after'] = True
        except BaseException as exc: report.update(status='fail', post_error_type=join.error(exc))
        allowed = {p.name for p in OUTPUT.iterdir()}
        rt.require(allowed <= {'proof.json', 'native.json', '.container.cid'}, 'Foreign blind output')
        for n in allowed:
            s = rt.canonical(OUTPUT/n).lstat()
            rt.require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == s.st_gid == 0
                and stat.S_IMODE(s.st_mode) == 0o400 and s.st_size <= MAX_CONTROL, 'Exact owned bounded blind leaf required')
        report['saved_outputs'] = {n: rt.identity(OUTPUT/n, MAX_CONTROL, empty=True) for n in allowed}
        report['invalid_empty_output_leaves'] = sorted(n for n, p in report['saved_outputs'].items() if p['bytes'] == 0)
        if report['invalid_empty_output_leaves']: report['status'] = 'fail'
        if report['status'] != 'pass': report['decision'] = 'CLOSED_BLIND_COST'
        def receipt_encode(value):
            if value['status'] != 'pass': value['decision'] = 'CLOSED_BLIND_COST'
            return join.encode(value)
        try:
            def publication_identity(path, maximum):
                empty = Path(path).name in report['invalid_empty_output_leaves']
                return rt.identity(path, maximum, empty=empty)
            publication.publish(OUTPUT, report, deadline, started, owner, allowed, encode=receipt_encode, identity=publication_identity,
                snapshot=join.snapshot, require=rt.require, check=check, sync=join.sync, error=join.error,
                maximum=MAX_CONTROL, report_maximum=16 << 20)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in handlers.items(): signal.signal(s, handler)
    return report


def validate_native(value, payload, pin):
    rt.require(value['schema'] == SCHEMA and value['stage'] == 'blind32_partition_native'
        and value['status'] == 'pass' and value['phase'] == 'complete'
        and value['producer_revision'] == payload['producer_revision'] and value['image_id'] == IMAGE
        and value['manifest'] == manifest() and value['proof_identity'] == pin
        and value['source_rehashed_after'] is True and all(value[n] is False for n in FALSE_FLAGS)
        and not any(n in value for n in ('error_type', 'post_error_type', 'publication_failed')),
        'Complete unrescued native profile receipt required')
    # Worker-specific full32/runtime/array/memory checks are wired after release.
    import vcoco_selector_blind_cost_native as worker
    worker.validate_receipt(rt, value, tuple(payload['selection']))


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--native', action='store_true'); p.add_argument('--code'); p.add_argument('--revision')
    p.add_argument('--proof-bytes', type=int); p.add_argument('--proof-sha256'); p.add_argument('--deadline', type=float)
    a = p.parse_args(); code = Path(a.code or os.environ['WR_CODE']); revision = a.revision or os.environ['WR_CODE_REVISION']
    rt.require(re.fullmatch('[0-9a-f]{40}', revision) and code == ROOT/'jobs'/revision/ENTRY/'code'
        and Path(__file__).resolve() == code/'infra/vcoco_selector_blind_cost.py', 'Exact new immutable caller required')
    if a.native:
        rt.require(type(a.deadline) is float and 0 < a.deadline-time.monotonic() <= BUDGET, 'Finite shared native deadline required')
        value = native(code, revision, dict(bytes=a.proof_bytes, sha256=a.proof_sha256), a.deadline)
    else:
        rt.require(a.code is a.revision is a.proof_bytes is a.proof_sha256 is a.deadline is None
            and os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02'
            and os.environ.get('DOCKER_HOST') == 'unix://'+str(ROOT/'docker.sock'), 'Actual VM02 root host only')
        node = ROOT/'jobs/.world-reward-h100.lock'; s = node.lstat(); fd = os.fstat(9)
        rt.require(not node.is_symlink() and stat.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino) == (fd.st_dev, fd.st_ino), 'Actual FD9 lease required')
        value = host(code, revision)
    print(join.encode(dict(status=value['status'], decision=value.get('decision'), error_type=value.get('error_type'))).decode(), end='')
    if value['status'] != 'pass': raise SystemExit(1)


if __name__ == '__main__': main()
