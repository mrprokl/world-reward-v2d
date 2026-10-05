"""Frozen external object-attribution metrics, Azure CPU only, no RGB/model."""
import argparse
from dataclasses import asdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time
import urllib.request

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_openimages_holds_evaluate'
CONFIG = 'configs/openimages_holds_evaluate_v1.json'
CONFIG_PIN = {'bytes': 2146, 'sha256': '61ec7c1cb6c7f2c0375dfc584cc5dd07da066069abdc88058e9cb62c3d10066d'}
BUDGET_SECONDS, CHILD_SECONDS, CLEANUP_SECONDS = 600, 300, 60
HELPERS = ('infra/openimages_holds_evaluate.py', 'infra/run_openimages_holds_evaluate.sh',
           CONFIG, 'infra/mediapipe_cpu_runtime_verify.py', 'infra/hoi_detr_model_qualify.py',
           'src/world_reward/hoi_object_ranking_evaluation.py', 'src/world_reward/openimages_holds_reference.py')


def require(value, message):
    if not value: raise ValueError(message)


def check(deadline):
    if time.monotonic() >= deadline: raise TimeoutError('Declared evaluator deadline exceeded')


def _alarm(*_):
    raise TimeoutError('Declared evaluator deadline exceeded')


def helper(code):
    p = code/'infra/mediapipe_cpu_runtime_verify.py'; b = p.read_bytes()
    require(len(b) == 23559 and hashlib.sha256(b).hexdigest() == '936ad97c5ffca3b7f86e3462a600a247b6fa540d9b669e748892ff45e12aedf2', 'Pinned readonly stdlib bootstrap required')
    s = importlib.util.spec_from_file_location('wr_oi_eval_rt', p)
    rt = importlib.util.module_from_spec(s); s.loader.exec_module(rt); return rt


def acquire_references(c, out, deadline=None, *, owned=None, sources=None):
    """Only after all predictions bind. No images/calibration/other tracks."""
    deadline = time.monotonic()+BUDGET_SECONDS if deadline is None else deadline
    owned = [] if owned is None else owned
    sources = [] if sources is None else sources
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args): raise ValueError('Reference redirect forbidden')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    for row in c['references']:
        check(deadline)
        require(row['url'].startswith('https://storage.googleapis.com/openimages/')
                and re.fullmatch(r'[a-z]+\.csv', row['file']) and type(row['maximum_bytes']) is int
                and 0 < row['maximum_bytes'] <= 256 << 20, 'Exact official Open Images reference URL required')
        with opener.open(row['url'], timeout=min(60, max(.001, deadline-time.monotonic()))) as response:
            chunks = []; size = 0
            while True:
                check(deadline); block = response.read(min(1 << 20, row['maximum_bytes']+1-size))
                if not block: break
                chunks.append(block); size += len(block)
                require(size <= row['maximum_bytes'], 'Bounded exact publisher reference response required')
            raw = b''.join(chunks)
            require(response.status == 200 and response.url == row['url'] and 0 < len(raw) <= row['maximum_bytes'], 'Bounded exact publisher reference response required')
            etag = response.headers.get('ETag', '').strip('"')
        pin = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        require(len(etag) == 32 and hashlib.md5(raw).hexdigest() == etag, 'Publisher reference ETag MD5 required')
        if row.get('identity') is not None: require(pin == row['identity'], 'Independently pinned vocabulary differs')
        path = out/row['file']
        fd = os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW, 0o600)
        t = os.fstat(fd); owned.append((path, (t.st_dev, t.st_ino, t.st_uid)))
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
        sources.append(dict(file=row['file'], url=row['url'], identity=pin, publisher_etag_md5=etag))
        check(deadline)
    return sources


def evaluate(c, out):
    import sys
    sys.path.insert(0, str(Path(os.environ['WR_CODE'])/'src'))
    import numpy as np
    from world_reward.hoi_detr_observations import HOIDetrObservations
    from world_reward.hoi_object_ranking_evaluation import evaluate_hoi_object_ranking, exact_image_sign_test
    from world_reward.openimages_holds_reference import parse_holds_reference, parse_holds_vocabulary
    require(os.geteuid() == 1000 and {x.name for x in Path('/sys/class/net').iterdir()} == {'lo'}, 'Offline CPU evaluator required')
    p = c['predictions']; acquisition = json.loads(Path(c['acquisition_manifest']['path']).read_bytes())
    observations = json.loads((ROOT/p['folder']/'observations_manifest.json').read_bytes())['observations']
    by_id = {r['image_id']: r for r in observations}
    ids = [r['image_id'] for r in acquisition['records']]; require(len(ids) == len(set(ids)) == 128, 'Complete fixed128 reference IDs required')
    def rows(file):
        with (out/file).open() as f:
            return [r for r in csv.DictReader(f) if r['ImageID'] in set(ids)]
    boxes, relations = rows('boxes.csv'), rows('relations.csv')
    with (out/'triplets.csv').open() as f: vocabulary = parse_holds_vocabulary(list(csv.reader(f)))
    results = []; deltas = []; retrieval_a = []; retrieval_b = []
    for meta in acquisition['records']:
        iid = meta['image_id']; record = dict(image_id=iid, acquisition_status=meta['status'])
        # Complete closed slots remain misses. Reference geometry is not needed
        # to invent absent predictions; cohort selection guaranteed positives.
        if iid not in by_id:
            record.update(retrieval_a=0., retrieval_b=0., informative=False, missing=True)
            retrieval_a.append(0.); retrieval_b.append(0.); results.append(record); continue
        row = by_id[iid]
        with np.load(ROOT/p['folder']/row['file'], allow_pickle=False) as arrays:
            obs = HOIDetrObservations(row['original_frame_index'], tuple(row['image_size']), **{k: arrays[k] for k in arrays.files})
        ref = parse_holds_reference([r for r in boxes if r['ImageID'] == iid], [r for r in relations if r['ImageID'] == iid])
        require(ref.total_positive_person_slots > 0, 'Original selected image must retain manual positive person endpoints')
        value = evaluate_hoi_object_ranking(obs, ref.persons, ref.objects, ref.holds, vocabulary, reference_coordinates='normalized')
        # Endpoint slots, not only successfully matched persons, define the
        # retrieval denominator. Any unresolved positive endpoint is a miss.
        aa, bb = [], []
        for person, unscorable in zip(ref.positive_person_indices, ref.positive_person_unscorable):
            aa.append(0. if person < 0 or unscorable else float(value.retrieval_a[person]))
            bb.append(0. if person < 0 or unscorable else float(value.retrieval_b[person]))
        a, b = float(np.mean(aa)), float(np.mean(bb)); retrieval_a.append(a); retrieval_b.append(b)
        record.update(missing=False, retrieval_a=a, retrieval_b=b, positive_person_slots=ref.total_positive_person_slots,
                      unscorable_positive_person_slots=ref.unbound_positive_person_count, binding=dict(ref.binding_diagnostics),
                      native_roles=[int((obs.class_ids == i).sum()) for i in range(3)], comparison_count=int(value.comparison_count.sum()),
                      matched_positive_count=int(value.matched_positive_count.sum()), matched_negative_count=int(value.matched_negative_count.sum()),
                      eligible_hand_count=int(value.eligible_hand_count.sum()), unknown_candidates=int(value.unknown_candidate_count.sum()))
        # Exclude reference-unscorable persons from conditional comparisons,
        # never their misses from the all-slot retrieval safety denominator.
        valid = sorted(set(int(i) for i, u in zip(ref.positive_person_indices, ref.positive_person_unscorable) if i >= 0 and not u))
        valid = [i for i in valid if value.comparison_count[i] > 0]
        record['informative'] = bool(valid)
        if valid:
            ca, cb = float(value.concordance_a[valid].mean()), float(value.concordance_b[valid].mean())
            deltas.append(cb-ca); record.update(concordance_a=ca, concordance_b=cb)
        results.append(record)
    test = asdict(exact_image_sign_test(np.array(deltas, np.float64))) if deltas else None
    a, b = float(np.mean(retrieval_a)), float(np.mean(retrieval_b))
    sufficient = len(deltas) >= 6 and test is not None and test['sufficient_images']
    decision = 'inconclusive' if not sufficient else ('promising_component_only' if test['mean_delta'] > 0 and test['p_one_sided'] <= .05 and b >= a else 'reject')
    result = dict(schema='world_reward.openimages_object_attribution.v1', status='pass', cohort_slots=128, results=results,
                  informative_images=len(deltas), test=test, retrieval_a=a, retrieval_b=b, decision=decision,
                  reference_geometry_evaluator_only=True, hand_ownership_verified=False, contact_verified=False,
                  training_overlap_verified=False, CARI4D_superiority_verified=False, challenge_inputs_used=False, adoption=False)
    with (out/'metrics.json').open('xb') as f: f.write((json.dumps(result, sort_keys=True, allow_nan=False)+'\n').encode()); f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o444)


def authenticate_input_qualification(rt, c, acquisition, deadline):
    """Recheck immutable original128 metadata, never RGB or model outputs."""
    check(deadline)
    original = c['original_acquisition_manifest']
    parent = rt.pinned(Path(original['path']), original['identity'], 256 << 10)
    module = Path(__file__).parent/'hoi_detr_model_qualify.py'
    spec = importlib.util.spec_from_file_location('wr_oi_eval_header_contract', module)
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    gate.validate_external_cohort_qualification(parent, acquisition, original['identity'],
                                               str(Path(original['path']).parent), 16 << 20)
    check(deadline)


def authenticate_predictions(rt, c, deadline):
    """Hash-only original source and ALL predictions before any reference I/O."""
    check(deadline); p = c['predictions']; folder = rt.canonical(ROOT/p['folder'])
    h = rt.pinned(folder/'report.json', p['host'], 32 << 10); n = rt.pinned(folder/'native.json', p['native'], 32 << 10)
    require(h['status'] == n['status'] == 'pass' and h['producer_revision'] == p['producer_revision']
            and n['native_model_loads'] == 1 and n['fixed_cohort_slots'] == 128 and n['reference_geometry_read'] is False,
            'Complete frozen native cohort PASS required before references')
    require(h['phase'] == 'complete' and h['native_report_identity'] == p['native'] and h['actual_model_qualified'] is True
            and n['strict_checkpoint']['keys'] == 1796 and n['strict_checkpoint']['weights_only'] is True and n['strict_checkpoint']['strict'] is True
            and all(h[k] is True for k in ('source_rehashed_after', 'inputs_rehashed_after', 'image_unchanged', 'owned_containers_removed', 'owned_overlay_removed'))
            and all(h[k] is False and n[k] is False for k in ('ground_truth_used', 'challenge_inputs_used', 'adoption', 'quality_verified')),
            'Actual full native strict-load/provenance/cleanup scope required')
    original = ROOT/'jobs'/p['producer_revision']/'run_hoi_detr_model_qualify/code'
    sb = h['source_binding']
    source_parent(rt, original)
    require(rt.source(ROOT, original, p['producer_revision'], 'run_hoi_detr_model_qualify', tuple(sb['helpers'])) == sb == n['source_binding'], 'Whole original prediction source differs')
    manifest = rt.pinned(folder/'observations_manifest.json', p['observations_manifest'], 256 << 10)
    require(n['observations'] == p['observations_manifest'] and h['observations_identity'] == n['observations'], 'Complete observation manifest must bind both host/native')
    require(len({r['image_id'] for r in manifest['observations']}) == len(manifest['observations'])
            and len({r['file'] for r in manifest['observations']}) == len(manifest['observations']), 'Unique complete native observation records required')
    for row in manifest['observations']:
        check(deadline)
        require(re.fullmatch(r'[0-9a-f]{16}\.npz', row['file']), 'Original observation file leaf required')
        require(rt.identity(folder/row['file'], 32 << 20) == row['observations'], 'Every original native observation pin required')
    require(len(manifest['observations']) == n['native_forward_calls'] == n['acquired_slots'], 'All acquired images completed before references')
    acquisition = rt.pinned(Path(c['acquisition_manifest']['path']), c['acquisition_manifest']['identity'], 256 << 10)
    authenticate_input_qualification(rt, c, acquisition, deadline)
    records = acquisition['records']; ids = [r['image_id'] for r in records]
    acquired = [r['image_id'] for r in records if r['status'] == 'acquired']
    require(len(ids) == len(set(ids)) == 128 and len(acquired) == n['acquired_slots'] == 61 and n['missing_slots'] == 67
            and set(r['image_id'] for r in manifest['observations']) == set(acquired)
            and all(r['file'] == r['image_id']+'.npz' and r['original_frame_index'] == 0 for r in manifest['observations']),
            'All and only61 exact qualified slots must be predicted; no partial cohort')
    check(deadline)
    return dict(host=h, native=n, manifest=manifest, acquisition=acquisition), original, folder


def source_parent(rt, code):
    rt.canonical(code)
    require({p.name for p in code.parent.iterdir()} == {'code', 'revision', 'source-sha256'}, 'Exact source-only parent inventory required')


def command(args, deadline, *, log=None):
    check(deadline)
    env = dict(PATH='/usr/bin:/bin', DOCKER_HOST='unix://'+str(ROOT/'docker.sock'))
    kwargs = dict(env=env, timeout=max(.001, min(15 if log is None else CHILD_SECONDS, deadline-time.monotonic())))
    if log is None:
        result = subprocess.run(args, capture_output=True, **kwargs)
        require(len(result.stdout) <= 1 << 20 and len(result.stderr) <= 1 << 20, 'Bounded Docker metadata required')
    else:
        fd = os.open(log, os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            result = subprocess.run(args, stdout=stream, stderr=stream, **kwargs)
            stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    require(result.returncode == 0, 'Owned evaluator Docker command failed')
    check(deadline)
    return result.stdout.decode().strip() if log is None else ''


def validate_container(value, name, revision, image_id, mounts):
    h, c = value['HostConfig'], value['Config']
    require(value['Image'] == image_id and value['Name'] == '/'+name
            and c['User'] == '1000:1000' and c['Labels'].get('world-reward.job') == ENTRY
            and c['Labels'].get('world-reward.revision') == revision
            and h['NetworkMode'] == 'none' and h['ReadonlyRootfs'] is True and not h['Privileged']
            and h['CapDrop'] == ['ALL'] and 'no-new-privileges' in h['SecurityOpt']
            and h['NanoCpus'] == 4_000_000_000 and h['Memory'] == 4*(1 << 30), 'Exact owned offline CPU evaluator required')
    require({(m['Source'], m['Destination'], not m['RW']) for m in value['Mounts'] if m['Type'] == 'bind'} == mounts
            and not any(m['Type'] == 'volume' for m in value['Mounts'])
            and not h.get('DeviceRequests') and not h.get('Devices') and not h.get('Binds')
            and not h.get('VolumesFrom'), 'Exact source/prediction/reference-only mounts required')


def cleanup_container(cidfile, name, revision, image_id, mounts, deadline):
    if not cidfile.exists():
        require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'No unknown evaluator without CID may be removed')
        return
    t = cidfile.lstat()
    require(stat.S_ISREG(t.st_mode) and t.st_nlink == 1 and t.st_uid == 0 and 64 <= t.st_size <= 65, 'Own regular evaluator CID required')
    raw = cidfile.read_bytes()
    require(re.fullmatch(b'[0-9a-f]{64}\n?', raw), 'Exact evaluator CID required'); cid = raw.decode().strip()
    ids = command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline)
    require(ids in ('', cid), 'Exact original CID inventory required')
    if ids:
        value = json.loads(command(['docker', 'inspect', cid, '--format', '{{json .}}'], deadline))
        validate_container(value, name, revision, image_id, mounts)
        require(command(['docker', 'rm', '-f', cid], deadline) == cid, 'Own evaluator removal failed')
    require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline)
            and not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Evaluator survived independent cleanup checks')
    cidfile.chmod(0o444)


def write_receipt(path, report, started, deadline):
    """An owned descriptor demotes any late/error PASS before final sealing."""
    def raw():
        b = (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode()
        require(len(b) <= 64 << 10, 'Bounded evaluator receipt required'); return b
    fd = os.open(path, os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w+b') as stream:
        try:
            stream.write(raw()); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
            if report['status'] == 'pass': check(deadline)
        except Exception as exc:
            report.update(status='fail', phase='receipt_sealing', receipt_error_type=type(exc).__name__, elapsed_seconds=time.monotonic()-started)
            stream.seek(0); stream.truncate(); stream.write(raw()); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)


def run(code, revision, child=False):
    started = time.monotonic(); deadline = started+BUDGET_SECONDS
    previous_handler = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(BUDGET_SECONDS)
    try:
        rt = helper(code); source_parent(rt, code); own = rt.source(ROOT, code, revision, ENTRY, HELPERS)
        c = rt.pinned(code/CONFIG, CONFIG_PIN, 16 << 10)
        require(re.fullmatch('sha256:[0-9a-f]{64}', c['image_id']), 'Exact immutable evaluator image ID required')
        for key, expected in (('budget_seconds', BUDGET_SECONDS), ('child_seconds', CHILD_SECONDS), ('cleanup_grace_seconds', CLEANUP_SECONDS)):
            require(c.get(key, expected) == expected, 'Frozen evaluator resource deadline required')
        out = rt.canonical(ROOT/c['output'])
        if child:
            deadline = min(deadline, float(os.environ['WR_EVALUATOR_DEADLINE']))
            signal.alarm(max(1, math.ceil(deadline-time.monotonic())))
            require(os.geteuid() == 1000, 'Offline native evaluator UID required')
            prior, _, _ = authenticate_predictions(rt, c, deadline)
            proof_pin = dict(bytes=int(os.environ['WR_REFERENCE_PROOF_BYTES']), sha256=os.environ['WR_REFERENCE_PROOF_SHA256'])
            proof = rt.pinned(out/'reference_proof.json', proof_pin, 64 << 10)
            require(proof['source_binding'] == own and proof['protocol_identity'] == CONFIG_PIN, 'Independent reference proof/source binding required')
            for row in proof['references']:
                check(deadline); require(rt.identity(out/row['file'], 256 << 20) == row['identity'], 'Complete private reference identity required')
            evaluate(c, out)  # Numerical function/acceptance gates are unchanged.
            require(authenticate_predictions(rt, c, deadline)[0] == prior, 'Original complete predictions changed in evaluator')
            for row in proof['references']:
                require(rt.identity(out/row['file'], 256 << 20) == row['identity'], 'Private references changed in evaluator')
            require(rt.source(ROOT, code, revision, ENTRY, HELPERS) == own, 'Native evaluator source changed')
            native = dict(stage='openimages_holds_evaluator_native', status='pass', source_binding=own,
                metrics=rt.identity(out/'metrics.json', 256 << 10), reference_geometry_evaluator_only=True,
                CPU_only=True, challenge_inputs_used=False, adoption=False, elapsed_seconds=time.monotonic()-started)
            write_receipt(out/'native.json', native, started, deadline)
            require(native['status'] == 'pass', 'Late native evaluator is not qualified')
            return native
        require(os.geteuid() == 0 and os.uname().nodename == 'world-reward-ncc-h100-02'
                and not out.exists() and out.parent.is_dir(), 'Fresh Azure evaluator namespace required')
        out.mkdir(mode=0o755); os.chown(out, 1000, 1000)
        t = out.lstat(); inode = (t.st_dev, t.st_ino, t.st_uid)
        return host_run(rt, code, revision, c, own, out, inode, started, deadline)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)


def host_run(rt, code, revision, c, own, out, inode, started, deadline):
    sources, downloads, prior = [], [], None
    name = 'world-reward-oi-eval-'+revision[:12]; cidfile = out/'container.cid'
    mounts = set(); attempted = False; image = None; failure = None
    report = dict(schema=c['schema'], stage='openimages_holds_evaluator_host', status='fail', phase='prediction_authentication',
        producer_revision=revision, source_binding=own, protocol_identity=CONFIG_PIN, budget_seconds=BUDGET_SECONDS,
        child_seconds=CHILD_SECONDS, cleanup_grace_seconds=CLEANUP_SECONDS, source_rehashed_after=False,
        predictions_rehashed_after=False, references_rehashed_after=False, owned_container_removed=False,
        reference_geometry_evaluator_only=True, CPU_only=True, challenge_inputs_used=False, adoption=False)
    try:
        prior, original, folder = authenticate_predictions(rt, c, deadline)
        image = json.loads(command(['docker', 'image', 'inspect', c['image_id'], '--format', '{{json .}}'], deadline))
        require(image['Id'] == c['image_id'] and image['Os'] == 'linux' and image['Architecture'] == 'amd64', 'Actual exact evaluator CPU image required')
        report['phase'] = 'reference_acquisition'
        acquire_references(c, out, deadline, owned=downloads, sources=sources)
        proof = dict(source_binding=own, protocol_identity=CONFIG_PIN, references=sources)
        rt.write(out/'reference_proof.json', (json.dumps(proof, sort_keys=True)+'\n').encode(), 0o444)
        pin = rt.identity(out/'reference_proof.json', 64 << 10)
        report['reference_proof_identity'] = pin
        require(not command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Existing evaluator never overridden')
        paths = (code.parent, original.parent, folder, Path(c['acquisition_manifest']['path']),
                 Path(c['original_acquisition_manifest']['path']))
        mounts = {(str(p), str(p), True) for p in paths} | {(str(out), str(out), False)}
        child_deadline = min(deadline, time.monotonic()+CHILD_SECONDS)
        args = ['docker', 'create', '--name', name, '--cidfile', str(cidfile), '--label', 'world-reward.job='+ENTRY,
            '--label', 'world-reward.revision='+revision, '--network', 'none', '--read-only', '--user', '1000:1000',
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '4', '--memory', '4g', '--tmpfs', '/tmp:rw,nosuid,size=64m']
        for src, dst, ro in sorted(mounts):
            args += ['--mount', f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
        args += ['--entrypoint', '/usr/bin/env', c['image_id'], '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin',
            'PYTHONDONTWRITEBYTECODE=1', 'WR_CODE='+str(code), 'WR_EVALUATOR_DEADLINE='+format(child_deadline, '.17g'),
            'WR_REFERENCE_PROOF_BYTES='+str(pin['bytes']), 'WR_REFERENCE_PROOF_SHA256='+pin['sha256'],
            '/opt/conda/bin/python', '-I', '-B', str(code/HELPERS[0]), '--code', str(code), '--revision', revision, '--child']
        report['phase'] = 'native_evaluation'; attempted = True
        command(args, deadline)
        validate_container(json.loads(command(['docker', 'inspect', name, '--format', '{{json .}}'], deadline)), name, revision, c['image_id'], mounts)
        command(['docker', 'start', '-a', name], child_deadline, log=out/'native.log')
        native_pin = rt.identity(out/'native.json', 64 << 10); native = rt.pinned(out/'native.json', native_pin, 64 << 10)
        require(native['status'] == 'pass' and native['source_binding'] == own and native['CPU_only'] is True, 'Complete native evaluator PASS required')
        require(rt.identity(out/'metrics.json', 256 << 10) == native['metrics'], 'Native metrics identity differs')
        report.update(status='pass', phase='complete', native_report_identity=native_pin, metrics=native['metrics'])
    except Exception as exc:
        failure = exc; report.update(error_type=type(exc).__name__)
        if type(exc) is ValueError and len(exc.args) == 1 and type(exc.args[0]) is str:
            # Only literal guards from this source, never arbitrary URLs/stdio.
            import ast
            literals = {n.value for n in ast.walk(ast.parse(Path(__file__).read_bytes())) if isinstance(n, ast.Constant) and type(n.value) is str}
            if exc.args[0] in literals and len(exc.args[0]) <= 200: report['requirement'] = exc.args[0]
    finally:
        signal.alarm(CLEANUP_SECONDS); grace = min(deadline+CLEANUP_SECONDS, time.monotonic()+CLEANUP_SECONDS)
        def own_output():
            t = out.lstat(); require(rt.canonical(out) == out and (t.st_dev, t.st_ino, t.st_uid) == inode, 'Owned evaluator output replaced')
        def attempt(key, action):
            nonlocal failure
            try: own_output(); action(); report[key] = True
            except Exception as exc: failure = failure or exc; report[key+'_error_type'] = type(exc).__name__
        def post_predictions():
            value = authenticate_predictions(rt, c, grace)[0]
            if prior is not None: require(value == prior, 'Full prior prediction ledger changed')
        def post_source():
            source_parent(rt, code); require(rt.source(ROOT, code, revision, ENTRY, HELPERS) == own, 'Full evaluator source changed')
        def post_references():
            for row in sources:
                check(grace); require(rt.identity(out/row['file'], 256 << 20) == row['identity'], 'Private reference changed')
            if 'reference_proof_identity' in report:
                require(rt.identity(out/'reference_proof.json', 64 << 10) == report['reference_proof_identity'], 'Original reference proof changed')
            if 'native_report_identity' in report:
                require(rt.identity(out/'native.json', 64 << 10) == report['native_report_identity']
                        and rt.identity(out/'metrics.json', 256 << 10) == report['metrics'], 'Original native metrics receipt changed')
        attempt('owned_container_removed', lambda: cleanup_container(cidfile, name, revision, c['image_id'], mounts, grace) if attempted else None)
        attempt('source_rehashed_after', post_source)
        attempt('predictions_rehashed_after', post_predictions)
        attempt('references_rehashed_after', post_references)
        if image is not None:
            attempt('image_unchanged', lambda: require(json.loads(command(['docker', 'image', 'inspect', c['image_id'], '--format', '{{json .}}'], grace)) == image, 'Original CPU image changed'))
        try:
            own_output()
            completed = {r['file'] for r in sources}
            for path, known in downloads:
                if path.name not in completed and path.exists():
                    t = path.lstat(); require(stat.S_ISREG(t.st_mode) and t.st_nlink == 1 and (t.st_dev, t.st_ino, t.st_uid) == known, 'Only own incomplete download removed'); path.unlink()
            if failure or time.monotonic() >= deadline: report['status'] = 'fail'
            if report['status'] == 'pass':
                t = (out/'native.log').lstat(); require(stat.S_ISREG(t.st_mode) and t.st_nlink == 1 and t.st_uid == 0, 'Only own native log removed'); (out/'native.log').unlink()
            for path in out.iterdir():
                t = path.lstat(); require(stat.S_ISREG(t.st_mode) and t.st_nlink == 1 and t.st_uid in (0, 1000), 'Only owned evaluator files sealed'); path.chmod(0o444)
            out.chmod(0o555)
        except Exception as exc:
            failure = failure or exc; report.update(status='fail', sealing_error_type=type(exc).__name__)
        report['elapsed_seconds'] = time.monotonic()-started; report['references'] = sources
        if failure or time.monotonic() >= deadline: report['status'] = 'fail'
        own_output(); write_receipt(out/'report.json', report, started, deadline)
    require(report['status'] == 'pass', 'Evaluator HOSTFAIL; original predictions unchanged')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(allow_abbrev=False); p.add_argument('--code', type=Path, required=True); p.add_argument('--revision', required=True); p.add_argument('--child', action='store_true')
    a = p.parse_args(); run(a.code, a.revision, a.child)
