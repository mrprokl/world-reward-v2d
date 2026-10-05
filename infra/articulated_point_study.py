"""One immutable Azure-only conditional study: manufacture, track, paired fit.

The host schedules existing source-authenticated operators across two qualified
images. It never supplies challenge records or predictor access to private truth.
Each completed stage is sealed once before the next stage can consume it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_point_authored_qualify as base

ROOT = base.ROOT
ENTRY = 'run_articulated_point_study'
PROTOCOL = 'configs/articulated_point_study_protocol_v1.json'
PROTOCOL_PIN = dict(bytes=8076, sha256='d1a3fd38cfd0496fd22a2a0280ff95cbc684d866eb345beb525bbbece921db0c')
BOOT_IMAGE = 'sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4'
STAGES = ('manufacture', 'tracks', 'fit')
IMAGES = dict(manufacture=base.IMAGE, tracks=BOOT_IMAGE, fit=base.IMAGE)
BUDGETS = dict(manufacture=600, tracks=900, fit=10800)
HELPERS = (*base.HELPERS, 'infra/articulated_point_study.py',
    'infra/run_articulated_point_study.sh', 'infra/articulated_point_native.py',
    'infra/articulated_point_tracks.py', 'infra/tracker_noise_experiment.py',
    'infra/robotap_boots_infer.py', 'infra/robotap_boots_acquire.py',
    'configs/robotap_boots_protocol.json', 'configs/robotap_boots_inference_pins.json',
    'configs/bootstapir_runtime_verify_pins.json',
    'src/world_reward/articulated_point_cohort.py', 'src/world_reward/authored_point_study.py', PROTOCOL)


def configuration(rt, code):
    c = base.protocol(rt, code)
    p = rt.pinned(code/PROTOCOL, PROTOCOL_PIN, 32 << 10)
    rt.require(p['schema'] == 'world_reward.articulated_point_study_protocol.v1'
        and p['stage_order'] == list(STAGES) and p['budgets_seconds'] == BUDGETS
        and p['base_protocol'] == dict(path=base.PROTOCOL, **base.PROTOCOL_PIN),
        'One exact prospective scientific protocol required')
    c['articulated_study'] = p
    return c


def output(revision):
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Exact immutable producer revision required')
    return ROOT/'validation/articulated_point_study_v1'/revision


def image(rt, identifier):
    value = rt.strict(base.control(['docker', 'image', 'inspect', identifier,
        '--format', '{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}']))
    rt.require(value['Id'] == identifier and value['Architecture'] == 'amd64'
        and value['Os'] == 'linux' and value['RootFS']['Type'] == 'layers', 'Exact immutable native image required')
    if identifier == BOOT_IMAGE:
        layers = value['RootFS']['Layers']
        pin = rt.pinned(Path(os.environ['WR_CODE'])/'configs/bootstapir_runtime_verify_pins.json',
            rt.identity(Path(os.environ['WR_CODE'])/'configs/bootstapir_runtime_verify_pins.json'), 16 << 10)
        rt.require(len(layers) == pin['child_layers'] == 47
            and hashlib.sha256(json.dumps(layers, separators=(',', ':')).encode()).hexdigest()
                == pin['ordered_rootfs_sha256'], 'Original qualified Boots layer sequence required')
    return value


def runtime_assets(rt, code):
    """Runtime-only ancestry: no RoboTAP pickle/public dataset is accessed."""
    import tracker_noise_experiment as ancestry
    import robotap_boots_acquire as acquire
    import robotap_boots_infer as boots
    _, evidence = ancestry.runtime_evidence(code)
    p = acquire.read_protocol(code/'configs/robotap_boots_protocol.json')
    assets = ROOT/boots.BASE/'assets'
    rows = {assets/'tapnet_source'/n: v for n, v in evidence['native_sources'].items()}
    rows[assets/p['checkpoint']['file']] = evidence['checkpoint']
    pins = rt.strict((code/'configs/robotap_boots_inference_pins.json').read_bytes())
    rows[ROOT/pins['runtime']['report_path']] = evidence['runtime']
    rt.require(all(rt.identity(path) == pin for path, pin in rows.items()), 'Complete original runtime bytes required')
    return evidence, rows


def prerequisites(rt, code, revision):
    c = configuration(rt, code)
    proof = base.prerequisites(rt, ROOT, code, revision, entry=ENTRY, source_helpers=HELPERS)
    boots, files = runtime_assets(rt, code)
    proof.update(boots_runtime=boots, boots_files={str(p): v for p, v in files.items()},
        images={s: image(rt, i) for s, i in IMAGES.items()}, protocol_identity=PROTOCOL_PIN)
    c['_articulated_proof'] = proof
    return c, proof


def lock_identity(rt, path, fd=None):
    s = rt.canonical(path).lstat()
    rt.require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Existing unaliased cooperative lock required')
    if fd is not None:
        rt.require((os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (s.st_dev, s.st_ino), 'Original lock inode changed')
    return s.st_dev, s.st_ino


def validate_stage(rt, value, stage, proof):
    """Typed execution census, not a claim that the scientific result improved."""
    required = dict(stage='articulated_point_'+stage+'_native_v1', status='pass', phase='complete',
        frames=144, source_binding=proof['source_binding'], protocol_identity=PROTOCOL_PIN,
        source_inputs_assets_rehashed_after=True, challenge_inputs_used=False, adoption=False)
    if stage == 'manufacture':
        required.update(native_decode_calls=6, native_decoded_frames=144,
            reconstitution_decode_calls=6, reconstitution_decoded_frames=144,
            primitive_render_calls=12, primitive_render_frames=288, tracker_executed=False,
            native_metadata_rehashed_after=True)
        rt.require(len(value['scenes']) == len(value['public_views']) == 6, 'All original authored scenes required')
    elif stage == 'tracks':
        required.update(model_loads=1, native_calls_attempted=6, native_calls_returned=6,
            native_calls_completed=6, private_truth_read=False, calibration_performed=False,
            runtime_rehashed_after=True, manufacture_rehashed_after=True, outputs_rehashed_after=True)
        rt.require(len(value['scenes']) == 6 and len(value['outputs']) == 12, 'Every native prediction/evidence required')
    else:
        required.update(constructor_attempts=10, constructor_returns=10, optimizer_run_attempts=8,
            optimizer_run_returns=8, actual_native_updates=2408, descriptive_four_pairs_only=True,
            calibrated_before_reserved_reads=True, both_results_frozen_before_private_evaluation=True, statistical_population_gain_verified=False,
            positive_weight_executed=True)
        rt.require(len(value['reserved_results']) == 4
            and all(row['status'] == 'pass' and row.get('both_results_frozen_before_private_evaluation') is True
                and set(row['results']) == {'A_original', 'B_point'}
                for row in value['reserved_results']), 'Every reserved pair must be complete')
    rt.require(all(type(value.get(k)) is type(v) and value[k] == v for k, v in required.items()),
        'Complete typed actual-stage proof required')


def stage_mounts(rt, code, c, proof, stage):
    paths = {code.parent}
    if stage != 'tracks':
        paths |= {ROOT/'vendor/video_to_data'/base.NATIVE, ROOT/'vendor/video_to_data'/base.BODY_SOURCE}
        paths |= {Path(n) for n in proof['frozen']}
    else:
        paths |= {Path(n) for n in proof['boots_files']}
    if stage != 'manufacture':
        import articulated_point_native as authored
        directory, report = authored.authenticate_manufacture(rt, c)
        paths.add(directory/'report.json')
        paths |= {directory/n for n in report['outputs']
                  if stage == 'fit' or n.endswith('_public.npz')}
    if stage == 'fit':
        directory = Path(c['articulated_study']['tracks']['path'])
        report = rt.pinned(directory/'report.json', c['articulated_study']['tracks']['report'], 1 << 20)
        paths |= {directory/'report.json', *(directory/n for n in report['outputs'])}
    for p in paths:
        rt.canonical(p)
        rt.require(p.exists() and ',' not in str(p) and '\n' not in str(p), 'Exact unambiguous mount required')
        rt.require('eval_private' not in p.parts and not p.is_relative_to(ROOT/'data'), 'No challenge/benchmark private mount permitted')
        if stage == 'tracks':
            rt.require(not p.name.endswith(('_private.npz', '_source.pth', '_object.obj')), 'Predictor must not access private controls')
    return sorted(paths)


def native(rt, code, revision, stage, proof_pin):
    target = output(revision)/stage/'native'
    proof = rt.pinned(Path('/opt/articulated-proof.json'), proof_pin, 4 << 20)
    rt.require(sys.platform == 'linux' and os.geteuid() == 1000
        and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'}
        and os.environ['WR_IMAGE_ID'] == IMAGES[stage] and not any(target.iterdir()), 'Fresh offline stage only')
    own = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    rt.require(own == proof['source_binding'], 'Actual source differs before native work')
    c = configuration(rt, code)
    c['articulated_study'].update(proof['prior_stages'], stage=stage)
    c['_articulated_proof'] = proof
    begin = time.monotonic()
    r = dict(stage='articulated_point_'+stage+'_native_v1', status='fail', phase='setup', outputs={})
    def persist(value):
        temporary = target/'report.tmp'
        with temporary.open('w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(target/'report.json')
    def expired(*_):
        raise TimeoutError('Inclusive fixed scientific stage budget')
    handlers = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, BUDGETS[stage])
    c['_persist'] = persist
    try:
        if stage == 'tracks':
            import articulated_point_tracks as operator
        else:
            import articulated_point_native as operator
        r = operator.native(rt, code, target, c)
        rt.require(r['status'] == 'pass', 'Actual scientific phase failed')
        rt.require(rt.source(ROOT, code, revision, ENTRY, HELPERS) == own, 'Own source changed during native work')
        if stage == 'tracks':
            _, files = runtime_assets(rt, code)
            rt.require({str(p): v for p, v in files.items()} == proof['boots_files'], 'Boots runtime changed')
        else:
            base.check_native_sources(rt, ROOT, proof)
        r['source_inputs_assets_rehashed_after'] = True
    except BaseException as error:
        if (target/'report.json').is_file() and r['phase'] == 'setup':
            r = rt.strict((target/'report.json').read_bytes())
        r.update(status='fail', error_type=type(error).__name__, error_context=str(error)[-400:])
    finally:
        try:
            r.update(source_binding=own, producer_revision=revision, protocol_identity=PROTOCOL_PIN,
                image_id=IMAGES[stage], challenge_inputs_used=False, adoption=False)
            import articulated_point_native as authored
            r['policy_sha256'] = authored.policy_sha256(c)
            r['outputs'] = {p.name: rt.identity(p) for p in target.iterdir() if p.name != 'report.json'}
            rt.require(time.monotonic()-begin < BUDGETS[stage], 'Stage seal exceeded fixed budget')
        except BaseException as error:
            r.update(status='fail', seal_error_type=type(error).__name__)
        r['elapsed_seconds'] = time.monotonic()-begin
        persist(r)
        for p in target.iterdir(): p.chmod(0o444)
        target.chmod(0o555)
        if time.monotonic()-begin >= BUDGETS[stage]:
            r.update(status='fail', seal_error_type='InclusiveDeadline')
            target.chmod(0o700); persist(r); (target/'report.json').chmod(0o444); target.chmod(0o555)
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, h in handlers.items(): signal.signal(s, h)
    return 0 if r['status'] == 'pass' else 1


def run_stage(rt, code, revision, stage, c, proof):
    begin = time.monotonic()
    parent = output(revision)/stage
    rt.canonical(parent); rt.require(not parent.exists(), 'Fresh stage, no overwrite/resume')
    parent.mkdir(mode=0o755); target = parent/'native'; target.mkdir(mode=0o700); os.chown(target, 1000, 1000)
    prior = {k: c['articulated_study'][k] for k in ('manufacture', 'tracks') if k in c['articulated_study']}
    stage_proof = dict(proof, prior_stages=prior)
    base.write(parent/'proof.json', (json.dumps(stage_proof, sort_keys=True)+'\n').encode())
    proof_pin = rt.identity(parent/'proof.json')
    name = 'world-reward-articulated-'+stage+'-'+revision[:12]
    rt.require(not base.control(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$']).strip(), 'Owned name occupied')
    cid = parent/'.container.cid'
    args = ['docker', 'run', '--name', name, '--cidfile', str(cid), '--label',
        'world_reward.authored_pair.owner='+revision, '--gpus', 'all', '--network', 'none',
        '--read-only', '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt',
        'no-new-privileges', '--cpus', '4', '--memory', '64g', '--tmpfs', '/tmp:rw,nosuid,size=2g']
    for p in stage_mounts(rt, code, c, proof, stage):
        args += ['--mount', f'type=bind,src={p},dst={p},readonly']
    args += ['--mount', f'type=bind,src={parent}/proof.json,dst=/opt/articulated-proof.json,readonly',
        '--mount', f'type=bind,src={target},dst={target}', '--entrypoint', '/usr/bin/env', IMAGES[stage],
        '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin', 'HOME=/tmp', 'HF_HUB_OFFLINE=1',
        'TRANSFORMERS_OFFLINE=1', 'MOMENTUM_ENABLED=0', 'OMP_NUM_THREADS=4',
        'PYTHONDONTWRITEBYTECODE=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
        'PYTHONPATH='+str(code/'infra')+':'+str(code/'src')+':'+str(ROOT/'vendor/video_to_data'/base.NATIVE)+':/workspace/v2d_sam3d_body/lib',
        'WR_ROOT='+str(ROOT), 'WR_CODE='+str(code), 'WR_CODE_REVISION='+revision,
        'WR_IMAGE_ID='+IMAGES[stage], 'python', '-B', str(code/'infra/articulated_point_study.py'),
        '--native', stage, str(proof_pin['bytes']), proof_pin['sha256']]
    r = dict(stage=stage, status='fail', producer_revision=revision, outputs={})
    failure = None
    try:
        with (parent/'native.log').open('xb') as log:
            os.fchmod(log.fileno(), 0o444)
            child = subprocess.run(args, stdout=log, stderr=log, timeout=BUDGETS[stage]+30)
        rt.require(child.returncode == 0, 'Actual native stage failed; retained Azure evidence')
        value = rt.pinned(target/'report.json', rt.identity(target/'report.json'), 1 << 20)
        validate_stage(rt, value, stage, proof)
        actual = {p.name: rt.identity(p) for p in target.iterdir() if p.name != 'report.json'}
        rt.require(actual == value['outputs'] and target.stat().st_mode & 0o777 == 0o555, 'Complete native seal differs')
        r.update(native_report_identity=rt.identity(target/'report.json'), outputs=actual)
    except BaseException as error:
        failure = error; r.update(error_type=type(error).__name__)
    finally:
        try:
            base.cleanup(rt, name, revision, cid, image=IMAGES[stage])
            r['owned_cleanup_verified'] = True
            _, after = prerequisites(rt, code, revision)
            rt.require(after == proof, 'Actual source/assets/images changed after stage')
            r['source_inputs_assets_rehashed_after'] = True
            rt.require(time.monotonic()-begin < BUDGETS[stage]+180, 'Host stage budget exceeded')
        except BaseException as error:
            failure = failure or error; r['post_error_type'] = type(error).__name__
        r.update(status='fail' if failure else 'pass', elapsed_seconds=time.monotonic()-begin)
        base.write(parent/'report.json', (json.dumps(r, sort_keys=True)+'\n').encode()); parent.chmod(0o555)
    if failure: raise ValueError('Conditional study stopped; no replay or partial acceptance')
    c['articulated_study'][stage] = dict(path=str(target), report=r['native_report_identity'])
    return r


def host(rt, code, revision):
    rt.require(sys.platform == 'linux' and os.geteuid() == 0
        and os.environ['DOCKER_HOST'] == 'unix://'+str(ROOT/'docker.sock'), 'Owned VM01 host required')
    c, proof = prerequisites(rt, code, revision)
    target = rt.canonical(output(revision)); rt.require(not target.exists(), 'Fresh whole lifecycle only')
    parent = target.parent
    if not parent.exists(): parent.mkdir(mode=0o755)
    lock = ROOT/'jobs/.world-reward-h100.lock'; before = lock_identity(rt, lock)
    fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
    r = dict(stage='articulated_point_study_host_v1', status='fail', producer_revision=revision,
        protocol_identity=PROTOCOL_PIN, source_binding=proof['source_binding'], stages=[],
        synthetic_oracle_controls=True, challenge_inputs_used=False, quality_verified=False, adoption=False)
    try:
        subprocess.run(['flock', '--timeout', '43200', str(fd)], check=True, pass_fds=(fd,), timeout=43210)
        rt.require(lock_identity(rt, lock, fd) == before, 'Original GPU lock changed')
        rt.require(not base.control(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).strip(), 'Foreign GPU compute present')
        target.mkdir(mode=0o755)
        for stage in STAGES:
            r['stages'].append(run_stage(rt, code, revision, stage, c, proof))
        rt.require(lock_identity(rt, lock, fd) == before, 'Original GPU lock changed after lifecycle')
        r['status'] = 'pass'
    finally:
        os.close(fd)
        if target.is_dir():
            base.write(target/'report.json', (json.dumps(r, sort_keys=True)+'\n').encode()); target.chmod(0o555)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--native', nargs=3)
    args = parser.parse_args(argv)
    code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']; rt = base.runtime(code)
    rt.require(Path(__file__).resolve() == code/'infra/articulated_point_study.py', 'Exact current caller required')
    if args.native:
        stage, count, digest = args.native
        rt.require(stage in STAGES, 'Explicit native stage required')
        return native(rt, code, revision, stage, dict(bytes=int(count), sha256=digest))
    host(rt, code, revision)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
