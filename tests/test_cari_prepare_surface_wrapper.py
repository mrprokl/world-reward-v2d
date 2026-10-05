"""Tiny manufactured host ledgers/shell stubs; no native, Docker or media calls."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'infra/run_cari_prepare.sh'
TEXT = SCRIPT.read_text()
HOST = TEXT.split("<<'PYSURFACEHOST'\n", 1)[1].split('\nPYSURFACEHOST', 1)[0]
FUNCTIONS = TEXT.split(' surface_inspect() {', 1)[1].split(' surface_finish() {', 1)[0]
FUNCTIONS = 'surface_inspect() {' + FUNCTIONS
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'


def pin(path):
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


@pytest.fixture
def host_case(tmp_path):
    root = tmp_path / 'root'; rev = 'a' * 40; episode = 26
    code = root / 'jobs' / rev / 'run_cari_prepare/code'
    base = root / 'outputs/episode_000026'; out = base / 'cari_inputs'
    control = root / 'results' / f'cari-native-prepare-surface-000026-{rev}'

    def write(path, value, mode=0o444):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        path.chmod(mode)
        return path

    for name in ('cari_prepare.py', 'cari_wrapper_common.sh', 'run_cari_prepare.sh'):
        write(code / 'infra' / name, b'# explicit manufactured source\n')
    write(code / 'infra/mediapipe_cpu_runtime_verify.py', (ROOT / 'infra/mediapipe_cpu_runtime_verify.py').read_bytes())
    for name, value in (('revision', rev), ('source-sha256', 'b' * 64)):
        write(code.parent / name, (value + '\n').encode())
    producer, qrev, firstrev = ('c' * 40, 'd' * 40, 'e' * 40)
    for r, entry in ((producer, 'run_object_budget_solid'), (qrev, 'run_surface_qslim_qualify'),
                     (firstrev, 'run_surface_identity_qualify')):
        old = root / 'jobs' / r / entry
        write(old / 'code/infra/original.py', b'# historical text, not executed\n')
        write(old / 'revision', (r + '\n').encode()); write(old / 'source-sha256', ('f' * 64 + '\n').encode())
        for p in (old / 'code', *(old / 'code').rglob('*')):
            p.chmod(0o555 if p.is_dir() else 0o444)
    q = dict(producer_revision=qrev); first = dict(producer_revision=firstrev)
    write(code / 'configs/surface_qslim_qualification_pins.json', q)
    write(code / 'configs/surface_identity_qualification_pins.json', first)
    proposal = base / ('object_budget_surface_' + producer)
    names = [proposal / n for n in ('report.json', 'native.json', 'geometry.npz',
        'object_fixed_canonical.glb', 'candidate_geometry.npz', 'mapping.json')]
    names += [base / 'object_grounded' / n for n in ('report.json', 'object.glb', 'transform.json', 'intrinsics.json')]
    names += [base / 'scale_smoke/report.json', root / 'results' / ('surface-qslim-qualify-' + qrev) / 'native.json',
        root / 'results' / ('surface-qslim-qualify-' + qrev) / 'report.json',
        root / 'results' / ('surface-identity-qualify-' + firstrev) / 'native.json',
        root / 'results/surface-qslim-independent-v2/report.json']
    for p in names: write(p, b'not geometry: source-bound test bytes')
    video = write(root / 'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000026.mp4', b'not a video')
    meta = write(root / 'data/track_1/meta/episodes.jsonl', b'{"episode_index":26,"length":3}\n')
    write(root / 'results/input-manifest.json', dict(track='track_1', repo_id='nvidia/video_to_data_challenge',
        revision='5f68335f3acc802033d1e80728c1633197521de8', files=[dict(path=str(p.relative_to(root / 'data')), **pin(p)) for p in (meta, video)]))
    for n in ('report.json', 'prompts.json'): write(base / 'automatic_masks' / n, dict(manufactured=True))
    frames = []
    for i in range(3):
        h = write(base / f'automatic_masks/masks/0/{i:06d}.png', b'not human PNG')
        o = write(base / f'automatic_masks/masks/1/{i:06d}.png', b'not object PNG')
        d = write(base / f'depth_full/{i:06d}.npz', b'not depth NPZ')
        frames.append(dict(frame_index=i, mask_sha256=pin(h)['sha256'], object_mask_sha256=pin(o)['sha256'], output_sha256=pin(d)['sha256']))
    common = dict(status='pass', episode_index=26, total_video_frames=3, input_track='track_1',
        input_sha256=pin(video)['sha256'], ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], frames=frames)
    body = write(base / 'body_full/report.json', dict(common, stage='sam3d_body_full_video_initializer'))
    write(base / 'depth_full/report.json', dict(common, stage='monocular_moge2_full_video'))
    adapter = write(base / 'body_full/cari_adapter/canonical_initializer.pkl', b'not a pickle')
    write(adapter.parent / 'report.json', dict(stage='native_cari_body_adapter_full_video', status='pass',
        episode_index=26, frames=3, input_track='track_1', ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        body_report_sha256=pin(body)['sha256'], canonical_initializer_sha256=pin(adapter)['sha256']))
    pp = code / 'configs/surface_mesh_000026_pins.json'
    write(pp, dict(schema='world_reward.surface_mesh_pins.v1', episode_index=26, input_sha256=pin(video)['sha256'],
        metric_scale_baked_once=.5, report=dict(producer_revision=producer, **pin(proposal / 'report.json')), files={str(p.relative_to(root)): pin(p) for p in names}, source_helpers={}))
    pose = base / 'object_pose_full_surface'; geo = write(pose / 'geometry_and_poses.npz', b'not a pose NPZ')
    copy = write(pose / 'object_fixed_canonical.glb', (proposal / 'object_fixed_canonical.glb').read_bytes())
    write(pose / 'report.json', dict(stage='fixed_scale_full_object_pose_initializer', status='pass',
        mesh_source='surface', episode_index=26, input_sha256=pin(video)['sha256'], execution_verified=True,
        original_frame_coverage_verified=True, fixed_shape=True, ground_truth_used=False, hand_labeled_test=False,
        oracle_modes=[], frames=frames, geometry_and_poses_sha256=pin(geo)['sha256'], fixed_canonical_mesh_sha256=pin(copy)['sha256'],
        topology_budget=dict(committed_pins_sha256=pin(pp)['sha256'])))
    pose.chmod(0o555)
    for p in (code, *code.rglob('*')): p.chmod(0o555 if p.is_dir() else 0o444)
    started = time.monotonic()

    def call(mode, status=0, post=0, cleanup=0, native=0, age=None, seal_advance=0):
        command = '''import os,time
from pathlib import Path
from types import SimpleNamespace
os.geteuid=lambda:0
original_lstat=Path.lstat
def manufactured_owned_stat(path):
 a=original_lstat(path)
 if path.parent==Path(%r):
  fields=('st_dev','st_ino','st_mode','st_nlink','st_gid','st_size','st_mtime_ns','st_ctime_ns')
  return SimpleNamespace(**{k:getattr(a,k)for k in fields},st_uid=0)
 return a
Path.lstat=manufactured_owned_stat
original_clock=time.monotonic
original_fsync=os.fsync
elapsed_advance=0
def manufactured_fsync(fd):
 global elapsed_advance
 original_fsync(fd)
 elapsed_advance=%r
os.fsync=manufactured_fsync
time.monotonic=lambda:original_clock()+elapsed_advance
exec(%r)
''' % (str(control), seal_advance, HOST)
        return subprocess.run([sys.executable, '-B', '-c', command, str(root), str(code), rev, '26', str(control),
            str(out), mode, str(started if age is None else time.monotonic()-age), str(status), str(post), str(cleanup), str(native)], capture_output=True, text=True)
    return dict(root=root, code=code, base=base, out=out, control=control, pins=pp, pose=pose,
        depth=base / 'depth_full/000001.npz', call=call, write=write, rev=rev)


def test_host_freezes_public_inputs_before_native_and_rechecks(host_case):
    r = host_case; before = r['call']('before'); assert before.returncode == 0, before.stderr
    after = r['call']('after'); assert after.returncode == 0, after.stderr
    assert before.stdout == after.stdout and not r['out'].exists()
    proof = json.loads((r['control'] / 'proof.json').read_text())
    assert proof['total_frames'] == 3 and len(proof['historical_sources']) == 3
    assert not any('weights' in n for n in proof['inputs'])
    r['depth'].chmod(0o644); r['depth'].write_bytes(b'mutated depth')
    assert r['call']('after').returncode != 0


@pytest.mark.parametrize('target', ['out', 'control'])
def test_existing_destination_abstains_before_any_write(host_case, target):
    r = host_case; r[target].mkdir()
    assert r['call']('before').returncode != 0
    assert not (r['control'] / 'proof.json').exists()


def test_wrong_surface_pose_pin_stops_before_reservation(host_case):
    r = host_case; p = r['pose'] / 'report.json'; value = json.loads(p.read_text())
    value['topology_budget']['committed_pins_sha256'] = '0' * 64
    p.chmod(0o644); p.write_text(json.dumps(value)); p.chmod(0o444)
    assert r['call']('before').returncode != 0 and not r['control'].exists()


def test_seal_never_promotes_native_only_pass(host_case):
    r = host_case; assert r['call']('before').returncode == 0
    assert r['call']('seal', status=0, post=1, cleanup=0, native=1).returncode != 0
    report = json.loads((r['control'] / 'report.json').read_text())
    assert report['status'] == 'fail' and report['owned_container_absence_verified'] is False
    assert r['control'].stat().st_mode & 0o777 == 0o555
    assert (r['control'] / 'report.json').stat().st_mode & 0o777 == 0o444


def test_complete_source_bound_native_receipt_and_host_seal(host_case):
    r = host_case; before = r['call']('before'); assert before.returncode == 0, before.stderr
    proof = json.loads((r['control'] / 'proof.json').read_text()); pose = r['pose']
    r['write'](r['out'] / 'report.json', dict(stage='world_reward_native_cari_inputs', status='pass', frames=3,
        episode_index=26, producer_revision=r['rev'], input_sha256=proof['video_sha256'],
        original_frame_coverage_verified=True, object_source='surface', ground_truth_used=False,
        hand_labeled_test=False, oracle_modes=[], script_sha256=proof['source']['helpers']['infra/cari_prepare.py']['sha256'],
        object_pose_source=dict(report=str((pose / 'report.json').relative_to(r['root'])),
            geometry_and_poses=str((pose / 'geometry_and_poses.npz').relative_to(r['root'])),
            geometry_and_poses_sha256=pin(pose / 'geometry_and_poses.npz')['sha256']),
        surface_geometry_validation=dict(source_rehashed_after=True)))
    complete = r['call']('complete'); assert complete.returncode == 0, complete.stderr
    assert complete.stdout == before.stdout
    sealed = r['call']('seal', post=1, cleanup=1, native=1); assert sealed.returncode == 0, sealed.stderr
    assert json.loads((r['control'] / 'report.json').read_text())['status'] == 'pass'


def test_compute_deadline_cannot_be_extended_by_reporting_grace(host_case):
    r = host_case; assert r['call']('before').returncode == 0
    result = r['call']('seal', post=1, cleanup=1, native=1, age=7201)
    assert result.returncode != 0
    assert json.loads((r['control'] / 'report.json').read_text())['status'] == 'fail'


def test_deadline_crossed_during_fsync_demotes_same_exclusive_receipt(host_case):
    r = host_case; assert r['call']('before').returncode == 0
    r['write'](r['out'] / 'report.json', dict(manufactured=True))
    result = r['call']('seal', post=1, cleanup=1, native=1, age=7199, seal_advance=2)
    assert result.returncode != 0
    report = json.loads((r['control'] / 'report.json').read_text())
    assert report['status'] == 'fail' and report['elapsed_seconds'] > 7200
    assert report['original_exit_status'] == 0 and report['failure_type'] == 'ValueError'
    assert r['control'].stat().st_mode & 0o777 == 0o555
    assert (r['control'] / 'report.json').stat().st_mode & 0o777 == 0o444


def test_existing_host_receipt_is_never_overwritten(host_case):
    r = host_case; assert r['call']('before').returncode == 0
    receipt = r['write'](r['control'] / 'report.json', b'original sealed receipt')
    before = pin(receipt)
    assert r['call']('seal', status=17).returncode != 0
    assert pin(receipt) == before


def test_changed_complete_source_rejects_postproof(host_case):
    r = host_case; assert r['call']('before').returncode == 0
    p = r['code'] / 'infra/cari_prepare.py'; p.chmod(0o644); p.write_bytes(b'# changed source\n'); p.chmod(0o444)
    assert r['call']('after').returncode != 0


@pytest.fixture
def cleanup_case(tmp_path):
    cid = 'a' * 64; rev = 'b' * 40; name = 'wr-cari-prepare-surface-000026-' + rev
    cidfile = tmp_path / '.container.cid'; cidfile.write_text(cid)
    bindir = tmp_path / 'bin'; bindir.mkdir(); calls = tmp_path / 'calls'; removed = tmp_path / 'removed'
    docker = bindir / 'docker'
    docker.write_text('''#!/bin/sh
echo "$1" >> "$CALLS"
if [ "$1" = rm ];then touch "$REMOVED";[ "$SCENARIO" != rm_fail ];exit;fi
case "$SCENARIO" in
 daemon) echo 'Cannot connect to daemon' >&2;exit 1;;
 timeout) exit 124;;
 malformed) echo '{}';exit 0;;
 absent) echo "Error: No such object: $2" >&2;exit 1;;
 absent_split) printf '[]\\n';printf '\\nerror: no such object: %s\\n' "$2" >&2;exit 1;;
 absent_format_newline) printf '\\n';printf 'error: no such object: %s\\n' "$2" >&2;exit 1;;
 absent_context) printf '[]\\n';printf '\\nDaemon unavailable\\nerror: no such object: %s\\n' "$2" >&2;exit 1;;
 absent_stdout) echo "Error: No such object: $2";exit 1;;
 absent_space) printf ' Error: No such object: %s\\n' "$2" >&2;exit 1;;
 live_stderr) echo "$2 /$NAME $IMAGE $REV";echo 'daemon warning' >&2;exit 0;;
 mismatch) echo "$2 /foreign $IMAGE $REV";exit 0;;
esac
if [ -e "$REMOVED" ] && [ "$SCENARIO" != still_live ];then echo "error: no such object: $2" >&2;exit 1;fi
echo "$2 /$NAME $IMAGE $REV"
'''); docker.chmod(0o755)
    timeout = bindir / 'timeout'; timeout.write_text('#!/bin/sh\nshift;exec "$@"\n'); timeout.chmod(0o755)
    python = bindir / 'python3'; python.symlink_to(sys.executable)

    def call(scenario, present=True, created=1):
        if not present and cidfile.exists(): cidfile.unlink()
        env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'], CALLS=str(calls),
            REMOVED=str(removed), SCENARIO=scenario, NAME=name, IMAGE=IMAGE, REV=rev)
        shell = 'set -u\nCID_FILE="$1";CREATED="$2";CLEANUP_VERIFIED=0\n' + FUNCTIONS + '\nsurface_cleanup;status=$?;echo "verified=$CLEANUP_VERIFIED";exit "$status"'
        result = subprocess.run(['bash', '-c', shell, 'test', str(cidfile), str(created)], env=env, capture_output=True, text=True)
        return result, calls.read_text().splitlines() if calls.exists() else []
    return dict(call=call, cidfile=cidfile)


@pytest.mark.parametrize('scenario', ['absent', 'absent_split', 'absent_format_newline', 'live'])
def test_cleanup_exact_absence_or_owned_removal(cleanup_case, scenario):
    result, calls = cleanup_case['call'](scenario)
    assert result.returncode == 0, result.stderr
    assert 'verified=1' in result.stdout
    assert calls == (['inspect', 'rm', 'inspect'] if scenario == 'live' else ['inspect'])


@pytest.mark.parametrize('scenario', ['daemon', 'timeout', 'malformed', 'mismatch', 'rm_fail', 'still_live',
    'absent_context', 'absent_stdout', 'absent_space', 'live_stderr'])
def test_cleanup_error_never_means_absence(cleanup_case, scenario):
    result, calls = cleanup_case['call'](scenario)
    assert result.returncode != 0
    if scenario not in ('rm_fail', 'still_live'): assert 'rm' not in calls


@pytest.mark.parametrize('value', ['foreign', 'a' * 63, 'a' * 64 + '\nextra'])
def test_cleanup_rejects_invalid_cid_without_docker(cleanup_case, value):
    cleanup_case['cidfile'].write_text(value)
    result, calls = cleanup_case['call']('live')
    assert result.returncode != 0 and calls == []


def test_missing_post_launch_cid_is_not_success(cleanup_case):
    result, calls = cleanup_case['call']('absent', present=False)
    assert result.returncode != 0 and calls == []


def test_failure_status_is_preserved_and_postchecks_are_required(tmp_path):
    finish = TEXT.split(' surface_finish() {', 1)[1].split(" trap surface_finish EXIT", 1)[0]
    for original, cleanup, post, expected in ((17, 1, 1, 17), (0, 1, 0, 1), (0, 0, 1, 1), (0, 0, 0, 0)):
        shell = 'BEFORE=proof;POST_VERIFIED=0;NATIVE_VERIFIED=0;CLEANUP_VERIFIED=0\n'
        shell += f'surface_cleanup(){{ return {cleanup}; }}\nsurface_host(){{ if [ "$1" != seal ]&&[ {post} = 1 ];then return 1;fi;[ "$1" = seal ]||echo proof; }}\n'
        shell += 'surface_finish() {' + finish + f'\ntrap surface_finish EXIT\nexit {original}\n'
        result = subprocess.run(['bash', '-c', shell], capture_output=True, text=True)
        assert result.returncode == expected


def test_surface_only_budget_image_no_gpu_and_legacy_tail_unchanged():
    before = subprocess.check_output(['rtk', 'proxy', 'git', 'show',
        '4e7d946ff5a3e84d0e414b4d5f75c9a3815b4f05:infra/run_cari_prepare.sh'], cwd=ROOT).decode()
    legacy = before[before.index('docker run --rm --network none \\\n'):]
    assert TEXT.endswith(legacy)
    branch = TEXT[TEXT.index(' REV="$WR_CODE_REVISION";'):TEXT.index(legacy)]
    assert '--name "$NAME" --cidfile "$CID_FILE"' in branch and '--label "world_reward.cari_prepare.owner=$REV"' in branch
    assert '7200-(SECONDS-SURFACE_START)' in branch and '7200+30-(SECONDS-SURFACE_START)' in branch
    assert '--gpus' not in branch and 'flock' not in branch and 'fd/9' not in branch
    assert '"$IMAGE" python "$CODE/infra/cari_prepare.py" "$@"' in branch
    assert 'mkdir "$OUT"' not in branch
    assert 'EXISTING="$(timeout 5s docker ps' in branch and ')" || exit 1' in branch
    subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
