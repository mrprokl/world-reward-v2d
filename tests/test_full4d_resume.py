"""Synthetic continuation gates only; no original clips, Docker or GPU calls."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

import full4d_resume as resume


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def pose_case(tmp_path):
    base, code = tmp_path/'episode_000009', tmp_path/'code'
    script = 'infra/object_pose_smoke.py'; script_sha = write(code/script, b'synthetic original producer')
    (code/script).chmod(0o444)
    item = dict(episode=9, total=96, video_pin=dict(sha256='a'*64, bytes=1))
    directory = base/'object_pose_full_surface'
    source = {}
    for field, relative in (('object_report_sha256','object_grounded/report.json'),
                           ('alignment_report_sha256','scale_smoke/report.json'),
                           ('full_depth_report_sha256','depth_full/report.json')):
        source[field] = write(base/relative, b'opaque synthetic upstream receipt')
    report = dict(status='pass', stage=resume.STAGE_NAMES['object_pose'], episode_index=9,
        input_track='track_1', ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        script_sha256=script_sha, input_sha256='a'*64,
        frames=[{'frame_index': index} for index in range(96)], **source)
    report['geometry_and_poses_sha256'] = write(directory/'geometry_and_poses.npz', b'opaque synthetic numerical payload')
    report['fixed_canonical_mesh_sha256'] = write(directory/'object_fixed_canonical.glb', b'opaque synthetic mesh')
    write(directory/'report.json', json.dumps(report).encode())
    return base, code, script, item, report


def test_existing_full_pose_pass_is_hashed_without_decode_or_rerun(pose_case):
    base, code, script, item, _ = pose_case
    evidence = resume.report_pass(base, 'object_pose', script, code, item)
    assert evidence['files'] == 2 and evidence['inference_replayed'] is False
    assert (base/'object_pose_full_surface/report.json').stat().st_mode & 0o222


@pytest.mark.parametrize('fault', ['failed', 'short_timeline', 'wrong_video', 'changed_source', 'changed_pose', 'wrong_oracle'])
def test_existing_pose_refuses_nonpass_or_changed_original_lineage(pose_case, fault):
    base, code, script, item, report = pose_case
    if fault == 'failed': report['status'] = 'fail'
    elif fault == 'short_timeline': report['frames'].pop()
    elif fault == 'wrong_video': report['input_sha256'] = 'b'*64
    elif fault == 'changed_source':
        (code/script).chmod(0o644)
        write(code/script, b'new numerical source must not replace original')
        (code/script).chmod(0o444)
    elif fault == 'changed_pose': write(base/'object_pose_full_surface/geometry_and_poses.npz', b'changed')
    else: report['oracle_modes'] = ['synthetic_forbidden_oracle']
    write(base/'object_pose_full_surface/report.json', json.dumps(report).encode())
    with pytest.raises(ValueError): resume.report_pass(base, 'object_pose', script, code, item)


def test_missing_stage_is_fresh_but_occupied_incomplete_stage_is_never_deleted(tmp_path):
    base = tmp_path/'episode'; base.mkdir()
    item = dict(episode=9, total=96)
    assert resume.report_pass(base, 'body_full', 'infra/body_smoke.py', tmp_path/'code', item) is None
    output = base/'body_full'; output.mkdir(); original = output/'predictions.npz'; original.write_bytes(b'partial')
    with pytest.raises(ValueError, match='complete PASS'):
        resume.report_pass(base, 'body_full', 'infra/body_smoke.py', tmp_path/'code', item)
    assert original.read_bytes() == b'partial'


@pytest.mark.parametrize('stage,expected_locks', [('object_pose', 0), ('body_full', 1), ('refined', 1)])
def test_cooperative_workers_use_original_command_and_serialize_learned_stages(monkeypatch, tmp_path, stage, expected_locks):
    calls = []; counters = {'locks': 0}
    @contextmanager
    def model_lock():
        counters['locks'] += 1
        yield
    oldcode = tmp_path/'old_code'; experiment = tmp_path/'experiment'; control = tmp_path/'continuation'
    (control/'logs').mkdir(parents=True)
    driver = SimpleNamespace(command=lambda *args: calls.append(args) or ['opaque', 'original', 'command'])
    monkeypatch.setattr(resume.subprocess, 'check_output', lambda *a, **k: '')
    monkeypatch.setattr(resume.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    monkeypatch.setattr(resume, 'run_process', lambda cmd, log, seconds, stop: 0)
    resume.native(driver, oldcode, experiment, {'body_image': 'exact_original_image'}, {'episode': 9},
        (stage, 'infra/original.py', (), 'body_image', stage, None), control, 600, 'c'*40, model_lock(), threading.Event())
    assert counters['locks'] == expected_locks
    assert calls[0][0] == oldcode and calls[0][-1] == resume.OLD
    assert (control/'logs'/f'9-{stage}.log').is_file()
    assert not (experiment/'logs').exists()


def test_worker_stops_before_launch_and_keeps_original_reports(monkeypatch, tmp_path):
    stop = threading.Event(); stop.set()
    driver = SimpleNamespace(command=lambda *args: ['not', 'executed'])
    monkeypatch.setattr(resume.subprocess, 'check_output', lambda *a, **k: '')
    monkeypatch.setattr(resume, 'run_process', lambda *a, **k: pytest.fail('no launch after stop'))
    with pytest.raises(ValueError, match='interrupted'):
        resume.native(driver, tmp_path, tmp_path, {'body_image': 'exact'}, {'episode': 9},
            ('refined', 'infra/original.py', (), 'body_image', 'refined', None), tmp_path,
            1, 'c'*40, threading.Lock(), stop)


def test_full_original_report_and_frozen_population_are_not_relabelled():
    assert resume.OLD == 'de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba'
    assert resume.ORIGINAL_REPORT == dict(bytes=8613,
        sha256='d20a4f074c0bf5ffeb2e489a995e66655f5a65773e445638cab46d6b197b1e40')
    source = Path(resume.__file__).read_text()
    assert "prior.get('status') == 'running'" in source
    assert 'other_workers_only_after_scout_pass=True' in source
    assert 'ThreadPoolExecutor(max_workers=2)' in source
    assert "with (ROOT/'jobs/.world-reward-h100.lock').open('a')" in source
    assert 'rmtree(' not in source
    assert "'resample'" not in source


def test_capacity_inputs_execute_new_source_but_import_original_pure_geometry_first(monkeypatch, tmp_path):
    old, new = tmp_path/'original/code', tmp_path/'continuation/code'
    control = tmp_path/'receipt'; (control/'logs').mkdir(parents=True)
    captured = []
    def command(*args):
        assert args[0] == new and args[-1] == 'c'*40
        return ['docker', '--entrypoint', 'env', 'original_image', '-i',
                'PYTHONPATH='+str(new/'src')+':'+str(new/'infra'), 'python', str(new/'infra/cari_prepare.py')]
    monkeypatch.setattr(resume.subprocess, 'check_output', lambda *a, **k: '')
    monkeypatch.setattr(resume.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    monkeypatch.setattr(resume, 'run_process', lambda argv, *a: captured.append(argv) or 0)
    resume.native(SimpleNamespace(command=command), old, tmp_path/'experiment', {'body_image':'original_image'},
        {'episode':1}, ('inputs','infra/cari_prepare.py',(),'body_image','inputs',None), control, 100,
        'c'*40, threading.Lock(), threading.Event(), capacity_code=new)
    assert f'type=bind,src={old.parent},dst={old.parent},readonly' in captured[0]
    assert 'PYTHONPATH='+str(old/'src')+':'+str(old/'infra')+':'+str(new/'src')+':'+str(new/'infra') in captured[0]
    assert str(new/'infra/cari_prepare.py') in captured[0]


def shared_fixture(tmp_path, role='prepare'):
    from full4d_pins import SHARED
    base = tmp_path/'experiment/outputs/episode_000009'; code = tmp_path/'code'
    script, stage, filenames = SHARED[role]
    script_sha = write(code/'infra'/script, b'opaque original source'); (code/'infra'/script).chmod(0o444)
    directory = base/('cari_shared_'+role+'_v1')
    report = dict(status='pass', stage=stage, phase='complete', episode_index=9, frames=96,
        input_track='track_1', ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        producer_revision=resume.OLD, script_sha256=script_sha, original_frame_indices=list(range(96)),
        source_inputs_assets_rehashed=True, source_helpers_rehashed=True,
        source_helpers={'infra/'+script:dict(bytes=22,sha256=script_sha)})
    # Source helper byte count is computed, never fabricated.
    report['source_helpers']['infra/'+script] = resume.identity(code/'infra'/script)
    report.update(body_assets={},inference_source_identity={},decoder_identity={})
    experiment = tmp_path/'experiment'
    input_path = experiment/'pins/cari_clip_000009_input_pins.json'
    refined_path = experiment/'pins/cari_clip_000009_shared_refined_pins.json'
    write(input_path,json.dumps({'source_files':{}}).encode()); input_path.chmod(0o444)
    preceding = {'refined':dict(sha256='d'*64), 'refined_files':{'refined.pth':dict(sha256='e'*64)}}
    if role == 'export':
        write(refined_path,json.dumps(preceding).encode()); refined_path.chmod(0o444)
        report.update(input_pins=resume.payload_identity(input_path), refined_pins=resume.payload_identity(refined_path),
            refined_report_sha256='d'*64,refined_bundle_sha256='e'*64,source_files={})
    files = {}
    for name in filenames-{'report.json'}:
        raw = b'opaque synthetic native payload'; write(directory/name, raw); (directory/name).chmod(0o444)
        files[name] = resume.payload_identity(directory/name)
    if role == 'export':
        report.update(output_files=dict(files),aligned_object_mesh_sha256=files['object_aligned.glb']['sha256'])
    raw = json.dumps(report).encode(); write(directory/'report.json', raw); (directory/'report.json').chmod(0o444)
    files['report.json'] = resume.payload_identity(directory/'report.json')
    pin = {role: files['report.json'] | dict(producer_revision=resume.OLD, script_sha256=script_sha),
           role+'_files':files}
    pinpath = experiment/'pins'/f'cari_clip_000009_shared_{role}_pins.json'
    write(pinpath,json.dumps(pin).encode()); pinpath.chmod(0o444)
    item = dict(episode=9,total=96)
    return base, experiment, code, item, report, pinpath


def stub_shared_native_gates(monkeypatch, experiment):
    """This fixture tests byte inventories; native lineage has its own tests."""
    import cari_full_forward as forward
    import cari_full_refine as refined
    import cari_full_export as export
    called = []
    def gate(role):
        def validate(root,code,spec,*pins):
            assert root == resume.ROOT and spec.episode_index == 9 and spec.total_frames == 96
            called.append(role)
            return {'report':dict(body_assets={},inference_source_identity={},decoder_identity={})}
        return validate
    monkeypatch.setattr(forward,'verify_prepare_artifacts',gate('prepare'))
    monkeypatch.setattr(forward,'verify_forward_artifacts',gate('forward'))
    monkeypatch.setattr(refined,'verify_refined_artifacts',gate('refined'))
    def check_export(report,spec):
        assert report['phase'] == 'complete' and spec.total_frames == 96
        called.append('export')
    monkeypatch.setattr(export,'validate_export_report',check_export)
    return called


@pytest.mark.parametrize('role', ['prepare','forward','refined','export'])
def test_existing_complete_shared_outputs_are_not_overwritten(tmp_path, role, monkeypatch):
    base, experiment, code, item, _, _ = shared_fixture(tmp_path, role)
    called = stub_shared_native_gates(monkeypatch,experiment)
    evidence = resume.saved_stage(base,role,experiment,code,item)
    assert evidence['inference_replayed'] is False
    assert called == (['export','refined'] if role == 'export' else [role])


@pytest.mark.parametrize('fault', ['changed_payload','missing_pin','failed_report','extra_output','wrong_producer'])
def test_existing_shared_outputs_fail_closed_not_deleted(tmp_path, fault, monkeypatch):
    base, experiment, code, item, report, pinpath = shared_fixture(tmp_path)
    stub_shared_native_gates(monkeypatch,experiment)
    directory = base/'cari_shared_prepare_v1'
    if fault == 'changed_payload':
        target = directory/'target.npy'; target.chmod(0o644); target.write_bytes(b'changed'); target.chmod(0o444)
    elif fault == 'missing_pin': pinpath.unlink()
    elif fault == 'extra_output': (directory/'unexpected').write_bytes(b'foreign')
    else:
        if fault == 'failed_report': report['status'] = 'fail'
        else: report['producer_revision'] = 'd'*40
        path = directory/'report.json'; path.chmod(0o644); path.write_text(json.dumps(report)); path.chmod(0o444)
    with pytest.raises((ValueError, FileNotFoundError)):
        resume.saved_stage(base,'prepare',experiment,code,item)
    assert directory.is_dir()


@pytest.mark.parametrize('fault', [None,'changed_video','short_timeline','missing_source'])
def test_existing_video_reuse_is_full_t_and_exact_source_bound(tmp_path, monkeypatch, fault):
    from full4d_pins import SHARED
    monkeypatch.setattr(resume,'ROOT',tmp_path)
    experiment = tmp_path/'experiment'; code = tmp_path/'code'
    base = experiment/'outputs/episode_000009'; directory = experiment/'videos/episode_000009'
    video = tmp_path/'data/original.mp4'
    paths = {code/name for name in ('infra/full4d_video.py','infra/camera_render.py',
        'infra/qwen4d_preview.py','src/world_reward/mesh_geometry.py')} | {
        experiment/'pins/cari_clip_000009_shared_export_pins.json',tmp_path/'results/input-manifest.json',video} | {
        base/'cari_shared_export_v1'/name for name in SHARED['export'][2]}
    sources = {}
    for path in paths:
        write(path,b'opaque synthetic saved source'); path.chmod(0o444)
        sources[str(path)] = resume.payload_identity(path)
    for suffix in ('mp4','jpg'):
        path=directory/f'episode_000009.{suffix}'; write(path,b'opaque tiny visual'); path.chmod(0o444)
    report = dict(schema='world_reward.full4d_video.v1',status='pass',producer_revision=resume.OLD,
        episode_index=9,frames_encoded=96,original_frames=96,original_frame_indices=list(range(96)),
        ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],source_rehashed_after=True,
        model_execution=False,optimizer_execution=False,input_track='track_1',per_frame_alignment=False,
        per_frame_camera=False,per_frame_centring=False,object_scale_applied_again=False,
        original_geometry_unchanged=True,sources=sources,
        video=resume.payload_identity(directory/'episode_000009.mp4'),
        poster=resume.payload_identity(directory/'episode_000009.jpg'))
    item = dict(episode=9,total=96,video=str(video),video_pin=sources[str(video)])
    if fault=='short_timeline': report['original_frame_indices'].pop()
    if fault=='missing_source': report['sources'].pop(str(video))
    path=directory/'report.json'; write(path,json.dumps(report).encode()); path.chmod(0o444)
    if fault=='changed_video':
        path=directory/'episode_000009.mp4'; path.chmod(0o644); path.write_bytes(b'changed'); path.chmod(0o444)
    if fault:
        with pytest.raises(ValueError): resume.saved_stage(base,'video',experiment,code,item)
    else:
        assert resume.saved_stage(base,'video',experiment,code,item)['inference_replayed'] is False
    assert directory.is_dir()


def test_missing_previous_scout_receipt_is_not_fabricated(tmp_path):
    assert resume.previous_continuation(tmp_path) is None


@pytest.fixture
def cleanup_case(tmp_path,monkeypatch):
    monkeypatch.setattr(resume,'ROOT',tmp_path)
    experiment=tmp_path/'experiments'/('full4d-v1-'+resume.OLD)
    base=experiment/'outputs/episode_000009'; directory=base/'cari_inputs'
    paths=['aligned_depth.h5','automatic_masks.h5','episode_000009.0.color.mp4','intrinsics.pkl',
        'object_metric.glb','own_object_poses.pkl','export/episode_000009/wild_export.json',
        'export/episode_000009/edex','export/episode_000009/images/front_stereo_camera_left.h5',
        'export/episode_000009/human_masks/front_stereo_camera_left.h5',
        'export/episode_000009/object_masks/front_stereo_camera_left.h5',
        'export/episode_000009/object_mesh/output_aligned.glb']
    for name in paths: write(directory/name,b'opaque interrupted serialization')
    paths.append('.aligned_depth.h5.storage-repack.lock')
    write(directory/paths[-1],b'')
    monkeypatch.setattr(resume,'INTERRUPTED_INPUT_CENSUS',
        dict(files=len(paths),bytes=sum((directory/name).stat().st_size for name in paths)))
    original=tmp_path/'jobs'/resume.OLD/'run_full4d_sample/code'
    item=dict(episode=9,total=415,video_pin=dict(bytes=1,sha256='a'*64))
    stopped=experiment/'continuations'/resume.SCOUT_CONTINUATION/'report.json'
    write(stopped,b'opaque stopped scout receipt')
    reportpin=resume.payload_identity(stopped,readonly=False)
    monkeypatch.setattr(resume,'INTERRUPTED_SCOUT_REPORT',reportpin)
    previous=dict(status='fail',source_rehashed_after=True,report=reportpin,
        episodes=[dict(episode=9,status='fail',phase='inputs',error='Continuation interrupted')])
    monkeypatch.setattr(resume,'previous_continuation',lambda _:previous)
    monkeypatch.setattr(resume.subprocess,'check_output',lambda *a,**k:'')
    monkeypatch.setattr(resume,'_cleanup_upstream',lambda *a:{'opaque_upstream':'unchanged'})
    monkeypatch.setattr(resume,'source',lambda *a:{'closure_sha256':'c'*64})
    # Synthetic ownership view only: production still checks the actual lstat.
    actual=Path.lstat
    def own(path):
        result=actual(path)
        if not path.is_relative_to(directory): return result
        return SimpleNamespace(**{key:getattr(result,key) for key in ('st_dev','st_ino','st_size','st_mode',
            'st_nlink','st_mtime_ns','st_ctime_ns')},st_uid=1000,st_gid=1000)
    monkeypatch.setattr(Path,'lstat',own)
    return experiment,base,original,item,previous,paths


def test_cleanup_records_inventory_before_removing_only_unfinished_inputs(cleanup_case):
    experiment,base,original,item,previous,paths=cleanup_case
    upstream=base/'body_full/cari_adapter/canonical_initializer.pkl'; write(upstream,b'original initializer untouched')
    recorded=[]
    def record(evidence):
        recorded.append(json.loads(json.dumps(evidence)))
        if len(recorded)==1:
            assert (base/'cari_inputs/aligned_depth.h5').exists()
            assert evidence['status']=='recorded_before_removal'
    result=resume.cleanup_interrupted_inputs(experiment,base,original,item,previous,record)
    assert not (base/'cari_inputs').exists() and upstream.read_bytes()==b'original initializer untouched'
    assert result['files']==len(paths) and result['status']=='removed_unpublished_generated_inputs'
    assert len(result['inventory_sha256'])==64 and len(recorded)==2
    assert result['original_completed_predictions_removed'] is False
    assert result['numerical_payload_decoded'] is False


@pytest.mark.parametrize('fault',['running','different_phase','different_episode','symlink','hardlink','foreign_file',
    'worker_alive','empty_payload','newer_output','wrong_interruption'])
def test_cleanup_refuses_non_interrupted_or_foreign_inputs(cleanup_case,monkeypatch,fault):
    experiment,base,original,item,previous,_=cleanup_case
    directory=base/'cari_inputs'
    if fault=='running': previous['status']='running'
    elif fault=='different_phase': previous['episodes'][0]['phase']='refined'
    elif fault=='different_episode': item['episode']=1
    elif fault=='symlink': (directory/'unknown').symlink_to(directory/'aligned_depth.h5')
    elif fault=='hardlink':
        import os
        os.link(directory/'aligned_depth.h5',directory/'alias')
    elif fault=='foreign_file': write(directory/'unknown.bin',b'foreign')
    elif fault=='empty_payload': (directory/'aligned_depth.h5').write_bytes(b'')
    elif fault=='wrong_interruption': previous['report']={'bytes':1,'sha256':'f'*64}
    elif fault=='newer_output':
        import os
        stopped=experiment/'continuations'/resume.SCOUT_CONTINUATION/'report.json'
        value=stopped.stat().st_mtime_ns+1_000_000
        os.utime(directory/'aligned_depth.h5',ns=(value,value))
    else: monkeypatch.setattr(resume.subprocess,'check_output',lambda *a,**k:'owned worker still alive')
    if fault=='different_episode':
        assert resume.cleanup_interrupted_inputs(experiment,base,original,item,previous,lambda _:None) is None
    else:
        with pytest.raises(ValueError):
            resume.cleanup_interrupted_inputs(experiment,base,original,item,previous,lambda _:pytest.fail('no removal receipt'))
    assert (directory/'aligned_depth.h5').exists()


@pytest.mark.parametrize('published',['pin','pass_report'])
def test_cleanup_never_removes_successful_or_pinned_inputs(cleanup_case,published):
    experiment,base,original,item,previous,_=cleanup_case
    if published=='pin': write(experiment/'pins/cari_clip_000009_input_pins.json',b'opaque published pin')
    else: write(base/'cari_inputs/report.json',b'{"status":"pass"}')
    assert resume.cleanup_interrupted_inputs(experiment,base,original,item,previous,lambda _:pytest.fail('never cleanup')) is None
    assert (base/'cari_inputs/aligned_depth.h5').exists()


def test_cleanup_stops_if_upstream_changes_after_durable_receipt(cleanup_case,monkeypatch):
    experiment,base,original,item,previous,_=cleanup_case
    calls=iter([{'immutable':'before'},{'immutable':'changed'}])
    monkeypatch.setattr(resume,'_cleanup_upstream',lambda *a:next(calls))
    recorded=[]
    with pytest.raises(ValueError,match='Original completed inputs changed'):
        resume.cleanup_interrupted_inputs(experiment,base,original,item,previous,recorded.append)
    assert len(recorded)==1 and (base/'cari_inputs/aligned_depth.h5').exists()
