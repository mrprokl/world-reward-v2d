"""Frozen protocol, byte reuse and native orchestration without services/GPU."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import full4d_coverage as coverage
import full4d_pins as pins
import cari_clip_pin_inventory as inventory
from test_cari_clip_inputs_latent import case as latent_case

REPO = Path(__file__).resolve().parents[1]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode()); path.chmod(0o444)
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def config(tmp_path):
    for name in (coverage.CONFIG, 'configs/full4d_sample_v1.json'):
        put(tmp_path/name, (REPO/name).read_bytes())
    return coverage.protocol(tmp_path)


@pytest.mark.parametrize('bad', [None, 'cohort', 'reroll', 'updates', 'oracle', 'edge', 'budget', 'scope'])
def test_exact_protocol_is_separate_and_not_quality_selected(tmp_path, bad):
    cfg = config(tmp_path)
    if bad is None:
        assert cfg['episodes'] == [1, 7] and cfg['cohort'] == [9, 1, 14, 7]
        assert cfg['allow_unobserved_poses'] and cfg['native_refinement_steps'] == 300
        return
    if bad == 'cohort': cfg['episodes'] = [9, 14]
    elif bad == 'reroll': cfg['resample_failed_clips'] = True
    elif bad == 'updates': cfg['native_refinement_steps'] = 3
    elif bad == 'oracle': cfg['ground_truth_used'] = True
    elif bad == 'edge': cfg['edge_policy'] = 'freeze_last_pose'
    elif bad == 'budget': cfg['budgets']['refined'] = 10
    else: cfg['selection'] = 'heldout_accuracy'
    path = tmp_path/coverage.CONFIG; path.chmod(0o644); put(path, cfg)
    with pytest.raises(ValueError): coverage.protocol(tmp_path)


def test_full_native_stages_follow_explicit_latent_initializer_and_fit():
    rows = list(coverage.stages()); names = [r[0] for r in rows]
    assert names == ['scale_smoke', 'object_grounded', 'surface', 'body_full', 'depth_full', 'adapter',
        'object_pose', 'inputs', 'prepare', 'forward', 'refined', 'export', 'video']
    latent = next(r for r in rows if r[0] == 'object_pose')
    assert '--full-video' in latent[2] and '--allow-unobserved-poses' in latent[2]
    assert latent[5] == 'object_pose_full_surface_latent'
    assert '--allow-unobserved-poses' in next(r[2] for r in rows if r[0] == 'inputs')
    assert names.index('refined') < names.index('export') < names.index('video')


def test_command_keeps_exact_native_readonly_input_and_changes_only_latent_output_mount(tmp_path, monkeypatch):
    cfg = config(tmp_path/'code'); root = tmp_path/'root'; code = tmp_path/'code'
    experiment = root/'experiments'/('full4d-v1-'+'a'*40)
    monkeypatch.setattr(coverage, 'ROOT', root)
    old = str(root/'outputs/episode_000001/object_pose_full_surface')
    original = ['docker', 'run', '--network', 'none', '--read-only', '--gpus', 'all', '--mount',
        'type=bind,src='+old+',dst='+old, 'native-pinned-image', '--full-video']
    monkeypatch.setattr(coverage, 'command', lambda *_a: original.copy())
    argv = coverage.coverage_command(code, experiment, cfg, 1, 'object_pose', 'x', (), 'img', 'owned', 'a'*40)
    assert argv[:8] == original[:8]
    assert argv[8] == 'type=bind,src='+old+'_latent,dst='+old+'_latent'
    assert argv[9:] == original[9:] and original[8].endswith(old)
    assert coverage.coverage_command(code, experiment, cfg, 1, 'refined', 'x', (), 'img', 'owned', 'a'*40) == original


def reuse_case(tmp_path, monkeypatch):
    root = tmp_path/'root'; old_experiment = root/'experiments'/('full4d-v1-'+coverage.BASELINE)
    base = old_experiment/'outputs/episode_000001'; dest = root/'experiments/new/outputs/episode_000001'; dest.mkdir(parents=True)
    item = dict(episode=1, total=4, video_pin=dict(bytes=1, sha256='a'*64))
    metadata = dict(stage='automatic_masks', status='pass', episode_index=1, frames=4, input_track='track_1',
        input_sha256='a'*64, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], empty_frames={'0':0,'1':1})
    area = {'0':[100]*4, '1':[100, 0, 100, 100]}
    records = [dict(frame_index=i, decoded_rgb_sha256=str(i)*64) for i in range(4)]
    for role in ('0', '1'):
        for i in range(4): put(base/f'automatic_masks/masks/{role}/{i:06d}.png', b'unit-test-placeholder-not-image'+role.encode()+bytes([i]))
    masks, aggregate, _ = coverage._inventory(base/'automatic_masks/masks', 4)
    metadata['mask_inventory'] = aggregate
    files = {'report.json':put(base/'automatic_masks/report.json', metadata),
        'tracking.json':put(base/'automatic_masks/tracking.json', dict(areas=area, records=records)),
        'prompts.json':put(base/'automatic_masks/prompts.json', {'prompts':[]}),
        'grounding.json':put(base/'automatic_masks/grounding.json', {}),
        'mask-inventory.json':put(base/'automatic_masks/mask-inventory.json', masks)}
    row = dict(frontend=dict(files=files, mask_inventory=aggregate)); oldcode = tmp_path/'oldcode'
    monkeypatch.setattr(coverage.os, 'chown', lambda *_a:None)
    return root, oldcode, old_experiment, base, dest, item, row, area, records


def test_byte_identical_reuse_never_relabels_or_changes_original_receipts(tmp_path, monkeypatch):
    root, code, exp, base, dest, item, row, _, _ = reuse_case(tmp_path, monkeypatch)
    before = {str(p):p.read_bytes() for p in (base/'automatic_masks').rglob('*') if p.is_file()}
    result = coverage.reuse(root, code, exp, row, item, dest, 'automatic_masks')
    assert result['model_calls'] == 0 and result['original_receipt_preserved']
    assert all(Path(path).read_bytes() == value for path, value in before.items())
    for name in result['files']:
        assert (dest/'automatic_masks'/name).read_bytes() == (base/'automatic_masks'/name).read_bytes()
        assert not (dest/'automatic_masks'/name).stat().st_mode & 0o222
    assert coverage._inventory(dest/'automatic_masks/masks', 4)[1] == row['frontend']['mask_inventory']
    with pytest.raises(FileExistsError): coverage.reuse(root, code, exp, row, item, dest, 'automatic_masks')


@pytest.mark.parametrize('bad', ['rootpin', 'mask_changed', 'leading', 'trailing', 'human_absent'])
def test_reuse_fails_before_outputs_on_missing_anchor_or_changed_bound_predictions(tmp_path, monkeypatch, bad):
    root, code, exp, base, dest, item, row, area, records = reuse_case(tmp_path, monkeypatch)
    if bad == 'rootpin': row['frontend']['files']['report.json']['sha256'] = 'f'*64
    elif bad == 'mask_changed':
        path = base/'automatic_masks/masks/1/000001.png'; path.chmod(0o644); put(path, b'changed original pixel bytes')
    else:
        area['0' if bad == 'human_absent' else '1'][1 if bad == 'human_absent' else (0 if bad == 'leading' else -1)] = 0
        path = base/'automatic_masks/tracking.json'; path.chmod(0o644)
        row['frontend']['files']['tracking.json'] = put(path, dict(areas=area, records=records))
    with pytest.raises(ValueError): coverage.reuse(root, code, exp, row, item, dest, 'automatic_masks')
    assert not (dest/'automatic_masks').exists()


def test_native_sam31_receipt_without_legacy_empty_frames_summary_is_reusable(tmp_path, monkeypatch):
    root, code, exp, base, dest, item, row, _, _ = reuse_case(tmp_path, monkeypatch)
    path = base/'automatic_masks/report.json'; report = json.loads(path.read_text())
    del report['empty_frames']; path.chmod(0o644)
    row['frontend']['files']['report.json'] = put(path, report)
    assert coverage.reuse(root, code, exp, row, item, dest, 'automatic_masks')['byte_identical_copy']


def test_sparse_body_depth_reuse_is_source_hash_bound_and_same_original_rgb(tmp_path, monkeypatch):
    root, code, exp, base, dest, item, row, _, tracking = reuse_case(tmp_path, monkeypatch)
    ids = [0, 2, 3]; common = dict(status='pass', episode_index=1, input_track='track_1', input_sha256='a'*64,
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], total_video_frames=4)
    bodysha = put(code/'infra/body_smoke.py', b'native-source')['sha256']
    depthsha = put(code/'infra/depth_smoke.py', b'depth-source')['sha256']
    bodyrows = [dict(frame_index=i, decoded_rgb_sha256=tracking[i]['decoded_rgb_sha256']) for i in ids]
    predicted = put(base/'body_smoke/predictions.npz', b'not actual arrays')
    asset = put(root/'weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/LICENSE', b'fake unit license')
    put(base/'body_smoke/report.json', dict(common, stage='sam3d_body_three_frame_smoke', script_sha256=bodysha,
        frames=bodyrows, mhr_geometry_forward_verified=True, geometry_units='metres', body_assets={'LICENSE':asset},
        mask_report_sha256=row['frontend']['files']['report.json']['sha256'],
        prompts_sha256=row['frontend']['files']['prompts.json']['sha256'], predictions_sha256=predicted['sha256']))
    depthrows = []
    for i in ids:
        pin = put(base/f'depth_smoke/{i:06d}.npz', b'not actual depth'+bytes([i]))
        depthrows.append(dict(frame_index=i, decoded_rgb_sha256=tracking[i]['decoded_rgb_sha256'], output_sha256=pin['sha256']))
    put(base/'depth_smoke/report.json', dict(common, stage='monocular_moge2_three_frame', frames=depthrows, script_sha256=depthsha))
    for stage in ('body_smoke', 'depth_smoke'):
        value = coverage.reuse(root, code, exp, row, item, dest, stage)
        assert value['model_calls'] == 0 and value['original_receipt_preserved']
    # A receipt cannot substitute another frame's original RGB.
    path = base/'depth_smoke/report.json'; altered = json.loads(path.read_text()); altered['frames'][1]['decoded_rgb_sha256']='f'*64
    path.chmod(0o644); put(path, altered)
    with pytest.raises(ValueError): coverage.reuse(root, code, exp, row, item, dest, 'depth_smoke')


def test_v4_inventory_can_publish_hash_only_explicit_latent_source(tmp_path, monkeypatch):
    spec, _, _ = latent_case(inventory.inputs, tmp_path/'root')
    root = tmp_path/'root'; code = tmp_path/'code'
    script_pin = put(code/'infra/cari_prepare.py', b'unit-test-producer')
    for name in ('cari_clip_pin_inventory.py', 'run_cari_clip_pin_inventory.sh', 'cari_clip_inputs.py', 'cari96_inputs.py'):
        put(code/'infra'/name, (REPO/'infra'/name).read_bytes())
    path = root/inventory.inputs.relative_paths(spec)['input_report']; report = json.loads(path.read_text())
    report.update(script_sha256=script_pin['sha256'], submission_eligible=False, challenge_performance_verified=False,
        mesh_pose_frame_roundtrip_max_error_m=0., jpeg_original_RGB_mean_absolute_error=1.)
    path.write_text(json.dumps(report))
    monkeypatch.setattr(inventory.inputs, 'verify_public_inputs', lambda *_a:pytest.fail('No payload decode in pin publisher'))
    value = inventory.inventory(root, code, spec, report['producer_revision'], script_pin['sha256'], object_source='surface_latent')
    assert value['schema'] == 'world-reward-cari-clip-input-pins-v4' and value['allow_unobserved_poses']
    assert len(value['source_files']) == 15
    assert inventory.parser().parse_args(['--episode','26','--frames','96','--height','2','--width','4',
        '--camera-name','front','--producer-revision','c'*40,'--producer-script-sha256','a'*64,
        '--object-source','surface_latent']).object_source == 'surface_latent'


def test_full4d_pin_wiring_is_explicit_v4_and_default_v3_unchanged(tmp_path, monkeypatch):
    from test_full4d_pins import fixture, write
    root, code, rev, _, pinroot, _ = fixture(monkeypatch, tmp_path)
    spec = pins.inputs.PublicClipSpec(17, 96, 'front_stereo_camera_left', 1152, 1536)
    names = pins.inputs.source_paths(spec, object_source='surface_latent')
    for name in names: write(root/name, b'unit-test-payload', mode=0o644)
    sourcepin = write(code/'infra/cari_prepare.py', b'unit-test-source')
    write(root/pins.inputs.relative_paths(spec)['input_report'], json.dumps(dict(stage='world_reward_native_cari_inputs',
        status='pass', episode_index=17, frames=96, producer_revision=rev, script_sha256=sourcepin['sha256'],
        object_source='surface', original_frame_coverage_verified=True, input_track='track_1', ground_truth_used=False,
        hand_labeled_test=False, oracle_modes=[])).encode(), mode=0o644)
    monkeypatch.setattr(pins.inputs,'validate_reports',lambda *_a:None)
    value = pins.input_pin(root, code, rev, 17, 96, allow_unobserved_poses=True)
    assert value['schema'] == 'world-reward-cari-clip-input-pins-v4' and value['object_source']=='surface_latent'
    assert set(value['source_files']) == names and len(names) == 15
    assert all(not (root/name).stat().st_mode & 0o222 for name in names)


def test_fresh_launcher_refuses_local_execution_before_any_source_or_data_io(monkeypatch):
    monkeypatch.setattr(coverage.sys, 'platform', 'darwin')
    monkeypatch.setattr(coverage, 'source', lambda *_a:pytest.fail('Local execution must fail before IO'))
    with pytest.raises(ValueError): coverage.run(SimpleNamespace(baseline_report_bytes=1, baseline_report_sha256='a'*64))
    wrapper = (REPO/'infra/run_full4d_coverage.sh').read_text()
    assert 'set +x' in wrapper and '28920s' in wrapper and '--baseline-report-sha256' in wrapper
    assert 'scenesmith-ncc-h100-01' in wrapper and 'run_full4d_coverage/code' in wrapper


def test_actual_original_source_closure_and_independent_root_report_required(tmp_path, monkeypatch):
    cfg = config(tmp_path/'config'); root = tmp_path/'root'
    oldcode = root/'jobs'/coverage.BASELINE/'run_gemini_full4d/code'
    put(oldcode/'infra/gemini_full4d.py', b'unit-test-original-producer')
    put(oldcode.parent/'revision', (coverage.BASELINE+'\n').encode())
    put(oldcode.parent/'source-sha256', ('b'*64+'\n').encode())
    for p in (oldcode, *oldcode.rglob('*')):
        if p.is_dir(): p.chmod(0o555)
    binding = coverage.source(root, oldcode, coverage.BASELINE, cfg['baseline_entry'], ())
    report = dict(producer_revision=coverage.BASELINE, source_binding=binding,
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], baseline_modified=False,
        episodes=[dict(episode=ep, status='fail', phase='scale_smoke') if ep in (1,7)
            else dict(episode=ep,status='complete_full4d') for ep in coverage.COHORT])
    path = root/'experiments'/('full4d-v1-'+coverage.BASELINE)/'report.json'; pin = put(path, report)
    out = coverage.old_source(root, cfg, pin)
    assert out[0] == oldcode and out[2] == binding and out[-1] == pin
    with pytest.raises(ValueError): coverage.old_source(root, cfg, pin | {'sha256':'f'*64})
    # Changing a baseline failure into a success cannot select a new record.
    report['episodes'][1]['status'] = 'complete_full4d'; path.chmod(0o644); newer = put(path, report)
    with pytest.raises(ValueError, match='original upstream failures'):
        coverage.old_source(root, cfg, newer)


def test_actual_new_dispatcher_source_closure_and_pin_entry_are_explicit(tmp_path, monkeypatch):
    root = tmp_path/'root'; rev = 'd'*40; code = root/'jobs'/rev/coverage.ENTRY/'code'
    for name in dict.fromkeys(coverage.HELPERS):
        assert (REPO/name).is_file(), name
        put(code/name, (REPO/name).read_bytes())
    put(code.parent/'revision', (rev+'\n').encode())
    put(code.parent/'source-sha256', ('e'*64+'\n').encode())
    for path in (code, *code.rglob('*')):
        if path.is_dir(): path.chmod(0o555)
    binding = coverage.source(root, code, rev, coverage.ENTRY, coverage.HELPERS)
    assert set(binding['helpers']) == set(coverage.HELPERS)
    monkeypatch.setattr(pins, 'ROOT', root)
    monkeypatch.setenv('WR_OUTPUT_PREFIX', f'experiments/full4d-v1-{rev}/outputs')
    assert pins._context(root, code, rev, 1) == (root, code, root/f'experiments/full4d-v1-{rev}/outputs/episode_000001')
    with pytest.raises(ValueError): pins._context(root, code, 'f'*40, 1)
    original = code/'infra/run_full4d_coverage.sh'; original.chmod(0o644)
    with pytest.raises(ValueError, match='Readonly complete source closure'):
        coverage.source(root, code, rev, coverage.ENTRY, coverage.HELPERS)
