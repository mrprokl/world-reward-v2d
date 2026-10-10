"""Tiny host provenance fixtures only; never native model/data correctness."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'infra'), str(REPO/'src')]
import gemini_full4d as graph
import full4d_pins as pins
import object_budget_solid as surface
import bridge_rgb_anchor_infer as model_resolver


def payload(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode() if type(value) is dict else value
    path.write_bytes(raw); path.chmod(0o444)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize('mutation', ['episodes', 'manual_labels', 'ground_truth_used',
    'jitter_fix_claimed', 'native_refinement_steps', 'resample_failed_clips', 'frontend_producer_revision'])
def test_config_refuses_scientific_override(tmp_path, mutation):
    cfg = json.loads((REPO/graph.CONFIG).read_bytes())
    cfg['frontend_producer_revision'] = 'e'*40
    cfg[mutation] = [0, 1, 2, 3] if mutation == 'episodes' else True
    payload(tmp_path/graph.CONFIG, cfg)
    payload(tmp_path/'configs/full4d_sample_v1.json', json.loads((REPO/'configs/full4d_sample_v1.json').read_bytes()))
    with pytest.raises(ValueError): graph.config(tmp_path)


def test_config_frozen_random_cohort_and_native_defaults(tmp_path):
    cfg = json.loads((REPO/graph.CONFIG).read_bytes()); cfg['frontend_producer_revision'] = 'e'*40
    payload(tmp_path/graph.CONFIG, cfg)
    payload(tmp_path/'configs/full4d_sample_v1.json', json.loads((REPO/'configs/full4d_sample_v1.json').read_bytes()))
    assert graph.config(tmp_path)['episodes'] == [9, 1, 14, 7]
    assert graph.STAGES == __import__('full4d_sample').STAGES


@pytest.mark.parametrize('entry', ['run_full4d_sample', 'run_gemini_full4d', 'run_unapproved'])
def test_pin_context_requires_actual_allowlisted_source(monkeypatch, tmp_path, entry):
    rev = 'a'*40; root = tmp_path/'root'; code = root/'jobs'/rev/entry/'code'
    code.mkdir(parents=True); code.chmod(0o555)
    payload(code.parent/'revision', (rev+'\n').encode()); payload(code.parent/'source-sha256', ('b'*64+'\n').encode())
    monkeypatch.setattr(pins, 'ROOT', root)
    monkeypatch.setenv('WR_OUTPUT_PREFIX', f'experiments/full4d-v1-{rev}/outputs')
    if entry == 'run_unapproved':
        with pytest.raises(ValueError): pins._context(root, code, rev, 9)
    else: assert pins._context(root, code, rev, 9)[1] == code


@pytest.mark.parametrize('entry', ['run_gemini_full4d', 'run_unapproved'])
def test_surface_source_honestly_names_new_dispatcher(monkeypatch, tmp_path, entry):
    rev = 'a'*40; root = tmp_path/'root'; code = root/'jobs'/rev/entry/'code'
    code.mkdir(parents=True)
    payload(code.parent/'revision', b'fixture'); payload(code.parent/'source-sha256', b'fixture')
    monkeypatch.setattr(surface, 'ROOT', root)
    monkeypatch.setattr(surface, '__file__', str(code/'infra/object_budget_solid.py'))
    monkeypatch.setenv('WR_OUTPUT_PREFIX', f'experiments/full4d-v1-{rev}/outputs')
    calls = []; rt = SimpleNamespace(source=lambda *args: calls.append(args) or dict(producer_revision=rev))
    if entry == 'run_unapproved':
        with pytest.raises(ValueError): surface.surface_source(code, rev, rt)
        assert not calls
    else:
        result = surface.surface_source(code, rev, rt)
        assert result['source_entry'] == entry and calls[0][3] == entry


def test_exact_copy_is_immutable_and_does_not_overwrite(monkeypatch, tmp_path):
    src, dst = tmp_path/'source', tmp_path/'new'
    pin = payload(src, b'tiny numeric cache fixture')
    monkeypatch.setattr(graph.os, 'chown', lambda *args: None)
    graph._copy(src, dst, pin)
    assert src.read_bytes() == dst.read_bytes() and src.stat().st_ino != dst.stat().st_ino
    with pytest.raises(ValueError): graph._copy(src, dst, pin)
    with pytest.raises(ValueError): graph._copy(src, tmp_path/'other', pin | {'sha256':'f'*64})


@pytest.mark.parametrize('mutation', [None, 'gt', 'model', 'timeline', 'payload', 'source', 'calibration'])
def test_rgb_only_depth_cache_preserves_original_receipt(monkeypatch, tmp_path, mutation):
    root = tmp_path/'root'; original = root/'jobs'/graph.OLD/'run_full4d_sample'/'code'
    code = tmp_path/'newcode'; script = b'# exact unchanged depth numerical producer fixture\n'
    payload(original/'infra/depth_smoke.py', script); payload(code/'infra/depth_smoke.py', script)
    monkeypatch.setattr(graph, 'source', lambda *args: dict(closure_sha256='a'*64))
    monkeypatch.setattr(graph.os, 'chown', lambda *args: None)
    monkeypatch.setattr(graph, 'reserve', lambda path, **kwargs: path.mkdir())
    directory = root/'experiments'/('full4d-v1-'+graph.OLD)/'outputs/episode_000009/depth_smoke'
    item = dict(episode=9, total=5, video_pin=dict(bytes=100, sha256='b'*64))
    weight = root/f'weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/{graph.DEPTH_MODEL}/model.pt'
    weight_pin = payload(weight, b'manufactured model identity')
    monkeypatch.setattr(graph, 'depth_weight_identity', lambda root, expected: dict(
        canonical_blob=str(weight),artifact=weight_pin,snapshot_alias=False))
    rows=[]
    for index in (0,2,4):
        pin = payload(directory/f'{index:06d}.npz', b'manufactured numeric output')
        rows.append(dict(frame_index=index, output_sha256=pin['sha256'], decoded_rgb_sha256='c'*64))
    report = dict(stage='monocular_moge2_three_frame', status='pass', episode_index=9,
        input_track='track_1', input_sha256='b'*64, input_dataset_revision=graph.DATASET,
        total_video_frames=5, model_revision=graph.DEPTH_MODEL,
        intrinsics_source='RGB_size_only_default_FOV_prior_not_calibration',
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], network='none',
        script_sha256=hashlib.sha256(script).hexdigest(), model_sha256=weight_pin['sha256'], frames=rows)
    if mutation == 'gt': report['ground_truth_used'] = True
    elif mutation == 'model': report['model_revision'] = 'f'*40
    elif mutation == 'timeline': report['frames'][-1]['frame_index'] = 3
    elif mutation == 'payload': report['frames'][0]['output_sha256'] = 'f'*64
    elif mutation == 'source':
        (code/'infra/depth_smoke.py').chmod(0o644)
        payload(code/'infra/depth_smoke.py', b'changed numerical helper')
    elif mutation == 'calibration': report['intrinsics_source'] = 'source_camera_calibration'
    original_report_pin=payload(directory/'report.json', report)
    base=tmp_path/'freshbase'; base.mkdir()
    if mutation:
        with pytest.raises(ValueError): graph.depth_reuse(root,code,dict(reuse_depth_only_if_exact_provenance=True),item,base,'depth_smoke')
        assert not (base/'depth_smoke').exists()
    else:
        receipt=graph.depth_reuse(root,code,dict(reuse_depth_only_if_exact_provenance=True),item,base,'depth_smoke')
        assert receipt['report']==original_report_pin and receipt['inference_replayed'] is False
        assert (directory/'report.json').read_bytes() == (base/'depth_smoke/report.json').read_bytes()


def test_downstream_numerical_stage_sources_are_not_patched():
    # Contract entry metadata is the only change; no learned/model/optimizer constants.
    assert graph.STAGES[8][2] == ('--full-video', '--mesh-source', 'surface')
    assert graph.STAGES[-3][0] == 'refined'
    assert graph.DEPTH_FIELDS.keys() == {'depth_smoke','depth_full'}


def test_source_import_is_stdlib_only():
    import subprocess
    result = subprocess.run([sys.executable, '-B', '-c',
        "import sys;sys.path[:0]=['infra','src'];import gemini_full4d;assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('mutation', [None, 'native_pin', 'manual_prompt', 'missing_frame', 'producer_script',
    'empty_object', 'diagnostic_count', 'boolean_area', 'record_index', 'record_rgb', 'visibility',
    'seed_rgb', 'interpolated', 'pose_source'])
def test_frontend_requires_exact_automatic_native_lineage_and_copies_pngs(monkeypatch, tmp_path, mutation):
    root=tmp_path/'root'; rev='e'*40
    monkeypatch.setattr(graph, 'ROOT', root)
    monkeypatch.setattr(graph.os, 'chown', lambda *args: None)
    monkeypatch.setattr(graph, 'reserve', lambda path, **kwargs: path.mkdir())
    binding=dict(producer_revision=rev, helpers={})
    monkeypatch.setattr(graph, 'source', lambda *args: binding)
    source_code=root/'jobs'/rev/graph.FRONTEND_ENTRY/'code'
    script_pin=payload(source_code/'infra/gemini_sam31_track.py', b'# tiny automatic frontend producer fixture\n')
    consumer_code=tmp_path/'consumer/code'
    payload(consumer_code/'infra/object_pose_smoke.py', (REPO/'infra/object_pose_smoke.py').read_bytes())
    if mutation=='pose_source':
        (consumer_code/'infra/object_pose_smoke.py').chmod(0o644)
        payload(consumer_code/'infra/object_pose_smoke.py',b'# different pose method\n')
    aggregate_root=root/'results'/('gemini-sam31-'+rev)
    directory=aggregate_root/'episode_000009/automatic_masks'
    for role in ('0','1'):
        for index in range(3): payload(directory/f'masks/{role}/{index:06d}.png', b'tiny PNG byte fixture')
    rows, inventory, raw=graph._inventory(directory/'masks',3)
    payload(directory/'mask-inventory.json', raw)
    prompts=dict(prompts=[dict(object_id=role,points=None,point_labels=None,mask_path=None) for role in (0,1)])
    if mutation=='manual_prompt': prompts['prompts'][0]['points']=[[0,0]]
    payload(directory/'prompts.json', prompts)
    for name in graph.FRONTEND_DIAGNOSTICS:
        payload(directory/name, b'tiny retained metadata fixture')
    areas={'0':[50,50,50], '1':[40,40,40]}
    if mutation=='empty_object': areas['1'][1]=0
    if mutation=='boolean_area': areas['1'][1]=True
    records=[dict(frame_index=index,decoded_rgb_sha256='c'*64,
        person_visible=True,object_visible=bool(areas['1'][index]),native_presence=[True,True]) for index in range(3)]
    if mutation=='record_index': records[-1]['frame_index']=1
    if mutation=='record_rgb': records[-1]['decoded_rgb_sha256']='not-a-SHA'
    if mutation=='visibility': records[1]['object_visible']=False
    tracking=dict(status='full_T_complete',episode=9,frames=3,areas=areas,records=records,
        original_frame_indices=[0,1,2],interpolation=mutation=='interpolated',quality_verified=False)
    tracking_pin=payload(directory/'tracking.json',tracking)
    grounding=dict(episode=9,total=3,video_pin=dict(bytes=10,sha256='b'*64),
        ground_truth_used=False,hand_labeled_test=False,manual_points=False,
        records=[dict(frame_index=0,rgb_sha256='c'*64)])
    if mutation=='seed_rgb': grounding['records'][0]['rgb_sha256']='d'*64
    payload(directory/'grounding.json',grounding)
    diagnostics={role:dict(frames=3,visible_frames=sum(n>0 for n in values),
        empty_frames=sum(n==0 for n in values),longest_empty_run=1 if any(n==0 for n in values) else 0,
        quality_verified=False,interpolation=False) for role,values in areas.items()}
    if mutation=='diagnostic_count': diagnostics['1']['empty_frames']=1
    report=dict(stage='automatic_masks',status='pass',episode_index=9,frames=3,
        input_track='track_1',input_sha256='b'*64,ground_truth_used=False,
        hand_labeled_test=False,oracle_modes=[],producer_revision=rev,
        script_sha256=script_pin['sha256'],mask_inventory=inventory,diagnostics=diagnostics)
    if mutation=='producer_script': report['script_sha256']='f'*64
    report_pin=payload(directory/'report.json',report)
    native=dict(schema='world_reward.gemini_sam31_tracking.v1',status='complete_diagnostic_not_quality_pass',
        producer_revision=rev,ground_truth_used=False,manual_labels=False,
        episodes=[dict(episode_index=9,status='pass',frames=3,report=report_pin)])
    native_pin=payload(aggregate_root/'native-report.json',native)
    host=dict(status='complete_diagnostic_not_quality_pass',producer_revision=rev,
        source_binding=binding,native_report=native_pin,ground_truth_used=False)
    if mutation=='native_pin': host['native_report']=native_pin | dict(sha256='f'*64)
    payload(aggregate_root/'report.json',host)
    if mutation=='missing_frame': (directory/'masks/1/000002.png').unlink()
    dest=tmp_path/'fresh/masks'; dest.parent.mkdir()
    item=dict(episode=9,total=3,video_pin=dict(bytes=10,sha256='b'*64))
    if mutation:
        with pytest.raises(ValueError) as failure:
            graph.frontend_adapter(dict(frontend_producer_revision=rev),item,dest,consumer_code=consumer_code)
        assert not dest.exists()
        if mutation=='empty_object':
            assert type(failure.value) is graph.EmptyObservationUnsupported
            evidence=failure.value.evidence
            assert failure.value.frame_indices==[1]
            assert evidence['empty_runs']==[dict(first_frame=1,last_frame=1,frames=1)]
            assert evidence['tracking']==tracking_pin and evidence['mask_inventory']==inventory
            assert evidence['pose_source']['sha256']==graph.POSE_SOURCE_SHA
            assert evidence['physical_object_absence_claimed'] is False
            assert evidence['prediction_filled'] is False and evidence['model_calls']==0
            assert evidence['native_producer_areas_bound'] is True
            assert evidence['mask_pixels_independently_decoded'] is False
    else:
        receipt=graph.frontend_adapter(dict(frontend_producer_revision=rev),item,dest,consumer_code=consumer_code)
        assert receipt['byte_identical_copy'] and receipt['model_calls']==0
        assert receipt['aggregate_report']==native_pin
        assert set(p.name for p in dest.iterdir())==graph.FRONTEND_FILES | {'masks'}
        assert graph._inventory(dest/'masks',3)[:2]==(rows,inventory)
        assert (dest/'report.json').read_bytes()==(directory/'report.json').read_bytes()
        assert receipt['native_pose_support_preflight']['frame_indices']==[]


@pytest.mark.parametrize('status,wanted', [('complete_full4d_visual_diagnostic_not_quality_pass',True),
    (graph.ABSENCE_STATUS,True),('fail',False),('pending',False)])
def test_scientific_absence_continues_frozen_cohort_but_wiring_failure_stops(status,wanted):
    assert graph.scout_can_continue(status) is wanted


def test_absence_contract_preserves_denominator_and_does_not_enable_latent_solver():
    raw=(REPO/'infra/gemini_full4d.py').read_text()
    assert "except EmptyObservationUnsupported" in raw and 'cohort_denominator=len(selected)' in raw
    assert 'failed_clip_replaced=False' in raw and 'full4d_produced=False, model_calls=0' in raw
    assert hashlib.sha256((REPO/'infra/object_pose_smoke.py').read_bytes()).hexdigest()==graph.POSE_SOURCE_SHA
    assert '--allow-unobserved-poses' not in raw


@pytest.mark.parametrize('mutation', [None, 'writable_receipt', 'writable_model', 'receipt_during_read', 'callback_other_path', 'outside', 'absolute', 'wrong_digest', 'parent_alias', 'blob_alias', 'contents', 'hardlink', 'acquisition_revision', 'acquisition_cache'])
def test_depth_model_reuses_audited_two_level_hf_xet_resolver(monkeypatch, tmp_path, mutation):
    root=tmp_path/'root'; cache=root/'weights/cari4d/hf_home/hub'; repo=cache/'models--Ruicheng--moge-2-vitl-normal'
    raw=b'tiny known HF model cache byte fixture'; digest=hashlib.sha256(raw).hexdigest(); xet='9f'+'c'*62
    monkeypatch.setattr(model_resolver,'MOGE_SHA',digest)
    monkeypatch.setattr(model_resolver,'MOGE_BYTES',len(raw))
    monkeypatch.setattr(model_resolver,'XET_SHA',xet)
    blob=cache/'blobs'/xet[:2]/xet; pin=payload(blob,raw)
    snapshot=repo/'snapshots'/graph.DEPTH_MODEL; snapshot.mkdir(parents=True)
    alias=snapshot/'model.pt'; target='../../blobs/'+digest
    repoblob=repo/'blobs'/digest; repoblob.parent.mkdir()
    repoblob.symlink_to('../../blobs/'+xet[:2]+'/'+xet)
    acquisition=dict(assets=[dict(repo_id='Ruicheng/moge-2-vitl-normal',revision=graph.DEPTH_MODEL,cache_dir=str(cache))])
    if mutation=='acquisition_revision': acquisition['assets'][0]['revision']='f'*40
    elif mutation=='acquisition_cache': acquisition['assets'][0]['cache_dir']=str(tmp_path/'othercache')
    receipt=root/'results/weights-acquisition.json'; receipt_pin=payload(receipt,acquisition)
    if mutation in ('writable_receipt', 'receipt_during_read'):
        receipt.chmod(0o644)
    if mutation=='writable_receipt':
        # Existing bridge callers still reject writable metadata by default.
        with pytest.raises(ValueError,match='readonly regular artifact'):
            model_resolver.moge_asset(root)
    elif mutation=='writable_model': blob.chmod(0o644)
    elif mutation=='receipt_during_read':
        original=model_resolver.binding.strict_json
        def changed_after_decode(raw):
            decoded=original(raw)
            receipt.write_text(json.dumps(acquisition | dict(unrelated_metadata=1)))
            return decoded
        monkeypatch.setattr(model_resolver.binding,'strict_json',changed_after_decode)
    elif mutation=='callback_other_path':
        other=root/'results/not-acquisition.json'; payload(other,acquisition)
        original=model_resolver.moge_asset
        def wrong_path(root, *, acquisition_identity=None):
            acquisition_identity(other,2_000_000)
            return original(root,acquisition_identity=acquisition_identity)
        monkeypatch.setattr(model_resolver,'moge_asset',wrong_path)
    if mutation=='outside':
        outside=tmp_path/'outside'; payload(outside,raw); target=str(outside)
    elif mutation=='absolute': target=str(blob)
    elif mutation=='wrong_digest':
        payload(repo/'blobs'/('f'*64),raw); target='../../blobs/'+'f'*64
    elif mutation=='contents': blob.chmod(0o644); blob.write_bytes(b'changed'); blob.chmod(0o444)
    elif mutation=='hardlink': (cache/'blobs/alias').hardlink_to(blob)
    elif mutation=='blob_alias':
        outside=tmp_path/'outside'; payload(outside,raw); blob.unlink(); blob.symlink_to(outside)
    elif mutation=='parent_alias':
        (cache/'blobs').rename(cache/'realblobs'); (cache/'blobs').symlink_to(cache/'realblobs',target_is_directory=True)
    alias.symlink_to(target)
    if mutation not in (None,'writable_receipt'):
        with pytest.raises(ValueError): graph.depth_weight_identity(root,digest)
    else:
        actual=graph.depth_weight_identity(root,digest)
        assert actual['snapshot_alias'] is True and actual['artifact']==pin
        assert actual['canonical_blob']==str(blob) and len(actual['link_graph'])==3
        assert actual['resolver']=='bridge_rgb_anchor_infer.moge_asset_and_host_moge_chain'
        assert actual['acquisition_receipt']==receipt_pin
        assert actual['acquisition_metadata_writable'] is (mutation=='writable_receipt')
        assert actual['acquisition_permission_immutability_claimed'] is False
        assert actual['model_readonly_verified'] is True
        assert receipt.stat().st_mode & 0o777 == (0o644 if mutation=='writable_receipt' else 0o444)
        assert blob.stat().st_mode & 0o777 == 0o444
        assert alias.is_symlink() and os.readlink(alias)==target
