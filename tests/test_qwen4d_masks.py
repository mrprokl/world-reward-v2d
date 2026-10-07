"""Tiny synthetic byte contracts only: no models, media decoding or downloads."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from qwen4d_masks import DATASET, MODEL, MODEL_REVISION, PRODUCER, _inventory, prepare_masks
from world_reward.task_grounding import build_task_grounding_prompt, fixed_frame_indices, parse_response


def put(path, value, *, raw=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if raw else (json.dumps(value, sort_keys=True)+'\n').encode()
    path.write_bytes(data); path.chmod(0o400)
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def replace(path, value, *, raw=False):
    path.chmod(0o600)
    return put(path, value, raw=raw)


@pytest.fixture
def pilot(tmp_path):
    root = tmp_path.resolve()/'azure'; code = root/'code'
    episode, total = 8, 9
    original = root/'results'/('task-grounding-pilot-'+PRODUCER)
    source = original/f'episode_{episode:06d}'
    destination = root/'experiments'/('qwen4d-v1-'+'a'*40)/'outputs'/f'episode_{episode:06d}'/'automatic_masks'
    destination.parent.mkdir(parents=True)
    object_prompt, action = 'one target object', 'interact with target object'
    metadata_values = {
        'episodes.jsonl': [{'episode_index': episode, 'length': total, 'tasks': [action]}],
        'episodes_metadata.jsonl': [{'episode_index': episode, 'object_prompt': object_prompt}],
        'tasks.jsonl': [{'task_index': 0, 'task': action}],
    }
    files, metadata_pins = [], {}
    for name, rows in metadata_values.items():
        relative = 'track_1/meta/'+name
        pin = put(root/'data'/relative, b''.join((json.dumps(r)+'\n').encode() for r in rows), raw=True)
        metadata_pins[name] = pin; files.append(dict(path=relative, **pin))
    relative = f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4'
    video = root/'data'/relative
    video_pin = put(video, b'TINY SYNTHETIC BYTE FIXTURE, NOT A VIDEO', raw=True)
    files.append(dict(path=relative, **video_pin))
    put(root/'results/input-manifest.json', dict(track='track_1', repo_id='nvidia/video_to_data_challenge',
        revision=DATASET, files=files))
    cfg = dict(schema='world_reward.task_grounding_pilot.v1', episodes=[8,9,26], views=9,
        model=MODEL, model_revision=MODEL_REVISION, training_overlap_verified=False,
        challenge_overlap_verified=False,
        sam2_checkpoint=dict(name='sam2.1_hiera_large.pt', bytes=10, sha256='b'*64))
    put(code/'configs/task_grounding_pilot_v1.json', cfg)
    indices = fixed_frame_indices(total, 9)
    response = json.dumps({'frames': [dict(frame_index=f, person_bbox=[0,0,500,500],
        object_bbox=[500,500,1000,1000]) for f in indices]})
    records = parse_response(response, indices, 10, 10)
    prompt = build_task_grounding_prompt(object_prompt, action, indices)
    ground = dict(episode=episode, total=total, video=str(video), video_pin=video_pin,
        object_prompt=object_prompt, action=action, metadata_pins=metadata_pins,
        indices=list(indices), width=10, height=10, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        calls=1, status='seed_available', records=[asdict(r) for r in records], seed=asdict(records[0]))
    ground_pin = put(source/'grounding.json', ground)
    put(source/'response.txt', response.encode(), raw=True)
    prompts = {'prompts': [dict(frame_index=0, object_id=o, points=None, point_labels=None,
        mask_path=None, box=dict(zip(('x0','y0','x1','y1'), box)))
        for o,box in ((0,[0.,0.,5.,5.]), (1,[5.,5.,10.,10.]))]}
    put(source/'prompts.json', prompts)
    tracked = dict(episode=episode, frames=total, status='full_T_complete',
        areas={'0':[1]*total, '1':[0,1,1,1,1,1,1,1,1]})
    tracking_pin = put(source/'tracking.json', tracked)
    for obj in ('0','1'):
        for f in range(total):
            put(source/'masks'/obj/f'{f:06d}.png', f'TINY {obj}/{f} BYTES NOT PNG'.encode(), raw=True)
    global_receipts = {
        'report.json': dict(status='fail', phase='preview',
            producer_revision=PRODUCER, source_rehashed_after=True, quality_verified=False,
            ground_truth_used=False, hand_labeled_test=False, oracle_modes=[]),
        'inference.json': dict(status='complete', model=MODEL, model_revision=MODEL_REVISION,
            quality_verified=False, episodes=[ground]),
        'tracking.json': dict(quality_verified=False,
            checkpoint={k:cfg['sam2_checkpoint'][k] for k in ('bytes','sha256')},
            episodes=[dict(episode=episode,status='full_T_complete',frames=total)]),
    }
    saved = dict(producer_revision=PRODUCER,
        files={name:put(original/name,value) for name,value in global_receipts.items()},
        episodes=[dict(episode=episode, frames=total, grounding=ground_pin, tracking=tracking_pin,
            baseline_report=dict(bytes=1,sha256='c'*64))])
    put(code/'configs/task_grounding_saved_pins.json', saved)
    baseline = root/'outputs'/f'episode_{episode:06d}'/'automatic_masks'
    put(baseline/'report.json', {'untouched':'wrong original baseline retained'})
    return dict(root=root,code=code,episode=episode,total=total,source=source,
        original=original,destination=destination,saved=saved,ground=ground,
        tracked=tracked,prompts=prompts,baseline=baseline)


def run(p):
    return prepare_masks(p['root'],p['code'],p['episode'],p['destination'])


def repin(p, name, value, *, global_receipt=False):
    path = p['original']/name if global_receipt else p['source']/name
    pin = replace(path,value)
    saved = p['saved']
    if global_receipt: saved['files'][name] = pin
    else: saved['episodes'][0][name.removesuffix('.json')] = pin
    replace(p['code']/'configs/task_grounding_saved_pins.json',saved)


def test_copies_full_t_and_honest_provenance_without_baseline_write(pilot):
    p=pilot; baseline=(p['baseline']/'report.json').read_bytes()
    before=_inventory(p['source']/'masks',p['total'])
    report=run(p)
    assert report['stage']=='automatic_masks' and report['status']=='pass'
    assert report['frames']==p['total'] and report['episode_index']==8
    assert report['grounding_model']==MODEL and report['action']=='interact with target object'
    assert report['ground_truth_used'] is False and report['hand_labeled_test'] is False
    assert report['oracle_modes']==[] and report['quality_verified'] is False
    assert report['source_mask_bytes_independently_pinned'] is False
    assert report['empty_frames']=={'0':0,'1':1}
    assert report['mask_inventory']['files']==18
    assert 'confidence' not in report and 'detector_revision' not in report
    assert _inventory(p['source']/'masks',p['total'])==before
    assert _inventory(p['destination']/'masks',p['total'])==before
    assert (p['baseline']/'report.json').read_bytes()==baseline
    assert (p['destination']/'prompts.json').read_bytes()==(p['source']/'prompts.json').read_bytes()
    assert (p['destination']/'provenance/grounding.json').read_bytes()==(p['source']/'grounding.json').read_bytes()
    assert all(not path.stat().st_mode&0o222 for path in p['destination'].rglob('*') if path.is_file())
    assert json.loads((p['destination']/'report.json').read_bytes())==report


def test_no_overwrite_or_implicit_reuse(pilot):
    p=pilot; run(p); original=(p['destination']/'report.json').read_bytes()
    with pytest.raises(ValueError): run(p)
    assert (p['destination']/'report.json').read_bytes()==original


@pytest.mark.parametrize('episode',[True,7,30,8.0,'8'])
def test_frozen_population_only(pilot,episode):
    with pytest.raises(ValueError):
        prepare_masks(pilot['root'],pilot['code'],episode,pilot['destination'])
    assert not pilot['destination'].exists()


@pytest.mark.parametrize('kind',['baseline','foreign','wrong_episode','bad_revision','symlink'])
def test_unsafe_destination_rejected_before_write(pilot,kind):
    p=pilot
    targets={'baseline':p['baseline'],'foreign':p['root'].parent/'arbitrary',
        'wrong_episode':p['destination'].parent.parent/'episode_000009'/'automatic_masks',
        'bad_revision':p['root']/'experiments/qwen4d-v1-main/outputs/episode_000008/automatic_masks'}
    if kind=='symlink':
        link=p['root']/'escape'; link.symlink_to(p['destination'].parent,target_is_directory=True)
        target=link/'automatic_masks'
    else: target=targets[kind]
    with pytest.raises(ValueError): prepare_masks(p['root'],p['code'],8,target)
    assert not p['destination'].exists()


@pytest.mark.parametrize('kind',['missing_person','missing_object','extra_file','extra_stream','hardlink','symlink'])
def test_full_t_exact_regular_stream_contract(pilot,kind):
    p=pilot; folder=p['source']/'masks'
    if kind=='missing_person': (folder/'0/000003.png').unlink()
    elif kind=='missing_object': (folder/'1/000008.png').unlink()
    elif kind=='extra_file': put(folder/'0/unrelated.txt',b'x',raw=True)
    elif kind=='extra_stream': (folder/'2').mkdir()
    elif kind=='hardlink':
        path=folder/'0/000003.png'; path.unlink(); path.hardlink_to(folder/'0/000002.png')
    else:
        path=folder/'0/000003.png'; path.unlink(); path.symlink_to(folder/'0/000002.png')
    with pytest.raises(ValueError): run(p)
    assert not p['destination'].exists()


@pytest.mark.parametrize('key,value',[('status','complete_diagnostic_not_quality_pass'),('phase','infer'),('ground_truth_used',True),
    ('hand_labeled_test',True),('oracle_modes',['gt']),('quality_verified',True),('source_rehashed_after',False)])
def test_original_failures_or_oracle_never_promoted(pilot,key,value):
    p=pilot; receipt=json.loads((p['original']/'report.json').read_bytes()); receipt[key]=value
    repin(p,'report.json',receipt,global_receipt=True)
    with pytest.raises(ValueError): run(p)
    assert not p['destination'].exists()
    assert json.loads((p['original']/'report.json').read_bytes())[key]==value


@pytest.mark.parametrize('kind',['manual_point','mask_path','object_id','seed_box','extra_prompt',
    'bool_object_id','bool_frame_index','bool_box'])
def test_only_saved_automatic_prompt_bytes(pilot,kind):
    p=pilot; prompts=p['prompts']
    if kind=='manual_point': prompts['prompts'][0]['points']=[[1,2]]
    elif kind=='mask_path': prompts['prompts'][0]['mask_path']='manual.png'
    elif kind=='object_id': prompts['prompts'][0]['object_id']=7
    elif kind=='seed_box': prompts['prompts'][0]['box']['x0']=1.
    elif kind=='extra_prompt': prompts['prompts'].append(dict(prompts['prompts'][0]))
    elif kind=='bool_object_id': prompts['prompts'][0]['object_id']=False
    elif kind=='bool_frame_index': prompts['prompts'][0]['frame_index']=False
    else: prompts['prompts'][0]['box']['x0']=False
    replace(p['source']/'prompts.json',prompts)
    with pytest.raises(ValueError): run(p)
    assert not p['destination'].exists()


@pytest.mark.parametrize('kind',['tamper_grounding','tamper_video','wrong_model','bad_area_count','empty_entity','response_repair'])
def test_pins_and_required_entity_integrity(pilot,kind):
    p=pilot
    if kind=='tamper_grounding': replace(p['source']/'grounding.json',{'modified':True})
    elif kind=='tamper_video': replace(Path(p['ground']['video']),b'changed',raw=True)
    elif kind=='wrong_model':
        row=json.loads((p['original']/'inference.json').read_bytes()); row['model']='alternate/model'
        repin(p,'inference.json',row,global_receipt=True)
    elif kind in ('bad_area_count','empty_entity'):
        row=p['tracked']; row['areas']['1']=[0]*(8 if kind=='bad_area_count' else 9)
        repin(p,'tracking.json',row)
    else:
        raw=(p['source']/'response.txt').read_bytes()
        replace(p['source']/'response.txt',b'```json\n'+raw+b'\n```',raw=True)
    with pytest.raises(ValueError): run(p)
    assert not p['destination'].exists()


def test_no_model_execution_or_baseline_mutation_source():
    src=(Path(__file__).parents[1]/'infra/qwen4d_masks.py').read_text()
    assert 'import torch' not in src and 'import cv2' not in src and 'import numpy' not in src
    assert 'subprocess' not in src and 'unlink(' not in src
    assert 'source_mask_bytes_independently_pinned=False' in src
    assert "destination.open('xb')" in src and 'models_rerun=False' in src
