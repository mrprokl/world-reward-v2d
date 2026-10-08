"""Tiny CPU contracts only; never fetch/decode challenge media or load models."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

import task_grounding_pilot as pilot
from task_grounding_pilot import config
from world_reward.task_grounding import build_task_grounding_prompt, fixed_frame_indices


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_population_and_cost():
    cfg=config(ROOT)
    assert cfg['scope']=='known_stress_development_visual_diagnostic_not_heldout'
    assert cfg['model_revision']=='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
    assert sum(r['bytes'] for r in cfg['files']) == 17545914364 < 20*1024**3
    assert cfg['inference_budget_seconds']==900 and cfg['tracking_budget_seconds']==1800
    assert cfg['sam2_checkpoint']['bytes']==898083611
    assert cfg['preview_bytes_per_episode']==180000


def test_source_declares_offline_and_no_retry():
    src=(ROOT/'infra/task_grounding_pilot.py').read_text(); tree=ast.parse(src)
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
    gen=[node for node in calls if isinstance(node.func,ast.Attribute) and node.func.attr=='generate']
    assert len(gen)==1
    options={k.arg:ast.literal_eval(k.value) for k in gen[0].keywords if k.arg in ('do_sample','num_beams')}
    assert options=={'do_sample':False,'num_beams':1}
    assert "'--network','none'" in src and 'trust_remote_code=False' in src
    assert 'HF_HUB_OFFLINE=1' in src and 'TRANSFORMERS_OFFLINE=1' in src
    assert 'json.loads(text' not in src  # All response parsing goes through strict transport.
    assert "root/'jobs/.world-reward-h100.lock'" in src
    assert "str(exc)" not in src  # Signed publisher redirect URLs must not enter receipts.


def test_same_prompt_for_any_lawful_task():
    cfg=config(ROOT)
    a=build_task_grounding_prompt('target chair','sit on chair',fixed_frame_indices(399,cfg['views']))
    assert 'hands, feet or the body' in a
    assert 'Original frame' not in a  # Frame-labelled images are assembled by driver.
    assert 'Episode' not in a and 'World Reward' not in a
    data=json.loads(a.split('Task JSON: ')[1])
    assert data['frame_indices']==[0,49,99,149,199,248,298,348,398]


@pytest.fixture
def synthetic_inputs(monkeypatch, tmp_path):
    """Opaque tiny synthetic files, not challenge recordings or manual labels."""
    runtime = tmp_path.resolve()
    monkeypatch.setattr(pilot, 'ROOT', runtime)
    population = (8, 9, 26, 1, 14, 7)
    rows = []
    def save(relative, contents):
        path = runtime / 'data' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        rows.append({'path': relative, 'bytes': len(contents), 'sha256': hashlib.sha256(contents).hexdigest()})
    metadata = {
        'episodes.jsonl': [{'episode_index': episode, 'length': 9 + episode, 'task_index': episode}
                           for episode in population],
        'episodes_metadata.jsonl': [{'episode_index': episode, 'object_prompt': 'synthetic object'}
                                   for episode in population],
        'tasks.jsonl': [{'task_index': episode, 'task': f'synthetic action {episode}'}
                       for episode in population],
    }
    for name, records in metadata.items():
        save('track_1/meta/' + name, b'\n'.join(json.dumps(row).encode() for row in records) + b'\n')
    for episode in population:
        save(f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4',
             f'opaque synthetic bytes {episode}'.encode())
    manifest = {'track': 'track_1', 'repo_id': 'nvidia/video_to_data_challenge',
                'revision': pilot.DATASET_REVISION, 'files': rows}
    (runtime / 'results').mkdir()
    (runtime / 'results/input-manifest.json').write_text(json.dumps(manifest))
    return runtime


def test_optional_sample_preserves_frozen_default_and_order(synthetic_inputs):
    assert [row['episode'] for row in pilot.inputs()] == [8, 9, 26]
    rows = pilot.inputs((9, 1, 14, 7))
    assert [row['episode'] for row in rows] == [9, 1, 14, 7]
    cfg, selected = pilot._runtime_inputs(ROOT, rows, config(ROOT))
    assert cfg == config(ROOT) and cfg['episodes'] == [8, 9, 26]
    assert selected == rows and selected is not rows
    # Explicit sample changes no grounding prompt, model, or numerical setting.
    assert 'Episode' not in build_task_grounding_prompt(rows[0]['object_prompt'], rows[0]['action'],
                                                       fixed_frame_indices(rows[0]['total'], cfg['views']))


@pytest.mark.parametrize('episodes', [(), (True,), (-1,), (30,), (9, 9), {9, 1}, '9'])
def test_optional_sample_rejects_nonoriginal_or_ambiguous_population(episodes):
    with pytest.raises(ValueError, match='Unique original Track1 episode'):
        pilot.inputs(episodes)


def test_selected_sample_hashes_only_its_original_videos(synthetic_inputs):
    unused = synthetic_inputs / 'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000026.mp4'
    unused.write_bytes(b'changed excluded synthetic video')
    assert [row['episode'] for row in pilot.inputs((9, 1, 14, 7))] == [9, 1, 14, 7]
    with pytest.raises(ValueError, match='Original input identity differs'):
        pilot.inputs()


@pytest.mark.parametrize('fault', ['manual_extra', 'wrong_video', 'short_T', 'missing_provenance', 'changed_settings'])
def test_reusable_runtime_refuses_unfrozen_inputs_or_settings(synthetic_inputs, fault):
    rows = copy.deepcopy(pilot.inputs((9, 1, 14, 7)))
    settings = copy.deepcopy(config(ROOT))
    if fault == 'manual_extra': rows[0]['human_bbox'] = [0, 0, 1, 1]
    elif fault == 'wrong_video': rows[0]['video'] = str(synthetic_inputs / 'data/track_2/video.mp4')
    elif fault == 'short_T': rows[0]['total'] = 8
    elif fault == 'missing_provenance': rows[0]['metadata_pins'].pop('tasks.jsonl')
    else: settings['min_pixels'] += 1
    with pytest.raises(ValueError):
        pilot._runtime_inputs(ROOT, rows, settings)
