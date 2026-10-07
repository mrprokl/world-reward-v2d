"""Tiny CPU contracts only; never fetch/decode challenge media or load models."""
import ast
import json
from pathlib import Path

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
