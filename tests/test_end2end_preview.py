from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
import end2end_preview as p

def test_original_random_denominator_not_success_resampled():
    assert p.COHORT == [9,1,14,7]
    assert 'FAIL / not adopted' in p.labels(9,False)[2]
    assert 'GT pending' in p.labels(14,True)[2]

def probe():
    return dict(streams=[dict(codec_name='h264',width=960,height=280,r_frame_rate='30/1',avg_frame_rate='30/1',nb_read_frames='415')])

def test_full_original_encoded_frames_and_sharedview():
    p.probe_contract(probe(),415)
    for key,value in [('nb_read_frames','414'),('avg_frame_rate','29/1'),('width',640)]:
        r=probe();r['streams'][0][key]=value
        with pytest.raises(ValueError):p.probe_contract(r,415)

def test_revision_no_urls_or_expressions():
    assert p.revision('a'*40)=='a'*40
    for value in ['HEAD','a'*39,'a'*40+'?sig=secret','../x']:
        with pytest.raises(ValueError):p.revision(value)

def test_render_source_has_no_modification_or_pixel_label_prompt():
    import ast,inspect
    tree=ast.parse(inspect.getsource(p))
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('refine_sequence','inference_pose') for n in ast.walk(tree))
    text=inspect.getsource(p)
    assert "floor = viewer.fixed_floor_height(ah)" in text
    assert "bh[frame]" in text and "range(total)" in text
    assert "str(video) in input_ledger" in text


def test_continuation_labels_and_kind_do_not_claim_fallback_gain():
    assert p.labels(9, True, 'continuation', 'accepted_native_continuation')[2].startswith('C contact-continuation')
    assert p.labels(14, True, 'continuation', 'dynamic_A_fallback_no_improvement')[2] == 'Baseline fallback / no gain'
    for kind in ['../x', 'oracle', None]:
        with pytest.raises(ValueError): p.candidate_kind(kind)
    with pytest.raises(ValueError): p.labels(9, True, 'continuation', 'made_up')


def continuation_report(status='accepted_native_continuation'):
    return dict(status=status, fitted_outputs_sealed_before_final_QA=True, full_original_frames=415,
        source_fps=30., native_direct136_generated=True, stale_B_pose_used=False,
        original_activations_unchanged=True, whole_hand_minimum_claimed=False, penetration_evaluated=False,
        hand_labeled_test=False, oracle_modes=[], decision={'C_vs_A': {'passed': True}},
        continuation=dict(status=status, full_original_frames=415, clip_global_alpha=True,
            ground_truth_used=False, private_truth_read=False, metric_improvement_claimed=False,
            alpha=0. if status == 'dynamic_A_fallback_no_improvement' else .5))


@pytest.mark.parametrize('bad', ['frames', 'seal', 'direct', 'stale', 'activity', 'wholehand', 'PEN', 'oracle', 'alpha', 'fallback_alpha'])
def test_actual_complete_continuation_contract_rejects_stale_or_untruthful_outputs(bad):
    r = continuation_report()
    if bad == 'frames': r['full_original_frames'] = 96
    elif bad == 'seal': r['fitted_outputs_sealed_before_final_QA'] = False
    elif bad == 'direct': r['native_direct136_generated'] = False
    elif bad == 'stale': r['stale_B_pose_used'] = True
    elif bad == 'activity': r['original_activations_unchanged'] = False
    elif bad == 'wholehand': r['whole_hand_minimum_claimed'] = True
    elif bad == 'PEN': r['penetration_evaluated'] = True
    elif bad == 'oracle': r['oracle_modes'] = ['GT']
    elif bad == 'alpha': r['continuation']['alpha'] = 0
    elif bad == 'fallback_alpha':
        r = continuation_report('dynamic_A_fallback_no_improvement'); r['continuation']['alpha'] = .5
    with pytest.raises(ValueError): p.complete_candidate(r, 'continuation', 415)


def test_preserved_joint_and_native_continuation_completion_ABIs():
    from world_reward.shared_identity import NATIVE_PARAMETER_DIMS
    assert set(p.FROZEN_NATIVE_KEYS) == set(NATIVE_PARAMETER_DIMS)-{'mhr_trans','mhr_body_pose_cont'}
    assert p.complete_candidate(dict(native_effective_updates=301, fitted_outputs_sealed_before_QA=True,
        source_inputs_rehashed=True, decision={'passed': False}), 'joint', 415) is False
    assert p.complete_candidate(continuation_report(), 'continuation', 415) is True
    assert p.complete_candidate(continuation_report('dynamic_A_fallback_no_improvement'), 'continuation', 415) is True
    wrapper = (Path(__file__).resolve().parents[1]/'infra/run_end2end_preview.sh').read_text()
    assert 'PREFIX=native-contact-continuation-real' in wrapper and 'run_native_contact_continuation_real,readonly' in wrapper
    assert '--candidate-kind' in wrapper and '--network none' in wrapper


def cohort_report():
    rev = 'a'*40
    helpers = {p:{'bytes':1,'sha256':'b'*64} for p in ('infra/native_contact_continuation_real.py',
        'infra/run_native_contact_continuation_real.sh','src/world_reward/native_contact_continuation.py')}
    r = dict(schema='world_reward.native_contact_continuation_real.v1',
        status='complete_native_continuation_diagnostic', producer_revision=rev,
        source_inputs_rehashed=True, unexpected_failures=0, native_layers_loaded=1,
        ground_truth_used=False, private_truth_read=False, baseline_modified=False, production_adopted=False,
        cohort=dict(random_seed=20261008,population=30,episodes=p.COHORT),
        source_binding={'helpers':helpers}, input_ledger={'original_RGB':{'bytes':1,'sha256':'b'*64}},
        episodes=[dict(episode=ep,status='accepted_native_continuation',report={'bytes':1,'sha256':'b'*64})
            if ep in (9,14) else dict(episode=ep,status='unsupported_original_frontend_unchanged',
                baseline_retained=True,rerolled=False,fabricated_predictions=False,reason='original unsupported')
            for ep in p.COHORT])
    host = dict(producer_revision=rev,container_absence_verified=True,process_exit_code=0,GPU_requested=True)
    return r, host, rev


@pytest.mark.parametrize('bad', [None,'failure','order','rehash','cleanup','source','GT','reroll','missing_ledger'])
def test_continuation_root_ledger_completion_source_and_cleanup_are_mandatory(tmp_path, monkeypatch, bad):
    import json
    r, host, rev = cohort_report()
    expected_source = r['source_binding'].copy()
    if bad == 'failure': r['unexpected_failures'] = 1
    elif bad == 'order': r['episodes'].reverse()
    elif bad == 'rehash': r['source_inputs_rehashed'] = False
    elif bad == 'cleanup': host['container_absence_verified'] = False
    elif bad == 'source': r['source_binding'] = {'helpers':{}}
    elif bad == 'GT': r['ground_truth_used'] = True
    elif bad == 'reroll': r['episodes'][1]['rerolled'] = True
    elif bad == 'missing_ledger': r['input_ledger'] = {}
    for name, value in [('report.json',r),('host-exit.json',host)]:
        path = tmp_path/name; path.write_text(json.dumps(value)); path.chmod(0o444)
    monkeypatch.setattr(p,'source',lambda *_:expected_source)
    from sequence_pose_probe import ArtifactLedger
    if bad is None:
        assert p.continuation_cohort(ArtifactLedger(),tmp_path,rev) == r
    else:
        with pytest.raises(ValueError): p.continuation_cohort(ArtifactLedger(),tmp_path,rev)
