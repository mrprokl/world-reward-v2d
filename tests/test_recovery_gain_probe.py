"""Owned tiny synthetic masks/RGB only, no challenge data or model loading."""
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import recovery_gain_probe as probe


def rectangle():
    mask=np.zeros((12,16),bool); mask[3:9,4:12]=True
    return mask


def frame(mask, raw=2., selected=True, evaluated=True):
    record=dict(choice='reverse_native_semantic_anchor_unverified' if selected else 'saved_forward_native',
        native_reverse_evaluated=evaluated,reverse_logits=None if not evaluated else
        dict(raw_presence_logit=raw,positive_pixels=int(mask.sum())))
    return record,dict(native_presence=[True,True if selected else False])


def test_gap_topology_keeps_original_indices_and_unknown_edges():
    assert probe.gaps(np.array([1,1,0,1,0,1,1,1],bool))==[
        dict(start=0,end=1,length=2),dict(start=3,end=3,length=1),dict(start=5,end=7,length=3)]
    assert probe.gaps(np.zeros(4,bool))==[]
    assert probe.gaps(np.ones(4,bool))==[dict(start=0,end=3,length=4)]


def test_largest_gap_middle_start_end_are_selected_without_manual_record_keys():
    empty=np.zeros(50,bool); empty[10:21]=True; empty[30:32]=True
    assert probe.qa_indices(empty,3)==[10,15,20]
    chosen=probe.qa_indices(empty)
    assert len(chosen)==12 and {9,10,15,20,21} <= set(chosen)
    assert chosen==probe.qa_indices(empty)


@pytest.mark.parametrize('n',[1,2,7,13,50])
def test_qa_global_budget_full_coverage_and_constant_dimensions(n):
    for mask in (np.zeros(n,bool),np.ones(n,bool)):
        for budget in (1,3,12):
            chosen=probe.qa_indices(mask,budget)
            assert len(chosen)==min(budget,n) and chosen==sorted(set(chosen))
            assert all(0<=i<n for i in chosen)


@pytest.mark.parametrize('limit',[0,13,True,1.5])
def test_qa_budget_invalid_rejected(limit):
    with pytest.raises(ValueError): probe.qa_indices(np.zeros(3,bool),limit)


def test_literal_positive_native_reverse_replaces_empty_without_synthetic_pixels():
    mask=rectangle(); record,timeline=frame(mask); empty=np.zeros_like(mask)
    assert probe.verify_frame(empty,mask,mask,record,timeline,False)


@pytest.mark.parametrize('mutation',['old_visible','chosen_erased','chosen_expanded','native_absence',
    'empty_reverse','unevaluated','nan_logit','wrong_native_pixels','unknown_choice'])
def test_recovery_fabrication_or_loss_of_native_honesty_is_rejected(mutation):
    mask=rectangle(); forward=np.zeros_like(mask); chosen=mask.copy(); reverse=mask.copy(); record,timeline=frame(mask)
    if mutation=='old_visible': forward=mask.copy()
    elif mutation=='chosen_erased': chosen[:]=False
    elif mutation=='chosen_expanded': chosen[0,0]=True
    elif mutation=='native_absence': record['reverse_logits']['raw_presence_logit']=-2.
    elif mutation=='empty_reverse': reverse[:]=False
    elif mutation=='unevaluated': record['native_reverse_evaluated']=False; record['reverse_logits']=None
    elif mutation=='nan_logit': record['reverse_logits']['raw_presence_logit']=float('nan')
    elif mutation=='wrong_native_pixels': record['reverse_logits']['positive_pixels']+=1
    else: record['choice']='reverse_amodal_invented'
    with pytest.raises(ValueError): probe.verify_frame(forward,chosen,reverse,record,timeline,False)


def test_negative_raw_logit_keeps_zero_reverse_even_when_segmentation_logit_positive():
    mask=rectangle(); empty=np.zeros_like(mask); record,timeline=frame(mask,raw=-2,selected=False)
    assert not probe.verify_frame(empty,empty,empty,record,timeline,False)


def test_unevaluated_is_unknown_and_literal_old_visibility_remains_unchanged():
    mask=rectangle(); empty=np.zeros_like(mask); record,timeline=frame(mask,selected=False,evaluated=False)
    timeline['native_presence'][1]=True
    assert not probe.verify_frame(mask,mask,empty,record,timeline,True)
    with pytest.raises(ValueError): probe.verify_frame(mask,empty,empty,record,timeline,True)


def test_morphology_and_person_overlap_are_diagnostics_not_mask_quality_gates():
    mask=rectangle(); person=np.zeros_like(mask); person[3:6,4:12]=True
    row=probe.mask_metrics(mask,person,48)
    assert row['area_pixels']==48 and row['area_vs_forward_visible_median']==1
    assert row['largest_component_fraction']==row['bbox_fill']==1
    assert row['person_mask_overlap_fraction']==.5 and 'passed' not in row
    mask[0,0]=True; row=probe.mask_metrics(mask,person,48)
    assert row['components']==2 and row['largest_component_fraction']==pytest.approx(48/49)


def test_no_common_observations_are_null_not_artificially_good_agreement():
    assert probe.summary([])==dict(count=0,minimum=None,median=None,p10=None,p95=None)
    assert probe.summary([0.,1.])['median']==.5


def test_qa_panels_share_same_original_rgb_and_global_layout(tmp_path):
    rgb=np.zeros((12,16,3),np.uint8); mask=rectangle(); empty=np.zeros_like(mask)
    tile=probe.comparison(rgb,empty,empty,mask,4)
    assert tile.size==(720,218)
    result=probe.write_qa(tmp_path/'qa.jpg',[tile]*12)
    assert result['bytes']<=180000
    assert result['sha256']==hashlib.sha256((tmp_path/'qa.jpg').read_bytes()).hexdigest()
    assert not (tmp_path/'qa.jpg').stat().st_mode & 0o222


def test_import_never_loads_models_or_gpu_code():
    root=Path(__file__).resolve().parents[1]
    script=r'''
import importlib.abc,sys
sys.path[:0]=[sys.argv[1]+'/src',sys.argv[1]+'/infra']
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','torchvision','sam3','transformers'}:
            raise AssertionError('Model import: '+fullname)
sys.meta_path.insert(0,Finder())
import recovery_gain_probe
assert recovery_gain_probe.RECOVERY=='0826d17bd1a300220a986b961fcf942d0036e2b4'
print('no-model-imports')
'''
    result=subprocess.run(['rtk','proxy',sys.executable,'-B','-c',script,str(root)],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='no-model-imports'


def test_wrapper_cpu_readonly_exact_receipts_and_separate_lease():
    path=Path(__file__).resolve().parents[1]/'infra/run_recovery_gain_probe.sh'
    script=path.read_text()
    result=subprocess.run(['rtk','proxy','bash','-n',str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    for token in ('--network none','--read-only','--cpus 4','--memory 8g','600s docker run',
                  'CUDA_VISIBLE_DEVICES=-1','--cidfile','container_absence_verified',
                  '.world-reward-recovery-gain.lock','results/sam31-recovery-$RECOVERY',
                  'jobs/$RECOVERY/run_sam31_recovery','results/input-manifest.json'):
        assert token in script
    assert '--gpus' not in script and '.world-reward-h100.lock' not in script
