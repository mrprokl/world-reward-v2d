"""Manufactured public receipts/arrays and private adapter gates, no real data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'infra'))
spec=importlib.util.spec_from_file_location('wr_ycbv_evaluate_test',REPO/'infra/ycbv_point_evaluate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def write(p,raw):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o400);return gate.files.identity(p)


def record(p,data,revision='b'*40,script='c'*64):
    data=dict(data,producer_revision=revision,script_sha256=script)
    return dict(**write(p,(json.dumps(data)+'\n').encode()),producer_revision=revision,script_sha256=script)


def fixture(tmp_path):
    base=tmp_path/gate.BASE;base.mkdir(parents=True)
    retention=dict(bytes=12,sha256='a'*64)
    acq=record(base/'report.json',dict(stage='external_ycbv_contiguous_rgb_only_acquisition',status='pass',phase='complete',
      dataset_revision='5c2c4aa229800355648cd268040aa814f8dc94f0',license='MIT',device='cpu',gpu_used=False,
      challenge_inputs_used=False,inference_performed=False,source_rehashed_after=True,disposable_archives_removed=True,
      selection_before_private_annotation_values=True,selected_frames=288,all_instances_retained=True,
      private_annotations_exported_as_inference_inputs=False,retention_receipt=retention))
    rows=[];predictions={};initial_masks={};maskrows=[]
    comp=base/'comparison_v1';comp.mkdir();write(comp/'.container.cid',b'e'*64)
    for scene in gate.SCENES:
        arrays=dict(frame_index=np.arange(96,dtype=np.int64),baseline_poses=np.tile(np.eye(4),(96,1,1)),candidate_poses=np.tile(np.eye(4),(96,1,1)))
        p=comp/f'scene_{scene:06d}.npz';np.savez_compressed(p,**arrays);p.chmod(0o400);identity=gate.files.identity(p);predictions[p.name]=identity
        ids={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest())for k,v in arrays.items()}
        rows.append(dict(file=p.name,scene_id=scene,frames=96,arrays=ids,**identity))
        m=np.zeros((480,640),np.uint8);m[:3,:4]=255
        p=base/f'automatic_masks_v1/scene_{scene:06d}/masks/1/000000.png';p.parent.mkdir(parents=True);Image.fromarray(m).save(p);p.chmod(0o400)
        mp=gate.files.identity(p);initial_masks[str(scene)]=mp
        for frame in range(96):maskrows.append(dict(scene_id=scene,frame_id=frame,file=f'scene_{scene:06d}/masks/1/{frame:06d}.png',**mp))
    track=record(comp/'report.json',dict(stage='public_ycbv_same_native25_pool_boots_point_comparison',status='pass',phase='complete',
      full_original_frame_coverage=True,same_native_valid_pool=True,native_pool_frozen_before_tracking_and_rankings=True,
      all_inputs_models_sources_after_reverified=True,ground_truth_used=False,private_annotations_read=False,sensor_depth_used=False,
      source_camera_calibration_used=False,human_scale_used=False,hand_labeled_test=False,oracle_initial_queries=False,oracle_modes=[],challenge_inputs_used=False,
      whole_pilot_GPU_elapsed_seconds=800.,outputs=rows))
    masks=record(base/'automatic_masks_v1/report.json',dict(stage='public_ycbv_point_native_object_masks',status='pass',phase='complete',
      private_annotations_read=False,query='object.',frames_completed=288,all_inputs_sources_assets_outputs_rehashed=True,masks=maskrows))
    pins=dict(schema='world-reward-ycbv-point-evaluation-pins-v1',acquisition_report=acq,retention=retention,track_report=track,
      masks_report=masks,predictions=predictions,initial_masks=initial_masks)
    return tmp_path,pins


def test_all_public_predictions_before_any_private_read(tmp_path,monkeypatch):
    root,pins=fixture(tmp_path);original=Path.open
    def safe(path,*args,**kwargs):
        assert 'eval_private'not in path.parts;return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',safe)
    rows,_=gate.public_predictions(root,pins);assert len(rows)==3 and all(x['baseline'].shape==(96,4,4)for x in rows)
    assert all(x['automatic_mask'].sum()==12 for x in rows)


def test_prediction_receipt_tamper_before_private(tmp_path):
    root,pins=fixture(tmp_path);p=root/gate.BASE/'comparison_v1/scene_000050.npz';p.chmod(0o600);p.write_bytes(b'bad');p.chmod(0o400)
    with pytest.raises(ValueError):gate.public_predictions(root,pins)


@pytest.mark.parametrize('fault',['missing','extra','producer','schema','floatbytes','floatframe','overbudget','GT'])
def test_public_or_pin_contract_stops(tmp_path,fault):
    root,pins=fixture(tmp_path)
    if fault=='missing':pins['predictions'].pop('scene_000050.npz')
    elif fault=='extra':write(root/gate.BASE/'comparison_v1/extra.json',b'{}')
    elif fault=='producer':pins['track_report']['producer_revision']='d'*40
    elif fault=='schema':pins['schema']='wrong'
    elif fault=='floatbytes':pins['retention']['bytes']=12.
    else:
        p=root/gate.BASE/'comparison_v1/report.json';data=json.loads(p.read_bytes())
        if fault=='floatframe':data['outputs'][2]['frames']=96.
        elif fault=='overbudget':data['whole_pilot_GPU_elapsed_seconds']=3601.
        else:data['ground_truth_used']=True
        p.chmod(0o600);pins['track_report']=record(p,data)
    with pytest.raises(ValueError):gate.public_predictions(root,pins)


def test_private_retention_exact_all_files_and_no_label_interpretation(tmp_path):
    base=tmp_path/gate.BASE/'eval_private';base.mkdir(parents=True,mode=0o700)
    files=[]
    for name in('hf_readme','publisher_readme','publisher_license','bop_format','bop_params'):
        path=f'source/licenses/{name}.txt';files.append(dict(file=path,**write(base/path,b'license fixture')))
    path='source/licenses/dataset_info.md';files.append(dict(file=path,**write(base/path,b'license fixture')))
    for scene in gate.SCENES:
        prefix=f'source/test/{scene:06d}/'
        for kind in('camera','gt','gt_info'):
            path=prefix+f'scene_{kind}.json';files.append(dict(file=path,**write(base/path,b'not JSON but hash only')))
        for frame in range(96):
            for folder,suffix in(('depth',''),('mask','_000000'),('mask_visib','_000000')):
                path=prefix+f'{folder}/{frame:06d}{suffix}.png';files.append(dict(file=path,**write(base/path,b'not PNG but hash only')))
    data=dict(files=files,all_instances_retained=True)
    rp=write(base/'retention-receipt.json',json.dumps(data).encode())
    result=gate.private_identities(tmp_path,dict(retention=rp));assert len(result)==879
    write(base/'source/unpinned.json',b'{}')
    with pytest.raises(ValueError):gate.private_identities(tmp_path,dict(retention=rp))


def test_wrapper_cpu_narrow_and_shell_syntax():
    shell=REPO/'infra/run_ycbv_point_evaluate.sh';subprocess.run(['bash','-n',str(shell)],check=True)
    text=shell.read_text();assert '--gpus'not in text and '--network none'in text and 'CUDA_VISIBLE_DEVICES='in text
    assert 'dst=$BASE/eval_private'in text or '"$BASE/eval_private"'in text
    assert '--cap-drop ALL'in text and '--read-only'in text and '--user 0:0'in text


@pytest.mark.parametrize('fault',[None,'marker','helper','revision'])
def test_host_producer_full_marker_linkage(tmp_path,fault):
    reports={};pins={}
    for name,key,job,entry in(('acquisition','acquisition_report','run_ycbv_point_acquire','infra/ycbv_point_acquire.py'),
      ('track','track_report','run_ycbv_point_track','infra/ycbv_point_track.py'),
      ('masks','masks_report','run_ycbv_point_masks','infra/ycbv_point_masks.py')):
        revision='b'*40;code=tmp_path/'jobs'/revision/job/'code';hp=write(code/entry,b'# fixed original helper\n')
        write(code.parent/'revision',(revision+'\n').encode());write(code.parent/'source-sha256',('d'*64+'\n').encode())
        markers={n:gate.files.identity(code.parent/n)for n in('revision','source-sha256')}
        pins[key]=dict(bytes=12,sha256='a'*64,producer_revision=revision,script_sha256=hp['sha256'])
        binding=dict(helpers={entry:hp},markers=markers)if name=='masks'else dict(files={entry:hp},markers=markers)
        reports[name]=dict(source_binding=binding)if name=='masks'else dict(source_helpers=binding)
    if fault=='marker':
        p=tmp_path/'jobs'/('b'*40)/'run_ycbv_point_masks/source-sha256';p.chmod(0o600);write(p,('e'*64+'\n').encode())
    elif fault=='helper':
        p=tmp_path/'jobs'/('b'*40)/'run_ycbv_point_track/code/infra/ycbv_point_track.py';p.chmod(0o600);write(p,b'# changed helper\n')
    elif fault=='revision':
        p=tmp_path/'jobs'/('b'*40)/'run_ycbv_point_acquire/revision';p.chmod(0o600);write(p,('e'*40+'\n').encode())
    if fault is None:assert set(gate.producer_sources(tmp_path,pins,reports))=={'acquisition','track','masks'}
    else:
        with pytest.raises(ValueError):gate.producer_sources(tmp_path,pins,reports)
