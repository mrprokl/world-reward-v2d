import json
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
import form_hoi_external_eval as e


def config():
    return json.loads((Path(__file__).resolve().parents[1]/e.CONFIG).read_bytes())


def geometry(count=3):
    rng=np.random.default_rng(1);h=rng.normal(size=(count,18439,3)).astype(np.float32)
    j=rng.normal(size=(count,127,3)).astype(np.float32)
    return dict(human_vertices=h,human_joints=j,human_faces=np.array([[0,1,2],[2,3,4]],np.int64),
        object_vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32),
        object_faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64),
        object_rotation=np.broadcast_to(np.eye(3),(count,3,3)).copy(),
        object_translation=np.zeros((count,3),np.float32),object_scale=np.array(1.,np.float32),
        frame_index=np.arange(count,dtype=np.int64))


def roles():return dict(body=np.arange(64),alignment=np.arange(64),hands=np.arange(512))


def test_frozen_external_contract_and_original_grid(monkeypatch):
    c=config()
    assert c['metrics']==list(e.METRICS) and c['body_joint_indices']==list(e.JOINTS)
    assert c['original_frame_indices']==list(range(96))
    assert not c['kaggle_score_equivalence_claimed'] and not c['training_overlap_verified']
    assert c['reserved_evaluated']==0
    assert 'not Kaggle scores' in c['scope']
    monkeypatch.setattr(e,'pinned',lambda path,pin,maximum:json.loads(Path(path).read_bytes()))
    assert e.configuration(Path(__file__).resolve().parents[1])[0]==c


def test_geometry_no_missing_frame_renumber_scale_or_so3_repair():
    original=geometry();result=e.geometry_arrays(original,count=3)
    assert np.array_equal(result['human_vertices'],original['human_vertices'])
    assert result['object_scale']==1.
    for key,value in [('frame_index',np.array([0,1,3],np.int64)),('object_scale',np.ones(3)),
                      ('object_rotation',np.zeros((3,3,3))),('human_joints',np.zeros((3,70,3)))]:
        bad=dict(original);bad[key]=value
        with pytest.raises(ValueError):e.geometry_arrays(bad,count=3)
    with pytest.raises(ValueError):e.geometry_arrays(dict(original,ground_truth=True),count=3)


def test_fps_baseline_geometry_only_ties_smallest_index():
    p=np.array([[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,0]])
    idx=np.arange(5,dtype=np.int64)
    assert e.farthest_vertices(p,idx,4).tolist()==[0,1,2,3]
    assert len(np.unique(e.farthest_vertices(np.zeros((5,3)),idx,5)))==5
    with pytest.raises(ValueError):e.farthest_vertices(p,idx[::-1],4)


def test_anatomical_hand_role_digest_and_exact512():
    c=config();f=np.array([[0,1,2]],np.int64)
    spec=dict(vertex_indices=np.arange(4636,dtype=np.int32).reshape(2,2318),
        sample_local_indices=np.broadcast_to(np.arange(256,dtype=np.int32),(2,256)).copy(),
        sample_assignments=np.zeros((2,2318),np.int32),hand_faces_left=np.zeros((4603,3),np.int32),
        hand_faces_right=np.zeros((4603,3),np.int32))
    keys=('vertex_indices','sample_local_indices','sample_assignments','hand_faces_left','hand_faces_right')
    face=e.array_sha(f.astype(np.int32));topology=e.array_sha(f.astype(np.int32),*(spec[k] for k in keys))
    c['hand_spec'].update(faces_sha256=face,topology_sha256=topology)
    spec.update(faces_sha256=np.array(face),topology_sha256=np.array(topology),mhr_model_sha256=np.array(c['hand_spec']['mhr_model_sha256']))
    full,hands=e.anatomical_hands(spec,f,c)
    assert full.shape==(2,2318) and hands.shape==(512,)
    assert hands[:256].tolist()==list(range(256)) and hands[256:].tolist()==list(range(2318,2574))
    spec['sample_local_indices'][0,0]=1
    with pytest.raises(ValueError):e.anatomical_hands(spec,f,c)


def package(n=4):return dict(full_source_frames=n,total=3,camera='front_stereo_camera_left',width=1536,height=1152)


def edex(n=4):
    camera=dict(transform=np.column_stack((np.eye(3),[1.,2,3])).tolist(),
        intrinsics=dict(focal=[1000.,1000],principal=[768.,576.],size=[1536,1152]))
    return [dict(frame_start=0,frame_end=n,cameras=[camera,copy.deepcopy(camera)]),
        dict(sequence=['images/rear/000000.png','images/front_stereo_camera_left/000000.png'])]


def test_edex_identifies_front_not_index0_camera_world_inverse():
    value=edex();value[0]['cameras'][0]['transform'][0][3]=99.
    T,K,index=e.front_camera(value,package())
    assert index==1 and np.array_equal(T[:3,3],[-1.,-2,-3])
    assert K[0,0]==1000
    for mutate in ('ambiguous','trimmed','missing'):
        value=edex()
        if mutate=='ambiguous':value[1]['sequence'][0]=value[1]['sequence'][1]
        elif mutate=='trimmed':value[0]['frame_start']=1
        else:value[1]['sequence'][1]='images/other/000000.png'
        with pytest.raises(ValueError):e.front_camera(value,package())


def test_reference_native_world_no_second_flip_or_pred_cam_t_guess():
    original=geometry(count=4);pose=np.broadcast_to(np.eye(4),(4,4,4)).copy();pose[:,:3,3]=[1.,2,4.]
    target,meta=e.reference_geometry(dict(pred_vertices=original['human_vertices'],faces=original['human_faces']),
        dict(pred_joint_coords=original['human_joints'],pred_cam_t=np.ones((4,3))*999),pose,
        original['object_vertices'],original['object_faces'],edex(),package())
    assert np.array_equal(target['human_vertices'],original['human_vertices'][:3].astype(float)-[1.,2,3])
    assert np.array_equal(target['object_translation'],np.broadcast_to([0.,0,1.],(3,3)))
    assert meta['YZ_flip_applied_again'] is False and not meta['interaction_trim_used']
    bad=pose.copy();bad[0,3,3]=0
    with pytest.raises(ValueError):e.reference_geometry(dict(pred_vertices=original['human_vertices'],faces=original['human_faces']),
        dict(pred_joint_coords=original['human_joints']),bad,original['object_vertices'],original['object_faces'],edex(),package())


def test_reference_loader_fail_closed_before_import_torch_or_read(monkeypatch):
    called=[];monkeypatch.setattr(e,'reference_paths',lambda _:called.append(True))
    with pytest.raises(ValueError,match='entire prediction'):e.load_reference({},dict(all_four_paired_predictions_sealed=True))
    assert called==[]


def test_public_operator_calls_firsthuman_common_scene_no_fpsfactor_or_dropping():
    a=e.geometry_arrays(geometry(),count=3);b=e.geometry_arrays(geometry(),count=3);calls=[]
    def sample(v,f,n,seed):
        calls.append(('sample',n,seed));return np.broadcast_to(v[0],(n,3)).copy()
    def metrics(pred,target,frames,joints):
        calls.append(('metrics',frames.copy(),joints));assert pred['mhr_joints'].shape==(3,127,3)
        assert pred['mhr_vertices'].shape==(3,64,3)
        return dict(cd_h_cm=1.,cd_o_cm=2.,cd_c_cm=3.,acc_h_cm=4.,acc_o_cm=5.)
    def penetration(pred,scene,target):
        assert scene['hands'].shape==(3,512,3)
        assert np.array_equal(scene['mesh_vertices'],a['object_vertices'])
        assert scene['scale']==a['object_scale']
        calls.append(('PEN',));return 6.
    modules=(None,SimpleNamespace(episode_metrics=metrics,episode_penetration=penetration),SimpleNamespace(sample_mesh_surface=sample))
    result=e.five_metrics(a,b,roles(),modules,'seq')
    assert result==dict(cd_h_cm=1.,cd_o_cm=2.,cd_c_cm=3.,acc_h_cm=4.,acc_o_cm=5.,interpenetration_cm=6.)
    assert calls[:2]==[('sample',64,e.object_seed('seq'))]*2
    assert np.array_equal(calls[2][1],np.arange(3)) and calls[2][2]==e.JOINTS


def test_nozero_motion_coverage_claims_contact_or_selects_frames():
    a=e.geometry_arrays(geometry(),count=3);report=e.motion_coverage(a,roles())
    assert report['all96_frames_scored'] and not report['bbox_overlap_is_contact_truth']
    assert report['object_translation_step_m']['max']==0
    assert report['PEN_may_be_uninformative_when_no_interaction']


def test_receipts_atomic_immutable_and_do_not_overwrite(tmp_path):
    p=tmp_path/'done.json';pin=e.seal_json(p,dict(status='complete'))
    assert e.identity(p)==pin and not p.stat().st_mode&0o222
    with pytest.raises(ValueError):e.seal_json(p,dict(status='other'))
    assert not (tmp_path/'.done.json.part').exists()


def test_gate_missing_allfour_never_reference_ready(tmp_path):
    c=config();c['input_gate']['dev_producer_revision']=None
    with pytest.raises(ValueError,match='producer pin pending'):e.freeze_gate(c,{},tmp_path)


def test_runtime_closure_no_local_results_or_models_dependency():
    root=Path(__file__).resolve().parents[1]
    assert all((root/name).is_file() for name in e.HELPERS)
    assert all(Path(name).parts[0] in ('infra','configs','src') for name in e.HELPERS)
    wrapper=(root/'infra/run_form_hoi_external_eval.sh').read_text()
    assert '--gpus' not in wrapper and '--network none' in wrapper and 'CUDA_VISIBLE_DEVICES=-1' in wrapper
    assert 'mhr_model.pt' not in wrapper and 'track_1' not in wrapper
    assert 'dst=$DEV,readonly' in wrapper and 'dst=$PREDROOT,readonly' in wrapper
    assert 'OPENBLAS_NUM_THREADS=1' in wrapper


def frozen_fixture(tmp_path, monkeypatch):
    root=tmp_path/'root';dev_data=tmp_path/'dev';rev='a'*40;prod='b'*40
    monkeypatch.setattr(e,'ROOT',root);monkeypatch.setattr(e,'DEV_DATA',dev_data)
    cfg=config();cfg['input_gate']['dev_producer_revision']=rev
    cfg['input_gate']['dev_report_identity']=dict(bytes=1,sha256='c'*64)
    pred=root/'results/predictions';pred.mkdir(parents=True)
    cfg['input_gate']['prediction_manifest_path']=str(pred/'manifest.json')
    cfg['input_gate']['prediction_manifest_identity']=dict(bytes=1,sha256='d'*64)
    cohort=dict(cohort=[dict(sequence_id=f'seq{i}',split='development') for i in range(4)])
    dev=dev_data/rev;records={};report=dict(status='pass',inference_ready=True,producer_revision=rev,
        source_before={'sha':'s'},source_after={'sha':'s'},reference_arrays_decoded=False,
        reference_masks_decoded=False,source_calibration_decoded=False,reserved_acquired=0,sequences=[])
    manifest=dict(schema='world_reward.form_hoi_external_prediction_set.v1',dataset_revision=cfg['dataset_revision'],
        split='development',ground_truth_used=False,private_truth_read=False,original_frame_indices=list(range(96)),producer_revision=prod,sequences=[])
    for i in range(4):
        sid=f'seq{i}';p=dev/sid/'inputs/input.json';p.parent.mkdir(parents=True)
        video_pin=dict(bytes=10,sha256='e'*64);input_pin=dict(bytes=20,sha256='f'*64)
        native=dict(schema='world_reward.external_rgb_input.v1',sequence_id=sid,dataset='nvidia/form-hoi',
            dataset_revision=cfg['dataset_revision'],split='development',video=str(p.parent/'rgb.mp4'),video_pin=video_pin,
            total=96,full_source_frames=100,camera=cfg['camera'],height=1152,width=1536,fps=30,original_frame_indices=list(range(96)),
            object_prompt='container',action='lift container',inference_ready=True,reference_inputs_present=False)
        p.write_text(json.dumps(native));report['sequences'].append(dict(sequence_id=sid,receipt=dict(bytes=1,sha256='0'*64)))
        records[dev/sid/'receipt.json']=dict(sequence_id=sid,producer_revision=rev,inference_ready=True,
            reference_arrays_decoded=False,reference_masks_decoded=False,source_calibration_decoded=False,
            alias_guard=dict(qualified=True,exact_content_duplicate_check_passed=True),input=input_pin,
            retained=[dict(role='rgb',file='inputs/rgb.mp4',**video_pin)])
        row=dict(sequence_id=sid,input=dict(path=str(p),pin=input_pin),variants={})
        for name in ('A','B'):
            base=pred/sid/name;base.mkdir(parents=True)
            geometry_pin=dict(bytes=1,sha256='1'*64);r=base/'report.json';g=base/'eval_geometry.npz'
            receipt=dict(status='complete',sequence_id=sid,producer_revision=prod,ground_truth_used=False,
                private_truth_read=False,original_frame_indices=list(range(96)),artifacts={'eval_geometry.npz':geometry_pin},
                full_predictions_sealed_before_evaluation=True,reference_inputs_mounted=False,oracle_modes=[])
            r.write_text(json.dumps(receipt))
            row['variants'][name]=dict(geometry=dict(path=str(g),pin=geometry_pin),report=dict(path=str(r),pin=dict(bytes=1,sha256='2'*64)))
        manifest['sequences'].append(row)
    records[dev/'report.json']=report;records[pred/'manifest.json']=manifest
    monkeypatch.setattr(e,'pinned',lambda path,pin,maximum:records[Path(path)])
    monkeypatch.setattr(e,'artifact_record',lambda record,maximum=1 << 30:Path(record['path']))
    code=Path(__file__).resolve().parents[1]
    return cfg,cohort,code,records,manifest,report


def test_allfour_pair_gate_produces_no_truth_and_reports_exact_guard_limit(tmp_path,monkeypatch):
    cfg,cohort,code,records,manifest,report=frozen_fixture(tmp_path,monkeypatch)
    rows,proof=e.freeze_gate(cfg,cohort,code)
    assert len(rows)==4 and all(set(r['variants'])=={'A','B'} for r in rows)
    assert proof['all_four_paired_predictions_sealed'] and not proof['reference_values_read']
    assert proof['exact_content_guard_only'] and not proof['near_alias_absence_verified']


@pytest.mark.parametrize('bad',['last_missing_B','last_uses_truth','bad_alias','three_only','private_mounted','not_sealed'])
def test_any_incomplete_or_gt_read_last_record_blocks_entire_eval(tmp_path,monkeypatch,bad):
    cfg,cohort,code,records,manifest,report=frozen_fixture(tmp_path,monkeypatch)
    last=manifest['sequences'][-1]
    if bad=='last_missing_B':del last['variants']['B']
    elif bad=='three_only':manifest['sequences'].pop()
    elif bad=='bad_alias':records[e.DEV_DATA/('a'*40)/'seq3/receipt.json']['alias_guard']['qualified']=False
    else:
        p=Path(last['variants']['B']['report']['path']);receipt=json.loads(p.read_bytes())
        key={'last_uses_truth':'private_truth_read','private_mounted':'reference_inputs_mounted','not_sealed':'full_predictions_sealed_before_evaluation'}[bad]
        receipt[key]=False if bad=='not_sealed' else True;p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):e.freeze_gate(cfg,cohort,code)


def test_actual_runtime_cid_marker_not_mistaken_for_prior_result(tmp_path):
    assert e.require_fresh_output(tmp_path)==tmp_path
    marker=tmp_path/'.container.cid';marker.write_text('a'*64+'\n')
    assert e.require_fresh_output(tmp_path)==tmp_path
    (tmp_path/'report.json').write_text('{}')
    with pytest.raises(ValueError,match='Fresh output'):e.require_fresh_output(tmp_path)
    (tmp_path/'report.json').unlink();marker.write_text('notacontainer')
    with pytest.raises(ValueError,match='docker CID'):e.require_fresh_output(tmp_path)


def official_runtime_fixture(tmp_path,monkeypatch,*,kernels=(object(),object())):
    """Tiny verified-import contract only; no kit download, JIT or GT data."""
    kit=tmp_path/'vendor/v2d_submission_kit/v2dlb';kit.mkdir(parents=True)
    modules=[]
    for name in ('mhr_metrics','mhr_submission','mesh_common'):
        path=kit/(name+'.py');path.write_text('# tiny import contract fixture\n')
        modules.append(SimpleNamespace(__file__=str(path)))
    modules[0].MHR_TABLE3_BODY_JOINT_INDICES=e.JOINTS
    modules[1]._PENETRATION_KERNELS=kernels
    by_name=dict(zip(('v2dlb.mhr_metrics','v2dlb.mhr_submission','v2dlb.mesh_common'),modules))
    monkeypatch.setattr(e,'identity',lambda path,maximum:e.OFFICIAL['v2dlb/'+Path(path).name])
    monkeypatch.setattr(e.importlib,'import_module',lambda name:by_name[name])
    monkeypatch.setitem(sys.modules,'v2dlb',SimpleNamespace(__path__=[str(kit)]))
    state=dict(threads=1,requests=[])
    def set_threads(count):state['threads']=count;state['requests'].append(count)
    numba=SimpleNamespace(set_num_threads=set_threads,get_num_threads=lambda:state['threads'])
    monkeypatch.setitem(sys.modules,'numba',numba)
    return modules,state,numba


def test_public_operator_runtime_uses_qualified_kernels_and_four_threads(tmp_path,monkeypatch):
    modules,state,_=official_runtime_fixture(tmp_path,monkeypatch)
    kernels=modules[1]._PENETRATION_KERNELS
    assert e.official_modules(tmp_path)==modules
    assert modules[1]._PENETRATION_KERNELS is kernels
    assert state==dict(threads=4,requests=[4])


@pytest.mark.parametrize('missing_attribute',[False,True])
def test_public_operator_no_silent_unqualified_penetration_fallback(tmp_path,monkeypatch,missing_attribute):
    modules,state,_=official_runtime_fixture(tmp_path,monkeypatch,kernels=None)
    if missing_attribute:del modules[1]._PENETRATION_KERNELS
    with pytest.raises(ValueError,match='kernels must be active before reference'):
        e.official_modules(tmp_path)
    assert state['requests']==[]


def test_public_operator_missing_numba_fails_before_reference_access(tmp_path,monkeypatch):
    _,state,_=official_runtime_fixture(tmp_path,monkeypatch)
    monkeypatch.setitem(sys.modules,'numba',None)
    with pytest.raises(ValueError,match='Numba runtime required before reference'):
        e.official_modules(tmp_path)
    assert state['requests']==[]


def test_public_operator_thread_runtime_must_observe_requested_four(tmp_path,monkeypatch):
    _,state,numba=official_runtime_fixture(tmp_path,monkeypatch)
    numba.get_num_threads=lambda:1
    with pytest.raises(ValueError,match='exactly four CPU threads'):
        e.official_modules(tmp_path)
    assert state['requests']==[4]
