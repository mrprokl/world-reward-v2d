"""Own synthetic CPU evidence only; never checkpoint/inference/private media."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO=Path.cwd()
STAGED=Path('/tmp/wr-h98-eval-staging/infra/root5_rgb_evaluate.py')
MODULE=Path(__file__).parents[1]/'infra/root5_rgb_evaluate.py'
if not MODULE.exists():MODULE=STAGED


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'infra'));monkeypatch.syspath_prepend(str(REPO/'src'));monkeypatch.syspath_prepend(str(MODULE.parent))
    spec=importlib.util.spec_from_file_location('h98_quality_test',MODULE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def frozen(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));path.chmod(0o444)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def quality_pins(gate,tmp_path):
    p=tmp_path/'public.json';frozen(p,{'explicit_public_only':True})
    pins=dict(schema='world-reward-root5-quality-pins-v1',public_pins_sha256=gate.sha256(p),
        fit_revision='a'*40,fit_report_sha256='b'*64,fit_report_bytes=100,fit_script_sha256='c'*64,
        replay_revision='d'*40,replay_report_sha256='e'*64,replay_report_bytes=100,replay_script_sha256='f'*64)
    q=tmp_path/'quality.json';frozen(q,pins);return p,q,pins


@pytest.mark.parametrize('fault',['none','missing','extra','sha','revision','size','bool','schema','public','writable'])
def test_explicit_pins_not_discovered_or_repaired(gate,tmp_path,fault):
    p,q,pins=quality_pins(gate,tmp_path)
    if fault=='none':assert gate.load_quality_pins(q,p)[0]==pins;return
    if fault=='missing':pins.pop('replay_report_sha256')
    elif fault=='extra':pins['private_labels']='forbidden'
    elif fault=='sha':pins['fit_script_sha256']='B'*64
    elif fault=='revision':pins['fit_revision']='main'
    elif fault=='size':pins['replay_report_bytes']=0
    elif fault=='bool':pins['fit_report_bytes']=True
    elif fault=='schema':pins['schema']='other'
    elif fault=='public':pins['public_pins_sha256']='1'*64
    q.chmod(0o644);q.write_text(json.dumps(pins));q.chmod(0o444)
    if fault=='writable':q.chmod(0o644)
    with pytest.raises(ValueError):gate.load_quality_pins(q,p)


def truth(gate,monkeypatch):
    hf=np.tile([0,1,2],(36874,1)).astype(np.int32);faces=hf.astype(np.int64)
    monkeypatch.setattr(gate,'HUMAN_FACES_SHA',hashlib.sha256(hf.tobytes()).hexdigest())
    ids=np.full((768,1024),-1,np.int64);ids.flat[:64]=0;ids.flat[64:128]=36874
    depth=np.full(ids.shape,np.nan,np.float32);depth[ids>=0]=2.
    data=dict(human_vertices_camera_m=np.tile([0.,0.,2.],(18439,1)),human_faces=hf,
        object_vertices_camera_m=np.tile([0.,0.,2.],(194,1)),object_faces=np.tile([0,1,2],(384,1)).astype(np.int64),
        camera_K=np.asarray(gate.COHORT.fixed_K,np.float64),scene_depth_m=depth,visible_face_indices=ids,
        clip_index=np.array(0,np.int64),frame_index=np.array(0,np.int64))
    return data,faces,dict(clip_index=0,frame_index=0)


@pytest.mark.parametrize('fault',['none','face64','facebytes','baseline32','K','ids','depth','missing','frames','object'])
def test_truth_native_i32_exact_not_cast_or_aligned(gate,monkeypatch,fault):
    data,faces,record=truth(gate,monkeypatch)
    if fault=='none':gate.validate_truth(data,record,faces);return
    if fault=='face64':data['human_faces']=data['human_faces'].astype(np.int64)
    elif fault=='facebytes':data['human_faces'][0,0]=3;faces[0,0]=3
    elif fault=='baseline32':faces=faces.astype(np.int32)
    elif fault=='K':data['camera_K'][0,0]=1279.
    elif fault=='ids':data['visible_face_indices'].flat[0]=40000
    elif fault=='depth':data['scene_depth_m'].flat[1]=0.
    elif fault=='missing':data.pop('clip_index')
    elif fault=='frames':data['frame_index']=np.array(1,np.int64)
    else:data['object_faces'][0,0]=194
    with pytest.raises(ValueError):gate.validate_truth(data,record,faces)


def test_manufacturing_full249_limits_and_clip_identity(gate):
    rig=dict(controls=np.zeros((15,204),np.float32),shape45=np.zeros((15,45),np.float32),parameter_limits=np.tile([-1.,1.],(249,1)))
    report=dict(parameter_names=[f'own_{i}'for i in range(249)])
    gate.validate_rig(rig,report)
    rig['parameter_limits'][207]=[.1,1.]
    with pytest.raises(ValueError):gate.validate_rig(rig,report)
    rig['parameter_limits'][207]=[-1,1];rig['shape45'][1,0]=.1
    with pytest.raises(ValueError):gate.validate_rig(rig,report)


def safeguard(before,after):
    change=after-before;clip=change.reshape(3,5).mean(1)
    return dict(baseline_iou=before.tolist(),candidate_iou=after.tolist(),per_frame_delta=change.tolist(),per_clip_mean_delta=clip.tolist(),
        worst_frame_delta=float(change.min()),worst_clip_mean_delta=float(clip.min()),limit=-.01,
        safeguard_pass=bool(np.all(change>=-.01)and np.all(clip>=-.01)),used_for_state_selection=False,
        candidate_export_blocked_by_safeguard=False,accuracy_verified=False)


def test_safeguard_one_bad_frame_rejects_without_frame_exclusion(gate):
    before=np.full(15,.8);after=before.copy();after[4]-=.02
    value=safeguard(before,after);assert not gate.safeguard_decision(value)['safeguard_pass']
    rows=[]
    for i in range(15):
        rows.append(dict(clip_index=i//5,frame_index=i%5,**{m:dict(human_pve_cm=v,per_hand_relative_vector_cm=[1.,1.])for m,v in(('raw',11.),('baseline',10.),('fitted',9.))}))
    clips,decision=gate.aggregate(rows);assert len(clips)==3 and decision['synthetic_root_refit_hypothesis_supported']
    value['used_for_state_selection']=True
    with pytest.raises(ValueError):gate.safeguard_decision(value)


def test_metrics_measure_vector_of_centroids_not_mean_norm(gate):
    target=np.tile([0.,0.,2.],(18439,1));predicted=target.copy();left=np.arange(18439)<100;right=np.arange(18439)>=18339
    predicted[left,0]=np.r_[np.ones(50),-np.ones(50)]
    result=gate.frame_metrics(dict(raw=predicted,baseline=predicted,fitted=predicted),target,np.tile([0.,0.,2.],(32,1)),np.array([0.,0.,2.]),left,right)
    assert result['baseline']['per_hand_relative_vector_cm'][0]==0.
    assert result['baseline']['per_hand_vertex_pve_cm'][0]==100.


def test_visible_object_median_all_pixels_original_half_pixel_rays(gate):
    depth=np.array([[1.,2.,3.]],np.float32);ids=np.array([[0,1,1]],np.int64)
    median,count=gate.visible_object_median(dict(scene_depth_m=depth,visible_face_indices=ids,human_faces=np.array([[0,1,2]]),camera_K=np.eye(3)))
    assert count==2 and np.array_equal(median,[5.25,1.25,2.5])


def test_public_audit_failure_cannot_open_private(gate,tmp_path,monkeypatch):
    p,q,_=quality_pins(gate,tmp_path);seen=[]
    monkeypatch.setattr(gate.public,'load_pins',lambda _: {})
    monkeypatch.setattr(gate,'public_fit',lambda *_:(_ for _ in()).throw(ValueError('public rejected')))
    monkeypatch.setattr(gate,'private_truth',lambda *_:seen.append('forbidden'))
    with pytest.raises(ValueError):gate.run(tmp_path,{},p,q)
    assert seen==[] and not(tmp_path/gate.BASE/'eval_private').exists()


def replay_fixture(gate,tmp_path,monkeypatch):
    p,q,pins=quality_pins(gate,tmp_path);records=[dict(file=gate.COHORT.frame_name(i))for i in range(15)]
    producer=dict(body_model={'verified':'synthetic'},native_head_source={'sha256':'1'*64});lineage={'frozen_files':[]};traces=[]
    frozen_fit=frozen(tmp_path/gate.public.OUT/'report.json',producer);pins['fit_report_sha256']=frozen_fit['sha256']
    source=tmp_path/'source.py';source.write_text('synthetic source only');source.chmod(0o444);pins['replay_script_sha256']=gate.sha256(source)
    monkeypatch.setattr(gate.public,'helper_identities',lambda fit_source:{'fit':gate.sha256(fit_source)})
    report=dict(stage=gate.REPLAY_STAGE,status='pass',phase='complete',operation='replay',producer_revision=pins['replay_revision'],
        image_id=gate.IMAGE,script_sha256=pins['replay_script_sha256'],network='none',device='cuda',frames=15,all_cases_retained=True,
        predictions_native_replay_verified=True,source_inputs_assets_rehashed=True,all_training_traces_verified=True,all_native_best_replays_verified=True,
        optimizer_updates=0,additional_training_forwards=0,native_backwards=0,private_truth_read=False,ground_truth_used=False,
        challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],shared_identity_constant=True,body_hands_fixed=True,camera_Z_fixed=True,
        intrinsics_fixed=True,root_bounds_only=True,dense_fixed_remainder_bounds_checked=False,lineage=lineage,accuracy_verified=False,quality_verified=False,
        adoption_authorized=False,deterministic_algorithms=True,warn_only=False,TF32=False,CUBLAS_WORKSPACE_CONFIG=':4096:8',seed=0,threads=4,
        body_model=producer['body_model'],native_head_source=producer['native_head_source'],trace_checks=traces,fit_report=frozen_fit,
        source_helpers={'fit':pins['replay_script_sha256']},torch_version='2.5.1+cu124',CUDA_version='12.4',
        counters=dict(final_native_heads_attempted=15,final_native_heads_returned=15,final_native_heads_validated=15,
        objective_native_heads_attempted=0,objective_native_heads_returned=0,objective_native_heads_validated=0,
        optimizer_backwards_attempted=0,optimizer_backwards_returned=0,adam_updates=0,proxy_rasters_attempted=0,proxy_rasters_completed=0,
        safeguard_rasters_attempted=0,safeguard_rasters_completed=0),
        records=[dict(file=r['file'],native_heads_attempted=1,native_heads_returned=1,native_heads_validated=1,
            replay_parity={k:0. for k in('vertices_m','keypoints_m','joints_m','controls','rotations')})for r in records])
    path=tmp_path/gate.REPLAY_OUT/'report.json'
    def save():
        if path.exists():path.chmod(0o644)
        identity=frozen(path,report);pins['replay_report_sha256']=identity['sha256'];pins['replay_report_bytes']=identity['bytes']
    save();return pins,records,producer,lineage,traces,source,report,save


@pytest.mark.parametrize('fault',['none','source','returned','partial','error','optimizer','truth','geometry','fitreceipt','retainederror'])
def test_exact_selected_native_replay_before_quality(gate,tmp_path,monkeypatch,fault):
    pins,records,producer,lineage,traces,source,row,save=replay_fixture(gate,tmp_path,monkeypatch)
    if fault=='source':row['source_helpers']={'fit':'0'*64}
    elif fault=='returned':row['counters']['final_native_heads_returned']=14
    elif fault=='partial':row['records'].pop()
    elif fault=='error':row['status']='fail'
    elif fault=='optimizer':row['optimizer_updates']=1
    elif fault=='truth':row['ground_truth_used']=True
    elif fault=='geometry':row['records'][0]['replay_parity']['vertices_m']=.00002
    elif fault=='fitreceipt':row['fit_report']['sha256']='0'*64
    elif fault=='retainederror':row['error']='prior failure'
    save()
    if fault=='none':assert gate.audit_replay(tmp_path,pins,records,producer,lineage,traces,source)['bytes']>0
    else:
        with pytest.raises(ValueError):gate.audit_replay(tmp_path,pins,records,producer,lineage,traces,source)


def test_imported_metrics_are_unchanged_aliases_and_no_old_stage_execution(gate):
    for name in('frame_metrics','aggregate','visible_object_median','heldout_diagnostics'):assert getattr(gate,name)is getattr(gate.metrics,name)
    tree=ast.parse(Path(gate.__file__).read_text());calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
    assert 'metrics.public_fit'not in calls and 'metrics.run'not in calls
    assert 'public.validate_schedule'in calls and 'audit_replay'in calls
    assert not any('torch'in(n.module or'')for n in ast.walk(tree)if isinstance(n,ast.ImportFrom))


def test_wrapper_zeroargs_required_mounts_dedup_and_error_propagation(gate):
    wrapper=Path(gate.__file__).with_name('run_root5_rgb_evaluate.sh');text=wrapper.read_text()
    subprocess.run(['bash','-n',str(wrapper)],check=True)
    assert '(( $# == 0 ))'in text and 'SOURCE_LIST="$(python3'in text and '< <('not in text
    assert 'paths=set()'in text and 'if False'not in text and '--gpus'not in text
    for name in('weights/grounding_dino','weights/sam2','results/image-grounding.json','baseline_v1/report.json','baseline_v2','root_replay_v1','eval_private'):
        assert name in text
    assert 'src=$SOURCE,dst=$SOURCE,readonly'in text and 'WR_CODE=$CODE'in text
    result=subprocess.run(['bash',str(wrapper),'unexpected'],capture_output=True)
    assert result.returncode==2


def test_source_selection_deduplicates_same_fit_replay_and_fails_bad_config(gate,tmp_path):
    wrapper=Path(gate.__file__).with_name('run_root5_rgb_evaluate.sh').read_text();source=wrapper.split("<<'PYSAFE'\n",1)[1].split('\nPYSAFE',1)[0]
    root=tmp_path/'root';root.mkdir();code=tmp_path/'code';code.mkdir();configs=code/'configs';configs.mkdir()
    public=dict(producer_revision=dict(masks='8084688a4d84bbad9ba8c0580e4d1b2803745511',baseline='feae71ea16a1d942f08e95ccafc131b6467dffb9',dwpose='feae71ea16a1d942f08e95ccafc131b6467dffb9'))
    pp=configs/'p.json';qp=configs/'q.json';frozen(pp,public);frozen(qp,dict(fit_revision='a'*40,replay_revision='a'*40))
    files=[root/'jobs'/('a'*40)/'run_root5_rgb_fit/code/infra/root5_rgb_fit.py']
    files+=[root/'jobs'/gate.RENDER_REVISION/'run_root5_rgb_prepare/code/infra'/n for n in('root5_rgb_render.py','identity_rgb_render.py','joint_rgb_render.py')]
    files+=[root/'jobs'/public['producer_revision']['masks']/'run_root5_rgb_observe/code/infra/root5_rgb_observe.py']
    for p in files:p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic');p.chmod(0o444)
    (root/'jobs'/public['producer_revision']['baseline']/'run_root5_rgb_observe/code/infra').mkdir(parents=True)
    argv=[sys.executable,'-c',source,str(root),str(code),str(root/gate.OUT),str(pp),str(qp)]
    result=subprocess.run(argv,capture_output=True,text=True,check=True);paths=result.stdout.splitlines()
    assert len(paths)==len(set(paths))==6
    qp.chmod(0o644);qp.write_text('{}');qp.chmod(0o444)
    assert subprocess.run(argv,capture_output=True).returncode!=0 and not(root/gate.OUT).exists()


def test_all15_scored_once_even_when_public_safeguard_failed(gate,tmp_path,monkeypatch):
    p,q,_=quality_pins(gate,tmp_path);data,faces,_=truth(gate,monkeypatch)
    private=tmp_path/'own_test_truth';private.mkdir();records=[];pairs=[];candidates=[];cases=[];observations=[]
    target=data['human_vertices_camera_m'];baseline=target.copy();baseline[:,0]+=.01
    left=np.arange(18439)<100;right=np.arange(18439)>=18339
    for i in range(15):
        record=dict(file=gate.COHORT.frame_name(i),clip_index=i//5,frame_index=i%5,sha256='a'*64);records.append(record)
        value={k:v.copy()for k,v in data.items()};value['clip_index']=np.array(i//5,np.int64);value['frame_index']=np.array(i%5,np.int64)
        path=private/(Path(record['file']).stem+'.npz')
        with path.open('xb')as stream:np.savez_compressed(stream,**value)
        path.chmod(0o444)
        cases.append(dict(file=record['file'],rgb_sha256=record['sha256'],clip_index=i//5,frame_index=i%5,truth_sha256=gate.sha256(path),
            human_faces_sha256=gate.HUMAN_FACES_SHA,object_faces_sha256=hashlib.sha256(value['object_faces'].tobytes()).hexdigest()))
        kp=np.tile([0.,0.,2.],(308,1)).astype(np.float32)
        pairs.append(dict(raw_vertices_camera_m=baseline,shared_vertices_camera_m=baseline,human_faces=faces,hand_mask_left=left,hand_mask_right=right,
            shared_keypoints_camera_m=kp,camera_K=np.asarray(gate.COHORT.fixed_K)))
        candidates.append(dict(vertices_camera_m=target,keypoints_camera_m=kp))
        observations.append(dict(keypoints=np.zeros((1,133,2)),scores=np.zeros((1,133))))
    proxies=[dict(object_points_camera_m=np.tile([0.,0.,2.],(32,1)))for _ in records]
    guard=safeguard(np.full(15,.8),np.full(15,.78));audit=[]
    monkeypatch.setattr(gate.public,'load_pins',lambda _: {})
    monkeypatch.setattr(gate,'source_bindings',lambda *_:{})
    def public_fit(*_):
        audit.append('fullpublic/replay');return records,pairs,candidates,proxies,observations,{'lineage':{}},[],guard
    def private_truth(*_):
        assert audit==['fullpublic/replay'];audit.append('private');return private,cases
    monkeypatch.setattr(gate,'public_fit',public_fit);monkeypatch.setattr(gate,'private_truth',private_truth)
    monkeypatch.setattr(gate.public,'public_predictions',lambda *_:(None,None,None,None,{}))
    report={};gate.run(tmp_path,report,p,q)
    assert report['status']=='pass'and report['all_frames_scored']and len(report['frame_metrics'])==15
    assert report['decision']['gates']['median_camera_pve_gain_5pct']
    assert not report['decision']['gates']['no_automatic_human_silhouette_degradation_over_1pp']
    assert not report['decision']['synthetic_root_refit_hypothesis_supported']
    assert report['decision']['adoption_authorized']is False


def test_reusable_metrics_preserve_exact_seven_historical_AST_and_constants(gate):
    old_path=REPO/'infra/keypoint_rgb_evaluate.py';old=ast.parse(old_path.read_text());new=ast.parse(Path(gate.metrics.__file__).read_text())
    names=('require_fields','regular','floating','visible_object_median','frame_metrics','aggregate','heldout_diagnostics')
    a={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)};b={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
    assert set(b)==set(names)
    for name in names:assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False)
    assert gate.sha256(old_path)==gate.metrics.LEGACY_SOURCE_SHA
    spec=importlib.util.spec_from_file_location('unchanged_historical_metrics',old_path);legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    for name in('CLIPS','FRAMES','VERTICES','WIDTH','HEIGHT','TRUTH_KEYS'):assert getattr(legacy,name)==getattr(gate.metrics,name)
    target=np.tile([0.,0.,2.],(18439,1));left=np.arange(18439)<100;right=np.arange(18439)>=18339
    values={m:target+np.array([offset,0.,0.])for m,offset in(('raw',.1),('baseline',.08),('fitted',.07))}
    args=(values,target,np.tile([0.,0.,2.],(32,1)),np.array([0.,0.,2.]),left,right)
    assert legacy.frame_metrics(*args)==gate.frame_metrics(*args)


def test_complete_public_fit_trace_fixture_before_any_private_access(gate,tmp_path,monkeypatch):
    public=gate.public;records=[];raw=[];pairs=[];observations=[];candidates=[];rows=[]
    coco=np.array(gate.metrics.policy.TRAIN_COCO);indices=np.array(gate.metrics.policy.TRAIN_MHR)
    xy=np.c_[np.linspace(300.,600.,10),np.linspace(200.,500.,10)];K=np.asarray(gate.COHORT.fixed_K)
    points=np.c_[((xy[:,0]-K[0,2])/K[0,0])*2,((xy[:,1]-K[1,2])/K[1,1])*2,np.full(10,2.)].astype(np.float32)
    J=np.zeros((20,5));J[::2,0]=192;J[1::2,1]=192
    J[:,2:]=np.random.default_rng(0).normal(size=(20,3))*50
    evidence=public.jacobian_evidence(J,10);assert evidence['minimum_maximum_ratio']>=1e-5
    for i in range(15):
        record=dict(file=gate.COHORT.frame_name(i),clip_index=i//5,frame_index=i%5,sha256='a'*64);records.append(record)
        observation=dict(keypoints=np.zeros((1,133,2)),scores=np.zeros((1,133)));observation['keypoints'][0,coco]=xy;observation['scores'][0,coco]=1
        observations.append(observation);kp=np.tile([0.,0.,2.],(308,1)).astype(np.float32);kp[indices]=points
        pairs.append(dict(shared_keypoints_camera_m=kp,camera_K=K));raw.append({})
        candidate=dict(latent=np.zeros(5,np.float32),physical_delta=np.zeros(5,np.float32),global_rot=np.zeros(3,np.float32));candidates.append(candidate)
        row=dict(file=record['file'],native_heads_attempted=61,native_heads_returned=61,native_heads_validated=61,
            training_validity=[True]*10,training_MHR_indices=indices.tolist(),training_points=10,adam_updates=59,heldout_used_for_fit=False,
            observation_compute_dtype='float32',heldout_COCO_indices=list(gate.metrics.policy.HELDOUT_COCO),
            evaluated_latents=np.zeros((60,5)).tolist(),evaluated_physical_deltas=np.zeros((60,5)).tolist(),
            evaluated_projected_points=np.tile(xy,(60,1,1)).tolist(),evaluated_losses=np.zeros(60).tolist(),
            update_gradients=np.zeros((59,5)).tolist(),initial_observation_jacobian_values=J.tolist(),
            best_evaluated_index=0,selected_total_objective=0.,selected_physical_delta=np.zeros(5).tolist(),
            initial_observation_jacobian=evidence,initial_parity={k:0. for k in('vertices_m','keypoints_m','joints_m','controls','rotations')},
            analytic_XY_jacobian=dict(rtol=1e-5,atol=1e-6,latent_is_zero=True,additional_native_calls=0,maximum_absolute_error=0.))
        rows.append(row)
    p,q,pins=quality_pins(gate,tmp_path);failed={'sha256':'1'*64}
    reference=tmp_path/'reference.pt';reference.write_bytes(b'public test reference');reference.chmod(0o644)
    lineage={'frozen_files':[dict(path=str(reference),sha256=gate.sha256(reference),bytes=reference.stat().st_size)]}
    code=tmp_path/'code.py';code.write_text('synthetic source');code.chmod(0o444);source=gate.regular(code)
    pins['fit_script_sha256']=source['sha256'];bindings={'fit':source,'replay':source}
    monkeypatch.setattr(gate,'source_bindings',lambda *_:bindings);monkeypatch.setattr(public,'helper_identities',lambda **_: {'source':'frozen'})
    monkeypatch.setattr(public,'previous_fit_failure',lambda _:failed);monkeypatch.setattr(public,'public_predictions',lambda *_:(records,raw,pairs,observations,lineage))
    monkeypatch.setattr(public,'frozen_candidates',lambda *_:candidates);monkeypatch.setattr(public,'frozen_proxies',lambda *_:[{}for _ in range(15)])
    counters=dict(objective_native_heads_attempted=900,objective_native_heads_returned=900,objective_native_heads_validated=900,
        final_native_heads_attempted=15,final_native_heads_returned=15,final_native_heads_validated=15,optimizer_backwards_attempted=885,
        optimizer_backwards_returned=885,adam_updates=885,proxy_rasters_attempted=15,proxy_rasters_completed=15,safeguard_rasters_attempted=15,
        safeguard_rasters_completed=15,initial_jacobian_rows_attempted=300,initial_jacobian_rows_completed=300)
    producer=dict(stage='public_root5_rgb_native_fixed_depth_refit',status='pass',phase='complete',operation='fit',producer_revision=pins['fit_revision'],
        script_sha256=pins['fit_script_sha256'],source_helpers={'source':'frozen'},lineage=lineage,image_id=gate.IMAGE,network='none',device='cuda',budget_seconds=300,
        optimizer=gate.OPTIMIZER,frames=15,all_cases_retained=True,private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,
        oracle_modes=[],shared_identity_constant=True,body_hands_fixed=True,camera_Z_fixed=True,intrinsics_fixed=True,root_bounds_only=True,
        dense_fixed_remainder_bounds_checked=False,inputs_unchanged=True,native_arrays_verified=True,all_candidates_frozen=True,object_proxies_frozen_before_fit=True,
        initial_jacobian_includes_priors=False,training_COCO_indices=coco.tolist(),training_MHR_indices=indices.tolist(),accuracy_verified=False,quality_verified=False,
        adoption_authorized=False,source_inputs_assets_rehashed=True,optimizer_updates=885,native_backwards=885,torch_version='2.5.1+cu124',CUDA_version='12.4',
        previous_failed_fit=failed,previous_failure_rewritten=False,pytorch3d_version='0.7.9',seed=0,TF32=False,CUBLAS_WORKSPACE_CONFIG=':4096:8',
        deterministic_algorithms=True,warn_only=False,threads=4,counters=counters,candidate_outputs=[],proxy_outputs=[],fit_records=rows,
        root_bounds=[[-1.,1.]]*3,silhouette_safeguard=safeguard(np.full(15,.8),np.full(15,.8)))
    path=tmp_path/public.OUT/'report.json';(path.parent/'candidates').mkdir(parents=True);(path.parent/'proxies').mkdir()
    def save():
        if path.exists():path.chmod(0o644)
        receipt=frozen(path,producer);pins['fit_report_sha256']=receipt['sha256'];pins['fit_report_bytes']=receipt['bytes']
    native_checks=[]
    def replay(*args):native_checks.append(args[5]);return source
    monkeypatch.setattr(gate,'audit_replay',replay);save()
    result=gate.public_fit(tmp_path,{},pins)
    assert len(result[0])==15 and len(native_checks[0])==15
    assert all(item['path']!=str(reference)for item in result[6])
    assert reference.stat().st_mode&0o222
    assert all(row['manual_Adam_and_public_objectives_verified']for row in native_checks[0])
    producer['native_backwards']=884;save()
    with pytest.raises(ValueError):gate.public_fit(tmp_path,{},pins)
    assert not(tmp_path/gate.BASE/'eval_private').exists()


def test_native_public_reference_policy_not_applied_to_frozen_candidates(gate,tmp_path):
    reference=tmp_path/'native_reference.pt';reference.write_bytes(b'synthetic reference');reference.chmod(0o644)
    assert gate.public.native.regular(reference)['bytes']>0
    with pytest.raises(ValueError):gate.regular(reference)
    candidate=tmp_path/'candidate.npz';candidate.write_bytes(b'frozen predicted result');candidate.chmod(0o444)
    assert gate.regular(candidate)['bytes']>0
    candidate.chmod(0o644)
    with pytest.raises(ValueError):gate.regular(candidate)


def test_paired_points_are_not_native_zero_points_for_strict_jacobian_gate(gate):
    native=np.array([[.1,.2,2.]],np.float32);pair=native.copy();pair[0,2]+=np.float32(.000009)
    K=np.asarray(gate.COHORT.fixed_K);J=np.zeros((2,5));J[0,0]=.3*K[0,0]/native[0,2];J[1,1]=.3*K[1,1]/native[0,2]
    assert np.linalg.norm(native-pair)<1e-5
    result=gate.paired_xy_diagnostic(J,pair,K)
    assert result['maximum_absolute_error']>1e-6 and result['used_as_gate']is False
    assert result['native_zero_points_are_identical_to_pair_verified']is False
