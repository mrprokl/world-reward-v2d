"""Data-free public-only conversion contracts; no Torch, truth or GPU needed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def convert(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/'infra'; monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location('test_public_hand_convert',infra/'hand_synthetic_convert.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def receipt(convert):
    records = [{'image_sha256': str(i)*64, 'mask_sha256': 'f'*64} for i in range(6)]
    hashes = {'input_manifest':'a'*64,'mask_report':'b'*64}
    calls = [{'case_index':i,'inference_type':mode,'image_sha256':records[i]['image_sha256'],
        'mask_sha256':records[i]['mask_sha256'],'decoded_rgb_sha256':'e'*64,'camera_intrinsics_supplied':False,
        'bbox_xyxy':[10.,20.,30.,40.],'native_forward_errors':dict.fromkeys(('vertices_m','joints_m','keypoints_m','controls'),0.)}
        for i in range(6) for mode in ('body','full')]
    r = {'stage':convert.infer.STAGE,'status':'pass','cases':6,'calls_completed':12,'actual_network_inference':True,
        'actual_native_parameter_block_forward':True,'private_truth_read':False,'challenge_inputs_used':False,
        'hand_labeled_test':False,'oracle_modes':[],'network':'none','native_control_dimension':204,'raw266_used':False,
        'rotation_source':'fresh_native_parameter_block_decode','geometry_units':'metres',
        'geometry_frame':'SAM_camera_x_right_y_down_z_forward','body_checkpoint_sha256':convert.infer.BODY_SHA,
        'prompt_mode':'automatic_binary_person_mask_and_derived_bbox_no_fallback','cam_int':'None_RGB_size_FOV',
        'script_sha256':convert.sha256(Path(convert.infer.__file__)),
        'body_helper_sha256':convert.sha256(Path(convert.infer.body.__file__)),'public_hashes':hashes,
        'producer_revision':'c'*40,'image_id':'sha256:'+'d'*64,'calls':calls,
        'prediction_sha256':{'body':'a'*64,'full':'b'*64},
        'model':{'body_revision':convert.infer.body.BODY_REVISION,'upstream_revision':convert.infer.body.UPSTREAM_REVISION,
            'dinov3_revision':convert.infer.body.DINOV3_REVISION,'inference_source_identity':{'sha256':'a'*64,'python_files':3},
            'checkpoint_loading':{'mode':'strict_network_and_head_state_with_explicit_asset_buffer_retention','unexpected_keys':[]},
            'body_assets':{'model.ckpt':{'sha256':convert.infer.BODY_SHA,'bytes':convert.infer.BODY_BYTES}},
            'hand_indices':{'left':list(range(68,95)),'right':list(range(95,122))}}}
    return r,records,hashes


def test_exact_actual_public_producer_receipt(convert):
    convert.validate_inference(*receipt(convert))


@pytest.mark.parametrize('kind',['status','boolcases','truth','manual','raw','stalerotation','source','oldhelper',
                                'revision','rgb','mask','order','singlemode','bbox','missingbbox','nativeerror','duplicatehand','badtypehand','hashes'])
def test_bad_inference_never_accepted(convert,kind):
    r,records,hashes = receipt(convert)
    if kind=='status':r['status']='running'
    if kind=='boolcases':r['cases']=True
    if kind=='truth':r['private_truth_read']=True
    if kind=='manual':r['hand_labeled_test']=True
    if kind=='raw':r['raw266_used']=True
    if kind=='stalerotation':r['rotation_source']='cached'
    if kind=='source':r['model']['upstream_revision']='0'*40
    if kind=='oldhelper':r['body_helper_sha256']='0'*64
    if kind=='revision':r['producer_revision']='main'
    if kind=='rgb':r['calls'][1]['decoded_rgb_sha256']='0'*64
    if kind=='mask':r['calls'][1]['mask_sha256']='0'*64
    if kind=='order':r['calls'][0]['case_index']=1
    if kind=='singlemode':r['calls'][1]['inference_type']='body'
    if kind=='bbox':r['calls'][1]['bbox_xyxy']=[0.,0.,1024.,768.]
    if kind=='missingbbox':del r['calls'][0]['bbox_xyxy']
    if kind=='nativeerror':r['calls'][0]['native_forward_errors']['controls']=1e-4
    if kind=='duplicatehand':r['model']['hand_indices']['right'][0]=68
    if kind=='badtypehand':r['model']['hand_indices']['left']=[float(i) for i in range(68,95)]
    if kind=='hashes':r['public_hashes']={}
    with pytest.raises(ValueError):convert.validate_inference(r,records,hashes)


def prediction():
    x = {k:np.zeros(shape,np.float32) for k,shape in {
        'vertices_camera_m':(6,18439,3),'joints_camera_m':(6,127,3),'mhr_model_params':(6,204),
        'global_rot':(6,3),'body_pose_params':(6,133),'hand_pose_params':(6,108),'scale_params':(6,28),
        'shape_params':(6,45),'expr_params':(6,72),'pred_cam_t':(6,3),'focal_length':(6,)}.items()}
    x.update(joint_global_rotations=np.broadcast_to(np.eye(3,dtype=np.float32),(6,127,3,3)).copy(),
             frame_index=np.arange(6,dtype=np.int64),faces=np.tile([0,1,2],(36874,1)).astype(np.int32))
    x['pred_cam_t'][:,2]=2.; x['focal_length'][:]=1280.
    return x


@pytest.mark.parametrize('kind',['nan','float16','wrongshape','frames','floatfaces','facebounds','expression','rotation','cam','focal','rawextra','masked','maskedfaces','maskedframes'])
def test_prediction_coverage_finite_topology_and_fresh_block_contract(convert,kind):
    x = prediction(); convert.prediction_arrays(x)
    if kind=='nan':x['mhr_model_params'][0,68]=np.nan
    if kind=='float16':x['mhr_model_params']=x['mhr_model_params'].astype(np.float16)
    if kind=='wrongshape':x['hand_pose_params']=x['hand_pose_params'][:,:54]
    if kind=='frames':x['frame_index'][1]=0
    if kind=='floatfaces':x['faces']=x['faces'].astype(np.float64)
    if kind=='facebounds':x['faces'][0,0]=18439
    if kind=='expression':x['expr_params'][0,0]=.1
    if kind=='rotation':x['joint_global_rotations'][0,0,0,0]=-1
    if kind=='cam':x['pred_cam_t'][0,2]=0
    if kind=='focal':x['focal_length'][0]=1920
    if kind=='rawextra':x['pred_pose_raw']=np.zeros((6,266),np.float32)
    if kind=='masked':x['mhr_model_params']=np.ma.array(x['mhr_model_params'],mask=False)
    if kind=='maskedfaces':x['faces']=np.ma.array(x['faces'],mask=False)
    if kind=='maskedframes':x['frame_index']=np.ma.array(x['frame_index'],mask=False)
    with pytest.raises(ValueError):convert.prediction_arrays(x)


def converted():
    p=np.arange(6*136,dtype=np.float64).reshape(6,136)/100; p[0,0]=-0.
    return {'pose':p,'scales':np.arange(68,dtype=np.float32)/20,'shape':np.zeros(45,np.float32),
        'valid_input':np.ones(6,bool),'per_frame_vertex_error_mm':np.full(6,.5),
        'report':{'frames':6,'fitted_frames':6,'invalid_input_frames':[],
                  'vertex_error_mm':{'mean':.5,'worst_frame_mean':.5,'max':4.,'p99':2.}}}


def test_one_shared_identity_and_54_absolute_finger_columns_only(convert):
    result=converted(); controls=np.arange(6*204,dtype=np.float32).reshape(6,204)/10
    original=result['pose'].tobytes(); source=controls.tobytes()
    official,proposal=convert.make_proposals(result,controls,np.arange(68,95)[::-1],np.arange(95,122)[::-1])
    fixed=np.r_[np.arange(68),np.arange(122,136)]
    assert proposal.pose[:,fixed].tobytes()==official.pose[:,fixed].tobytes()
    assert np.array_equal(proposal.pose[:,68:122],controls[:,68:122])
    assert official.scales.shape==(68,) and official.shape.shape==(45,) and official.expression.shape==(72,)
    assert not official.expression.any() and proposal.report['adoption_performed'] is False
    assert result['pose'].tobytes()==original and controls.tobytes()==source


@pytest.mark.parametrize('kind',['badframe','residual','perframeidentity','boolvalid'])
def test_official_invalid_frames_or_identity_never_repaired(convert,kind):
    x=converted()
    if kind=='badframe':x['valid_input'][0]=False
    if kind=='residual':x['per_frame_vertex_error_mm'][0]=2.01
    if kind=='perframeidentity':x['shape']=np.zeros((6,45))
    if kind=='boolvalid':x['valid_input']=np.ones(6,np.int32)
    with pytest.raises(ValueError):convert.make_proposals(x,np.zeros((6,204)),np.arange(68,95),np.arange(95,122))


def test_reference_units_no_second_flip_or_translation_and_per_frame_gate(convert):
    v=np.full((6,18439,3),1000.,np.float64); j=np.full((6,127,3),-2.,np.float64)
    m,jm=convert.reference_geometry(v,j)
    assert np.all(m==1.) and np.all(jm==-2.)
    target=m.copy(); target[3,:,0]+=.002001
    errors,_=convert.vertex_residual_mm(m,target)
    with pytest.raises(ValueError):convert.require_fidelity(errors)
    target=m.copy(); target[0,0,0]+=1.
    errors,maximum=convert.vertex_residual_mm(m,target)
    assert maximum==1000. and convert.require_fidelity(errors)['worst_frame_mean_mm']<2.


def test_stubbed_exact_official_convert_forward_and_proposal_archive(convert,monkeypatch,tmp_path):
    """Interface wiring only: the stub is not a real-model fidelity claim."""
    r,_,hashes=receipt(convert); body=prediction(); full=prediction()
    body['vertices_camera_m'][:,0,0]=.2;full['mhr_model_params'][:,68:122]=.7
    arrays={'body':body,'full':full};output=tmp_path/'official_proposals';output.mkdir();path=output/'report.json'
    report={'public_hashes':hashes,'inference_report_sha256':'a'*64};path.write_text(json.dumps(report))
    monkeypatch.setattr(convert,'public_predictions',lambda root:(arrays,r,'a'*64,hashes))
    monkeypatch.setattr(convert.infer.body,'_source_identity',lambda root:r['model']['inference_source_identity'])
    monkeypatch.setattr(convert.infer.body,'_body_assets',lambda root:(tmp_path,r['model']['body_assets']))
    checked=[];monkeypatch.setattr(convert,'require_hash',lambda path,sha:checked.append(sha))
    class Tensor:
        def __init__(self,x):self.x=np.asarray(x).copy()
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.x
    noop=lambda *args,**kwargs:None
    torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True),float64=np.float64,
        set_num_threads=noop,use_deterministic_algorithms=noop,is_tensor=lambda x:isinstance(x,Tensor),
        tensor=lambda x,**kw:Tensor(x),backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()),cudnn=SimpleNamespace()),
        load=lambda *a,**kw:{'state_dict':{'head_pose.hand_joint_idxs_'+side:Tensor(r['model']['hand_indices'][side]) for side in ('left','right')}})
    monkeypatch.setitem(sys.modules,'torch',torch);calls=[]
    def official_convert(vertices,model,**kwargs):
        assert vertices is body['vertices_camera_m']
        assert kwargs['device']=='cuda' and kwargs['precision']=='float32' and kwargs['model_batch']==256
        calls.append('convert');return converted()
    class Reference:
        def __init__(self,model,device,*,chunk,precision):
            assert device=='cuda' and chunk==16 and precision=='float32'
            self.model=SimpleNamespace(character_torch=SimpleNamespace(mesh=SimpleNamespace(faces=Tensor(body['faces']))))
        def run(self,pose,identity):
            calls.append((pose.x.copy(),identity.x.copy()))
            vertices=body['vertices_camera_m'].astype(np.float64)*1000.
            if len(calls)==3:vertices[:,:,0]+=.1
            return Tensor(vertices),Tensor(body['joints_camera_m'])
    official=SimpleNamespace(convert=official_convert,MHR=Reference)
    monkeypatch.setattr(convert.importlib.util,'spec_from_file_location',lambda *args:SimpleNamespace(loader=SimpleNamespace(exec_module=noop)))
    monkeypatch.setattr(convert.importlib.util,'module_from_spec',lambda spec:official)
    convert.run(tmp_path,report,path)
    assert checked==[convert.REFERENCE_MODEL_SHA256,convert.CONVERTER_SHA256]
    assert len(calls)==3 and np.array_equal(calls[1][1],calls[2][1]) and calls[1][1].shape==(1,113)
    with np.load(output/'proposals.npz',allow_pickle=False) as archive:
        assert set(archive.files)=={'frame_index','baseline_pose','candidate_pose','scales','shape','expression','faces',
            'baseline_vertices_camera_m','candidate_vertices_camera_m','baseline_joints_camera_m','candidate_joints_camera_m',
            'hand_indices_left','hand_indices_right'}
        assert np.all(archive['candidate_pose'][:,68:122]==np.float32(.7))
        assert np.array_equal(archive['baseline_vertices_camera_m'],body['vertices_camera_m'])
        assert np.array_equal(archive['baseline_joints_camera_m'],body['joints_camera_m'])
    final=json.loads(path.read_text())
    assert final['status']=='pass' and final['conversion_fidelity_verified'] is True and final['shared_identity_preserved'] is True
    assert final['transfer']['adoption_performed'] is False and not final['rotations_exported']


def test_no_symlink_public_receipt_is_opened(convert,monkeypatch,tmp_path):
    base=tmp_path/'validation/hands_rgb_v1/predictions-v2'; base.mkdir(parents=True)
    private=tmp_path/'private.json'; private.write_text('must not read')
    (base/'report.json').symlink_to(private)
    monkeypatch.setattr(convert.infer,'public_inputs',lambda root:([],{}))
    monkeypatch.setattr(convert,'sha256',lambda path:pytest.fail('symlink must fail before reading bytes'))
    with pytest.raises(ValueError,match='regular'):convert.public_predictions(tmp_path)


def offline_main(convert,monkeypatch,tmp_path,reserved=True):
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE_REVISION','c'*40)
    monkeypatch.setenv('WR_IMAGE_ID','sha256:'+'d'*64)
    monkeypatch.setattr(convert.platform,'system',lambda:'Linux')
    original=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('lo')]) if str(p)=='/sys/class/net' else original(p))
    monkeypatch.setattr(sys,'argv',[convert.__file__])
    r,_,hashes=receipt(convert)
    monkeypatch.setattr(convert,'public_predictions',lambda root:({},r,'a'*64,hashes))
    path=tmp_path/'validation/hands_rgb_v1/official_proposals/report.json'
    path.parent.parent.mkdir(parents=True)
    if reserved:path.parent.mkdir();monkeypatch.setenv('WR_HAND_CONVERT_OUTPUT_RESERVED','1')
    return path


def test_parent_partial_timeout_nonce_and_frozen_worker(convert,monkeypatch,tmp_path):
    path=offline_main(convert,monkeypatch,tmp_path)
    def child(args,*,check,timeout,env):
        assert check is False and 0<timeout<=180. and len(env['WR_HAND_CONVERT_NONCE'])==64
        r=json.loads(path.read_text());r['phase']='official_shared_identity_conversion';path.write_text(json.dumps(r))
        raise subprocess.TimeoutExpired(args,timeout)
    monkeypatch.setattr(convert.subprocess,'run',child)
    with pytest.raises(subprocess.TimeoutExpired):convert.main()
    r=json.loads(path.read_text());assert r['status']=='fail' and r['phase']=='official_shared_identity_conversion'
    assert r['private_truth_read'] is False and r['adoption_performed'] is False
    before=path.read_bytes()
    with pytest.raises(FileExistsError):convert.main()
    monkeypatch.setattr(sys,'argv',[convert.__file__,'--worker'])
    with pytest.raises(RuntimeError,match='immutable'):convert.main()
    assert path.read_bytes()==before


def test_unreserved_empty_output_is_not_resume(convert,monkeypatch,tmp_path):
    path=offline_main(convert,monkeypatch,tmp_path,reserved=False);path.parent.mkdir()
    with pytest.raises(FileExistsError):convert.main()


def test_wrapper_public_mounts_scoped_chown_fidelity_not_private_quality():
    root=Path(__file__).resolve().parents[1];p=root/'infra/run_hand_synthetic_convert.sh';s=p.read_text()
    for name in ('inputs','automatic_masks','predictions-v2'):
        assert f'src=$BASE/{name},dst=$BASE/{name},readonly' in s
    assert 'src=$BASE/official_proposals,dst=$BASE/official_proposals"' in s
    assert 'src=$ROOT/validation,dst=' not in s and 'eval_private' not in s
    assert 'src=$ROOT/data' not in s and 'src=$ROOT/outputs' not in s
    assert 'weights/mhr' in s and '--memory 16g --cpus 4' in s and '--network none' in s
    assert 'chown "$(id -u scenesmith):$(id -g scenesmith)" "$BASE/official_proposals"' in s and 'chown -R' not in s
    assert s.index('mkdir "$BASE/official_proposals"')<s.index('chown ')<s.index('docker run')
    assert 'timeout --signal=TERM --kill-after=10s 190s' in s
    subprocess.run(['bash','-n',str(p)],check=True)
