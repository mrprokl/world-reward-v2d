"""Tiny procedural arrays/stdlib fakes only; no model, media or CUDA execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bridge_anchor_infer_test',ROOT/'infra/bridge_rgb_anchor_infer.py')
infer=importlib.util.module_from_spec(spec);spec.loader.exec_module(infer)


@pytest.fixture
def tiny(monkeypatch):
 monkeypatch.setattr(infer,'WIDTH',16);monkeypatch.setattr(infer,'HEIGHT',16);monkeypatch.setattr(infer,'FOCAL',20.)


def metadata_fixture():
 return [f'joint_{i}'for i in range(127)],[f'control_{i}'for i in range(249)],np.r_[-1,np.zeros(126,dtype=np.int64)],np.zeros((889,249),np.float32)


def test_actual_named_metadata_bounded_tree_without_numeric_guessing():
 names,controls,parents,matrix=metadata_fixture();before=matrix.copy()
 result=infer.named_metadata(names,controls,parents,matrix)
 assert result['joint_names']==names and result['parameter_names']==controls
 assert result['parameter_transform_sha256']==hashlib.sha256(matrix.tobytes()).hexdigest()
 assert result['hand_anatomy_accuracy_verified']is False and np.array_equal(matrix,before)


@pytest.mark.parametrize('bad',['joint_count','duplicate_name','parameter_count','bool_parent','two_roots','cycle','escape','matrix_shape','nan'])
def test_metadata_rejects_invalid_semantics(bad):
 names,controls,parents,matrix=metadata_fixture()
 if bad=='joint_count':names.pop()
 elif bad=='duplicate_name':names[1]=names[0]
 elif bad=='parameter_count':controls.pop()
 elif bad=='bool_parent':parents=parents.astype(bool)
 elif bad=='two_roots':parents[1]=-1
 elif bad=='cycle':parents[1]=2;parents[2]=1
 elif bad=='escape':parents[1]=127
 elif bad=='matrix_shape':matrix=matrix[:888]
 elif bad=='nan':matrix[1,1]=np.nan
 with pytest.raises(ValueError):infer.named_metadata(names,controls,parents,matrix)


def test_bbox_derived_only_from_real_binary_mask(tiny):
 rgb=np.zeros((16,16,3),np.uint8);mask=np.zeros((16,16),np.uint8);mask[3:8,4:10]=255
 before=mask.copy();box,prompt=infer.bbox(rgb,mask)
 assert box.tolist()==[4,3,10,8] and prompt.shape==(16,16,1)
 assert prompt.dtype==np.uint8 and set(np.unique(prompt))=={0,1} and np.array_equal(mask,before)


@pytest.mark.parametrize('bad',['empty','nonbinary','float','shape','thin'])
def test_bbox_no_fullimage_or_ideal_mask_fallback(tiny,bad):
 rgb=np.zeros((16,16,3),np.uint8);mask=np.zeros((16,16),np.uint8);mask[3:8,4:10]=255
 if bad=='empty':mask[:]=0
 elif bad=='nonbinary':mask[0,0]=1
 elif bad=='float':mask=mask.astype(float)
 elif bad=='shape':mask=mask[:15]
 elif bad=='thin':mask[:]=0;mask[0,:]=255
 with pytest.raises(ValueError):infer.bbox(rgb,mask)


def depth_fixture():
 yy,xx=np.indices((16,16));depth=np.ones((16,16),np.float32)*2
 points=np.stack(((xx+.5-8)*depth/20,(yy+.5-8)*depth/20,depth),-1).astype(np.float32)
 valid=np.ones((16,16),bool);norm=np.array([[20/16,0,.5],[0,20/16,.5],[0,0,1]],np.float32)
 person=np.zeros((16,16),bool);person[4:12,4:12]=True
 return depth,points,valid,norm,person,~person


def test_valid_native_pointmap_and_excluded_nan_unchanged(tiny):
 args=list(depth_fixture());args[2][0,0]=False;args[0][0,0]=np.nan;args[1][0,0]=np.nan
 before=[v.copy()for v in args];K,proof=infer.depth_contract(*args)
 assert proof['valid_pixels']==255 and proof['pixel_sample']=='x_plus_0.5_y_plus_0.5'
 assert K[0,0]==20 and proof['metric_scale_accuracy_verified']is False
 assert all(np.array_equal(a,b,equal_nan=True)for a,b in zip(args,before))


@pytest.mark.parametrize('bad',['valid_nan','negative_Z','range_not_Z','integer_grid','wrong_dtype','wrong_K','no_valid','mask_dtype'])
def test_depth_native_contract_rejects_without_repair(tiny,bad):
 args=list(depth_fixture())
 if bad=='valid_nan':args[1][1,1]=np.nan
 elif bad=='negative_Z':args[1][1,1,2]=-1
 elif bad=='range_not_Z':args[0][1,1]+=1
 elif bad=='integer_grid':args[1][:,:,:2]-=.5*args[0][...,None]/20
 elif bad=='wrong_dtype':args[0]=args[0].astype(np.float64)
 elif bad=='wrong_K':args[3][0,0]+=1
 elif bad=='no_valid':args[2][:]=False
 elif bad=='mask_dtype':args[4]=args[4].astype(np.uint8)
 with pytest.raises(ValueError):infer.depth_contract(*args)


class Tensor:
 def __init__(self,value):self.a=np.asarray(value)
 @property
 def dtype(self):return self.a.dtype
 @property
 def shape(self):return self.a.shape
 def float(self):return Tensor(self.a.astype(np.float32))
 def unsqueeze(self,index):return Tensor(np.expand_dims(self.a,index))
 def squeeze(self,index):return Tensor(np.squeeze(self.a,index))
 def sum(self):return Tensor(self.a.sum())
 def item(self):return self.a.item()
 def __gt__(self,other):return Tensor(self.a>other)
 def __getitem__(self,key):return Tensor(self.a[key])
 def detach(self):return self
 def cpu(self):return self
 def numpy(self):return self.a


class Context:
 def __enter__(self):return self
 def __exit__(self,*args):return False


def fake_torch():
 return SimpleNamespace(bool=np.dtype(bool),is_tensor=lambda value:isinstance(value,Tensor),inference_mode=Context,
  nn=SimpleNamespace(functional=SimpleNamespace(interpolate=lambda v,size,mode:Tensor(v.a))))


@pytest.mark.parametrize('support',[0,1,2,256])
def test_support_diag_precedes_gate_and_native_method_restored(tiny,support):
 valid=np.zeros((1,16,16),bool);valid.reshape(-1)[:support]=True;points=Tensor(np.zeros((1,16,16,3),np.float32))
 calls=[];diagnostics=[];persisted=[]
 def original(p,mask,focal=None,downsample_size=None):calls.append((p,mask,focal,downsample_size));return 'native'
 module=SimpleNamespace(recover_focal_shift=original)
 def native_infer(tensor,**kwargs):
  assert kwargs['apply_mask']is False
  return module.recover_focal_shift(points,Tensor(valid),focal=1.,downsample_size=(64,64))
 model=SimpleNamespace(infer=native_infer)
 if support<2:
  with pytest.raises(ValueError,match='two sampled'):infer.checked_depth(fake_torch(),module,model,Tensor(np.zeros((3,16,16))),diagnostics,lambda:persisted.append(copy.deepcopy(diagnostics)),0)
  assert not calls and persisted[0][0]['nearest64_valid_pixels']==support
 else:
  assert infer.checked_depth(fake_torch(),module,model,Tensor(np.zeros((3,16,16))),diagnostics,lambda:persisted.append(copy.deepcopy(diagnostics)),0)=='native'
  assert len(calls)==1 and diagnostics[0]['original_returned']is True
 assert module.recover_focal_shift is original


def test_native_solver_exception_never_retried_and_restored(tiny):
 module=SimpleNamespace(recover_focal_shift=lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError('native failed')));old=module.recover_focal_shift
 def run(*args,**kwargs):return module.recover_focal_shift(Tensor(np.zeros((1,16,16,3))),Tensor(np.ones((1,16,16),bool)))
 with pytest.raises(RuntimeError,match='native failed'):infer.checked_depth(fake_torch(),module,SimpleNamespace(infer=run),Tensor(np.zeros((3,16,16))),[],lambda:None,0)
 assert module.recover_focal_shift is old


def test_projection_is_model_consistency_not_independent_support(tiny):
 kp=np.ones((70,3),np.float32);kp[:,2]=2;translation=np.array([0,0,1],np.float32)
 pixels=20*kp[:,:2]/3+8
 prediction={'pred_keypoints_3d':Tensor(kp),'pred_keypoints_2d':Tensor(pixels)}
 result=infer.body_projection(prediction,dict(focal_length=np.array(20.,np.float32),pred_cam_t=translation))
 assert result['max_projection_error_px']<.05 and result['independent_RGB_hand_support_verified']is False
 prediction['pred_keypoints_2d']=Tensor(pixels+1)
 with pytest.raises(ValueError):infer.body_projection(prediction,dict(focal_length=np.array(20.,np.float32),pred_cam_t=translation))


def test_direct_actual_blob_independent_content_pin(monkeypatch,tmp_path):
 receipt=tmp_path/'results/weights-acquisition.json';receipt.parent.mkdir();receipt.write_text(json.dumps(dict(assets=[dict(repo_id='Ruicheng/moge-2-vitl-normal',revision=infer.MOGE_REV,cache_dir=str(tmp_path/'weights/cari4d/hf_home/hub'))])))
 blob=infer.moge_path(tmp_path);blob.parent.mkdir(parents=True);blob.write_bytes(b'not-a-checkpoint')
 def identity(path,*args):return dict(bytes=123,sha256='a'*64) if path==receipt else dict(bytes=infer.MOGE_BYTES,sha256=infer.MOGE_SHA)
 monkeypatch.setattr(infer.binding,'identity',identity)
 assert infer.moge_asset(tmp_path)[0]==blob
 monkeypatch.setattr(infer.binding,'identity',lambda *args:dict(bytes=infer.MOGE_BYTES,sha256='b'*64))
 with pytest.raises(ValueError,match='content bytes'):infer.moge_asset(tmp_path)


def test_original_symlink_graph_preserved_readonly(monkeypatch,tmp_path):
 cache=tmp_path/'weights/cari4d/hf_home/hub';repo=cache/'models--Ruicheng--moge-2-vitl-normal'
 snapshot=repo/'snapshots'/infer.MOGE_REV/'model.pt';snapshot.parent.mkdir(parents=True)
 repoblob=repo/'blobs'/infer.MOGE_SHA;repoblob.parent.mkdir();xet=infer.moge_path(tmp_path);xet.parent.mkdir(parents=True);xet.write_bytes(b'tiny')
 snapshot.symlink_to('../../blobs/'+infer.MOGE_SHA);repoblob.symlink_to('../../blobs/9f/'+infer.XET_SHA)
 monkeypatch.setattr(infer.binding,'identity',lambda *args:dict(bytes=infer.MOGE_BYTES,sha256=infer.MOGE_SHA))
 result=infer.host_moge_chain(tmp_path)
 assert len(result)==3 and snapshot.is_symlink()and repoblob.is_symlink()
 snapshot.unlink();snapshot.symlink_to(str(xet))
 with pytest.raises(ValueError,match='relative link'):infer.host_moge_chain(tmp_path)


def test_no_private_or_renderer_import_no_fake_track1():
 source=(ROOT/'infra/bridge_rgb_anchor_infer.py').read_text()
 assert 'import camera_render'not in source and 'import joint_rgb_infer'not in source and 'import bridge_rgb_anchor_render'not in source
 assert '_validate_inputs('not in source and 'inference_type=\'full\''in source
 assert 'apply_mask=False'in source and "joint_global_rotations_native"in source
 assert 'independent_RGB_hand_support_verified=False'in source and 'human_identity_clip_constant=False'in source
 assert 'bit_exact_replay_claimed=False'in source and 'torch.use_deterministic_algorithms(False,warn_only=False)'in source


def test_wrapper_owned_cleanup_source_preflight_and_tight_mounts():
 wrapper=(ROOT/'infra/run_bridge_rgb_anchor_infer.sh').read_text()
 assert '--user 0:0 --memory 12g --cpus 4 --read-only'in wrapper and '--gpus all --network none'in wrapper
 assert 'flock --nonblock 9'in wrapper and 'exec 9<"$LOCK"'in wrapper
 assert 'docker stop --time 3 "$NAME"'in wrapper and 'docker rm --force "$NAME"'in wrapper
 assert 'host --preflight'in wrapper and 'host --verify'in wrapper and 'WR_MOGE_ORIGINAL_CHAIN="$CHAIN"'in wrapper
 assert 'src=$ROOT,dst=$ROOT'not in wrapper and 'docker system prune'not in wrapper
 assert 'trap \'exit 130\' INT' in wrapper and 'trap \'exit 143\' TERM' in wrapper


@pytest.mark.parametrize('failure',[False,True])
def test_receipt_sealed400_after_real_operation_success_or_failure(tmp_path,failure):
 report=dict(status='fail',phase='integrity')
 def operation(persist):
  report['attempts']=1;persist()
  if failure:raise ValueError('native precondition failed')
  report.update(status='pass',phase='complete')
 if failure:
  with pytest.raises(ValueError):infer.persist_run(tmp_path,report,operation)
 else:infer.persist_run(tmp_path,report,operation)
 path=tmp_path/'report.json';actual=json.loads(path.read_text())
 assert path.stat().st_mode&0o777==0o400 and actual['status']==('fail'if failure else'pass')
 assert actual['attempts']==1 and actual['elapsed_seconds']>=0
 with pytest.raises(FileExistsError):infer.persist_run(tmp_path,{},operation)


def test_secret_like_upstream_exception_not_in_receipt(tmp_path):
 def fail(persist):raise ValueError('Bearer SUPERSECRET')
 with pytest.raises(ValueError):infer.persist_run(tmp_path,dict(status='fail'),fail)
 assert 'SUPERSECRET'not in (tmp_path/'report.json').read_text()
