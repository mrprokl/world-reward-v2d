"""Fake original modules only; no local Torch/CUDA/model/real checkpoint."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import ast
import json
import sys

import numpy as np
import pytest
import hoi_detr_native_model as p


class Tensor:
    def __init__(self,*,shape=(1,),dtype='float32',finite=True):self.shape,self.dtype,self.finite=shape,dtype,finite
    def numel(self):return 1
    def element_size(self):return 4


class Model:
    def __init__(self,events):
        self.events=events;self.state={f'original.layer_{i}.weight':Tensor()for i in range(898)};self.training=True
    def state_dict(self):return self.state
    def load_state_dict(self,state,strict):
        self.events.append(('strict_load',len(state),strict));return SimpleNamespace(missing_keys=[],unexpected_keys=[])
    def to(self,**kwargs):self.events.append(('to',kwargs));return self
    def eval(self):self.events.append(('eval',));self.training=False;return self


def fixture(tmp_path,monkeypatch):
    events=[];repo=Path(p.__file__).resolve().parents[1]
    protocol=json.loads((repo/'configs/hoi_detr_model_qualify_v1.json').read_bytes())
    policy=protocol['native_config'];buffers=protocol['checkpoint_buffers'];model=Model(events)
    cfg_dict=dict(load_from=policy['original_load_from'],resume_from=None,
        model=dict(type='CoDETR',train_cfg=[dict(native=True)],backbone=dict(use_act_checkpoint=True,pretrained=None,init_cfg=None),
            query_head=dict(num_query=1500,num_classes=3),roi_head=[dict(original=True)]),
        test_cfg=dict(original='unmodified'),data=dict(test=dict(pipeline=[dict(type='LoadImageFromFile'),dict(type='Normalize',native=True)])),
        custom_hooks=[buffers['native_hook']])
    class Config:
        _cfg_dict=cfg_dict
        model=cfg_dict['model'];custom_hooks=cfg_dict['custom_hooks'];data=SimpleNamespace(test=SimpleNamespace(pipeline=cfg_dict['data']['test']['pipeline']))
        def get(self,key):return cfg_dict.get(key)
        @staticmethod
        def fromfile(path,import_custom_modules):events.append(('config',path,import_custom_modules));return Config()
    def build(model_cfg,*,test_cfg):events.append(('build',deepcopy(model_cfg),deepcopy(test_cfg)));return model
    class EMA:
        def __init__(self,momentum):events.append(('ema_init',momentum));self.skip_buffers=False;self.checkpoint=None
        def before_run(self,runner):
            events.append(('register',));self.param_ema_buffer={k:'ema_'+k.replace('.','_')for k in list(runner.model.state)}
            runner.model.state.update({v:Tensor()for v in self.param_ema_buffer.values()})
        def after_train_iter(self,*a):pytest.fail('EMA updated')
        def after_train_epoch(self,*a):pytest.fail('EMA swapped')
    prepared=[];pipeline=[]
    class Compose:
        def __init__(self,steps):pipeline.extend(deepcopy(steps));events.append(('compose',))
        def __call__(self,value):prepared.append(value['img']);return dict(img=value['img'],img_metas='native')
    collate=lambda *a,**k:'native_collate';scatter=lambda *a,**k:'native_scatter'
    bbox=lambda *a,**k:'native_bbox';nms=lambda *a,**k:'native_nms'
    attributes={'mmcv':dict(Config=Config),'mmcv.parallel':dict(collate=collate,scatter=scatter),
        'mmcv.ops':dict(batched_nms=nms),'mmdet.models.builder':dict(build_detector=build),
        'mmdet.datasets.pipelines':dict(Compose=Compose),'mmdet.core.bbox.transforms':dict(bbox_cxcywh_to_xyxy=bbox),
        'mmdet.core.hook.ema':dict(ExpMomentumEMAHook=EMA),'projects.models':{}}
    names=set(attributes)
    for name in list(names):
        parts=name.split('.')
        for i in range(1,len(parts)):names.add('.'.join(parts[:i]))
    modules={name:ModuleType(name)for name in names}
    for name,module in modules.items():
        module.__path__=[]
        for key,value in attributes.get(name,{}).items():setattr(module,key,value)
        monkeypatch.setitem(sys.modules,name,module)
    for name,module in modules.items():
        if '.'in name:
            parent,leaf=name.rsplit('.',1);setattr(modules[parent],leaf,module)
    checkpoint={};loaded=[]
    def load(path,**kwargs):
        events.append(('load',path,kwargs));loaded.append(kwargs)
        checkpoint.update(state_dict={k:Tensor()for k in model.state},meta=dict(original=True))
        return checkpoint
    torch=SimpleNamespace(Tensor=Tensor,float32='float32',load=load,device=lambda name:name,
        cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:1,get_device_capability=lambda:(9,0)),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),cudnn=SimpleNamespace(allow_tf32=True)),
        is_autocast_enabled=lambda *a:False,isfinite=lambda value:SimpleNamespace(all=lambda:value.finite),
        no_grad=lambda:None,cat=lambda *a:None,as_tensor=lambda *a:None)
    runtime=SimpleNamespace(check=lambda deadline:events.append(('check',deadline)))
    kwargs=dict(torch=torch,np=np,runtime=runtime,deadline=99,source_root=tmp_path/'source',checkpoint_path=tmp_path/'checkpoint.pth',
        native_config=deepcopy(policy),checkpoint_buffers=deepcopy(buffers))
    return kwargs,model,events,cfg_dict,pipeline,prepared,checkpoint,modules


def test_exact_author_sequence898_backups1796_strict_cpu_cuda_eval(tmp_path,monkeypatch):
    kwargs,model,events,cfg,pipeline,prepared,checkpoint,_=fixture(tmp_path,monkeypatch)
    before=deepcopy(cfg);result=p.build_original_hoi(**kwargs)
    names=[e[0]for e in events]
    assert names==['check','config','build','check','ema_init','register','load','strict_load','to','eval','check','compose']
    assert events[1][2]is False and events[6][2]==dict(map_location='cpu',weights_only=True)
    assert events[7]==('strict_load',1796,True)and events[8]==('to',dict(device='cuda',dtype='float32'))
    before['load_from']=None;before['model']['train_cfg']=None
    assert cfg==before and pipeline[0]==dict(type='LoadImageFromWebcam')
    assert cfg['data']['test']['pipeline'][0]==dict(type='LoadImageFromFile')
    assert result.model is model and model.training is False and len(model.state)==1796
    assert result.provenance['checkpoint_buffer_schema']['ema_swapped']is False
    assert result.provenance['strict_checkpoint']['keys']==1796
    assert result.provenance['source_authenticated']is result.provenance['runtime_qualified']is result.provenance['weights_authenticated']is False
    assert kwargs['torch'].backends.cuda.matmul.allow_tf32 is kwargs['torch'].backends.cudnn.allow_tf32 is False
    assert not prepared


def test_rgb_bgr_once_nonmutating_and_native_operations_same(tmp_path,monkeypatch):
    kwargs,_,_,_,_,prepared,_,modules=fixture(tmp_path,monkeypatch);result=p.build_original_hoi(**kwargs)
    rgb=np.array([[[10,20,30],[40,50,60]],[[70,80,90],[100,110,120]]],dtype=np.uint8);before=rgb.copy();rgb.flags.writeable=False
    value=result.operations.prepare_rgb(rgb)
    np.testing.assert_array_equal(rgb,before);np.testing.assert_array_equal(prepared[0],before[:,:,::-1])
    prepared[0][0,0,0]=0;np.testing.assert_array_equal(rgb,before)
    assert set(value)=={'img','img_metas'}and result.operations.device=='cuda:0'
    assert result.operations.collate is modules['mmcv.parallel'].collate
    assert result.operations.scatter is modules['mmcv.parallel'].scatter
    assert result.operations.batched_nms is modules['mmcv.ops'].batched_nms
    assert result.operations.bbox_cxcywh_to_xyxy is modules['mmdet.core.bbox.transforms'].bbox_cxcywh_to_xyxy


@pytest.mark.parametrize('fault',['checkpointing','query','classes','nested_loader','locator','override'])
def test_original_configuration_policy_used_without_new_source_fork(tmp_path,monkeypatch,fault):
    kwargs,_,events,cfg,*_=fixture(tmp_path,monkeypatch)
    if fault=='checkpointing':cfg['model']['backbone']['use_act_checkpoint']=False
    elif fault=='query':cfg['model']['query_head']['num_query']=1000
    elif fault=='classes':cfg['model']['query_head']['num_classes']=1
    elif fault=='nested_loader':cfg['model']['roi_head'][0]['load_from']='remote'
    elif fault=='locator':cfg['load_from']='other'
    else:kwargs['native_config']['inference_overrides']['load_from']='remote'
    with pytest.raises(ValueError):p.build_original_hoi(**kwargs)
    assert not any(e[0]=='build'for e in events)


@pytest.mark.parametrize('fault',['missing','extra','dtype','shape','nan','prefix'])
def test_strict1796_checkpoint_no_partial_or_remapping(tmp_path,monkeypatch,fault):
    kwargs,_,events,*_=fixture(tmp_path,monkeypatch);load=kwargs['torch'].load
    def changed(*a,**k):
        value=load(*a,**k);state=value['state_dict'];name=next(iter(state))
        if fault=='missing':state.pop(name)
        elif fault=='extra':state['foreign']=Tensor()
        elif fault=='dtype':state[name]=Tensor(dtype='float16')
        elif fault=='shape':state[name]=Tensor(shape=(2,))
        elif fault=='nan':state[name]=Tensor(finite=False)
        else:state['module.'+name]=state.pop(name)
        return value
    kwargs['torch'].load=changed
    with pytest.raises(ValueError):p.build_original_hoi(**kwargs)
    assert not any(e[0]in('strict_load','to','eval')for e in events)


@pytest.mark.parametrize('fault',['wrong_count','swap','skip','hook','resume'])
def test_native_ema_registration_only_no_skip_resume_swap(tmp_path,monkeypatch,fault):
    kwargs,model,events,cfg,*_=fixture(tmp_path,monkeypatch)
    if fault=='wrong_count':model.state.pop(next(iter(model.state)))
    elif fault=='swap':kwargs['checkpoint_buffers']['swap_or_update']=True
    elif fault=='skip':kwargs['checkpoint_buffers']['skip_buffers']=True
    elif fault=='hook':cfg['custom_hooks'][0]['type']='foreign'
    else:
        module=sys.modules['mmdet.core.hook.ema'];parent=module.ExpMomentumEMAHook
        class Bad(parent):
            def __init__(self,**kwargs):super().__init__(**kwargs);self.checkpoint='remote'
        module.ExpMomentumEMAHook=Bad
    with pytest.raises(ValueError):p.build_original_hoi(**kwargs)
    assert not any(e[0]=='load'for e in events)


@pytest.mark.parametrize('fault',['no_cuda','multi_gpu','wrong_gpu','autocast','train_api','foreign_path','loader'])
def test_native_runtime_and_preprocessing_fail_closed(tmp_path,monkeypatch,fault):
    kwargs,_,events,cfg,*_=fixture(tmp_path,monkeypatch);torch=kwargs['torch']
    if fault=='no_cuda':torch.cuda.is_available=lambda:False
    elif fault=='multi_gpu':torch.cuda.device_count=lambda:2
    elif fault=='wrong_gpu':torch.cuda.get_device_capability=lambda:(8,0)
    elif fault=='autocast':torch.is_autocast_enabled=lambda *a:True
    elif fault=='train_api':monkeypatch.setitem(sys.modules,'mmdet.apis.inference',ModuleType('mmdet.apis.inference'))
    elif fault=='foreign_path':monkeypatch.setattr(sys,'path',[*sys.path,'/lus/unaudited'])
    else:cfg['data']['test']['pipeline'][0]=dict(type='oracle_loader')
    with pytest.raises(ValueError):p.build_original_hoi(**kwargs)
    if fault!='loader':assert not any(e[0]=='build'for e in events)


def test_bad_prepare_outputs_no_target_fields(tmp_path,monkeypatch):
    kwargs,*_=fixture(tmp_path,monkeypatch)
    module=sys.modules['mmdet.datasets.pipelines']
    class Bad:
        def __init__(self,pipeline):pass
        def __call__(self,value):return dict(value,interaction_targets='forbidden')
    module.Compose=Bad;result=p.build_original_hoi(**kwargs)
    with pytest.raises(ValueError,match='image-only'):result.operations.prepare_rgb(np.zeros((1,1,3),np.uint8))


def test_original_helper_functions_not_copied_and_no_profile_runtime_imports():
    tree=ast.parse(Path(p.__file__).read_bytes());names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert names=={'build_original_hoi'}
    calls=[n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
    assert all(name in calls for name in ('configuration_policy','register_original_checkpoint_buffers','strict_checkpoint'))
    assert not any(n in calls for n in ('gpu_model','native_context','source','protocol','derive_source','load_helpers','infer_hoi_detr_frame'))
    top_imports=[a.name for n in tree.body if isinstance(n,ast.Import)for a in n.names]
    assert 'torch'not in top_imports and 'numpy'not in top_imports and 'mmcv'not in top_imports
