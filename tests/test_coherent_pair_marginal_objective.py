"""Tiny manufactured loss/VJP checks; fake Torch only, never native Torch."""
from contextlib import nullcontext
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType,SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from world_reward import coherent_route_scorer as core
from world_reward.coherent_pair_marginal import MarginalPairScores
from world_reward.coherent_pair_packed_torch import TorchMarginalPairScores
from world_reward import coherent_pair_marginal_objective as objective


THETA=np.arange(17,dtype=np.float64)/32.
X=(np.arange(6*17,dtype=np.float64).reshape(2,3,17)%13-6)/16.
M=np.array([[.25,-.375,.125],[-.5,.25,.375]])


def scores(theta=THETA,alpha=.3125,supported=None,*,shift=0.):
    good=np.ones((2,3),bool) if supported is None else supported
    a=X@theta+shift;b=a+alpha*M
    a=np.where(good,a,np.nan);b=np.where(good,b,np.nan)
    native=np.repeat(good[:,None,:],2,axis=1)
    identity=MappingProxyType(dict(source_person_ids=('p0','p1'),source_object_ids=('o0','o1','o2'),
        person_members=(np.array([0]),np.array([1])),object_members=tuple(np.array([i]) for i in range(3)),
        person_to_group=np.arange(2),object_to_group=np.arange(3)))
    da=np.where(good[...,None],X,0.);dm=np.where(good,M,0.)
    return MarginalPairScores(identity,'a'*64,core._fingerprint((theta,.8125,float(alpha))),.8125,float(alpha),
        np.repeat(a[:,None,:],2,axis=1),np.repeat(b[:,None,:],2,axis=1),native,native.copy(),a,b,good,da,da.copy(),dm)


def positive():
    mask=np.zeros((2,3),bool);mask[0,0]=mask[1,2]=True;return mask


@pytest.mark.parametrize('arm',['A','B'])
def test_independent_enumerated_loss_and_gradient(arm):
    s=scores();mask=positive();lam=.25;out=objective.marginal_objective((s,),(mask,),THETA,alpha=.3125,regularization=lam,arm=arm)
    x=(s.scores_a if arm=='A' else s.scores_b).ravel();p=mask.ravel()
    z=np.exp(x.astype(np.longdouble));w=z/z.sum();v=z[p]/z[p].sum()
    expected=float(np.log(z.sum()/z[p].sum()))
    d=X.reshape(-1,17) if arm=='A' else M.reshape(-1,1);param=THETA if arm=='A' else np.array([.3125])
    np.testing.assert_allclose(out.loss,expected+.5*lam*(param@param),rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(out.gradient,np.asarray(w@d-v@d[p],float)+lam*param,rtol=1e-13,atol=1e-13)
    assert out.status=='complete' and out.counts['records']==out.counts['used']==1
    assert out.record_statuses==('used',) and out.unlisted_alternatives_verified_negative is False


@pytest.mark.parametrize('arm',['A','B'])
def test_gradient_finite_differences(arm):
    alpha=.3125;lam=.125;mask=positive();out=objective.marginal_objective((scores(),),(mask,),THETA,alpha=alpha,regularization=lam,arm=arm)
    for i in range(17 if arm=='A' else 1):
        h=1e-6;delta=np.zeros(17);delta[i if arm=='A' else 0]=h
        t1,t0=(THETA+delta,THETA-delta) if arm=='A' else (THETA,THETA)
        a1,a0=(alpha,alpha) if arm=='A' else (alpha+h,alpha-h)
        hi=objective.marginal_objective((scores(t1,a1),),(mask,),t1,alpha=a1,regularization=lam,arm=arm).loss
        lo=objective.marginal_objective((scores(t0,a0),),(mask,),t0,alpha=a0,regularization=lam,arm=arm).loss
        np.testing.assert_allclose((hi-lo)/(2*h),out.gradient[i],rtol=1e-7,atol=1e-8)


def test_right_alpha_zero_and_regularize_only_active_parameters():
    mask=positive();out=objective.marginal_objective((scores(alpha=0.),),(mask,),THETA,alpha=0.,regularization=7.,arm='B')
    h=1e-6
    f=lambda a:objective.marginal_objective((scores(alpha=a),),(mask,),THETA,alpha=a,regularization=7.,arm='B').loss
    np.testing.assert_allclose((-3*f(0.)+4*f(h)-f(2*h))/(2*h),out.gradient[0],atol=1e-8)
    assert out.loss==objective.marginal_objective((scores(alpha=0.),),(mask,),THETA,alpha=0.,arm='B').loss


def test_fixed_population_missing_not_renormalized_and_regularizer_once():
    s=scores();mask=positive();unsupported=scores(supported=np.zeros((2,3),bool));lam=.25
    one=objective.marginal_objective((s,),(mask,),THETA,alpha=.3125)
    out=objective.marginal_objective((s,unsupported),(mask,mask),THETA,alpha=.3125,regularization=lam)
    np.testing.assert_allclose(out.loss,one.loss/2+.5*lam*(THETA@THETA))
    np.testing.assert_allclose(out.gradient,one.gradient/2+lam*THETA)
    assert out.status=='partial' and out.counts['records']==2 and out.counts['used']==1
    assert out.counts['missing_positive']==1 and np.isnan(out.record_losses[1])
    assert out.record_statuses==('used','missing_positive')


@pytest.mark.parametrize('kind',['none','unsupported','all_positive','partial_positive'])
def test_explicit_edge_coverage(kind):
    mask=positive();good=np.ones((2,3),bool)
    if kind=='none':mask[:]=False
    if kind=='unsupported':good[:]=False
    if kind=='all_positive':mask[:]=True
    if kind=='partial_positive':good[0,0]=False
    out=objective.marginal_objective((scores(supported=good),),(mask,),THETA,alpha=.3125)
    if kind=='partial_positive':
        assert out.status=='partial' and out.counts['partially_missing_positive']==1 and out.counts['used']==1
    else:
        assert out.status=='no_informative' and out.counts['used']==0 and out.loss==0. and not out.gradient.any()
    if kind=='all_positive':assert out.record_losses[0]==0. and out.counts['no_alternative']==1


def test_empty_group_population_is_explicit_missing_without_inventing_score():
    s=scores();empty=replace(s,identity=MappingProxyType(dict(source_person_ids=(),source_object_ids=(),
        person_members=(),object_members=(),person_to_group=np.empty(0,np.int64),object_to_group=np.empty(0,np.int64))),supported=np.empty((0,0),bool),native_supported=np.empty((0,2,0),bool),
        native_route_supported=np.empty((0,2,0),bool),native_scores_a=np.empty((0,2,0)),native_scores_b=np.empty((0,2,0)),
        scores_a=np.empty((0,0)),scores_b=np.empty((0,0)),geometry_derivatives_a=np.empty((0,0,17)),
        geometry_derivatives_b=np.empty((0,0,17)),alpha_derivatives_b=np.empty((0,0)))
    out=objective.marginal_objective((empty,),(np.empty((0,0),bool),),THETA,alpha=.3125)
    assert out.status=='no_informative' and out.counts['no_positive']==1 and out.counts['supported_groups']==0


def test_extreme_common_offset_preserves_logprob_not_zero_from_cancellation():
    theta=np.zeros(17);s=scores(theta,alpha=0.,shift=1e300)
    out=objective.marginal_objective((s,),(positive(),),theta,alpha=0.)
    np.testing.assert_allclose(out.loss,np.log(3.),rtol=1e-15)


@pytest.mark.parametrize('arm',['A','B'])
def test_common_extreme_derivative_cancels_exactly_cpu_and_fake_torch(monkeypatch,arm):
    theta=np.zeros(17);theta[1]=1.;s=scores(theta,alpha=0.)
    values=np.arange(6,dtype=np.float64).reshape(2,3);d=np.zeros((2,3,17));d[...,0]=1e300;d[...,1]=values
    s=replace(s,scores_a=values,scores_b=values.copy(),native_scores_a=np.repeat(values[:,None,:],2,axis=1),
        native_scores_b=np.repeat(values[:,None,:],2,axis=1),geometry_derivatives_a=d,geometry_derivatives_b=d.copy(),
        alpha_derivatives_b=np.full((2,3),1e300))
    mask=np.zeros((2,3),bool);mask[0,0]=mask[1,2]=True;before=core._fingerprint((s,mask,theta))
    cpu=objective.marginal_objective((s,),(mask,),theta,alpha=0.,arm=arm)
    assert cpu.gradient[0]==0. and core._fingerprint((s,mask,theta))==before
    monkeypatch.setitem(sys.modules,'torch',fake_torch())
    device=objective.marginal_objective_torch((device_scores(s),),(mask,),theta,alpha=0.,arm=arm)
    assert device.gradient.array[0]==0.
    np.testing.assert_allclose(device.gradient.array,cpu.gradient,rtol=1e-14,atol=1e-14)


def test_mixed_temperature_with_valid_individual_fingerprints_rejected():
    s=scores();other=replace(s,temperature=1.,parameter_fingerprint=core._fingerprint((THETA,1.,.3125)))
    with pytest.raises(ValueError,match='uniform temperature'):
        objective.marginal_objective((s,other),(positive(),positive()),THETA,alpha=.3125)


def test_extreme_derivative_range_overflow_fails_cpu_and_fake_torch(monkeypatch):
    s=scores();d=s.geometry_derivatives_a.copy();d[0,0,0]=1e308;d[0,1,0]=-1e308
    s=replace(s,geometry_derivatives_a=d)
    with pytest.raises(FloatingPointError):objective.marginal_objective((s,),(positive(),),THETA,alpha=.3125)
    monkeypatch.setitem(sys.modules,'torch',fake_torch())
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError,match='Centered device derivative overflow'):
            objective.marginal_objective_torch((device_scores(s),),(positive(),),THETA,alpha=.3125)


def test_overflow_fails_not_clamped():
    s=scores();a=s.scores_a.copy();a.flat[0]=1e308;a.flat[1]=-1e308;s=replace(s,scores_a=a)
    with pytest.raises(FloatingPointError):objective.marginal_objective((s,),(positive(),),THETA,alpha=.3125)


def test_borrowed_arrays_and_outputs_immutable():
    s=scores();mask=positive();theta=THETA.copy();before=core._fingerprint((s,mask,theta))
    out=objective.marginal_objective((s,),(mask,),theta,alpha=.3125)
    assert before==core._fingerprint((s,mask,theta))
    for a in (out.gradient,out.record_losses):
        with pytest.raises(ValueError):a.setflags(write=True)
    with pytest.raises(TypeError):out.counts['used']=99


def test_native_alias_slots_add_no_extra_loss_mass():
    s=scores();identity=dict(s.identity);identity.update(source_object_ids=('o0','o1','o2','alias-o0'),
        object_members=(np.array([0,3]),np.array([1]),np.array([2])),object_to_group=np.array([0,1,2,0]))
    alias=replace(s,identity=MappingProxyType(identity),
        native_supported=s.native_supported[:,:,[0,1,2,0]],native_route_supported=s.native_route_supported[:,:,[0,1,2,0]],
        native_scores_a=s.native_scores_a[:,:,[0,1,2,0]],native_scores_b=s.native_scores_b[:,:,[0,1,2,0]])
    base=objective.marginal_objective((s,),(positive(),),THETA,alpha=.3125)
    aliased=objective.marginal_objective((alias,),(positive(),),THETA,alpha=.3125)
    assert base.loss==aliased.loss and base.gradient.tobytes()==aliased.gradient.tobytes()
    assert aliased.counts['positive_groups']==2


@pytest.mark.parametrize('change',[
    dict(person_to_group=np.array([0,0])),dict(object_members=(np.array([0,0]),np.array([1]),np.array([2]))),
    dict(object_members=(np.array([0]),np.array([1]))),dict(source_person_ids=('p0','p0')),
])
def test_inconsistent_alias_identity_rejected(change):
    s=scores();identity=dict(s.identity);identity.update(change)
    with pytest.raises(ValueError):objective.marginal_objective((replace(s,identity=identity),),(positive(),),THETA,alpha=.3125)


@pytest.mark.parametrize('bad',[
    {'theta':np.zeros(17,np.float32)},{'theta':np.zeros(16)}, {'theta':np.full(17,np.nan)},
    {'alpha':-1.},{'alpha':True},{'alpha':float('inf')},{'regularization':-1.},
    {'regularization':float('nan')},{'regularization':True},{'arm':'C'},
    {'positive_masks':(np.zeros((2,2),bool),)},{'positive_masks':(np.zeros((2,3),int),)},
    {'scores':()},{'scores':(SimpleNamespace(supported=np.ones((2,3),bool)),)},
])
def test_invalid_explicit_inputs(bad):
    args=dict(scores=(scores(),),positive_masks=(positive(),),theta=THETA,alpha=.3125)
    args.update(bad)
    with pytest.raises(ValueError):objective.marginal_objective(**args)


@pytest.mark.parametrize('field,value',[
    ('parameter_fingerprint','b'*64),('distribution_fingerprint','not-a-hash'),('alpha',.5),
    ('temperature',0.),('scores_a',np.ones((2,3),np.float32)),('geometry_derivatives_a',np.zeros((2,3,16))),
    ('alpha_derivatives_b',np.full((2,3),np.nan)),('native_route_supported',np.ones((1,2,3),bool)),
])
def test_malformed_score_abi(field,value):
    with pytest.raises(ValueError):objective.marginal_objective((replace(scores(),**{field:value}),),(positive(),),THETA,alpha=.3125)


class Tensor:
    def __init__(self,array,device='cuda:0'):self.array=np.asarray(array);self.device=device;self.requires_grad=False
    @property
    def shape(self):return self.array.shape
    @property
    def dtype(self):return self.array.dtype
    def reshape(self,*shape):return Tensor(self.array.reshape(*shape),self.device)
    def __getitem__(self,key):
        if isinstance(key,tuple):key=tuple(k.array if type(k) is Tensor else k for k in key)
        elif type(key) is Tensor:key=key.array
        return Tensor(self.array[key],self.device)
    def __bool__(self):return bool(self.array)
    def __int__(self):return int(self.array)
    def __invert__(self):return Tensor(~self.array)
    def __and__(self,x):return Tensor(self.array & x.array)
    def any(self):return Tensor(self.array.any())
    def all(self):return Tensor(self.array.all())
    def sum(self):return Tensor(self.array.sum())
    def max(self):return Tensor(self.array.max())
    def __add__(self,x):return Tensor(self.array+(x.array if type(x) is Tensor else x))
    __radd__=__add__
    def __sub__(self,x):return Tensor(self.array-(x.array if type(x) is Tensor else x))
    def __mul__(self,x):return Tensor(self.array*(x.array if type(x) is Tensor else x))
    __rmul__=__mul__
    def __truediv__(self,x):return Tensor(self.array/(x.array if type(x) is Tensor else x))
    def __matmul__(self,x):return Tensor(self.array@x.array)


def fake_torch():
    return SimpleNamespace(Tensor=Tensor,float64=np.dtype('float64'),bool=np.dtype('bool'),no_grad=nullcontext,
        tensor=lambda a,dtype,device:Tensor(np.array(a,dtype=dtype,copy=True),device),
        zeros_like=lambda a:Tensor(np.zeros_like(a.array),a.device),
        zeros=lambda shape,dtype,device:Tensor(np.zeros(shape,dtype=dtype),device),
        full=lambda shape,value,dtype,device:Tensor(np.full(shape,value,dtype=dtype),device),
        stack=lambda x:Tensor(np.stack([a.array for a in x])),
        isfinite=lambda x:Tensor(np.isfinite(x.array)),isnan=lambda x:Tensor(np.isnan(x.array)),
        exp=lambda x:Tensor(np.exp(x.array)),log=lambda x:Tensor(np.log(x.array)))


def device_scores(s):
    fields={name:(Tensor(value.copy()) if type(value) is np.ndarray else value) for name,value in vars(s).items()}
    return TorchMarginalPairScores(**fields,device='cuda:0')


def test_primary_torch_multidimensional_mask_none_requires_selection_first(monkeypatch):
    # PyTorch v2.5.1 primary source, no Torch import/kernel execution:
    # TensorIndexing.h L312-319/L460-463 records one tensor index then inserts
    # None at dimension1; IndexingUtils.h L30-35 subsequently expands a bool
    # mask and requires all its dimensions to match that intermediate tensor.
    # Source SHA256s: 44d5c6cebb19a4bd620391c64fa2e83009b15beb10a046d455fa787854258056
    # and 2e85c9ea26e416dda38405b25c78af172d60cbb4a3d88aded224f8eeeeb68fe4.
    original=Tensor.__getitem__;rejected=[]
    def primary_index(self,key):
        if (isinstance(key,tuple) and len(key)==2 and type(key[0]) is Tensor
                and key[1] is None and key[0].array.dtype==np.bool_):
            mask=key[0].array;shape=list(self.array.shape);shape.insert(1,1)
            if tuple(shape[:mask.ndim])!=mask.shape:
                rejected.append((self.array.shape,mask.shape,tuple(shape)))
                raise IndexError('Primary multidimensional mask/None shape mismatch')
        return original(self,key)
    monkeypatch.setattr(Tensor,'__getitem__',primary_index)
    monkeypatch.setitem(sys.modules,'torch',fake_torch())
    for alpha in (0.,.3125):
        s=scores(alpha=alpha);device=device_scores(s)
        with pytest.raises(IndexError):
            device.alpha_derivatives_b[device.supported,None]
        for arm in ('A','B'):
            actual=objective.marginal_objective_torch((device,),(positive(),),THETA,
                alpha=alpha,regularization=.125,arm=arm)
            expected=objective.marginal_objective((s,),(positive(),),THETA,
                alpha=alpha,regularization=.125,arm=arm)
            np.testing.assert_allclose(actual.loss.array,expected.loss,rtol=1e-14,atol=1e-14)
            np.testing.assert_allclose(actual.gradient.array,expected.gradient,rtol=1e-13,atol=1e-13)
            assert actual.gradient.shape==(17 if arm=='A' else 1,)
    assert rejected==[((2,3),(2,3),(2,1,3))]*2


@pytest.mark.parametrize('arm',['A','B'])
def test_lazy_torch_equivalent_entire_fixed_population_with_fake_only(monkeypatch,arm):
    monkeypatch.setitem(sys.modules,'torch',fake_torch())
    bank=(scores(),scores(supported=np.zeros((2,3),bool)),scores());masks=(positive(),positive(),np.ones((2,3),bool))
    cpu=objective.marginal_objective(bank,masks,THETA,alpha=.3125,regularization=.125,arm=arm)
    gpu=objective.marginal_objective_torch(tuple(map(device_scores,bank)),masks,THETA,alpha=.3125,regularization=.125,arm=arm)
    np.testing.assert_allclose(gpu.loss.array,cpu.loss,atol=1e-14)
    np.testing.assert_allclose(gpu.gradient.array,cpu.gradient,atol=1e-14)
    np.testing.assert_allclose(gpu.record_losses.array,cpu.record_losses,equal_nan=True)
    assert gpu.counts==cpu.counts and gpu.record_statuses==cpu.record_statuses and gpu.status==cpu.status


def test_lazy_torch_rejects_foreign_device_and_forged_dtype(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',fake_torch());s=device_scores(scores());s.scores_a.array=np.zeros((2,3),np.float32)
    with pytest.raises(ValueError):objective.marginal_objective_torch((s,),(positive(),),THETA,alpha=.3125)
    s=device_scores(scores());s.scores_a.device='cpu'
    with pytest.raises(ValueError):objective.marginal_objective_torch((s,),(positive(),),THETA,alpha=.3125)


def test_real_packed_cpu_scorer_abi_no_reconstruction_or_side_pooling():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('objective_original_fixture',root/'infra/coherent_pair_cost_probe.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    from world_reward.coherent_pair_learning import PairScale
    from world_reward.coherent_pair_cache import prepare_pair_cache
    from world_reward.coherent_pair_packed import prepare_marginal_packed
    from world_reward.coherent_pair_packed_score import score_pair_marginal_packed
    iterator=fixture.fixtures(np,object_count=2)
    try:
        bank=next(iterator);packed=prepare_marginal_packed(prepare_pair_cache(bank,PairScale(np.ones(12),np.ones(12,bool))))
        s=score_pair_marginal_packed(packed,THETA,temperature=.8125,alpha=.3125)
        mask=np.zeros(s.supported.shape,bool);mask[0,0]=True
        out=objective.marginal_objective((s,),(mask,),THETA,alpha=.3125,arm='B')
        assert out.counts['records']==1 and out.counts['positive_groups']==1 and out.gradient.shape==(1,)
        assert out.score_references[0]==(packed.packed_fingerprint,s.parameter_fingerprint)
    finally:iterator.close()


def test_torch_source_has_no_optimizer_or_autograd_dependency():
    import ast
    tree=ast.parse(Path(objective.__file__).read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr in ('backward','grad','fit','step') for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names) for n in tree.body)


def test_module_import_does_not_import_torch_or_run_optimizer():
    path=Path(objective.__file__)
    code="""import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]=='torch':raise RuntimeError('Eager Torch forbidden')
sys.meta_path.insert(0,Deny());sys.path.insert(0,sys.argv[1])
from world_reward.coherent_pair_marginal_objective import marginal_objective
assert 'torch' not in sys.modules
"""
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(path.parents[1])],capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr.decode()
