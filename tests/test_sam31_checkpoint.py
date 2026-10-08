"""CPU-only manufactured tensors; no Torch install, weights, GPU or source data."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
from sam31_runtime import verify_checkpoint_coverage


class Tensor:
    def __init__(self,value):self.value=np.asarray(value)
    @property
    def shape(self):return self.value.shape
    def detach(self):return self
    def cpu(self):return self
    def is_complex(self):return np.iscomplexobj(self.value)
    @property
    def real(self):return Tensor(self.value.real)
    @property
    def imag(self):return Tensor(self.value.imag)


class Model:
    def __init__(self,native,learned):self.native=native;self.learned=learned
    def state_dict(self):return self.native
    def named_parameters(self):return [(k,self.native[k])for k in self.learned]


PREFIX='detector.backbone.visual.trunk.blocks.0.attn.'


def fixture(monkeypatch):
    base=Tensor(np.array([[1+2j,3+4j]],dtype=np.complex64));learned={'detector.weight'}
    native={'detector.weight':Tensor([.3,.7]),PREFIX+'freqs_cis':base,
            PREFIX+'freqs_cis_real':base.real,PREFIX+'freqs_cis_imag':base.imag}
    checkpoint={k:v for k,v in native.items()if not k.endswith(('_real','_imag'))}
    fake=SimpleNamespace(load=lambda *a,**k:checkpoint,
        equal=lambda a,b:np.array_equal(a.value,b.value))
    monkeypatch.setitem(sys.modules,'torch',fake)
    return Model(native,learned),checkpoint


def test_exact_derived_views_proven_against_checkpoint(monkeypatch):
    model,_=fixture(monkeypatch);r=verify_checkpoint_coverage(model,'manufactured.pt')
    assert r['complete_learned_key_shape_coverage'] is True
    assert r['complete_key_shape_coverage'] is False
    assert r['derived_buffer_count']==2
    assert all(p['derived_equals_checkpoint_view']for p in r['derived_buffer_proofs'])


def test_complete_checkpoint_no_exception_needed(monkeypatch):
    model,checkpoint=fixture(monkeypatch);checkpoint.update(model.native)
    r=verify_checkpoint_coverage(model,'manufactured.pt')
    assert r['complete_key_shape_coverage'] is True and r['derived_buffer_count']==0


def test_missing_learned_parameter_fatal(monkeypatch):
    model,checkpoint=fixture(monkeypatch);del checkpoint['detector.weight']
    with pytest.raises(ValueError,match='missing learned'):verify_checkpoint_coverage(model,'manufactured.pt')


def test_arbitrary_missing_buffer_fatal(monkeypatch):
    model,_=fixture(monkeypatch);model.native['detector.other_buffer']=Tensor([1])
    with pytest.raises(ValueError,match='other than whitelisted'):verify_checkpoint_coverage(model,'manufactured.pt')


def test_unexpected_checkpoint_key_fatal(monkeypatch):
    model,checkpoint=fixture(monkeypatch);checkpoint['foreign']=Tensor([1])
    with pytest.raises(ValueError,match='unexpected'):verify_checkpoint_coverage(model,'manufactured.pt')


def test_learned_shape_mismatch_fatal(monkeypatch):
    model,checkpoint=fixture(monkeypatch);checkpoint['detector.weight']=Tensor([1,2,3])
    with pytest.raises(ValueError,match='incorrect-shaped'):verify_checkpoint_coverage(model,'manufactured.pt')


@pytest.mark.parametrize('part',['real','imag'])
def test_incorrect_native_derived_buffer_fatal(monkeypatch,part):
    model,_=fixture(monkeypatch);model.native[PREFIX+'freqs_cis_'+part]=Tensor([[100,200]])
    with pytest.raises(ValueError,match='differs from deterministic'):verify_checkpoint_coverage(model,'manufactured.pt')


def test_checkpoint_base_differs_from_generated_native_fatal(monkeypatch):
    model,checkpoint=fixture(monkeypatch);checkpoint[PREFIX+'freqs_cis']=Tensor([[100+200j,300+400j]])
    with pytest.raises(ValueError,match='differs from deterministic'):verify_checkpoint_coverage(model,'manufactured.pt')


def test_missing_checkpoint_complex_base_fatal(monkeypatch):
    model,checkpoint=fixture(monkeypatch);del checkpoint[PREFIX+'freqs_cis']
    with pytest.raises(ValueError):verify_checkpoint_coverage(model,'manufactured.pt')


def test_same_suffix_outside_detector_attention_not_allowed(monkeypatch):
    model,_=fixture(monkeypatch);model.native['tracker.freqs_cis_real']=Tensor([1])
    with pytest.raises(ValueError,match='other than whitelisted'):verify_checkpoint_coverage(model,'manufactured.pt')


def test_derived_tensor_cannot_be_named_learned_parameter(monkeypatch):
    model,_=fixture(monkeypatch);model.learned.add(PREFIX+'freqs_cis_real')
    with pytest.raises(ValueError,match='missing learned'):verify_checkpoint_coverage(model,'manufactured.pt')
