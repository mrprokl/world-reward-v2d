"""Pure manufactured algebra; no native model, CUDA, data or cohort replay."""
from dataclasses import FrozenInstanceError
import hashlib

import numpy as np
import pytest

from world_reward.scale_reference import propose_scale_reference, UnsupportedScaleReference


def fixture():
    c=np.zeros((28,68),np.float32);m=np.zeros(68,np.float32)
    b=np.column_stack((np.full(68,-np.inf,np.float32),np.full(68,np.inf,np.float32)))
    b[:3]=0
    for index,coordinate in ((11,11),(12,12),(15,15),(16,16)):
        c[index,coordinate]=2;m[coordinate]=-1;b[coordinate]=0
    return c,m,b


def test_exact_supported_equalities_and_unconstrained_coefficients_zero():
    c,m,b=fixture();r=propose_scale_reference(c,m,b)
    assert r.fixed_coordinates==(0,1,2,11,12,15,16)
    assert r.fixed_coefficient_indices==(-1,-1,-1,11,12,15,16)
    assert r.coefficients.dtype==np.float32 and r.cpu_expanded_scale.dtype==np.float32
    assert np.count_nonzero(r.coefficients)==4 and np.all(r.coefficients[[11,12,15,16]]==.5)
    assert not r.violating_coordinates and r.fixed_residuals_float64==(0.,)*7
    assert r.cpu_literal_bounds_passed and r.status=='PROPOSAL_CPU_BOUNDS_PASS'
    assert r.proposal_only and not r.native_expansion_verified and not r.adoption
    assert not r.rounded_coefficients_minimum_norm_certified


def test_literal_fixed_residual_failed_candidate_is_retained_no_neighbor_search():
    c,m,b=fixture()
    m[12]=np.float32(float.fromhex('-0x1.c2be6ap-14'))
    c[12,12]=np.float32(float.fromhex('0x1.8e47fap-8'))
    r=propose_scale_reference(c,m,b)
    expected=np.float32((np.float64(b[12,0])-np.float64(m[12]))/np.float64(c[12,12]))
    assert r.coefficients[12].tobytes()==expected.tobytes()
    assert float(r.coefficients[12]).hex()=='0x1.21b8940000000p-6'
    assert r.cpu_expanded_scale[12]==np.float32(2.**-37)
    assert r.fixed_residuals_float64[4]==2.**-37
    assert r.violating_coordinates==(12,)and not r.cpu_literal_bounds_passed
    assert r.status=='PROPOSAL_CPU_BOUNDS_FAIL'and not r.native_expansion_verified


def test_all_original_bounds_not_only_fixed_rows_are_checked():
    c,m,b=fixture();c[11,22]=4;b[22]=[-1,1]
    r=propose_scale_reference(c,m,b)
    assert r.violating_coordinates==(22,)and r.cpu_expanded_scale[22]==2
    assert np.all(r.cpu_expanded_scale[list(r.fixed_coordinates)]==0)


@pytest.mark.parametrize('kind',['dense','duplicate','zero_wrong','no_equalities'])
def test_unsupported_policy_has_no_qp_or_clamp_fallback(kind):
    c,m,b=fixture()
    if kind=='dense':c[1,11]=1
    elif kind=='duplicate':c[12,12]=0;c[11,12]=1
    elif kind=='zero_wrong':m[0]=1
    else:b[:]=[-1,1]
    with pytest.raises(UnsupportedScaleReference):propose_scale_reference(c,m,b)


def test_original_bytes_and_returned_arrays_are_immutable_no_alias():
    values=fixture();before=tuple(a.tobytes()for a in values)
    r=propose_scale_reference(*values)
    assert before==tuple(a.tobytes()for a in values)
    assert tuple(row[3]for row in r.input_identities)==tuple(hashlib.sha256(a).hexdigest()for a in before)
    for a in values:a.fill(42)
    assert r.coefficients[11]==.5 and r.cpu_expanded_scale[11]==0
    for a in(r.coefficients,r.cpu_expanded_scale):
        with pytest.raises(ValueError):a.flags.writeable=True
        with pytest.raises(ValueError):a.flat[0]=1
    with pytest.raises(FrozenInstanceError):r.status='ADOPTED'


@pytest.mark.parametrize('kind',['dtype','shape','masked','nan_mean','inf_component','invalid_bound','nan_bound'])
def test_strict_original_array_contract(kind):
    c,m,b=fixture()
    if kind=='dtype':c=c.astype(np.float64)
    elif kind=='shape':c=c.T
    elif kind=='masked':m=np.ma.array(m,mask=False)
    elif kind=='nan_mean':m[20]=np.nan
    elif kind=='inf_component':c[0,20]=np.inf
    elif kind=='invalid_bound':b[20]=[1,0]
    else:b[20,0]=np.nan
    with pytest.raises(ValueError):propose_scale_reference(c,m,b)


def test_overflow_rejected_without_clipping_or_zero_filling():
    c,m,b=fixture();c[11,11]=np.nextafter(np.float32(0),np.float32(1));m[11]=-1
    with pytest.raises(UnsupportedScaleReference,match='finite float32'):propose_scale_reference(c,m,b)


def test_signed_zero_source_identity_and_subnormal_proposal_are_preserved():
    c,m,b=fixture();m[0]=-0.;c[11,11]=1;m[11]=-np.nextafter(np.float32(0),np.float32(1))
    r=propose_scale_reference(c,m,b)
    assert r.coefficients[11]==np.nextafter(np.float32(0),np.float32(1))
    assert r.cpu_literal_bounds_passed
    assert r.input_identities[1][3]==hashlib.sha256(m.tobytes()).hexdigest()
