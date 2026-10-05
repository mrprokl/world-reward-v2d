from fractions import Fraction
import itertools

import numpy as np
import pytest

from world_reward.exact_normal_dot import positive_normal_dot


def reference(before, after):
    """Independent determinant expansion, not the production cross helper."""
    p = [[Fraction(float(x)) for x in row] for row in before]
    q = [[Fraction(float(x)) for x in row] for row in after]
    def determinant(rows, i, j):
        return (rows[0][i]*(rows[1][j]-rows[2][j])
                + rows[1][i]*(rows[2][j]-rows[0][j])
                + rows[2][i]*(rows[0][j]-rows[1][j]))
    return sum(determinant(p,i,j)*determinant(q,i,j) for i,j in ((0,1),(0,2),(1,2))) > 0


def check(a, b):
    original=(a.tobytes(),b.tobytes())
    result, report=positive_normal_dot(a,b)
    expected=np.array([reference(x,y)for x,y in zip(a,b)],np.bool_)
    assert np.array_equal(result,expected)
    assert original==(a.tobytes(),b.tobytes())
    assert report['triangles']==len(a)
    assert sum(report[k]for k in ('interval_positive_rows','interval_nonpositive_rows','exact_dyadic_fallback_rows'))==len(a)
    assert report['positive_rows']==int(expected.sum())
    assert not report['area_or_length_tolerance_used'] and not report['coordinate_normalization_performed']
    assert not result.flags.writeable
    with pytest.raises(ValueError):result.setflags(write=True)
    return result,report


def base(dtype=np.float64):
    return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype)


def test_positive_reversed_orthogonal_and_all_exact_zero():
    a=np.repeat(base()[None],5,axis=0)
    b=a.copy();b[1]=b[1,::-1];b[2]=[[0,0,0],[1,0,0],[0,0,1]]
    b[3]=0.;b[4]=[[0,0,0],[1,0,0],[2,0,0]]
    result,report=check(a,b)
    assert result.tolist()==[True,False,False,False,False]
    assert report['interval_positive_rows']>=1 and report['interval_nonpositive_rows']>=1
    assert report['exact_dyadic_fallback_rows']>=3


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_extreme_represented_scales_subnormal_and_overflow(dtype):
    info=np.finfo(dtype)
    scales=[dtype(1),dtype(info.max),dtype(info.tiny),np.nextafter(dtype(0),dtype(1)),dtype(2**-30)]
    a=[];b=[]
    for scale in scales:
        tri=base(dtype)*scale
        for other in (tri,tri[::-1],np.zeros((3,3),dtype)):
            a.append(tri);b.append(other)
    _,report=check(np.array(a,dtype),np.array(b,dtype))
    # Every zero counterpart is ambiguous. F32 extremes can be certified after
    # exact F64 promotion; do not incorrectly require F64 underflow there.
    assert report['exact_dyadic_fallback_rows']>=len(scales)


def test_overflowing_coordinate_differences_still_use_original_fraction():
    m=np.finfo(np.float64).max
    a=np.array([[[-m,0.,0.],[m,0.,0.],[0.,m,0.]]])
    b=a.copy()
    result,report=check(a,b)
    assert result.tolist()==[True] and report['exact_dyadic_fallback_rows']==1


def test_signed_zero_repeated_points_and_fraction_cancellation():
    z=-0.
    a=np.array([[[z,0,0],[1,z,0],[0,1,z]],[[0,0,0],[1,1,1],[2,2,2]],
        [[1.,1.,1.],[1.+2**-50,1.,1.],[1.,1.+2**-50,1.]]])
    b=a.copy();b[0,0,0]=0.
    result,_=check(a,b)
    assert result.tolist()==[True,False,True]


def test_mixed_dtype_is_original_F64_vs_F32_not_recased_both():
    a=np.array([[[1.,0.,0.],[1.+2**-30,0.,0.],[1.,1.,0.]]])
    b=a.astype(np.float32)
    result,report=check(a,b)
    assert not result[0] and report['exact_dyadic_fallback_rows']==1
    assert check(a,a)[0][0]


def test_independent_random_reference_and_batch_permutation():
    rng=np.random.default_rng(81425)
    a=rng.normal(size=(257,3,3));b=a+rng.normal(size=a.shape)*.5
    expected,report=check(a,b)
    order=rng.permutation(len(a))
    permuted,_=check(a[order],b[order])
    assert np.array_equal(permuted,expected[order])
    assert report['exact_dyadic_fallback_rows']==0


def test_simultaneous_corner_axis_permutations_preserve_dot_sign():
    a=np.array([base(),base()]);b=np.array([base(),base()[::-1]])
    original=check(a,b)[0]
    for corners in itertools.permutations(range(3)):
        for axes in itertools.permutations(range(3)):
            transformed,_=check(a[:,corners][:,:,axes],b[:,corners][:,:,axes])
            assert np.array_equal(transformed,original)


def test_dyadic_translation_and_common_scale_no_normalization():
    a=np.array([base(),base()]);b=np.array([base(),base()[::-1]])
    original=check(a,b)[0]
    for scale in (2**-20,2**20):
        shifted=2**-8+scale*a;other=2**-8+scale*b
        assert np.array_equal(check(shifted,other)[0],original)


def test_input_views_readonly_and_no_alias():
    a=np.repeat(base()[None],4,axis=0);b=a.copy()
    a.setflags(write=False)
    result,_=check(a[::-1],b[::-1])
    b[:]=0.
    assert result.all()


def test_empty_batch_not_geometry_acceptance():
    result,report=check(np.empty((0,3,3),np.float32),np.empty((0,3,3),np.float64))
    assert not len(result) and report['triangles']==report['positive_rows']==0


@pytest.mark.parametrize('kind',['list','masked','int','shape','nan','inf','mismatch'])
def test_invalid_input_rejected(kind):
    a=base()[None];b=a.copy()
    if kind=='list':a=a.tolist()
    elif kind=='masked':a=np.ma.array(a)
    elif kind=='int':a=a.astype(np.int64)
    elif kind=='shape':a=a.reshape(1,9)
    elif kind=='nan':a[0,0,0]=np.nan
    elif kind=='inf':a[0,0,0]=np.inf
    else:b=np.repeat(b,2,axis=0)
    with pytest.raises(ValueError):positive_normal_dot(a,b)
