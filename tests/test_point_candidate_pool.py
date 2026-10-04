"""Native scalar-loop equivalence on authored arrays/callbacks, no media/GPU."""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward import point_candidate_pool as pool
from world_reward.rigid_alignment import RigidAlignment, align_observed_points


def scene(frames=3,height=8,width=8):
    vertices=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.2,0.],[0.,0.,.3]])
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)
    sampled=np.tile(vertices,(2048,1))
    # mesh.centroid deliberately not vertex mean; caller must preserve native value.
    centroid=np.array([.037,.062,.081])
    pointmap=np.random.default_rng(3).normal(0,.04,(height,width,3))+[.2,-.3,2.4]
    return dict(frame_index=np.arange(frames,dtype=np.int64),vertices=vertices,faces=faces,surface_points=sampled,
                mesh_centroid=centroid,initial_rotation=Rotation.from_rotvec([.2,-.13,.04]).as_matrix(),
                initial_translation=np.array([.1,.2,3.]),camera_K=np.array([[400.,0.,width/2],[0.,400.,height/2],[0.,0.,1.]]),width=width,height=height),pointmap,np.ones((height,width),bool)


class Backend:
    def __init__(self,mask,fail=(),runtime=None):
        self.mask=mask;self.next=0;self.active=None;self.fail=set(fail);self.runtime=runtime;self.events=[]
    def raster(self,vertices,faces,K,width,height):
        assert (height,width)==self.mask.shape
        if self.active is None:
            number=self.next;self.next+=1;self.active=number;stage='initial'
        else:number=self.active;self.active=None;stage='fitted'
        self.events.append((stage,number,vertices.copy()))
        if self.runtime==stage:raise RuntimeError('native backend failure')
        if number in self.fail and stage=='fitted':raise ValueError('fitted near-plane failure')
        result=self.mask.copy()
        if (stage=='initial'and number%4 in (1,2))or(stage=='fitted'and number%4==0):
            result.flat[::2]=False
        return result
    def align(self,sampled,observed,R,t):
        number=self.active
        self.events.append(('align',number,sampled.copy(),observed.copy(),R.copy(),t.copy()))
        if self.runtime=='align':raise RuntimeError('native alignment failure')
        if number in self.fail and number%2==0:
            self.active=None;raise ValueError('underconstrained support')
        fitR=Rotation.from_rotvec([0.,0.,.003]).as_matrix()@R
        fitT=t+np.array([.002,0.,0.])
        return RigidAlignment(fitR,fitT,'improved',.1,.2 if number%4==2 else .01,32,2)


def independent_native(inputs,pointmap,mask,backend):
    """Original loop, independent of pool class/sampling/IoU helpers."""
    vertices,faces=inputs['vertices'],inputs['faces'];sampled=inputs['surface_points']
    initialR,t0=inputs['initial_rotation'],inputs['initial_translation'];center=inputs['mesh_centroid']
    orientations=Rotation.create_group('O').as_matrix();previous=initialR.copy();rows=[]
    def iou(a,b):return float(np.count_nonzero(a&b)/np.count_nonzero(a|b))
    for index in inputs['frame_index']:
        points=pointmap[int(index)] if pointmap.ndim==4 else pointmap
        visible=mask&np.isfinite(points).all(-1)&(points[...,2]>0);observed=points[visible]
        if len(observed)<40:raise ValueError('observed<40')
        rng=np.random.default_rng(0)
        if len(observed)>2048:observed=observed[rng.choice(len(observed),2048,replace=False)]
        candidates=[];rejected=[]
        for number,initial_R in enumerate([initialR@O for O in orientations]+[previous]):
            initial_t=t0 if index==0 else np.median(observed,axis=0)-center@initial_R.T
            try:
                initial_iou=iou(backend.raster(vertices@initial_R.T+initial_t,faces,inputs['camera_K'],inputs['width'],inputs['height']),mask)
                fit=backend.align(sampled,observed,initial_R,initial_t)
                fitted_iou=iou(backend.raster(vertices@fit.rotation.T+fit.translation,faces,inputs['camera_K'],inputs['width'],inputs['height']),mask)
                accepted=fitted_iou>=initial_iou and fit.final_residual<=fit.initial_residual
                chosenR,chosenT=(fit.rotation,fit.translation)if accepted else(initial_R,initial_t)
                candidates.append((number,chosenR,chosenT,initial_iou,fitted_iou,fitted_iou if accepted else initial_iou,
                                   fit.initial_residual,fit.final_residual,fit.final_residual if accepted else fit.initial_residual,accepted))
            except ValueError as exc:rejected.append((number,str(exc)))
        if not candidates:raise ValueError('no candidates')
        winner=max(candidates,key=lambda c:(c[5],-c[8],-c[0]));previous=np.asarray(winner[1]);rows.append((candidates,rejected,winner[0],len(observed),int(visible.sum())))
    return rows


@pytest.mark.parametrize('large',[False,True])
def test_exact_native_candidates_seed_translation_observations_and_greedy(large):
    data,points,mask=scene(height=50 if large else 8,width=50 if large else 8)
    points[0,0]=np.nan;points[0,1,2]=-1;mask[0,2]=False
    before={k:v.copy()for k,v in data.items()if isinstance(v,np.ndarray)}
    original=Backend(mask,fail=(4,7));expected=independent_native(data,points,mask,original)
    actual=Backend(mask,fail=(4,7));builder=pool.NativeCandidatePool(**data)
    for index,(candidates,rejected,greedy,sample_count,visible_count) in enumerate(expected):
        row=builder.add_frame(index,points,mask,actual.raster,align=actual.align)
        assert row.rejected==tuple(rejected) and row.greedy_index==greedy
        assert row.sampled_observations==sample_count and row.visible_point_pixels==visible_count
        assert np.flatnonzero(row.valid_candidates).tolist()==[c[0]for c in candidates]
        for c in candidates:
            number=c[0]
            np.testing.assert_array_equal(row.rotations[number],c[1]);np.testing.assert_array_equal(row.translations[number],c[2])
            for array,value in zip((row.initial_iou,row.fitted_iou,row.selected_iou,row.initial_residual,row.fitted_residual,row.selected_residual,row.icp_accepted),c[3:],strict=True):
                assert array[number]==value
            assert row.image_costs[number]==1-c[5]
        assert all(not v.flags.writeable for v in vars(row).values()if isinstance(v,np.ndarray))
    frozen=builder.finalize();assert frozen.rotations.shape==(3,25,3,3) and frozen.image_costs.shape==(3,25)
    assert all(not v.flags.writeable for v in vars(frozen).values())
    assert len(original.events)==len(actual.events)
    for e1,e2 in zip(original.events,actual.events,strict=True):
        assert e1[:2]==e2[:2]
        for a,b in zip(e1[2:],e2[2:],strict=True):np.testing.assert_array_equal(a,b)
    for k,old in before.items():np.testing.assert_array_equal(data[k],old)
    # Initial frame ALL25 seeds preserve generative translation; later uses sampled median and native centroid.
    for event in [e for e in actual.events if e[0]=='align'][:25]:
        np.testing.assert_array_equal(event[-1],data['initial_translation'])


def test_acceptance_both_gates_and_lowest_index_exact_ties():
    data,points,mask=scene();builder=pool.NativeCandidatePool(**data);backend=Backend(mask)
    row=builder.add_frame(0,points,mask,backend.raster,align=backend.align)
    assert not row.icp_accepted[0] and row.icp_accepted[1] and not row.icp_accepted[2] and row.icp_accepted[3]
    assert row.greedy_index==1 # slots1/3/... exactIoU/residual tie, earliestindex.
    np.testing.assert_array_equal(row.translations[0],data['initial_translation'])
    assert not np.array_equal(row.translations[1],data['initial_translation'])
    old=builder._previous_rotation.copy();backend=Backend(mask)
    second=builder.add_frame(1,points,mask,backend.raster,align=backend.align)
    alignrows=[e for e in backend.events if e[0]=='align']
    np.testing.assert_array_equal(alignrows[-1][-2],old)
    np.testing.assert_array_equal(alignrows[-1][-1],np.median(points.reshape(-1,3),axis=0)-data['mesh_centroid']@old.T)
    assert second.frame_index==1


@pytest.mark.parametrize('runtime',['initial','fitted','align'])
def test_backend_failures_not_invalid_candidates_or_retries(runtime):
    data,points,mask=scene();builder=pool.NativeCandidatePool(**data);backend=Backend(mask,runtime=runtime)
    with pytest.raises(RuntimeError,match='native'):builder.add_frame(0,points,mask,backend.raster,align=backend.align)
    assert not builder._frames


def test_fitted_failure_discards_whole_initial_slot_and_empty_frame_fails():
    data,points,mask=scene();builder=pool.NativeCandidatePool(**data);backend=Backend(mask,fail=range(25))
    with pytest.raises(ValueError,match='No finite supported'):builder.add_frame(0,points,mask,backend.raster,align=backend.align)
    assert not builder._frames
    backend=Backend(mask,fail=(7,));row=builder.add_frame(0,points,mask,backend.raster,align=backend.align)
    assert not row.valid_candidates[7] and row.image_costs[7]==0
    np.testing.assert_array_equal(row.rotations[7],np.eye(3));np.testing.assert_array_equal(row.translations[7],[0,0,0])


@pytest.mark.parametrize('fault',['frame_indices','less3','skip','short_sample','badR','badK','badfaces','pointshape','maskdtype','masked','low40','allnonpositive'])
def test_invalid_inputs_fail_before_callbacks(fault):
    data,points,mask=scene();events=[]
    if fault=='frame_indices':data['frame_index']=np.array([0,2,3],np.int64)
    elif fault=='less3':data['frame_index']=np.arange(2,dtype=np.int64)
    elif fault=='short_sample':data['surface_points']=data['surface_points'][:8191]
    elif fault=='badR':data['initial_rotation']=np.zeros((3,3))
    elif fault=='badK':data['camera_K'][0,0]=0
    elif fault=='badfaces':data['faces'][0]=[0,0,1]
    elif fault=='pointshape':points=points[0]
    elif fault=='maskdtype':mask=mask.astype(np.uint8)
    elif fault=='masked':points=np.ma.array(points,mask=False)
    elif fault=='low40':mask.flat[39:]=False
    elif fault=='allnonpositive':points[...,2]=0
    with pytest.raises(ValueError):
        builder=pool.NativeCandidatePool(**data)
        builder.add_frame(1 if fault=='skip'else 0,points,mask,lambda *a:events.append(a))
    assert not events


def test_sampling_rng_resets_per_frame_and_no_pointmap_retained():
    data,points,mask=scene(height=50,width=50);builder=pool.NativeCandidatePool(**data);samples=[]
    def align(sampled,observed,R,t):
        samples.append(observed.copy());return RigidAlignment(R,t,'unchanged',.1,.1,32,0)
    for i in range(3):builder.add_frame(i,points,mask,lambda *a:mask,align=align)
    expected=points.reshape(-1,3)[np.random.default_rng(0).choice(2500,2048,replace=False)]
    for value in samples:np.testing.assert_array_equal(value,expected)
    assert not any(v is points for v in vars(builder).values())
    builder.finalize()
    with pytest.raises(ValueError):builder.finalize()
    with pytest.raises(ValueError):builder.add_frame(3,points,mask,lambda *a:mask,align=align)


def test_default_icp_is_original_callable_without_keyword_override(monkeypatch):
    data,points,mask=scene();seen=[]
    def align(*args,**kwargs):
        assert not kwargs;seen.append(args)
        return align_observed_points(*args)
    monkeypatch.setattr(pool,'align_observed_points',align)
    data['surface_points']=np.random.default_rng(7).normal(0,.07,(8192,3))
    points=(data['surface_points'][:64]@data['initial_rotation'].T+data['initial_translation']).reshape(8,8,3)
    builder=pool.NativeCandidatePool(**data)
    row=builder.add_frame(0,points,mask,lambda *a:mask)
    assert len(seen)==25 and row.valid_candidates.any()
    assert all(len(args)==4 and args[0].shape==(8192,3)for args in seen)


def test_silhouette_math_ast_exact_native_and_no_query_arguments():
    source=(Path(__file__).resolve().parents[1]/'infra/camera_render.py').read_text()
    native=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='silhouette_iou')
    actual=ast.parse(inspect.getsource(pool._iou)).body[0]
    def expressions(node):
        return [ast.dump(n,include_attributes=False)for n in ast.walk(node)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='count_nonzero']
    assert expressions(native)==expressions(actual)
    assert not any(name in inspect.signature(pool.NativeCandidatePool.add_frame).parameters for name in ('tracks','cost','queries','ground_truth'))


def test_complete_original_coverage_and_owned_fixed_inputs():
    data,points,mask=scene();builder=pool.NativeCandidatePool(**data)
    with pytest.raises(ValueError):builder.finalize()
    data['vertices'][:]=123;data['surface_points'][:]=456;data['initial_rotation'][:]=0
    assert not np.all(builder.vertices==123) and not np.all(builder.surface_points==456)
    def align(sampled,observed,R,t):return RigidAlignment(R,t,'unchanged',.1,.1,32,0)
    for i in range(3):
        builder.add_frame(i,points,mask,lambda *a:mask,align=align)
        if i<2:
            with pytest.raises(ValueError):builder.finalize()
    result=builder.finalize();np.testing.assert_array_equal(result.frame_index,np.arange(3,dtype=np.int64))
    np.testing.assert_allclose(result.rotations@result.rotations.swapaxes(-1,-2),np.broadcast_to(np.eye(3),(3,25,3,3)),atol=1e-6,rtol=0)


def test_callback_keeps_masked_or_nonproper_candidates_invalid_not_nan_slots():
    data,points,mask=scene();builder=pool.NativeCandidatePool(**data)
    def bad_align(sampled,observed,R,t):return RigidAlignment(np.zeros((3,3)),t,'bad',.1,0.,32,0)
    with pytest.raises(ValueError,match='No finite supported'):builder.add_frame(0,points,mask,lambda *a:mask,align=bad_align)
    assert not builder._frames
    with pytest.raises(ValueError,match='No finite supported'):
        builder.add_frame(0,points,mask,lambda *a:np.ma.array(mask,mask=False),align=bad_align)
