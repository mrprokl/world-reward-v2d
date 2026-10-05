"""Tiny authored exact tables/incidences/prior checks; no full-bank benchmark."""
from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from world_reward.coherent_pair_packed import prepare_marginal_packed
from world_reward.coherent_pair_marginal import MarginalPairReference
from world_reward import coherent_pair_learning as old
from world_reward.coherent_pair_cache import prepare_pair_cache

spec = importlib.util.spec_from_file_location('authored_marginal_fixtures', Path(__file__).with_name('test_coherent_pair_marginal.py'))
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)


def prepared(b, variable=True):
    c = prepare_pair_cache(b, old.PairScale(np.arange(1., 13.), np.full(12, variable, bool)))
    return c, prepare_marginal_packed(c)


def bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def geometry(arrays, geometry_id):
    b, l, r, route = arrays['geometry_components'][geometry_id]
    raw = np.r_[arrays['base_raw'][b], arrays['local_raw'][l], arrays['bridge_raw'][r]].astype(np.float64)
    ok = np.r_[arrays['base_supported'][b], arrays['local_supported'][l], arrays['bridge_supported'][r]]
    x = np.r_[arrays['base_values'][b, :6], arrays['local_values'][l, :4], arrays['bridge_values'][r],
              arrays['base_values'][b, 6:], arrays['local_values'][l, 4:]]
    return (raw.tobytes()+ok.tobytes()+x[12:].astype(bool).tobytes()+bytes([route])).hex(), x


def unpack(packed, prefix, target):
    a = packed.arrays; result = {}
    for incidence in range(a[prefix+'_offsets'][target], a[prefix+'_offsets'][target+1]):
        gid = a[prefix+'_geometry_ids'][incidence]; key, x = geometry(a, gid)
        start, end = a[prefix+'_margin_offsets'][incidence:incidence+2]
        mids = a[prefix+'_margin_ids'][start:end]; margins = a['margin_values'][mids]
        entries = np.arange(start, end)
        refs = a['route_refs'][np.isin(a[prefix+'_route_to_margin_entry'], entries)]
        result[key] = (x, margins, tuple(map(tuple, refs)))
    return result


@pytest.mark.parametrize('n,o,hoi,missing,aliases', [(2,3,True,False,False), (2,3,True,True,True),
    (1,2,False,True,False), (0,0,True,False,False), (0,2,True,False,False), (1,0,True,False,False)])
def test_native_and_group_csr_exact_raw_geometries_margins_refs_against_oracle(n,o,hoi,missing,aliases):
    c, p = prepared(fixtures.bank(n,o,no_hoi=not hoi,missing=missing,aliases=aliases))
    oracle = MarginalPairReference(c); before = old.core._fingerprint(c)
    assert set(p.arrays) and p.arrays['native_offsets'].shape == (n*2*o+1,)
    for prefix, targets in (('native', oracle.native_states), ('group', tuple(states for sides in oracle.group_sides for states in sides))):
        for target, states in enumerate(targets):
            actual = unpack(p, prefix, target)
            assert actual.keys() == {state.geometry_key for state in states}
            for state in states:
                x, margins, refs = actual[state.geometry_key]
                bits(x, state.values); bits(margins, state.margins); assert refs == state.native_route_refs
    bits(p.arrays['native_supported'], c.factors['good'])
    supported_sides = np.array([[bool(states) for states in sides] for sides in oracle.group_sides]).reshape(-1,2)
    count = supported_sides.sum(axis=1); expected = np.zeros(supported_sides.shape)
    np.divide(supported_sides,count[:,None],out=expected,where=count[:,None]>0)
    bits(p.arrays['group_side_prior'].reshape(-1,2),expected)
    assert old.core._fingerprint(c) == before


def test_complete_hierarchical_prior_incidences_and_margin_duplicates():
    b = fixtures.with_margins(fixtures.bank(1,1), np.array([-2.,3.,3.,3.]))
    c,p = prepared(b); o = MarginalPairReference(c)
    assert p.arrays['route_refs'].shape == (8,4) and p.arrays['geometry_components'].shape[1] == 4
    for target in range(2):
        states = unpack(p,'group',target); assert len(states)==1
        _,margins,refs = next(iter(states.values())); bits(margins,np.array([-2.,3.]))
        assert len(refs)==4
        # A target prior has one geometry, two distinct margins, not four route copies.
        start,end = p.arrays['group_offsets'][target:target+2]
        mass = 0.
        for incidence in range(start,end):
            lo,hi = p.arrays['group_margin_offsets'][incidence:incidence+2]
            mass += (hi-lo)/(end-start)/(hi-lo)
        assert mass == 1.
    assert len(o.group_sides[0][0][0].margins)==2


def test_raw_components_not_disabled_normalized_values_define_geometry():
    b = fixtures.bank(1,1); e=b.evidence; h=e.hoi_evidence; x=h.features.copy()
    x[:,0]=np.tile([0.,-0.,.25,.25],2)
    c,p = prepared(replace(b,evidence=replace(e,hoi_evidence=replace(h,features=x))),variable=False)
    oracle=MarginalPairReference(c)
    assert len(oracle.group_sides[0][0])==2
    for target in range(2):
        actual=unpack(p,'group',target); assert len(actual)==2
        assert all(np.array_equal(x[:12],np.zeros(12)) for x,_,_ in actual.values())
    assert len(np.unique(p.arrays['local_component_ids'][0,0]))==2


def test_alias_and_native_duplicate_multiplicity_no_extra_geometry_or_margin_mass():
    _,one=prepared(fixtures.bank(1,1,copies=1));_,many=prepared(fixtures.bank(3,2,copies=3,aliases=True))
    for side in range(2):
        a,z=unpack(one,'group',side),unpack(many,'group',side);assert a.keys()==z.keys()
        for key in a:
            bits(a[key][0],z[key][0]);bits(a[key][1],z[key][1]);assert len(a[key][2])<len(z[key][2])
    bits(one.arrays['group_side_prior'],many.arrays['group_side_prior'])
    assert len(many.arrays['route_refs'])==3*2*2*9


def test_base_state_route_minus_one_and_absence_not_phantom_native_route():
    c,p=prepared(fixtures.bank(1,2,no_hoi=True,missing=True))
    assert np.all(p.arrays['route_refs'][:,3]==-1) and not p.arrays['native_route_supported'].any()
    assert np.all(p.arrays['geometry_components'][:,3]==0) and p.arrays['native_supported'].all()
    assert p.identity['native_pair_slots'].shape==(0,)
    for gid in range(len(p.arrays['geometry_components'])):
        _,x=geometry(p.arrays,gid);assert np.array_equal(x[6:12],np.zeros(6)) and np.array_equal(x[15:],np.zeros(2))


def test_missing_anchor_keeps_slot_component_maps_without_distribution_mass():
    b=fixtures.bank(1,2,no_hoi=True);e=b.evidence;boxes=e.objects.boxes_original_xyxy.copy();boxes[0,2:]=boxes[0,:2]
    person=fixtures.PersonPoseObservations(2,(20,30),('p0',),np.array([[0.,0.,25.,18.]]),np.array([.9]),
        np.full((1,133,2),5.),np.zeros((1,133),np.float32))
    bad=old.pair_route_bank(person,fixtures.build_interaction_candidate_evidence(person,replace(e.objects,boxes_original_xyxy=boxes)))
    c,p=prepared(bad);assert p.arrays['base_component_ids'].shape==(1,2,2)
    assert not p.arrays['native_supported'][:,:,0].any() and not p.arrays['group_supported'][0,0]
    assert not (p.arrays['route_refs'][:,2]==0).any()
    assert p.identity['source_object_ids']==('o0','o1')


def test_permutation_and_signedzero_canonical_component_tables_and_incidence_sets():
    b=fixtures.with_margins(fixtures.bank(1,1),np.array([-0.,0.,3.,-2.]));e=b.evidence;h=e.hoi_evidence
    order=np.arange(4)[::-1];arrays=dict(h.arrays)
    for name,x in arrays.items():
        if x.shape[:1]==(8,):arrays[name]=x.reshape(1,2,4,*x.shape[1:])[:,:,order].reshape(x.shape)
    arrays['native_pair_slots']=h.arrays['native_pair_slots']
    routes={name:x.reshape(4,1,*x.shape[1:])[order].reshape(x.shape) for name,x in e.hoi_routes.items()}
    routes['native_pair_slots']=e.hoi_routes['native_pair_slots']
    other=replace(b,evidence=replace(e,hoi_routes=routes,hoi_evidence=replace(h,arrays=arrays,
        features=h.features.reshape(1,2,4,15)[:,:,order].reshape(-1,15),
        feature_supported=h.feature_supported.reshape(1,2,4,15)[:,:,order].reshape(-1,15))))
    _,a=prepared(b);_,z=prepared(other)
    for name in ('base_raw','local_raw','bridge_raw','geometry_components','margin_values','group_geometry_ids','group_margin_ids','group_side_prior'):
        bits(a.arrays[name],z.arrays[name])
    assert np.sum(a.arrays['margin_values']==0.)==1 and not np.signbit(a.arrays['margin_values'][a.arrays['margin_values']==0.]).any()


def test_constructor_source_only_no_forged_tables_immutable_and_fingerprints():
    c,p=prepared(fixtures.bank());before=old.core._fingerprint(c)
    with pytest.raises(ValueError):replace(p,arrays=dict(p.arrays))
    with pytest.raises(TypeError):p.arrays['margin_values']=np.zeros(1)
    with pytest.raises(FrozenInstanceError):p.cache=None
    for x in p.arrays.values():
        assert type(x)is np.ndarray and not x.flags.writeable
        with pytest.raises(ValueError):x.flags.writeable=True
    with pytest.raises(ValueError):prepare_marginal_packed(object())
    for key in ('native_pair_slots','native_query_ids','native_detection_slots','native_retained_nms_positions','native_flat_keep'):
        bits(p.identity[key],getattr(c.native_template,key))
    assert old.core._fingerprint(c)==before and p.packed_fingerprint==old.core._fingerprint((p.arrays,p.identity))
    z=replace(p);assert z.packed_fingerprint==p.packed_fingerprint


def test_all_native_cell_group_side_and_margin_inverse_incidence_maps():
    c,p=prepared(fixtures.with_margins(fixtures.bank(2,3,aliases=True),np.array([-2.,0.,3.,0.])))
    a=p.arrays;n,o,k=c.counts;refs=a['route_refs'];i,side,j,route=refs.T
    bits(a['route_native_cells'],(i*2+side)*o+j)
    bits(a['route_group_sides'],(c.factors['person_to_group'][i]*len(c.object_members)+c.factors['object_to_group'][j])*2+side)
    for prefix,target in [('native',a['route_native_cells']),('group',a['route_group_sides'])]:
        entry=a[prefix+'_route_to_margin_entry'];geometry_incidence=np.searchsorted(a[prefix+'_margin_offsets'][1:],entry,side='right')
        targets=np.searchsorted(a[prefix+'_offsets'][1:],geometry_incidence,side='right')
        bits(target,targets)
        bits(a['route_geometry_ids'],a[prefix+'_geometry_ids'][geometry_incidence])
        bits(a['route_margin_ids'],a[prefix+'_margin_ids'][entry])
    bits(a['margin_values'][a['route_margin_ids']],c.factors['local'][i,side,route,7])
    assert a['local_component_ids'].shape==(n,2,k) and a['bridge_component_ids'].shape==(k,o)


def test_no_native_pairs_with_hoi_preserve_slots_and_invalid_person_anchor_rejected():
    c,p=prepared(fixtures.bank(1,2,copies=0));assert c.counts==(1,2,0)
    assert p.arrays['route_refs'].shape==(4,4) and np.all(p.arrays['route_refs'][:,3]==-1)
    with pytest.raises(ValueError,match='positive-area'):
        fixtures.PersonPoseObservations(2,(20,30),('p0',),np.zeros((1,4)),np.array([.9]),
            np.full((1,133,2),5.),np.zeros((1,133),np.float32))


def test_no_per_route_python_state_factory_or_full_route_feature17_table():
    _,p=prepared(fixtures.bank())
    assert all(type(x)is np.ndarray for x in p.arrays.values())
    assert p.arrays['base_values'].shape[1]==9 and p.arrays['local_values'].shape[1]==6 and p.arrays['bridge_values'].shape[1]==2
    assert p.arrays['route_refs'].shape[1]==4 and not any('vectors' in key for key in p.arrays)
