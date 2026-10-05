"""Manufactured transport cases only; no model, GT or ownership validation."""
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

import numpy as np
import pytest

from world_reward.person_pose_observations import PersonPoseObservations
from world_reward.hoi_detr_observations import HOIDetrObservations
from world_reward.interaction_tuple_evidence import build_interaction_tuple_evidence

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wr_test_person_bank_evidence", ROOT/"infra/hoi_person_bank_evidence.py")
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def saved(n=2):
    rgb = np.arange(16*24*3, dtype=np.uint8).reshape(16, 24, 3)
    ids = tuple(f"episode:000009/frame:000011/person/retained:{i:06d}" for i in range(n))
    xy = np.full((n, 133, 2), 3., np.float64)
    scores = np.full((n, 133), .4, np.float32)
    if n:
        xy[0, 9] = [100., -3.]  # valid but off-grid, retained as a diagnostic
        scores[0, 91] = 0.     # miss: don't invent support
    p = PersonPoseObservations(11, (16, 24), ids, np.tile([[0., 0., 24., 16.]], (n, 1)).astype(np.float64),
        np.full(n, .8, np.float64), xy, scores)
    arrays = {k:getattr(p,k).copy() for k in gate.PERSON_ARRAYS}
    row = dict(episode=9, frame_index=11, person_ids=list(ids), persons=n,
        prediction_file="episode_000009_frame_000011.npz", prediction={"bytes":1,"sha256":"f"*64},
        decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        keypoints=gate.array_identity(p.keypoints_original_xy), scores=gate.array_identity(p.raw_scores))
    return p, row, arrays, rgb


def hoi(classes=(0, 1, 0, 1, 2)):
    c = np.asarray(classes, np.int64); n = len(c)
    boxes = np.tile([[5., 3., 9., 7.]], (n, 1)).astype(np.float32)
    scores = np.full(n, .5, np.float32); pairs = []
    for a,b in ((0,1),(1,2)):
        left,right=np.flatnonzero(c==a),np.flatnonzero(c==b)
        pairs.append(np.column_stack((np.repeat(left,len(right)),np.tile(right,len(left)))).astype(np.int64))
    logits = [np.arange(len(p)*2,dtype=np.float32).reshape(len(p),2) for p in pairs]
    return HOIDetrObservations(11,(16,24),np.zeros((1500,3),np.float32),np.zeros((1500,4),np.float32),
        np.zeros((1500,256),np.float32),np.column_stack((boxes,scores)),np.arange(n,dtype=np.int64)[::-1].copy(),
        np.arange(n,dtype=np.int64),np.full(n,7,np.int64),c,boxes,scores,scores.copy(),pairs[0],logits[0],pairs[1],logits[1])


def test_actual_saver_fields_full_bank_flags_and_no_mutation():
    p,row,arrays,rgb=saved(3); original={k:v.tobytes() for k,v in arrays.items()}; r=gate.reconstruct_saved_person(row,arrays,rgb)
    assert r.person_ids==p.person_ids and r.original_frame_index==11 and r.image_size==(16,24)
    assert not r.in_original_image[0,9] and r.native_valid[0,9] and not r.native_valid[0,91]
    for k,v in arrays.items():
        assert getattr(r,k).tobytes()==v.tobytes() and not getattr(r,k).flags.writeable
        assert v.tobytes()==original[k]
    assert r.keypoints_original_xy[0,9].tolist()==[100.,-3.]


@pytest.mark.parametrize("fault",["extra","missing","valid","grid","booldtype","pointdtype","scorehash","pointhash",
    "rgb","rgbdtype","ids","duplicateid","count","frame","prediction","boxoffgrid"])
def test_malformed_stored_banks_fail_closed(fault):
    p,row,a,rgb=saved()
    if fault=="extra":a["foreign"]=a["raw_scores"]
    if fault=="missing":del a["native_valid"]
    if fault=="valid":a["native_valid"][0,91]=True
    if fault=="grid":a["in_original_image"][0,9]=True
    if fault=="booldtype":a["native_valid"]=a["native_valid"].astype(np.uint8)
    if fault=="pointdtype":a["keypoints_original_xy"]=a["keypoints_original_xy"].astype(np.float32)
    if fault=="scorehash":row["scores"]["sha256"]="f"*64
    if fault=="pointhash":row["keypoints"]["sha256"]="f"*64
    if fault=="rgb":rgb=rgb.copy();rgb[0,0,0]^=1
    if fault=="rgbdtype":rgb=rgb.astype(np.float32)
    if fault=="ids":row["person_ids"].reverse()
    if fault=="duplicateid":row["person_ids"][1]=row["person_ids"][0]
    if fault=="count":row["persons"]+=1
    if fault=="frame":row["frame_index"]+=1
    if fault=="prediction":row["prediction_file"]="foreign.npz"
    if fault=="boxoffgrid":a["boxes_original_xyxy"][0,0]=-1.
    with pytest.raises(ValueError):gate.reconstruct_saved_person(row,a,rgb)


@pytest.mark.parametrize("n,classes",[(0,()),(2,()),(0,(0,1)),(2,(0,)),(2,(1,)),(2,(0,1,0,1,2))])
def test_saved_complete_tuple_and_raw_bank_npz_empty_or_duplicate(n,classes,tmp_path):
    p,row,a,rgb=saved(n);p=gate.reconstruct_saved_person(row,a,rgb);h=hoi(classes)
    before=[x.tobytes()for x in (p.keypoints_original_xy,h.query_tokens,h.query_ids)]
    expected=build_interaction_tuple_evidence(p,h);stem=tmp_path/"all_bank"
    meta=gate.save_tuple_evidence(stem,p,h)
    assert meta["tuple_rows"]==n*2*len(h.hand_object_pairs)
    assert meta["source_person_ids"]==list(p.person_ids)and meta["feature_names"]==list(expected.feature_names)
    assert not meta["selection_performed"]and not meta["anatomical_ownership_verified"]and not meta["accuracy_verified"]
    path=Path(str(stem)+".npz")
    assert meta["prediction"]==dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with np.load(path,allow_pickle=False)as z:
        assert set(z.files)==set(meta["arrays"])
        for k in z.files:assert gate.array_identity(z[k])==meta["arrays"][k]
        for k,v in expected.arrays.items():np.testing.assert_array_equal(z["tuple__"+k],v)
        np.testing.assert_array_equal(z["tuple__features"],expected.features)
        np.testing.assert_array_equal(z["tuple__feature_supported"],expected.feature_supported)
        assert z["hoi__query_tokens"].shape==(1500,256)
        np.testing.assert_array_equal(z["hoi__query_ids"],h.query_ids)
        np.testing.assert_array_equal(z["person__keypoints_original_xy"],p.keypoints_original_xy)
    assert json.loads(Path(str(stem)+".json").read_bytes())==meta
    assert all(stat.S_IMODE(Path(str(stem)+suffix).stat().st_mode)==0o444 for suffix in (".npz",".json"))
    assert before==[x.tobytes()for x in (p.keypoints_original_xy,h.query_tokens,h.query_ids)]
    if meta["tuple_rows"]:
        assert np.isnan(expected.features[:,2:5]).any()  # missing point stays NaN, not imputed


def test_reuse_path_never_overwrites_and_frame_grid_mismatch(tmp_path):
    p,_,_,_=saved();h=hoi((0,1));stem=tmp_path/"old"
    gate.save_tuple_evidence(stem,p,h);pin=Path(str(stem)+".npz").read_bytes()
    with pytest.raises(ValueError):gate.save_tuple_evidence(stem,p,h)
    assert Path(str(stem)+".npz").read_bytes()==pin
    with pytest.raises(ValueError):gate.save_tuple_evidence(tmp_path/"bad",p,replace(h,original_frame_index=12))
    with pytest.raises(ValueError):gate.save_tuple_evidence(tmp_path/"bad",p,replace(h,image_size=(17,24)))


def test_save_failure_removes_only_owned_partial(tmp_path,monkeypatch):
    p,_,_,_=saved();stem=tmp_path/"partial"
    def fail(*_,**kwargs):raise OSError("bounded test")
    monkeypatch.setattr(gate.np,"savez",fail)
    with pytest.raises(OSError):gate.save_tuple_evidence(stem,p,hoi((0,1)))
    assert not Path(str(stem)+".npz").exists()and not Path(str(stem)+".json").exists()


def test_actual_npz_mapping_reconstruction_and_wrong_grid(tmp_path):
    p,row,a,rgb=saved(2);path=tmp_path/"native.npz"
    with path.open("xb")as stream:np.savez(stream,**a)
    with np.load(path,allow_pickle=False)as stored:
        recovered=gate.reconstruct_saved_person(row,stored,rgb)
        assert recovered.person_ids==p.person_ids
    wrong=np.zeros((18,26,3),np.uint8);row["decoded_RGB_sha256"]=hashlib.sha256(wrong.tobytes()).hexdigest()
    # Recompute genuine original native ingrid flags: changed grid cannot be silently accepted.
    a["keypoints_original_xy"][1,10]=[25.,17.]
    a["in_original_image"][1,10]=False
    row["keypoints"]=gate.array_identity(a["keypoints_original_xy"])
    with pytest.raises(ValueError):gate.reconstruct_saved_person(row,a,wrong)
