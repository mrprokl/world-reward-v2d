"""Tiny manufactured audit arrays; no challenge media or local render."""
import numpy as np
import pytest

import full4d_diagnose as audit


def test_saved_pose_diagnostic_is_read_only_and_keeps_timeline():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    poses=np.repeat(np.eye(4)[None],5,axis=0);poses[:,:3,3]=[0.,0.,3.]
    before=poses.copy()
    result=audit.signature_pose(v,poses,5,1.)
    assert np.array_equal(result['frame_index'],np.arange(5))
    assert np.array_equal(poses,before)
    assert result['statistics']['surface_rms_acceleration_m_s2']['rms']==0
    assert result['predictions_modified'] is False


def test_jsonable_carries_full_arrays_and_statistics_not_only_summary():
    value={'x':np.array([0,1,2]),'y':np.float32(.5),'z':(np.int64(3),)}
    assert audit.jsonable(value)=={'x':[0,1,2],'y':.5,'z':[3]}


def test_fixed_qa_population_is_not_resampled_after_user_feedback():
    assert audit.EPISODES==(9,1,14)
    assert audit.BASELINE=='de62258a3f0ca1f12dd0a151c8fe96f0256ea3ba'
    assert audit.MAX_JPEG<=180000


def test_frontend_writable_origin_is_hash_bound_without_chmod(tmp_path):
    path=tmp_path/'frontend.json';path.write_bytes(b'{"frame":0}')
    before=path.stat().st_mode
    sources=audit.Sources()
    assert sources.json(path)=={'frame':0}
    sources.verify()
    assert path.stat().st_mode==before
    path.write_bytes(b'{"frame":1}')
    with pytest.raises(ValueError,match='Source changed'):sources.verify()


def test_sources_require_complete_expected_identity_not_sha_string(tmp_path):
    path=tmp_path/'owned.json';path.write_bytes(b'{}')
    pin=audit.identity(path,readonly=False)
    with pytest.raises(ValueError,match='identity differs'):audit.Sources().bind(path,pin['sha256'])
    assert audit.Sources().bind(path,pin)==path
