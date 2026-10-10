"""Tiny offline protocol tests; no actual models, GPU or dataset reads."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'infra'))
import contact_bvh_qualification as qa


def manifest():
    root=qa.ROOT; sid='synthetic_DEV_schema_only'; base=root/'results'/('form-hoi-external-predict-'+'a'*40)/sid
    row=lambda p,pin=None:dict(path=str(p),pin=dict(bytes=10,sha256='b'*64) if pin is None else pin)
    return dict(schema='world_reward.contact_bvh_qualification_input.v1',ground_truth_used=False,private_truth_read=False,
        prediction_base=str(base),input=row(Path('/srv/world-reward-data/form_hoi_external_dev_v1')/('c'*40)/sid/'inputs/input.json'),
        body_report=row(base/'body_depth/report.json'),object_report=row(base/'object/report.json'),
        hand_spec=row(root/'weights/cari4d/refinement/mhr_hand_surface_spec.npz',qa.HAND_SPEC_PIN.copy()))


def test_exact_public_manifest_and_mount_allowlist_only():
    value=manifest();base=qa.validate_manifest(value)
    code=qa.ROOT/'jobs'/('d'*40)/qa.ENTRY/'code'; out=qa.ROOT/'results'/('contact-bvh-qualification-'+'d'*40)
    mounts=qa.readonly_mounts(value,qa.ROOT/'results/probe-input.json',code,out)
    paths={str(p) for p,ro in mounts}
    assert str(base/'object/object.glb') in paths and str(base/'body_depth/body.npz') in paths
    assert len(mounts)==10 and mounts[-1]==(out,False)
    assert all(ro for _,ro in mounts[:-1])
    assert not any('rgb.mp4' in p or 'weights/sam3d' in p or 'eval_private' in p or 'track_1' in p for p in paths)


@pytest.mark.parametrize('key,value',[('ground_truth_used',True),('private_truth_read',True),
                                     ('schema','wrong'),('prediction_base','/tmp/unpinned')])
def test_malformed_or_oracle_manifest_fails(key,value):
    m=manifest();m[key]=value
    with pytest.raises(ValueError):qa.validate_manifest(m)


def test_private_or_unpinned_input_rejected():
    m=manifest();m['input']['path']=str(qa.ROOT/'results/eval_private/input.json')
    with pytest.raises(ValueError):qa.validate_manifest(m)
    m=manifest();m['hand_spec']['pin']['sha256']='0'*64
    with pytest.raises(ValueError):qa.validate_manifest(m)
    m=manifest();m['body_report']['pin']['bytes']=True
    with pytest.raises(ValueError):qa.validate_manifest(m)


def test_small_parity_arrays_finite_shape_tolerance_and_bytes():
    a=np.arange(6,dtype=np.float32).reshape(2,3)
    same=qa.compare_arrays(a,a.copy(),atol=1e-7)
    assert same['pass_tolerance'] and same['byte_equal'] and same['max_abs_error']==0
    assert not qa.compare_arrays(a,a+.01,atol=1e-7)['pass_tolerance']
    with pytest.raises(ValueError):qa.compare_arrays(a,np.arange(6),atol=1e-7)
    with pytest.raises(ValueError):qa.compare_arrays(a,a*np.nan,atol=1e-7)


def test_native_topology_hash_shape_and_dtype_sensitive():
    a=np.arange(6,dtype=np.int32).reshape(2,3)
    assert qa.array_sha256(a)==qa.array_sha256(a.copy())
    assert qa.array_sha256(a)!=qa.array_sha256(a.astype(np.int64))
    assert qa.array_sha256(a)!=qa.array_sha256(a.reshape(3,2))


def test_public_package_uses_actual_total_not_invented_total96(monkeypatch,tmp_path):
    value=manifest();base=Path(value['prediction_base']); manifest_path=tmp_path/'manifest.json'
    payloads={manifest_path:json.dumps(value).encode()}
    public=dict(schema='world_reward.external_rgb_input.v1',dataset='nvidia/form-hoi',split='development',
        sequence_id=base.name,total=96,original_frame_indices=list(range(96)),inference_ready=True,reference_inputs_present=False)
    payloads[Path(value['input']['path'])]=json.dumps(public).encode()
    for row,stage,names in [('body_report','body_depth',['body.npz']),('object_report','object',['object.glb','transform.json'])]:
        payloads[Path(value[row]['path'])]=json.dumps(dict(status='complete',stage=stage,
            ground_truth_used=False,private_truth_read=False,input_pin=value['input']['pin'],
            artifacts={name:dict(bytes=10,sha256='b'*64) for name in names})).encode()
    pins={manifest_path:dict(bytes=10,sha256='b'*64),**{Path(value[n]['path']):value[n]['pin']
        for n in ('input','body_report','object_report','hand_spec')}}
    monkeypatch.setattr(qa,'canonical',lambda p:Path(p))
    monkeypatch.setattr(qa,'identity',lambda p,*args,**kwargs:pins.get(Path(p),dict(bytes=10,sha256='b'*64)))
    monkeypatch.setattr(qa,'public_input_identity',lambda p,*args:pins.get(Path(p),dict(bytes=10,sha256='b'*64)))
    monkeypatch.setattr(Path,'read_bytes',lambda p:payloads[p])
    result,_,_=qa.load_inputs(manifest_path,pins[manifest_path])
    assert result==value
    public.pop('total');public['total96']=96
    payloads[Path(value['input']['path'])]=json.dumps(public).encode()
    with pytest.raises(ValueError,match='DEV96'):qa.load_inputs(manifest_path,pins[manifest_path])


def test_report_exclusive_and_readonly(tmp_path):
    out=tmp_path/'report.json';qa.seal_json(out,dict(status='pass'))
    assert not out.stat().st_mode&0o222
    with pytest.raises(ValueError):qa.seal_json(out,dict(status='other'))


def test_only_exact_hand_asset_can_be_owner_writable(monkeypatch,tmp_path):
    monkeypatch.setattr(qa,'ROOT',tmp_path)
    path=tmp_path/'weights/cari4d/refinement/mhr_hand_surface_spec.npz'
    path.parent.mkdir(parents=True);path.write_bytes(b'pinned fake asset fixture');path.chmod(0o644)
    monkeypatch.setattr(qa,'HAND_SPEC_PIN',qa.identity(path,readonly=False))
    assert qa.public_input_identity(path,100)==qa.HAND_SPEC_PIN
    assert path.stat().st_mode&0o777==0o644
    report=tmp_path/'report.json';report.write_text('{}');report.chmod(0o644)
    with pytest.raises(ValueError):qa.public_input_identity(report,100)
    for mode in (0o664,0o646):
        path.chmod(mode)
        with pytest.raises(ValueError,match='group/world'):qa.public_input_identity(path,100)


def test_hand_asset_mutation_or_wrong_pin_still_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(qa,'ROOT',tmp_path)
    path=tmp_path/'weights/cari4d/refinement/mhr_hand_surface_spec.npz'
    path.parent.mkdir(parents=True);path.write_bytes(b'fixture');path.chmod(0o644)
    monkeypatch.setattr(qa,'HAND_SPEC_PIN',qa.identity(path,readonly=False))
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='authentic'):qa.public_input_identity(path,100)


def test_qualification_frames_operator_and_no_fit_adoption_contract():
    assert qa.FRAMES==(0,47,95) and qa.NATIVE_MIN_TRIANGLE_AREA==.005
    root=Path(__file__).resolve().parents[1]
    assert all((root/p).is_file() for p in qa.HELPERS)
    source=(root/'infra/contact_bvh_qualification.py').read_text()
    assert 'native_C.point_face_dist_forward' in source and 'native_C.point_face_dist_backward' in source
    assert 'torch.optim.Adam' in source and 'production_adopted=False' in source
    shell=(root/'infra/run_contact_bvh_qualification.sh').read_text()
    assert '.world-reward-h100.lock' in shell and '--network none' in shell and 'env "$IMAGE"' in shell
