"""Public-file/source/grid contracts only; no Torch/network/model inference."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def infer(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/"infra"
    monkeypatch.syspath_prepend(str(infra));monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec=importlib.util.spec_from_file_location("test_perspective_rgb_infer",infra/"perspective_rgb_infer.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def public(infer,tmp_path):
    folder=tmp_path/infer.BASE/"inputs";folder.mkdir(parents=True);images=[]
    for index in range(9):
        file=f"case_{index:02d}.png";(folder/file).write_bytes(f"tiny RGB {index}".encode())
        images.append({"file":file,"sha256":infer.native.assets.digest(folder/file),"width":1024,"height":768})
    (folder/"manifest.json").write_text(json.dumps({"schema":infer.SCHEMA,"images":images}))
    return tmp_path,folder


def test_exact_public_nine_only_no_private_read(infer,public):
    root,folder=public;(root/infer.BASE/"eval_private").mkdir()
    (root/infer.BASE/"eval_private"/"truth.npz").write_bytes(b"never read")
    records,hashes=infer.public_inputs(root)
    assert [r["case_index"] for r in records]==list(range(9))
    assert [r["file"] for r in records]==[f"case_{i:02d}.png" for i in range(9)]
    assert all(r["image_path"].parent==folder for r in records)
    assert hashes["public_inputs_sha256"]==infer.native.assets.digest(folder/"manifest.json")


@pytest.mark.parametrize("bad",["schema","extra_gt","count","file_order","traversal","grid","gridbool","sha","rgb_changed","extra_file","symlink"])
def test_invalid_public_firewall(infer,public,bad):
    root,folder=public;path=folder/"manifest.json";data=json.loads(path.read_text())
    if bad=="schema":data["schema"]="world-reward-joint-rgb-v1"
    if bad=="extra_gt":data["images"][0]["camera_K"]=[1]
    if bad=="count":data["images"].pop()
    if bad=="file_order":data["images"][0]["file"]="case_01.png"
    if bad=="traversal":data["images"][0]["file"]="../eval_private/truth.npz"
    if bad=="grid":data["images"][0]["width"]=512
    if bad=="gridbool":data["images"][0]["height"]=True
    if bad=="sha":data["images"][0]["sha256"]="0"*64
    if bad=="rgb_changed":(folder/"case_00.png").write_bytes(b"changed")
    if bad=="extra_file":(folder/"calibration.npz").write_bytes(b"private")
    if bad=="symlink":
        p=folder/"case_00.png";p.rename(folder/"other");p.symlink_to(folder/"other")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):infer.public_inputs(root)


def d82_fixture(infer,tmp_path,monkeypatch):
    root=tmp_path;path=root/infer.native.REPORT;path.parent.mkdir()
    binding={"receipt_sha256":infer.ASSET_RECEIPT_SHA,"files":{"geocalib-pinhole.tar":{"sha256":infer.WEIGHT_SHA}}}
    data={"stage":"standalone_geocalib_native_frontend_smoke","status":"pass","phase":"complete",
        "code_revision":infer.D82_REV,"script_sha256":infer.NATIVE_SHA,"image_id":infer.native.IMAGE_ID,
        "source_revision":infer.native.assets.SOURCE_REV,"bit_exact_replay":True,"actual_frontend_calls":2,
        "state_load_strict":True,"state_all_finite":True,"parameter_keys":748,"buffer_keys":141,
        "state_key_mapping":"none","missing_keys":[],"unexpected_keys":[],"assets_rechecked":True,
        "full_geocalib_package_imported":False,"ground_truth_used":False,"private_truth_read":False,
        "challenge_inputs_used":False,"oracle_modes":[],"accuracy_verified":False,"assets":binding,
        "state_inventory":[{}]*889,"checkpoint_state_keys":[str(i) for i in range(889)]}
    path.write_text(json.dumps(data));monkeypatch.setattr(infer,"D82_SHA",infer.native.assets.digest(path))
    return root,path,data,binding


def test_actual_d82_binding_before_gpu(infer,tmp_path,monkeypatch):
    root,path,_,binding=d82_fixture(infer,tmp_path,monkeypatch)
    assert infer.validate_d82(root,binding)["sha256"]==infer.native.assets.digest(path)


@pytest.mark.parametrize("bad",["receipt_bytes","source","strict","calls","boolean","remap","buffers","inventory","asset","weight"])
def test_d82_bad_proof_rejected(infer,tmp_path,monkeypatch,bad):
    root,path,data,binding=d82_fixture(infer,tmp_path,monkeypatch)
    if bad=="receipt_bytes":path.write_text("{}");
    if bad=="source":data["script_sha256"]="0"*64
    if bad=="strict":data["state_load_strict"]=False
    if bad=="calls":data["actual_frontend_calls"]=3
    if bad=="boolean":data["bit_exact_replay"]=1
    if bad=="remap":data["state_key_mapping"]="remove_second_component_once"
    if bad=="buffers":data["buffer_keys"]=140
    if bad=="inventory":data["state_inventory"].pop()
    if bad=="asset":binding["receipt_sha256"]="0"*64
    if bad=="weight":binding["files"]["geocalib-pinhole.tar"]["sha256"]="0"*64
    if bad!="receipt_bytes":
        path.write_text(json.dumps(data));monkeypatch.setattr(infer,"D82_SHA",infer.native.assets.digest(path))
    with pytest.raises(ValueError):infer.validate_d82(root,binding)


class Tensor:
    def __init__(self,array):self.array=array
    def float(self):return Tensor(self.array.astype(np.float32))
    def __getitem__(self,key):return Tensor(self.array[key])
    def __truediv__(self,value):return Tensor(self.array/value)
    def contiguous(self):return Tensor(np.ascontiguousarray(self.array))


def test_native_preprocess_exact_resize_crop_rgb_without_extra_calls(infer):
    calls=[]
    def interpolate(tensor,**kwargs):
        calls.append((tensor.array.copy(),kwargs))
        y,x=np.indices((320,426));return Tensor(np.broadcast_to(x,(1,3,320,426)).astype(np.float32).copy())
    torch=SimpleNamespace(from_numpy=lambda a:Tensor(a),nn=SimpleNamespace(functional=SimpleNamespace(interpolate=interpolate)))
    rgb=np.zeros((768,1024,3),np.uint8);rgb[:,:,0]=255
    result=infer.preprocess(rgb,torch).array
    assert result.shape==(1,3,320,416) and result.flags.c_contiguous
    assert np.array_equal(result[0,0,0],np.arange(5,421))
    original,options=calls[0]
    assert options==dict(size=(320,426),mode="bilinear",align_corners=False,antialias=True)
    assert original.shape==(1,3,768,1024) and original[0,0,0,0]==1 and original[0,1,0,0]==0


def test_preprocess_rejects_not_original_rgb(infer):
    with pytest.raises(ValueError):infer.preprocess(np.zeros((320,416,3),np.uint8),None)


def fields():
    up=np.zeros((2,320,416),np.float32);up[1]=-1
    return {"up_field":up,"latitude_field":np.zeros((1,320,416),np.float32),
            "up_confidence":np.full((320,416),.5,np.float32),"latitude_confidence":np.full((320,416),.2,np.float32)}


def test_native_fields_source_grid_and_summaries(infer):
    data=fields();report=infer.validate_fields(data)
    assert report["up_unit_max_error"]==0 and report["up_confidence"]["mean"]==.5
    assert set(data)==set(infer.FIELD_SHAPES)


@pytest.mark.parametrize("bad",["shape","missing","extra","nan","dtype","conf","latitude","unit","masked"])
def test_native_fields_bad_contract(infer,bad):
    data=fields()
    if bad=="shape":data["latitude_field"]=data["latitude_field"][0]
    if bad=="missing":data.pop("up_field")
    if bad=="extra":data["camera"]=np.eye(3)
    if bad=="nan":data["latitude_field"][0,0,0]=np.nan
    if bad=="dtype":data["up_confidence"]=data["up_confidence"].astype(np.float64)
    if bad=="conf":data["up_confidence"][0,0]=1.01
    if bad=="latitude":data["latitude_field"][0,0,0]=2
    if bad=="unit":data["up_field"][:,0,0]=0
    if bad=="masked":data["up_field"]=np.ma.array(data["up_field"],mask=False)
    with pytest.raises(ValueError):infer.validate_fields(data)


@pytest.mark.parametrize("focal",[1280.,900.,1600.])
def test_original_edge_camera_transport_not_gt_correction(infer,focal):
    result=SimpleNamespace(focal_px=focal,K=((426/1024*focal,0,207.5),(0,320/768*focal,159.5),(0,0,1)))
    camera=infer.original_camera(result)
    assert camera==pytest.approx(np.array([[focal,0,512],[0,focal,384],[0,0,1]]))


def test_mismatched_pixel_center_is_not_silently_repaired(infer):
    result=SimpleNamespace(focal_px=1280.,K=((532.5,0,208),(0,1280/2.4,160),(0,0,1)))
    with pytest.raises(ValueError):infer.original_camera(result)


def test_wrapper_scoped_public_mounts_and_budget(infer):
    source=Path(infer.__file__).with_name("run_perspective_rgb_infer.sh").read_text()
    assert "--network none" in source and "183s docker run" in source and "--memory 32g --cpus 4" in source
    assert "eval_private" not in source and "automatic_masks" not in source
    assert 'src=$BASE/inputs,dst=$BASE/inputs,readonly' in source and 'src=$OUT,dst=$OUT' in source
    assert "geocalib-frontend-smoke-v1.json" in source and infer.native.IMAGE_ID in source
    assert "-L \"$ROOT/validation\"" in source and "predictions_v1" in source
