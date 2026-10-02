"""Nine RGB-only native field predictions and independent camera fits.

No truth, renderer metadata, masks or camera labels are accessible. Native
frontend fields and solver abstentions are frozen before private evaluation.
An accepted field fit is not evidence of metric camera accuracy or adoption.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np

import geocalib_frontend as native
from world_reward.perspective_calibration import fit_perspective_calibration
import world_reward.perspective_calibration as calibration

WIDTH, HEIGHT, CASES = 1024, 768, 9
SCHEMA = "world-reward-perspective-rgb-v1"
BASE = "validation/perspective_rgb_v1"
STAGE = "public_perspective_rgb_native_fields_and_own_camera_predictions"
D82_SHA = "056f2d84882b301a4e8410e3672ffa5b5c646e5f3b030d9d1621f4b9a3f6eb30"
D82_REV = "e1e8e92a63c690174f5bc40733939235fe46d04a"
NATIVE_SHA = "0121c396461bdcae773921cafdbbdb110fa4188ddd5b10d64378fe6d6c23c88f"
ASSET_RECEIPT_SHA = "77ed2ed2e65f7011eb930425f37020f57f226c25d1bca5be381119bcbf79b83c"
WEIGHT_SHA = "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac"
PREPROCESS = {"original_width": WIDTH, "original_height": HEIGHT, "resized_width": 426,
              "resized_height": 320, "crop_left": 5, "crop_top": 0}
FIELD_SHAPES = {"up_field": (2,320,416), "latitude_field": (1,320,416),
                "up_confidence": (320,416), "latitude_confidence": (320,416)}


def public_inputs(root):
    folder = native.assets.safe_path(Path(root)/BASE/"inputs")
    manifest = native.assets.safe_path(folder/"manifest.json")
    if not folder.is_dir() or not manifest.is_file(): raise ValueError("Require regular RGB-only public directory/manifest")
    data = json.loads(manifest.read_text())
    if not isinstance(data,dict) or set(data)!={"schema","images"} or data["schema"]!=SCHEMA or not isinstance(data["images"],list) or len(data["images"])!=CASES:
        raise ValueError("Require exact nine RGB-only manifest")
    records = []
    for index,row in enumerate(data["images"]):
        filename = f"case_{index:02d}.png"
        if (not isinstance(row,dict) or set(row)!={"file","sha256","width","height"} or row["file"]!=filename
                or type(row["width"]) is not int or type(row["height"]) is not int or (row["width"],row["height"])!=(WIDTH,HEIGHT)
                or not isinstance(row["sha256"],str) or not re.fullmatch(r"[0-9a-f]{64}",row["sha256"])):
            raise ValueError("Public RGB record order/grid/identity mismatch")
        image = native.assets.safe_path(folder/filename)
        if not image.is_file() or native.assets.digest(image)!=row["sha256"]: raise ValueError("Public RGB SHA differs")
        records.append({"case_index":index,"file":filename,"rgb_sha256":row["sha256"],"image_path":image})
    if {p.name for p in folder.iterdir()}!={"manifest.json",*[r["file"] for r in records]}:
        raise ValueError("Public directory contains non-RGB extras")
    return records,{"public_inputs_sha256":native.assets.digest(manifest),"public_inputs_bytes":manifest.stat().st_size}


def validate_d82(root, binding):
    path = native.assets.safe_path(Path(root)/native.REPORT)
    if not path.is_file() or native.assets.digest(path)!=D82_SHA: raise ValueError("Require frozen actual D82 receipt")
    data = json.loads(path.read_text())
    expected = {"stage":"standalone_geocalib_native_frontend_smoke","status":"pass","phase":"complete",
                "code_revision":D82_REV,"script_sha256":NATIVE_SHA,"image_id":native.IMAGE_ID,
                "source_revision":native.assets.SOURCE_REV,"bit_exact_replay":True,"actual_frontend_calls":2,
                "state_load_strict":True,"state_all_finite":True,"parameter_keys":748,"buffer_keys":141,
                "state_key_mapping":"none","missing_keys":[],"unexpected_keys":[],"assets_rechecked":True,
                "full_geocalib_package_imported":False,"ground_truth_used":False,"private_truth_read":False,
                "challenge_inputs_used":False,"oracle_modes":[],"accuracy_verified":False}
    if any(type(data.get(k)) is not type(v) or data.get(k)!=v for k,v in expected.items()):
        raise ValueError("D82 native execution contract differs")
    if (data.get("assets")!=binding or binding.get("receipt_sha256")!=ASSET_RECEIPT_SHA
            or binding.get("files",{}).get("geocalib-pinhole.tar",{}).get("sha256")!=WEIGHT_SHA
            or native.assets.digest(Path(native.__file__))!=NATIVE_SHA
            or len(data.get("state_inventory",[]))!=889 or len(data.get("checkpoint_state_keys",[]))!=889):
        raise ValueError("D82 source/strict inventory/asset binding differs")
    return {"sha256":D82_SHA,"code_revision":D82_REV,"native_script_sha256":NATIVE_SHA}


def preprocess(rgb, torch):
    if rgb.shape!=(HEIGHT,WIDTH,3) or rgb.dtype!=np.uint8: raise ValueError("Require original-grid RGB uint8")
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()[None]/255.
    tensor = torch.nn.functional.interpolate(tensor,size=(320,426),mode="bilinear",align_corners=False,antialias=True)
    return tensor[:,:,:,5:421].contiguous()


def validate_fields(fields):
    if not isinstance(fields,dict) or set(fields)!=set(FIELD_SHAPES): raise ValueError("Unexpected native field keys")
    summary = {}
    for name,shape in FIELD_SHAPES.items():
        value = fields[name]
        if np.ma.isMaskedArray(value) or not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():
            raise ValueError("Native field shape/dtype/finite failure: "+name)
        low,high = float(value.min()),float(value.max())
        if (name.endswith("confidence") and not 0<=low<=high<=1) or (name=="latitude_field" and not -np.pi/2<=low<=high<=np.pi/2):
            raise ValueError("Native field bounds failure: "+name)
        summary[name] = {"min":low,"max":high,"mean":float(value.mean()),"shape":list(shape)}
    error = float(np.abs(np.linalg.norm(fields["up_field"],axis=0)-1).max())
    if error>32*np.finfo(np.float32).eps: raise ValueError("Up field not native unit-normalized")
    summary["up_unit_max_error"] = error
    return summary


def original_camera(result):
    """Analytical processed-integer -> original-edge transport, no truth fit."""
    K = np.asarray(result.K,dtype=np.float64)
    if K.shape!=(3,3) or not np.isfinite(K).all() or not np.isfinite(result.focal_px) or result.focal_px<=0:
        raise ValueError("Invalid solver camera")
    sx,sy = 426/WIDTH,320/HEIGHT
    expected = np.array([[sx*result.focal_px,0.,208-.5],[0.,sy*result.focal_px,160-.5],[0.,0.,1.]])
    if not np.allclose(K,expected,rtol=1e-12,atol=1e-12): raise ValueError("Solver camera/grid transport differs")
    original = np.array([[K[0,0]/sx,0.,(K[0,2]+5+.5)/sx],
                         [0.,K[1,1]/sy,(K[1,2]+.5)/sy],[0.,0.,1.]])
    if not np.allclose(original,[[result.focal_px,0,WIDTH/2],[0,result.focal_px,HEIGHT/2],[0,0,1]],rtol=1e-12,atol=1e-12):
        raise ValueError("Original camera is not shared focal/center")
    return original


def run(root,out,report,persist):
    records,inputs = public_inputs(root); target,binding = native.validate_assets(root)
    report.update(inputs=inputs,d82=validate_d82(root,binding),assets=binding); persist()
    import torch
    from PIL import Image
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0): raise RuntimeError("Require H100")
    report.update(torch_version=str(torch.__version__),cuda_version=torch.version.cuda,device_name=torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    native.seed_all(torch);model=native.build_frontend(target,torch)
    checkpoint=torch.load(target/"geocalib-pinhole.tar",map_location="cpu",weights_only=True)
    loading={};native.strict_checkpoint(model,checkpoint,torch,loading);del checkpoint
    if (not loading["state_load_strict"] or len(loading["state_inventory"])!=889 or loading["state_key_mapping"]!="none"):
        raise RuntimeError("Current strict model state differs from D82")
    report["checkpoint_load"]={k:loading[k] for k in ("state_load_strict","state_all_finite","parameter_keys","buffer_keys","state_key_mapping")}
    model=model.eval().cuda();torch.cuda.reset_peak_memory_stats()
    for row in records:
        report.update(phase="frontend",active_case=row["case_index"]);persist()
        with Image.open(row["image_path"]) as image:
            if image.mode!="RGB" or image.size!=(WIDTH,HEIGHT): raise ValueError("Original RGB PNG mode/grid differs")
            rgb=np.asarray(image).copy()
        image=preprocess(rgb,torch).cuda();native.seed_all(torch);report["seeds_reset_calls"]+=1
        with torch.inference_mode():fields=model({"image":image})
        torch.cuda.synchronize();report["actual_frontend_calls"]+=1
        values={k:v[0].detach().cpu().numpy().copy() for k,v in fields.items()};del fields,image
        field_summary=validate_fields(values);report["phase"]="own_cpu_calibration";persist()
        fit=fit_perspective_calibration(values["up_field"],values["latitude_field"],values["up_confidence"],values["latitude_confidence"],**PREPROCESS)
        K=original_camera(fit);name=Path(row["file"]).stem+".npz";path=out/name
        with path.open("xb") as stream:np.savez_compressed(stream,**values)
        path.chmod(0o444)
        report["outputs"].append({"case_index":row["case_index"],"file":row["file"],"rgb_sha256":row["rgb_sha256"],
            "fields_file":name,"fields_sha256":native.assets.digest(path),"fields_bytes":path.stat().st_size,
            "fields_summary":field_summary,"camera_K":K.tolist(),"calibration":fit.to_dict()});persist()
    if public_inputs(root)!=(records,inputs) or native.validate_assets(root)[1]!=binding: raise RuntimeError("Public/source assets changed")
    validate_d82(root,binding)
    if report["actual_frontend_calls"]!=CASES or report["seeds_reset_calls"]!=CASES or len(report["outputs"])!=CASES:
        raise RuntimeError("Exactly nine seeded native calls and preserved cases required")
    if any(k=="geocalib" or k.startswith("geocalib.") for k in sys.modules):raise RuntimeError("Forbidden full package import")
    report.update(status="pass",phase="complete",frames=CASES,all_cases_retained=True,inputs_assets_rechecked=True,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),accepted_cases=sum(r["calibration"]["accepted"] for r in report["outputs"]),
                  full_geocalib_package_imported=False,perspective_fields_imported=False,lm_optimizer_imported=False)
    report.pop("active_case",None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=native.assets.safe_path(Path(os.environ["WR_ROOT"]));out=native.assets.safe_path(root/BASE/"predictions_v1")
    revision=os.environ.get("WR_CODE_REVISION","")
    if (platform.system()!="Linux" or root!=Path("/srv/scenesmith/world-reward") or not root.is_dir()
            or not re.fullmatch(r"[0-9a-f]{40}",revision) or os.environ.get("WR_IMAGE_ID")!=native.IMAGE_ID
            or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}
            or not out.is_dir() or any(out.iterdir())):raise ValueError("Require fresh reserved offline immutable prediction run")
    report={"stage":STAGE,"status":"fail","phase":"prerequisites","outputs":[],"actual_frontend_calls":0,"seeds_reset_calls":0,
        "code_revision":revision,"producer_revision":revision,"image_id":native.IMAGE_ID,"script_sha256":native.assets.digest(Path(__file__)),
        "native_helper_sha256":native.assets.digest(Path(native.__file__)),"solver_sha256":native.assets.digest(Path(calibration.__file__)),
        "preprocess":{**PREPROCESS,"output_width":416,"output_height":320,"resize":"bilinear_antialias_align_corners_false","crop":"center_left5"},
        "network":"none","ground_truth_used":False,"private_truth_read":False,"challenge_inputs_used":False,"oracle_modes":[],
        "accuracy_verified":False,"adoption_authorized":False,"budget_seconds":180,"original_camera_convention":"edge_center_W/2_H/2"}
    started=time.perf_counter();path=out/"report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,indent=2,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Whole nine-case inference exceeds180s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(180)
        try:persist();run(root,out,report,persist)
        except Exception as error:report.update(status="fail",error_type=type(error).__name__,error=str(error)[:1000]);raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
