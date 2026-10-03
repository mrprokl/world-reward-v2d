"""Explicit-cohort H98 RGB observers, never fitting or private evaluation.

Existing native loaders/decoders/validators are called as functions; no legacy
stage main/public-input reader or renderer is executed. Model validity and ABI
replay are not geometric accuracy or final challenge eligibility.
"""
import argparse
from dataclasses import asdict
import gc
from importlib import metadata, util
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

import numpy as np
import root5_rgb_protocol as protocol
import hand_synthetic_masks as masks
import keypoint_rgb_baseline as body
import keypoint_rgb_dwpose as dw_helper
import depth_camera_support as depth_camera
from world_reward.prompt_selection import BoxDetection
from world_reward.data import sha256

native=body.native
smoke=dw_helper.smoke
STAGES={"masks":"public_root5_rgb_automatic_masks", "baseline":"public_root5_rgb_body_depth_first_rgb_identity_baseline",
        "dwpose":"public_root5_rgb_native_dwpose133_observations"}
FOLDERS={"masks":"automatic_masks","baseline":"baseline_v1","dwpose":"dwpose_v1"}
BUDGETS={"masks":180,"baseline":600,"dwpose":180}
QUERIES=(("human","person."),("object","bottle."))


def require_fields(row,expected):
    if not isinstance(row,dict)or any(type(row.get(k))is not type(v)or row[k]!=v for k,v in expected.items()):
        raise ValueError("Exact explicit-cohort producer contract required")


def cohort_contract(cohort):
    if not isinstance(cohort,protocol.PublicCohort)or(cohort.count,cohort.width,cohort.height)!=(15,native.WIDTH,native.HEIGHT):
        raise ValueError("Explicit preregistered compatible public cohort required")
    if not np.array_equal(np.asarray(cohort.fixed_K),native.CAMERA_K):raise ValueError("Same fixed camera ABI required")
    return asdict(cohort)


def source_identity():
    modules={"protocol":protocol,"mask_helper":masks,"baseline_helper":body,
             "dwpose_helper":dw_helper,"depth_support":depth_camera}
    return {"observer":sha256(Path(__file__))}|{k:sha256(Path(m.__file__))for k,m in modules.items()}|body.helper_identities()|{"dwpose_runtime":smoke.source_identity()}


def completed_masks(report,cohort):
    require_fields(report,dict(actual_detector_calls=2*cohort.count,actual_sam2_calls=2*cohort.count,
                               actual_sam2_image_encoder_calls=cohort.count))
    rows=report.get("records")
    if not isinstance(rows,list)or len(rows)!=cohort.count:raise ValueError("All ordered automatic masks required")
    for i,row in enumerate(rows):
        require_fields(row,dict(file=cohort.frame_name(i),clip_index=i//cohort.frames,frame_index=i%cohort.frames))
        for label,query in QUERIES:
            require_fields(row,{label+"_query":query,label+"_mask_file":Path(cohort.frame_name(i)).stem+"_"+label+".png"})
            if (not re.fullmatch("[0-9a-f]{64}",str(row.get(label+"_mask_sha256","")))or type(row.get(label+"_mask_bytes"))is not int
                    or row[label+"_mask_bytes"]<=0 or type(row.get(label+"_mask_pixels"))is not int
                    or not 0<row[label+"_mask_pixels"]<=cohort.width*cohort.height):raise ValueError("Original-grid mask evidence required")


def public_inputs(root,cohort,labels=("human","object")):
    """Read public RGB and requested automatic PNGs only; DWPose needs human only."""
    contract=cohort_contract(cohort);base=Path(root)/cohort.base
    records,manifest=protocol.public_inputs(base/"inputs",cohort)
    if labels not in (("human",),("human","object")):raise ValueError("Explicit public mask modalities required")
    path=base/"automatic_masks/report.json";receipt=protocol.identity(path);report=json.loads(path.read_text())
    require_fields(report,dict(stage=STAGES["masks"],status="pass",phase="complete",cohort=contract,frames=cohort.count,
        script_sha256=sha256(Path(__file__)),source_helpers=source_identity(),network="none",private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],human_query="person.",object_query="bottle.",
        input_manifest_sha256=manifest["sha256"],input_manifest_bytes=manifest["bytes"],all_cases_retained=True,
        actual_automatic_inference_verified=True,confidence=masks.CONFIDENCE,text_threshold=masks.TEXT_THRESHOLD,
        nms_iou=masks.NMS_IOU,ambiguity_margin=masks.AMBIGUITY_MARGIN))
    if not re.fullmatch("[0-9a-f]{40}",str(report.get("producer_revision",""))):raise ValueError("Immutable mask producer revision required")
    completed_masks(report,cohort)
    assets=report.get("model_assets")
    if not isinstance(assets,dict)or set(assets)!=set(masks.ASSETS):raise ValueError("Pinned detector/SAM asset inventory required")
    for name,(digest,size)in masks.ASSETS.items():
        require_fields(assets[name],dict(path=str(Path(root)/"weights"/name),sha256=digest,bytes=size))
    for record,row in zip(records,report["records"]):
        require_fields(row,dict(rgb_sha256=record["sha256"]))
        for label in labels:
            target=base/"automatic_masks"/row[label+"_mask_file"];identity=protocol.identity(target)
            if(identity["sha256"],identity["bytes"])!=(row[label+"_mask_sha256"],row[label+"_mask_bytes"]):raise ValueError("Automatic PNG bytes changed")
            record[label+"_mask_path"]=target;record[label+"_mask_pixels"]=row[label+"_mask_pixels"]
    if labels==("human","object")and{p.name for p in path.parent.iterdir()}!={"report.json",*[r[k+"_mask_path"].name for r in records for k in labels]}:
        raise ValueError("Only exact public masks may be exposed")
    return records,dict(manifest=manifest,automatic_masks=receipt)


def read_rgb(record,cohort,Image):
    with Image.open(record["path"])as image:
        if image.format!="PNG"or image.mode!="RGB"or image.size!=(cohort.width,cohort.height):raise ValueError("Original RGB decode required")
        return np.asarray(image).copy()


def read_public_masks(record,Image,labels=("human","object")):
    decoded=tuple(native.joint.read_mask(record[k+"_mask_path"],Image)for k in labels)
    if any(int(np.count_nonzero(value))!=record[k+"_mask_pixels"]for k,value in zip(labels,decoded)):
        raise ValueError("Original automatic mask pixel area differs")
    return decoded


def run_masks(root,out,cohort,report,persist):
    records,receipt=protocol.public_inputs(root/cohort.base/"inputs",cohort);assets=masks.validate_assets(root,report["image_id"])
    report.update(input_manifest_sha256=receipt["sha256"],input_manifest_bytes=receipt["bytes"],model_assets=assets,phase="detector_load");persist()
    import torch
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from PIL import Image
    if not torch.cuda.is_available():raise ValueError("Actual CUDA masks required")
    dist=metadata.distribution("SAM-2");direct=dist.read_text("direct_url.json")
    sam_source=masks.sam2_source_identity(Path(sam2.__file__).parent,json.loads(direct)if direct else{})
    report.update(sam2_source=sam_source,torch_version=str(torch.__version__));persist()
    processor=AutoProcessor.from_pretrained(root/"weights/grounding_dino",local_files_only=True)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(root/"weights/grounding_dino",local_files_only=True).cuda().eval()
    try:
        for item in records:
            row={k:item[k]for k in("file","clip_index","frame_index")}|dict(rgb_sha256=item["sha256"]);report["records"].append(row)
            rgb=Image.fromarray(read_rgb(item,cohort,Image));report["phase"]="automatic_detection"
            for label,query in QUERIES:
                entry=processor(images=rgb,text=query,return_tensors="pt").to("cuda")
                with torch.inference_mode():result=processor.post_process_grounded_object_detection(detector(**entry),entry.input_ids,
                    threshold=masks.CONFIDENCE,text_threshold=masks.TEXT_THRESHOLD,target_sizes=[(cohort.height,cohort.width)])[0]
                report["actual_detector_calls"]+=1
                boxes=[BoxDetection(tuple(b),float(s))for b,s in zip(result["boxes"].cpu().tolist(),result["scores"].cpu().tolist())]
                chosen=masks.select_person(boxes,cohort.width,cohort.height)
                row.update({label+"_query":query,label+"_candidate_count":len(boxes),label+"_box":list(chosen.box),label+"_detector_score":chosen.score})
                del entry,result;persist()
    finally:del detector,processor;torch.cuda.empty_cache()
    predictor=SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",str(root/"weights/sam2/sam2.1_hiera_large.pt"),device="cuda",mode="eval"))
    try:
        for item,row in zip(records,report["records"]):
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                predictor.set_image(read_rgb(item,cohort,Image));report["actual_sam2_image_encoder_calls"]+=1
                for label,_ in QUERIES:
                    predicted,scores,_=predictor.predict(box=np.array(row[label+"_box"],np.float32),multimask_output=False)
                    report["actual_sam2_calls"]+=1;mask=masks.binary_mask(predicted,scores,cohort.width,cohort.height)
                    filename=Path(item["file"]).stem+"_"+label+".png";target=out/filename
                    with target.open("xb")as handle:Image.fromarray(mask,mode="L").save(handle,format="PNG")
                    target.chmod(0o444);identity=protocol.identity(target)
                    row.update({label+"_mask_file":filename,label+"_mask_sha256":identity["sha256"],label+"_mask_bytes":identity["bytes"],
                                label+"_mask_pixels":int(np.count_nonzero(mask)),label+"_sam2_predicted_score":float(scores[0])});persist()
                predictor.reset_predictor()
    finally:del predictor;torch.cuda.empty_cache()
    if (records,receipt)!=protocol.public_inputs(root/cohort.base/"inputs",cohort)or assets!=masks.validate_assets(root,report["image_id"]):raise ValueError("RGB/model bytes changed")
    if sam_source!=masks.sam2_source_identity(Path(sam2.__file__).parent,json.loads(direct)if direct else{}):raise ValueError("SAM source changed")
    completed_masks(report,cohort)
    for row in report["records"]:
        for label,_ in QUERIES:
            identity=protocol.identity(out/row[label+"_mask_file"])
            if(identity["sha256"],identity["bytes"])!=(row[label+"_mask_sha256"],row[label+"_mask_bytes"]):raise ValueError("Mask output changed")
    if {p.name for p in out.iterdir()}!={"report.json",*[r[k+"_mask_file"]for r in report["records"]for k,_ in QUERIES]}:raise ValueError("Exact mask output inventory required")
    report.update(actual_automatic_inference_verified=True)


def run_baseline(root,out,cohort,report,persist):
    records,inputs=public_inputs(root,cohort);K=np.asarray(cohort.fixed_K,np.float64)
    asset,acquisition,asset_identity=native.depth_model.model_asset(root)
    model_path=root/"weights/mhr/mhr_model.pt";tool=root/"vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"
    native.regular(model_path,native.REFERENCE_MODEL_SHA256);native.regular(tool,native.CONVERTER_SHA256)
    semantic_path=root/"results/mhr-finger-semantics-v4.json";semantic_id=protocol.identity(semantic_path)
    semantic=json.loads(semantic_path.read_text());native.regions_helper.require_semantic_report(semantic)
    if semantic.get("source_image_id")!=native.IMAGE_ID:raise ValueError("Actual native semantic image required")
    report.update(public_inputs=inputs,acquisition_report=acquisition,MoGe_model_asset=asset_identity,semantic_report_sha256=semantic_id["sha256"],
                  model_sha256=native.REFERENCE_MODEL_SHA256,converter_sha256=native.CONVERTER_SHA256,phase="model_load");persist()
    if "torch"in sys.modules:raise ValueError("CUBLAS configuration must precede Torch")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    import torch
    import moge
    from moge.model import v2 as depth_module
    from moge.model.v2 import MoGeModel
    from moge.utils import geometry_torch
    from PIL import Image
    if not torch.cuda.is_available():raise ValueError("Actual offline CUDA baseline required")
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    dist=metadata.distribution("moge");direct=dist.read_text("direct_url.json")
    source=native.depth_model.installed_source(Path(moge.__file__).parent,json.loads(direct)if direct else{})
    if sha256(Path(geometry_torch.__file__))!=depth_camera.GEOMETRY_SHA or depth_module.recover_focal_shift is not geometry_torch.recover_focal_shift:raise ValueError("Native focal solver source differs")
    model,estimator,faces,body_source=native.human.load_model(root,torch)
    if model.head_pose.enable_hand_model:raise ValueError("Body head only; no wrist-centric head")
    joints=model.head_pose.mhr.get_joint_names();lbs=model.head_pose.mhr.get_lbsw()
    if joints!=semantic["joint_names"]or not isinstance(lbs,tuple)or len(lbs)!=2:raise ValueError("Actual native LBS/metadata differs")
    regions=native.regions_helper.lbs_regions(*(v.detach().cpu().numpy()for v in lbs),joints)
    report.update(body_model=body_source,MoGe_source=source,joint_names=joints,torch_version=str(torch.__version__),seed=0,TF32=False,
        CUBLAS_WORKSPACE_CONFIG=":4096:8",hand_regions_native_verified=True,
        hand_region_sha256={n:hashlib.sha256(regions[s]["vertex_mask"].tobytes()).hexdigest()for n,s in(("left","l"),("right","r"))})
    network=MoGeModel.from_pretrained(str(asset)).cuda().eval();camera=torch.as_tensor(K.astype(np.float32),device="cuda")[None]
    for record in records:
        report.update(phase="raw_inference",active_file=record["file"]);persist();rgb=read_rgb(record,cohort,Image)
        hm,om=read_public_masks(record,Image);box,prompt=native.human.derived_bbox(rgb,hm)
        with torch.inference_mode():predictions=estimator.process_one_image(img=rgb,bboxes=box[None],masks=prompt,cam_int=camera,inference_type="body")
        if not isinstance(predictions,list)or len(predictions)!=1:raise ValueError("Exactly one automatically selected Body actor required")
        values,parity=native.human.decode_prediction(torch,model,predictions[0],"body")
        report["body_calls_completed"]+=1;report["raw_parity_head_calls_completed"]+=1
        kp,rotation=body.raw_keypoint_decode(torch,model.head_pose,predictions[0],values);report["raw_keypoint_head_calls_completed"]+=1
        if not np.isclose(float(values["focal_length"]),cohort.prior_focal,atol=1e-4,rtol=1e-6):raise ValueError("Native Body ignored fixed public camera")
        tensor=torch.from_numpy(rgb).cuda().permute(2,0,1).float()/255
        depth=depth_camera.checked_depth_infer(torch,depth_module,network,tensor,float(np.degrees(2*np.arctan(cohort.width/(2*cohort.prior_focal)))),
            report["native_focal_solver_calls"],{k:record[k]for k in("file","clip_index","frame_index")});report["MoGe_calls_completed"]+=1
        z,points,valid,normalized=(depth[k][0].detach().cpu().numpy()for k in("depth","points","mask","intrinsics"))
        checks=native.joint.pointmap_contract(z,points,valid,normalized,K)
        silhouette,rendered=native.raster_camera_mesh(values["vertices_camera_m"],faces,K,cohort.width,cohort.height)
        raw={k:values[k]for k in native.BLOCKS}|dict(raw_depth=z,raw_points=points,validity=valid,rendered_depth=rendered.cpu().numpy(),
            silhouette=silhouette.cpu().numpy(),human_mask=hm>0,object_mask=om>0,raw_vertices_camera_m=values["vertices_camera_m"],
            raw_joints_camera_m=values["joints_camera_m"],raw_keypoints_camera_m=kp,raw_joint_global_rotations=rotation,human_faces=faces,camera_K=K.copy(),
            clip_index=np.array(record["clip_index"],np.int64),frame_index=np.array(record["frame_index"],np.int64))
        body.validate_raw(raw,record);report["raw_outputs"].append(native.save(out,"raw",record,raw))
        report["calls"].append(dict(file=record["file"],bbox_xyxy=box.tolist(),native_forward_errors=parity,pointmap_checks=checks,
                                   decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()));persist()
        del predictions,depth,tensor,raw,values
    del network;torch.cuda.empty_cache()
    raw_frames=native.frozen_rows(out,report["raw_outputs"],records,"raw",body.validate_raw)
    report.update(raw_frozen_before_shared=True,phase="shared_native_decode");persist();pairs=[]
    for record,raw in zip(records,raw_frames):
        first=raw_frames[record["clip_index"]*cohort.frames];v,kp,j,controls,r=body.shared_decode(torch,model.head_pose,raw,first)
        report["shared_head_calls_completed"]+=1
        pair=dict(raw_vertices_camera_m=raw["raw_vertices_camera_m"],shared_vertices_camera_m=v,raw_keypoints_camera_m=raw["raw_keypoints_camera_m"],
            shared_keypoints_camera_m=kp,raw_joints_camera_m=raw["raw_joints_camera_m"],shared_joints_camera_m=j,
            raw_joint_global_rotations=raw["raw_joint_global_rotations"],shared_joint_global_rotations=r,human_faces=faces,
            hand_mask_left=regions["l"]["vertex_mask"],hand_mask_right=regions["r"]["vertex_mask"],camera_K=K.copy(),pred_cam_t=raw["pred_cam_t"],
            raw_model_controls=raw["mhr_model_params"],shared_model_controls=controls,raw_shape_params=raw["shape_params"],shared_shape_params=first["shape_params"],
            raw_scale_params=raw["scale_params"],shared_scale_params=first["scale_params"],expression=np.zeros(72,np.float32),clip_index=raw["clip_index"],frame_index=raw["frame_index"])
        body.validate_pair(pair,record,raw);pairs.append(pair)
    body.validate_clip_constants(pairs,raw_frames)
    for record,pair in zip(records,pairs):report["paired_outputs"].append(native.save(out,"paired",record,pair))
    pairs=native.frozen_rows(out,report["paired_outputs"],records,"paired",body.validate_pair)
    native.validate_region_masks(pairs,report);report.update(paired_frozen_before_reference=True,phase="independent_reference_forward");persist()
    spec=util.spec_from_file_location("world_reward_root5_official_reference",tool);official=util.module_from_spec(spec);spec.loader.exec_module(official)
    reference=official.MHR(str(model_path),"cuda",chunk=16,precision="float32")
    if not np.array_equal(reference.model.character_torch.mesh.faces.detach().cpu().numpy(),faces):raise ValueError("Reference/native topology differs")
    controls=np.stack([p["shared_model_controls"]for p in pairs]);shapes=np.stack([p["shared_shape_params"]for p in pairs])
    vertices_mm,joints_m=reference.run(torch.tensor(controls[:,:136],dtype=torch.float64,device="cuda"),torch.tensor(np.c_[controls[:,136:],shapes],dtype=torch.float64,device="cuda"))
    report["official_reference_calls"]+=1
    report["reference_fidelity"]=native.reference_errors(vertices_mm.detach().cpu().numpy(),joints_m.detach().cpu().numpy(),
        np.stack([p["shared_vertices_camera_m"]for p in pairs]),np.stack([p["shared_joints_camera_m"]for p in pairs]),np.stack([p["pred_cam_t"]for p in pairs]))
    report["official_reference_per_frame_mean_mm"]=report["reference_fidelity"]["per_frame_mean_mm"]
    native.frozen_rows(out,report["raw_outputs"],records,"raw",body.validate_raw);native.frozen_rows(out,report["paired_outputs"],records,"paired",body.validate_pair)
    if public_inputs(root,cohort)!=(records,inputs):raise ValueError("Public RGB/masks changed")
    if (native.human.body._source_identity(root)!=body_source["inference_source_identity"]or native.human.body._body_assets(root)[1]!=body_source["body_assets"]
            or native.depth_model.model_asset(root)[1:]!=(acquisition,asset_identity)
            or native.depth_model.installed_source(Path(moge.__file__).parent,json.loads(direct)if direct else{})!=source):raise ValueError("Body/MoGe assets/source changed")
    native.regular(model_path,native.REFERENCE_MODEL_SHA256);native.regular(tool,native.CONVERTER_SHA256)
    if protocol.identity(semantic_path)!=semantic_id:raise ValueError("Native semantic receipt changed")
    if any(report[k]!=cohort.count for k in("body_calls_completed","MoGe_calls_completed","raw_parity_head_calls_completed","raw_keypoint_head_calls_completed","shared_head_calls_completed"))or report["official_reference_calls"]!=1:raise ValueError("All15 Body/depth/decode calls required")
    if {p.name for p in out.iterdir()}!={"raw","paired","report.json"}:raise ValueError("Exact baseline output inventory required")
    report.update(sources_assets_rechecked=True,conversion_fidelity_verified=True,actual_body_inference=True,actual_MoGe_inference=True,actual_shared_native_forward=True,
        body_inference_type="body",camera_K=K.tolist(),focal_fitted=False,scale_fit=False,inverse_fit_performed=False,
        identity_source="first_original_RGB_shape45_and_scale28_per_five_frame_clip",expression_zero=True,keypoint_count=308,joint_count=127)


def run_dwpose(root,out,cohort,report,persist,started):
    records,inputs=public_inputs(root,cohort,("human",));evidence=dw_helper.validate_smoke(root);assets=smoke.validate_assets(root)
    report.update(public_inputs=inputs,capability_evidence=evidence,assets=assets,dependencies=smoke.dependency_identity(),phase="private_prefix_install");persist()
    if any(name in sys.modules or util.find_spec(name)is not None for name in("onnxruntime","flatbuffers")):raise ValueError("Fresh disposable ORT runtime required")
    from PIL import Image
    with smoke.private_prefix(report,persist)as prefix:
        command=smoke.pip_argv(prefix,root);report["pip_argv"]=command;persist()
        subprocess.run(command,check=True,timeout=max(.001,BUDGETS["dwpose"]-(time.perf_counter()-started)),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        report["private_prefix_packages_installed"]=True;ort,origins=smoke.import_runtime(prefix);report.update(runtime_imports=origins,phase="native_source_load")
        source=root/smoke.acquisition.BASE/"source/onnxpose.py";spec=util.spec_from_file_location("world_reward_root5_native_onnxpose",source)
        implementation=util.module_from_spec(spec);spec.loader.exec_module(implementation)
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        session=ort.InferenceSession(str(root/smoke.acquisition.BASE/smoke.acquisition.ASSETS[0][0]),sess_options=options,providers=["CPUExecutionProvider"])
        session.disable_fallback();graph=smoke.session_metadata(session);smoke.validate_session(session,graph=graph);smoke.validate_options(session,ort)
        report.update(actual_sessions=1,graph=graph,phase="native_rgb_inference");persist();proxy=smoke.SessionProxy(session,report["calls"],persist)
        for record in records:
            rgb=read_rgb(record,cohort,Image);human,=read_public_masks(record,Image,("human",))
            box=smoke.actor_bbox(human);points,scores=implementation.inference_pose(proxy,box.copy(),np.ascontiguousarray(rgb))
            data=dict(keypoints=points,scores=scores,validity=smoke.validate_prediction(points,scores),bbox=box,
                      clip_index=np.array(record["clip_index"],np.int64),frame_index=np.array(record["frame_index"],np.int64))
            dw_helper.validate_prediction(data,record);filename=Path(record["file"]).stem+".npz";path=out/filename
            with path.open("xb")as stream:np.savez(stream,**data)
            path.chmod(0o444)
            report["records"].append({k:record[k]for k in("file","clip_index","frame_index")}|dict(rgb_sha256=record["sha256"],prediction_file=filename,
                prediction=smoke.identity(path),raw_simcc=report["calls"][-1]["raw_simcc"],positive_score_count=int(data["validity"].sum()),
                **{k:smoke.array_identity(data[k])for k in("keypoints","scores","validity","bbox")}));persist()
        del proxy,session;gc.collect();dw_helper.validate_artifacts(out,report["records"])
        if report["actual_sessions"]!=1 or len(report["calls"])!=cohort.count or any(not r["run_completed"]for r in report["calls"]):raise ValueError("One session/all15 native RGB calls required")
        if public_inputs(root,cohort,("human",))!=(records,inputs)or evidence!=dw_helper.validate_smoke(root)or assets!=smoke.validate_assets(root):raise ValueError("Public observations/assets/capability changed")
        report.update(final_inputs_source_assets_rehashed=True,native_cpu_abi_verified=True)
    if not report["private_prefix_removed"]:raise ValueError("Disposable runtime prefix not removed")


def main(argv=None,cohort=protocol.COHORT):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument("stage",choices=tuple(STAGES));args=parser.parse_args(argv)
    contract=cohort_contract(cohort);root=Path(os.environ.get("WR_ROOT",""));out=root/cohort.base/FOLDERS[args.stage]
    revision=os.environ.get("WR_CODE_REVISION","");image=os.environ.get("WR_IMAGE_ID","")
    if(platform.system()!="Linux"or root!=Path("/srv/scenesmith/world-reward")or root.resolve()!=root.absolute()or os.geteuid()!=1000
        or{p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}or not re.fullmatch("[0-9a-f]{40}",revision)
        or not out.is_dir()or any(out.iterdir())or out.resolve()!=out.absolute()or any(p.is_symlink()for p in(out,*out.parents))
        or(args.stage!="masks"and image!=native.IMAGE_ID)or(args.stage=="dwpose"and os.environ.get("CUDA_VISIBLE_DEVICES")!="")):
        raise ValueError("Fresh canonical offline explicit-cohort observer required")
    if args.stage=="baseline":(out/"raw").mkdir();(out/"paired").mkdir()
    report=dict(stage=STAGES[args.stage],status="fail",phase="public_integrity",cohort=contract,frames=cohort.count,producer_revision=revision,
        image_id=image,script_sha256=sha256(Path(__file__)),source_helpers=source_identity(),network="none",budget_seconds=BUDGETS[args.stage],
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        accuracy_verified=False,quality_verified=False,adoption_authorized=False,full_HOI_verified=False,all_cases_retained=False,
        records=[],calls=[],human_query="person.",object_query="bottle.",confidence=masks.CONFIDENCE,text_threshold=masks.TEXT_THRESHOLD,
        nms_iou=masks.NMS_IOU,ambiguity_margin=masks.AMBIGUITY_MARGIN,actual_detector_calls=0,actual_sam2_calls=0,actual_sam2_image_encoder_calls=0,
        raw_outputs=[],paired_outputs=[],native_focal_solver_calls=[],body_calls_completed=0,MoGe_calls_completed=0,raw_parity_head_calls_completed=0,
        raw_keypoint_head_calls_completed=0,shared_head_calls_completed=0,official_reference_calls=0,raw_frozen_before_shared=False,
        paired_frozen_before_reference=False,actual_sessions=0,private_prefix_packages_installed=False,private_prefix_removed=False,
        native_cpu_abi_verified=False,native_source_modified=False,own_feed_cast=False,channel_swap=False,full_image_fallback=False,
        raw_scores_clamped=False,confidence_threshold_applied=False,wrapper_neck134_used=False,global_image_modified=False,gpu_used=args.stage!="dwpose",
        fitting_performed=False,training_overlap_excluded=False,license_clearance_verified=False)
    started=time.perf_counter();path=out/"report.json"
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n")
            stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Whole bounded public observer exceeded declared stage budget")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGETS[args.stage])
        try:
            persist()
            if args.stage=="masks":run_masks(root,out,cohort,report,persist)
            elif args.stage=="baseline":run_baseline(root,out,cohort,report,persist)
            else:run_dwpose(root,out,cohort,report,persist,started)
            if source_identity()!=report["source_helpers"]:raise ValueError("Observer/protocol/native helper source changed")
            report.update(status="pass",phase="complete",all_cases_retained=True);persist()
        except Exception as error:report.update(error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
