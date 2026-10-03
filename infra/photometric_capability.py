"""H100 public-only native photometric mechanism and repeated-input control.

No quality label, private scene, objective optimization, camera fit or object
prediction is read. A selected proposal is an existing native BODY block, not
an averaged pose. Six ordinary Body inferences test original/gamma/SHAM paths.
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np
import hand_synthetic_masks as masks
import keypoint_rgb_baseline as baseline
import photometric_render as manufacture
from world_reward.data import sha256
from world_reward.photometric_consistency import GAMMAS, gamma_rgb, geometric_medoid
from world_reward.prompt_selection import BoxDetection

native=baseline.native
BASE="validation/photometric_native_v1"
STAGES={"masks":"public_photometric_automatic_masks","body":"public_photometric_native_capability"}
FOLDERS={"masks":"automatic_masks","body":"capability_v1"}
BUDGETS={"masks":120,"body":180}
IMAGE=native.IMAGE_ID
MASK_IMAGE="sha256:53b33bc4b60e0e3e8f83b401775b4701b18eef54408fd585fbe3a5d376c042e1"
K=np.asarray(((1280.,0.,512.),(0.,1280.,384.),(0.,0.,1.)),np.float64)
WIDTH,HEIGHT=1024,768
FIXED_BLOCKS=("global_rot","pred_cam_t","hand_pose_params","shape_params","scale_params","expr_params")
FIXED_CONTROLS=np.r_[np.arange(6),np.arange(68,122),np.arange(136,204)]
PROPOSAL_KEYS=frozenset(native.BLOCKS)|{"vertices_camera_m","keypoints_camera_m","joints_camera_m","joint_global_rotations"}
ALL_BRANCHES=(("original",1.),("gamma_08",.8),("gamma_12",1.2),("sham_0",1.),("sham_1",1.),("sham_2",1.))


def require_fields(row,expected):
    if not isinstance(row,dict)or any(type(row.get(k))is not type(v)or row[k]!=v for k,v in expected.items()):raise ValueError("Exact public capability contract required")


def identity(path):return native.regular(path,immutable=True)


def helper_hashes():
    from world_reward import photometric_consistency
    return dict(capability=sha256(Path(__file__)),manufacture=sha256(Path(manufacture.__file__)),
        masks=sha256(Path(masks.__file__)),baseline=sha256(Path(baseline.__file__)),
        policy=sha256(Path(photometric_consistency.__file__)),native_helpers=baseline.helper_identities())


def public_rgb(root):
    records,receipt=manufacture.public_inputs(Path(root)/BASE/"inputs")
    if len(records)!=1 or records[0]["file"]!="frame_000.png":raise ValueError("One new original public RGB required")
    return records[0],receipt


def read_rgb(record,Image):
    with Image.open(record["path"])as image:
        if image.format!="PNG"or image.mode!="RGB"or image.size!=(WIDTH,HEIGHT):raise ValueError("Original RGB grid required")
        return np.array(image,dtype=np.uint8,copy=True)


def public_mask(root,revision):
    record,manifest=public_rgb(root);out=Path(root)/BASE/"automatic_masks";path=out/"report.json";receipt=identity(path)
    report=json.loads(path.read_text());require_fields(report,dict(stage=STAGES["masks"],status="pass",phase="complete",producer_revision=revision,
        script_sha256=sha256(Path(__file__)),source_helpers=helper_hashes(),image_id=MASK_IMAGE,network="none",private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        human_query="person.",actual_detector_calls=1,actual_sam2_calls=1,actual_sam2_image_encoder_calls=1,
        confidence=masks.CONFIDENCE,text_threshold=masks.TEXT_THRESHOLD,nms_iou=masks.NMS_IOU,ambiguity_margin=masks.AMBIGUITY_MARGIN,
        input_manifest_sha256=manifest["sha256"],input_manifest_bytes=manifest["bytes"],rgb_sha256=record["sha256"],
        all_cases_retained=True,source_inputs_assets_rehashed=True))
    if any(key in report for key in("error","error_type")):raise ValueError("Passing mask producer cannot retain errors")
    expected_assets={name:dict(path=str(Path(root)/"weights"/name),sha256=digest,bytes=size)for name,(digest,size)in masks.ASSETS.items()}
    require_fields(report,dict(model_assets=expected_assets))
    target=out/"human.png";mask=identity(target)
    if mask!=report.get("human_mask_identity"):raise ValueError("Frozen automatic mask bytes changed")
    if type(report.get("human_mask_pixels"))is not int or not 20<=report["human_mask_pixels"]<=WIDTH*HEIGHT:raise ValueError("Actual complete human mask required")
    if {p.name for p in out.iterdir()}!={"human.png","report.json"}:raise ValueError("Exact public automatic mask inventory required")
    record.update(human_mask_path=target,human_mask_pixels=report["human_mask_pixels"])
    return record,dict(manifest=manifest,automatic_mask_report=receipt,automatic_mask=mask)


def validate_proposal(data,anchor=None):
    if set(data)!=PROPOSAL_KEYS:raise ValueError("Complete compact native proposal required")
    for key,shape in native.BLOCKS.items():native.array(data[key],shape,"float32")
    if np.any(data["expr_params"])or np.any(data["mhr_model_params"][:3])or data["pred_cam_t"][2]<=0:raise ValueError("Relative-root, zero-face, positive-camera native contract required")
    if data["global_rot"].tobytes()!=data["mhr_model_params"][3:6].tobytes():raise ValueError("Actual returned native Euler controls required")
    for key,count in(("vertices_camera_m",18439),("keypoints_camera_m",308),("joints_camera_m",127)):
        if np.any(native.array(data[key],(count,3),"float32")[:,2]<=0):raise ValueError("Every native camera point must remain positiveZ")
    baseline.rotations(data["joint_global_rotations"])
    if anchor is not None:
        for key in FIXED_BLOCKS:
            if data[key].tobytes()!=anchor[key].tobytes():raise ValueError("Fixed camera/identity/hand/face input changed: "+key)
        if data["mhr_model_params"][FIXED_CONTROLS].tobytes()!=anchor["mhr_model_params"][FIXED_CONTROLS].tobytes():raise ValueError("Actual fixed native root/hand/expanded-scale controls changed")
    return data


def fixed_decode(torch,head,proposal,anchor):
    """Keep the original nativeBODY proposal, anchor every other input block.

    body133 includes six pass-through skeletal translations. This is not an
    arm-only rotation experiment or a promise that every world vertexZ is fixed.
    """
    blocks={k:anchor[k].copy()for k in native.BLOCKS};blocks["body_pose_params"]=proposal["body_pose_params"].copy()
    v,kp,j,c,r=baseline.shared_decode(torch,head,blocks,anchor)
    result=blocks|dict(vertices_camera_m=v,keypoints_camera_m=kp,joints_camera_m=j,mhr_model_params=c,joint_global_rotations=r)
    validate_proposal(result,anchor);return result


def parity(candidate,expected,*,exact=False):
    validate_proposal(candidate);validate_proposal(expected)
    errors={k:float(np.linalg.norm(candidate[k].astype(np.float64)-expected[k],axis=-1).max())for k in("vertices_camera_m","keypoints_camera_m","joints_camera_m")}
    errors.update(controls=float(np.max(np.abs(candidate["mhr_model_params"].astype(np.float64)-expected["mhr_model_params"]))),
        rotations=float(np.max(np.abs(candidate["joint_global_rotations"].astype(np.float64)-expected["joint_global_rotations"]))))
    if any(v>1e-5 or not np.isfinite(v)for v in errors.values()):raise ValueError("Independent native replay exceeds1e-5")
    byte_equal=all(candidate[k].dtype==expected[k].dtype and candidate[k].tobytes()==expected[k].tobytes()for k in PROPOSAL_KEYS)
    if exact and not byte_equal:raise ValueError("SHAM repeated input did not exactly reproduce frozen original native blocks/geometry")
    return dict(maximum_errors=errors,all_native_arrays_byte_equal=byte_equal)


def save_proposal(out,name,data):
    if not re.fullmatch(r"(?:raw|fixed)_(?:original|gamma_08|gamma_12|sham_[0-2])|selected_(?:baseline|sham|tta)",name):raise ValueError("Fixed capability artifact name required")
    validate_proposal(data);path=out/(name+".npz")
    with path.open("xb")as stream:np.savez_compressed(stream,**data)
    path.chmod(0o444);return dict(file=path.name,identity=identity(path))


def reread_proposals(out,rows,anchor):
    if not isinstance(rows,list)or len(rows)!=15:raise ValueError("All6raw/6fixed/3selected native artifacts required")
    values=[];names=[]
    for row in rows:
        name=row.get("file")
        if type(name)is not str or Path(name).name!=name or not name.endswith(".npz"):raise ValueError("Canonical compact artifact required")
        path=out/name
        if identity(path)!=row.get("identity"):raise ValueError("Frozen native artifact bytes changed")
        with np.load(path,allow_pickle=False)as saved:data={k:saved[k]for k in saved.files}
        values.append(validate_proposal(data,None if name.startswith("raw_")else anchor));names.append(name)
    if len(set(names))!=15 or {p.name for p in out.iterdir()}!={"report.json",*names}:raise ValueError("Exact capability output inventory required")
    return values


def run_masks(root,out,report,persist):
    record,manifest=public_rgb(root);report.update(input_manifest_sha256=manifest["sha256"],input_manifest_bytes=manifest["bytes"],rgb_sha256=record["sha256"],
        model_assets=masks.validate_assets(root,report["image_id"]),phase="automatic_detector_load");persist()
    import torch
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
    from PIL import Image
    if not torch.cuda.is_available():raise ValueError("Actual CUDA inference required")
    dist=metadata.distribution("SAM-2");direct=dist.read_text("direct_url.json")
    report["sam2_source"]=masks.sam2_source_identity(Path(sam2.__file__).parent,json.loads(direct)if direct else{});persist()
    rgb=read_rgb(record,Image);processor=AutoProcessor.from_pretrained(root/"weights/grounding_dino",local_files_only=True)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(root/"weights/grounding_dino",local_files_only=True).cuda().eval()
    report["phase"]="automatic_human_detection";report["detector_attempts"]+=1;persist()
    inputs=processor(images=Image.fromarray(rgb),text="person.",return_tensors="pt").to("cuda")
    with torch.inference_mode():result=processor.post_process_grounded_object_detection(detector(**inputs),inputs.input_ids,
        threshold=masks.CONFIDENCE,text_threshold=masks.TEXT_THRESHOLD,target_sizes=[(HEIGHT,WIDTH)])[0]
    report["actual_detector_calls"]+=1
    detections=[BoxDetection(tuple(b),float(s))for b,s in zip(result["boxes"].cpu().tolist(),result["scores"].cpu().tolist())]
    chosen=masks.select_person(detections,WIDTH,HEIGHT);report.update(box_xyxy=list(chosen.box),detector_score=chosen.score,candidate_count=len(detections));persist()
    del inputs,result,detector,processor;torch.cuda.empty_cache()
    predictor=SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",str(root/"weights/sam2/sam2.1_hiera_large.pt"),device="cuda",mode="eval"))
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        report["image_encoder_attempts"]+=1;persist();predictor.set_image(rgb);report["actual_sam2_image_encoder_calls"]+=1
        report["sam2_attempts"]+=1;persist();predicted,scores,_=predictor.predict(box=np.asarray(chosen.box,np.float32),multimask_output=False);report["actual_sam2_calls"]+=1
    mask=masks.binary_mask(predicted,scores,WIDTH,HEIGHT);path=out/"human.png"
    with path.open("xb")as stream:Image.fromarray(mask).save(stream,format="PNG")
    path.chmod(0o444);report.update(human_mask_pixels=int(np.count_nonzero(mask)),human_mask_identity=identity(path))
    if public_rgb(root)!=(record,manifest):raise ValueError("Original public RGB changed")
    if masks.validate_assets(root,report["image_id"])!=report["model_assets"]:raise ValueError("Pinned automatic detector assets changed")
    if masks.sam2_source_identity(Path(sam2.__file__).parent,json.loads(direct)if direct else{})!=report["sam2_source"]:raise ValueError("Installed automatic SAM2 source changed")
    require_fields(report,dict(detector_attempts=1,actual_detector_calls=1,image_encoder_attempts=1,actual_sam2_image_encoder_calls=1,sam2_attempts=1,actual_sam2_calls=1))


def reset_seed(torch):
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)


def run_body(root,out,report,persist,revision,*,deterministic_algorithms=True,public_input_reader=None,native_model_loader=None):
    if type(deterministic_algorithms)is not bool:raise ValueError("Explicit native runtime mode required")
    if public_input_reader is None:public_input_reader=public_mask
    if native_model_loader is None:native_model_loader=native.human.load_model
    record,inputs=public_input_reader(root,revision);report.update(public_inputs=inputs,phase="native_model_load");persist()
    if "torch"in sys.modules:raise ValueError("Strict CUBLAS setup must precede Torch")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    import torch
    from PIL import Image
    if not torch.cuda.is_available()or str(torch.__version__)!="2.5.1+cu124"or torch.version.cuda!="12.4":raise ValueError("Pinned actual native CUDA required")
    reset_seed(torch);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(deterministic_algorithms,warn_only=False)
    semantic_path=root/"results/mhr-finger-semantics-v4.json";semantic_id=native.regular(semantic_path)
    semantic=json.loads(semantic_path.read_text());native.regions_helper.require_semantic_report(semantic)
    if semantic.get("source_image_id")!=IMAGE:raise ValueError("Pinned semantic source image required")
    model,estimator,faces,model_source=native_model_loader(root,torch);model.eval();model.requires_grad_(False)
    if model.head_pose.enable_hand_model is not False:raise ValueError("Unmodified ordinary Body-only head required")
    load=model_source.get("checkpoint_loading")
    require_fields(load,dict(mode="strict_network_and_head_state_with_explicit_asset_buffer_retention",parameter_tensors_loaded=1101,unexpected_keys=[]))
    retained=load.get("retained_mhr_asset_buffer_names")
    if not isinstance(retained,list)or len(retained)!=113 or len(set(retained))!=113:raise ValueError("Exact113 immutable native rig buffers required")
    report["actual_native_hand_indices"]=model_source["hand_indices"]
    report.update(body_model=model_source,semantic_report_sha256=semantic_id["sha256"],torch_version=str(torch.__version__),CUDA_version=torch.version.cuda,
        seed=0,TF32=False,deterministic_algorithms=deterministic_algorithms,warn_only=False,CUBLAS_WORKSPACE_CONFIG=":4096:8",phase="photometric_body_inference");persist()
    camera=torch.as_tensor(K.astype(np.float32),device="cuda")[None];rgb=read_rgb(record,Image)
    human=native.joint.read_mask(record["human_mask_path"],Image)
    if int(np.count_nonzero(human))!=record["human_mask_pixels"]:raise ValueError("Frozen automatic mask area changed")
    box,prompt=native.human.derived_bbox(rgb,human);raw=[];fixed=[];artifacts=[];anchor=None
    for name,gamma in ALL_BRANCHES:
        if (torch.are_deterministic_algorithms_enabled()!=deterministic_algorithms
            or torch.is_deterministic_algorithms_warn_only_enabled()or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32):raise ValueError("Frozen actual native runtime mode changed")
        variant=gamma_rgb(rgb,gamma);row=dict(name=name,gamma=gamma,source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            transformed_RGB_sha256=hashlib.sha256(variant.tobytes()).hexdigest(),original_bbox=box.tolist(),original_mask=inputs["automatic_mask"],camera_K=K.tolist())
        if gamma==1. and variant.tobytes()!=rgb.tobytes():raise ValueError("Original/SHAM RGB identity must be byte-exact")
        reset_seed(torch);report["body_attempts"]+=1;report["active_branch"]=name;persist()
        with torch.inference_mode():predictions=estimator.process_one_image(img=variant,bboxes=box[None],masks=prompt,cam_int=camera,inference_type="body")
        report["body_calls_completed"]+=1;persist()
        if not isinstance(predictions,list)or len(predictions)!=1:raise ValueError("Exactly one actual native actor required")
        report["parity_head_attempts"]+=1;persist();values,errors=native.human.decode_prediction(torch,model,predictions[0],"body");report["parity_heads_completed"]+=1
        report["keypoint_head_attempts"]+=1;persist();keypoints,rotations=baseline.raw_keypoint_decode(torch,model.head_pose,predictions[0],values);report["keypoint_heads_completed"]+=1
        if not np.isclose(float(values["focal_length"]),1280.,atol=1e-4,rtol=1e-6):raise ValueError("Ordinary Body ignored public fixedK")
        proposal={k:values[k]for k in native.BLOCKS}|dict(vertices_camera_m=values["vertices_camera_m"],keypoints_camera_m=keypoints,joints_camera_m=values["joints_camera_m"],joint_global_rotations=rotations)
        validate_proposal(proposal);raw.append(proposal);artifacts.append(save_proposal(out,"raw_"+name,proposal))
        if anchor is None:anchor=proposal
        report["fixed_head_attempts"]+=1;persist();candidate=fixed_decode(torch,model.head_pose,proposal,anchor);report["fixed_heads_completed"]+=1
        fixed.append(candidate);artifacts.append(save_proposal(out,"fixed_"+name,candidate));row.update(native_forward_errors=errors,
            raw_identity=artifacts[-2],fixed_identity=artifacts[-1],fixed_native_blocks_verified=True)
        if name=="original":row["original_native_replay"]=parity(candidate,anchor)
        elif name.startswith("sham"):
            row["raw_SHAM_parity"]=parity(proposal,anchor,exact=True);row["fixed_SHAM_parity"]=parity(candidate,fixed[0],exact=True)
        report["branches"].append(row);persist();del predictions,values
    tta=geometric_medoid(np.stack([v["vertices_camera_m"]for v in fixed[:3]]));sham=geometric_medoid(np.stack([v["vertices_camera_m"]for v in fixed[3:]]))
    if sham.index!=0 or np.any(sham.scores)or np.any(sham.pairwise_squared_distances):raise ValueError("Repeated-input SHAM must not manufacture geometry change")
    for mode,candidate in(("baseline",fixed[0]),("sham",fixed[3+sham.index]),("tta",fixed[tta.index])):
        report["selected_replay_attempts"]+=1;persist();replay=fixed_decode(torch,model.head_pose,candidate,anchor);report["selected_replays_completed"]+=1
        check=parity(replay,candidate);artifacts.append(save_proposal(out,"selected_"+mode,candidate))
        report["selected_replays"].append(dict(mode=mode,native_replay=check,artifact=artifacts[-1]));persist()
    report.update(selection=dict(gammas=list(GAMMAS),selected_gamma_index=tta.index,selected_gamma=GAMMAS[tta.index],
        pairwise_squared_camera_vertex_distance_m2=tta.pairwise_squared_distances.tolist(),medoid_scores_m2=tta.scores.tolist(),
        existing_native_proposal_selected=True,geometry_averaged=False,accuracy_verified=False),
        SHAM=dict(index=sham.index,medoid_scores_m2=sham.scores.tolist(),native_arrays_byte_equal=True),artifacts=artifacts,
        fixed_input_blocks=list(FIXED_BLOCKS),fixed_native_control_indices=FIXED_CONTROLS.tolist(),
        native_body_block_scope="133 proposal slots; native first130 used, hands overwritten; includes6 skeletal translation controls130:136",
        hard249_bounds_gate_applied=False,original_bbox=box.tolist(),camera_K=K.tolist())
    reread_proposals(out,artifacts,anchor)
    if public_input_reader(root,revision)[1]!=inputs or native.regular(semantic_path)!=semantic_id:raise ValueError("Original RGB/mask/semantic bytes changed")
    if native.human.body._source_identity(root)!=model_source["inference_source_identity"]or native.human.body._body_assets(root)[1]!=model_source["body_assets"]:raise ValueError("Original Body model/source changed")
    require_fields(report,{k:6 for k in("body_attempts","body_calls_completed","parity_head_attempts","parity_heads_completed","keypoint_head_attempts","keypoint_heads_completed","fixed_head_attempts","fixed_heads_completed")}|dict(selected_replay_attempts=3,selected_replays_completed=3))
    report.update(actual_native_photometric_mechanism_verified=True,all_artifacts_frozen_and_reloaded=True,
        original_native_replay_verified=True,SHAM_exact_replay_verified=True,actual_body_inference=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument("--stage",choices=tuple(STAGES),required=True);args=parser.parse_args(argv)
    root=Path(os.environ.get("WR_ROOT",""));revision=os.environ.get("WR_CODE_REVISION","");image=os.environ.get("WR_IMAGE_ID","");out=root/BASE/FOLDERS[args.stage]
    if (platform.system()!="Linux"or root!=Path("/srv/scenesmith/world-reward")or os.geteuid()!=1000
        or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}or not re.fullmatch("[0-9a-f]{40}",revision)
        or image!=(MASK_IMAGE if args.stage=="masks"else IMAGE)or not out.is_dir()or any(out.iterdir())
        or out.resolve()!=out.absolute()or any(p.is_symlink()for p in(out,*out.parents))):raise ValueError("Fresh canonical offline H100 capability required")
    report=dict(stage=STAGES[args.stage],status="fail",phase="public_integrity",producer_revision=revision,image_id=image,
        script_sha256=sha256(Path(__file__)),source_helpers=helper_hashes(),network="none",budget_seconds=BUDGETS[args.stage],
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        quality_verified=False,accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,all_cases_retained=False,
        source_inputs_assets_rehashed=False,fitting_performed=False,camera_fit_performed=False,geometry_averaged=False,
        human_query="person.",confidence=masks.CONFIDENCE,text_threshold=masks.TEXT_THRESHOLD,nms_iou=masks.NMS_IOU,ambiguity_margin=masks.AMBIGUITY_MARGIN,
        detector_attempts=0,actual_detector_calls=0,image_encoder_attempts=0,actual_sam2_image_encoder_calls=0,sam2_attempts=0,actual_sam2_calls=0,
        body_attempts=0,body_calls_completed=0,parity_head_attempts=0,parity_heads_completed=0,keypoint_head_attempts=0,keypoint_heads_completed=0,
        fixed_head_attempts=0,fixed_heads_completed=0,selected_replay_attempts=0,selected_replays_completed=0,branches=[],selected_replays=[])
    started=time.perf_counter();path=out/"report.json"
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Public photometric stage exceeded frozen budget")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGETS[args.stage])
        try:
            persist()
            if args.stage=="masks":run_masks(root,out,report,persist)
            else:run_body(root,out,report,persist,revision)
            if helper_hashes()!=report["source_helpers"]:raise ValueError("Own full native helper sources changed")
            report.update(status="pass",phase="complete",all_cases_retained=True,source_inputs_assets_rehashed=True);report.pop("active_branch",None)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
