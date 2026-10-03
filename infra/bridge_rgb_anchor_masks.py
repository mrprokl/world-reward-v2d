"""Four NEW RGB anchors → automatic person/bottle masks, never truth prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import bridge_frontend_bindings as binding
import hand_synthetic_masks as original
from world_reward.prompt_selection import BoxDetection

ENTRY = "run_bridge_rgb_anchor_masks"
STAGE = "bridge_rgb_anchor_automatic_masks"
BUDGET = 120


def read_rgb(record):
    import numpy as np
    from PIL import Image
    with Image.open(record["path"]) as image:
        binding.require(image.format == "PNG" and image.mode == "RGB" and image.size == (640,480), "Original RGB PNG grid differs")
        value = np.asarray(image).copy()
    binding.require(value.dtype == np.uint8 and value.shape == (480,640,3), "Original RGB dtype differs")
    return value


def observe(records,out,report,persist,*,detect,segment):
    """Injectable real model callbacks; all eight detections precede SAM masks."""
    import numpy as np
    from PIL import Image
    rows = []
    for record in records:
        rgb = read_rgb(record); row = {k:v for k,v in record.items() if k != "path"}
        row["decoded_rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest(); rows.append(row)
        for label,query in (("person","person."),("object","bottle.")):
            report.update(phase="automatic_detection",current_clip_id=record["clip_id"],current_query=query); persist()
            boxes = detect(rgb,query); report["detector_calls"] += 1
            chosen = original.select_person(boxes,640,480)
            row[label] = dict(query=query,candidate_count=len(boxes),box=list(chosen.box),detector_score=chosen.score)
    report["images"] = rows
    for record,row in zip(records,rows):
        rgb = read_rgb(record)
        binding.require(hashlib.sha256(rgb.tobytes()).hexdigest() == row["decoded_rgb_sha256"], "RGB changed between model stages")
        for label in ("person","object"):
            report.update(phase="automatic_segmentation",current_clip_id=record["clip_id"],current_query=row[label]["query"]); persist()
            masks,scores = segment(rgb,np.asarray(row[label]["box"],np.float32)); report["sam2_calls"] += 1
            mask = original.binary_mask(masks,scores,640,480)
            name = label+"_"+record["file"]; path = out/name
            fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
            with os.fdopen(fd,"wb") as stream:
                Image.fromarray(mask,mode="L").save(stream,format="PNG");stream.flush();os.fsync(stream.fileno())
            pin = binding.identity(path,2_000_000)
            row[label].update(mask_file=name,mask_sha256=pin["sha256"],mask_bytes=pin["bytes"],
                mask_pixels=int(np.count_nonzero(mask)),sam2_predicted_score=float(scores[0]));persist()
    binding.require(report["detector_calls"] == report["sam2_calls"] == 8, "Exactly eight automatic detector/SAM calls required")
    return rows


def run(root,code,out,report,persist):
    proof = binding.authenticate(root,code,ENTRY)
    records,frozen = binding.public_inputs(root,code/binding.INPUT_PINS)
    assets = binding.selected_assets(proof["selected_contract"],original.ASSETS)
    report.update(input_files={str(p.relative_to(root)) if p.is_relative_to(root) else str(p):pin for p,pin in frozen.items()},
        model_assets=assets,source_binding=proof["source_binding"],build_report_identity=proof["build_report_identity"],
        kernel_report_identity=proof["kernel_report_identity"],phase="model_load");persist()
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
    binding.require(torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(), "Actual Azure H100 required")
    report["sam2_source_binding"] = binding.installed_sam2(proof)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    processor = AutoProcessor.from_pretrained(binding.DEST/"weights/grounding_dino",local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(binding.DEST/"weights/grounding_dino",local_files_only=True).to("cuda").eval()
    predictor = None
    def detect(rgb,query):
        value = processor(images=Image.fromarray(rgb),text=query,return_tensors="pt").to("cuda")
        with torch.inference_mode():
            detected = processor.post_process_grounded_object_detection(detector(**value),value.input_ids,
                threshold=original.CONFIDENCE,text_threshold=original.TEXT_THRESHOLD,target_sizes=[(480,640)])[0]
        return tuple(BoxDetection(tuple(box),float(score)) for box,score in zip(detected["boxes"].cpu().tolist(),detected["scores"].cpu().tolist()))
    def segment(rgb,box):
        nonlocal predictor,detector,processor
        if predictor is None:
            # Same original two-stage loading order, never overlapping models.
            del detector,processor;torch.cuda.empty_cache()
            predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
                str(binding.DEST/"weights/sam2/sam2.1_hiera_large.pt"),device="cuda",mode="eval"))
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            predictor.set_image(rgb);masks,scores,_ = predictor.predict(box=box,multimask_output=False)
        predictor.reset_predictor();return masks,scores
    try: observe(records,out,report,persist,detect=detect,segment=segment)
    finally:
        del predictor;torch.cuda.empty_cache()
    binding.recheck(frozen)
    binding.require(binding.authenticate(root,code,ENTRY) == proof
        and binding.selected_assets(proof["selected_contract"],original.ASSETS) == assets,
        "Original source/assets changed during automatic masks")
    expected = {"report.json",*[label+"_"+name for label in ("person","object") for name in binding.FILENAMES]}
    binding.require({p.name for p in out.iterdir()} == expected, "Complete exclusive eight-mask output required")
    report.update(status="pass",phase="complete",original_inputs_sources_assets_rehashed=True,
        mask_files_rehashed={p.name:binding.identity(p,2_000_000) for p in out.iterdir() if p.suffix == ".png"})


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root = Path(os.environ["WR_ROOT"]);code = Path(os.environ["WR_CODE"]);rev = os.environ["WR_CODE_REVISION"]
    binding.require(sys.platform == "linux" and os.geteuid() == 0 and {p.name for p in Path("/sys/class/net").iterdir()} == {"lo"}
        and root == binding.ROOT and code == root/"jobs"/rev/ENTRY/"code" and Path(__file__).resolve() == code/"infra/bridge_rgb_anchor_masks.py"
        and os.environ["WR_IMAGE_ID"] == binding.IMAGE,"Actual source-bound VM02 offline new mask runtime required")
    out = binding.canonical(root/binding.BASE/"automatic_masks")
    binding.require(out.is_dir() and not any(out.iterdir()) and out.stat().st_mode & 0o777 == 0o700,"Fresh private automatic mask output required")
    report = dict(schema="world_reward.bridge_rgb_anchor_masks.v1",stage=STAGE,status="fail",phase="provenance",producer_revision=rev,
        script_sha256=binding.identity(Path(__file__))["sha256"],image_id=binding.IMAGE,frames=4,budget_seconds=BUDGET,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],network="none",accuracy_verified=False,
        adoption_authorized=False,replica_ready=False,license_eligibility_verified=False,training_overlap_verified=False,
        confidence=original.CONFIDENCE,text_threshold=original.TEXT_THRESHOLD,nms_iou=original.NMS_IOU,ambiguity_margin=original.AMBIGUITY_MARGIN,
        detector_calls=0,sam2_calls=0,images=[])
    fd = os.open(out/"report.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400);started=time.monotonic()
    with os.fdopen(fd,"w") as stream:
        def persist():
            report["elapsed_seconds"] = time.monotonic()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Frozen120s new anchor-mask budget exceeded")
        old = signal.signal(signal.SIGALRM,expired);term = signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,code,out,report,persist)
        except BaseException as error:
            report.update(status="fail",error_type=type(error).__name__,error="New automatic masks abstained; inspect recorded phase")
            raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,old);signal.signal(signal.SIGTERM,term);persist()
    print(json.dumps({k:report[k] for k in ("stage","status","frames","detector_calls","sam2_calls","elapsed_seconds")}))


if __name__ == "__main__": main()
