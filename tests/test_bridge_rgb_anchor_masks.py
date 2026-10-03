"""Automatic four-image model adapters on tiny generated arrays, no GPU."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"));monkeypatch.syspath_prepend(str(ROOT/"src"))
    spec=importlib.util.spec_from_file_location("bridge_masks_test",ROOT/"infra/bridge_rgb_anchor_masks.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def observations(gate,tmp_path,monkeypatch):
    # The real image-grid reader has its own test; this operator fixture uses
    # only generated arrays without storing a single source image locally.
    monkeypatch.setattr(gate,"read_rgb",lambda r:np.full((480,640,3),r["clip_id"],np.uint8))
    records=[dict(clip_id=i,frame_id=0,file=gate.binding.FILENAMES[i],sha256=str(i)*64,bytes=1,width=640,height=480,path=tmp_path/"unused") for i in range(4)]
    out=tmp_path/"masks";out.mkdir(mode=0o700); report=dict(detector_calls=0,sam2_calls=0,images=[]);calls=[]
    def detect(rgb,query):
        calls.append(("detect",int(rgb[0,0,0]),query));return (gate.BoxDetection((10.,10.,30.,30.),.9),)
    def segment(rgb,box):
        calls.append(("segment",int(rgb[0,0,0]),tuple(box)))
        mask=np.zeros((1,480,640),bool);mask[:,10:30,10:30]=True;return mask,np.array([.8],np.float32)
    return records,out,report,calls,detect,segment


def test_all_queries_automatic_and_detection_complete_before_any_segmentation(gate,observations):
    records,out,report,calls,detect,segment=observations
    rows=gate.observe(records,out,report,lambda:None,detect=detect,segment=segment)
    assert len(rows)==4 and report["detector_calls"]==report["sam2_calls"]==8
    assert [r[2] for r in calls[:8]]==["person.","bottle."]*4
    assert all(c[0]=="detect" for c in calls[:8]) and all(c[0]=="segment" for c in calls[8:])
    assert len(list(out.iterdir()))==8
    for row in rows:
        for label in ("person","object"):
            item=row[label];path=out/item["mask_file"]
            assert item["mask_pixels"]==400 and item["candidate_count"]==1
            assert not path.stat().st_mode&0o222 and gate.binding.identity(path)==dict(bytes=item["mask_bytes"],sha256=item["mask_sha256"])
            with Image.open(path) as image:
                value=np.asarray(image);assert image.mode=="L" and image.size==(640,480) and set(np.unique(value))=={0,255}


@pytest.mark.parametrize("fault",["missingperson","missingbottle","ambiguous","badbox","lowscore"])
def test_missing_ambiguous_target_abstains_without_any_fallback_mask(gate,observations,fault):
    records,out,report,calls,detect,segment=observations
    def bad(rgb,query):
        if int(rgb[0,0,0])!=2:return detect(rgb,query)
        if fault=="missingperson" and query=="person." or fault=="missingbottle" and query=="bottle.":return ()
        if fault=="ambiguous":return (gate.BoxDetection((1.,1.,10.,10.),.9),gate.BoxDetection((100.,100.,110.,110.),.89))
        if fault=="badbox":return (gate.BoxDetection((-1.,1.,10.,10.),.9),)
        if fault=="lowscore":return (gate.BoxDetection((1.,1.,10.,10.),.29),)
        return detect(rgb,query)
    with pytest.raises(ValueError):gate.observe(records,out,report,lambda:None,detect=bad,segment=segment)
    assert not list(out.iterdir()) and report["sam2_calls"]==0


@pytest.mark.parametrize("fault",["empty","shape","nan","masked"])
def test_segmentation_failure_retained_without_frame_dropping(gate,observations,fault):
    records,out,report,calls,detect,segment=observations
    def bad(rgb,box):
        masks,scores=segment(rgb,box)
        if int(rgb[0,0,0])==1:
            if fault=="empty":masks[:]=False
            elif fault=="shape":masks=masks[:,:,:639]
            elif fault=="nan":scores[0]=np.nan
            else:masks=np.ma.array(masks,mask=False)
        return masks,scores
    with pytest.raises(RuntimeError):gate.observe(records,out,report,lambda:None,detect=detect,segment=bad)
    assert len(list(out.iterdir()))==2 and report["detector_calls"]==8 and report["sam2_calls"]==3


def test_no_output_overwrite_and_fixed_original_primitives(gate,observations):
    records,out,report,calls,detect,segment=observations
    path=out/("person_"+records[0]["file"]);path.write_bytes(b"user work")
    with pytest.raises(FileExistsError):gate.observe(records,out,report,lambda:None,detect=detect,segment=segment)
    assert path.read_bytes()==b"user work"
    assert (gate.original.CONFIDENCE,gate.original.TEXT_THRESHOLD,gate.original.NMS_IOU,gate.original.AMBIGUITY_MARGIN)==(.3,.25,.7,.05)
    assert gate.BUDGET==120


def test_pins_and_measured_controls_precede_model_imports(gate):
    tree=ast.parse(Path(gate.__file__).read_text());run=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=="run")
    text=ast.unparse(run)
    assert text.index("binding.authenticate(")<text.index("import torch")
    assert text.index("binding.public_inputs(")<text.index("import torch")
    assert text.index("binding.selected_assets(")<text.index("import torch")
    assert text.index("binding.installed_sam2(")<text.index("AutoProcessor.from_pretrained(")
    assert "original.validate_assets" not in text and "direct_url" not in text and "render" not in text


def test_wrapper_exact_offline_image_owned_cleanup_and_no_private_cohort(gate):
    path=Path(gate.__file__).with_name("run_bridge_rgb_anchor_masks.sh")
    subprocess.run(["bash","-n",str(path)],check=True)
    text=path.read_text()
    assert "--network none" in text and "--user 0:0" in text and "123s docker run" in text
    assert "flock --nonblock 9 || exit 1" in text and 'exec 9<"$LOCK"' in text and "os.fstat(9)" in text
    assert "--read-only --cap-drop ALL --security-opt no-new-privileges" in text
    assert "--entrypoint /usr/bin/env" in text and "-i PATH=/opt/conda/bin:/usr/bin:/bin" in text
    assert gate.binding.IMAGE in text and "live=True" in text and "--no-trunc" in text
    assert "bridge_rgb_anchor_render" not in text and "eval_private" not in text and "src=$ROOT/validation,dst=" not in text
    assert "src=$DEST,dst=" not in text and "rmtree" not in text and "docker stop --time 3 \"$NAME\"" in text
    for name in gate.binding.FILENAMES:assert name in text


def test_unknown_argument_rejected_before_runtime(gate):
    with pytest.raises(SystemExit)as error:gate.main(["--episode","0"])
    assert error.value.code==2
