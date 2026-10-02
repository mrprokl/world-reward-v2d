"""Tiny fifteen-frame input/mask contracts; no native GPU model execution."""
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/"infra"
    monkeypatch.syspath_prepend(str(infra));monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec=importlib.util.spec_from_file_location("own_identity_masks",infra/"identity_rgb_masks.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


@pytest.fixture
def inputs(module,tmp_path):
    directory=tmp_path/"inputs";directory.mkdir();records=[]
    for index in range(15):
        clip,frame=divmod(index,5);name=f"clip_{clip:02d}_frame_{frame:03d}.png"
        path=directory/name;path.write_bytes(("tiny_rgb_only_"+str(index)).encode())
        records.append(dict(file=name,sha256=module.masks.identity(path)["sha256"],width=1024,height=768))
    manifest=dict(schema=module.SCHEMA,images=records)
    def save():(directory/"manifest.json").write_text(json.dumps(manifest))
    save();return directory,manifest,save


def test_fifteen_original_rgb_records_and_no_global_module_mutation(module,inputs):
    directory,_,_=inputs;rows,receipt=module.validate_inputs(directory)
    assert len(rows)==15 and receipt["sha256"]==module.masks.identity(directory/"manifest.json")["sha256"]
    assert [(r["clip_index"],r["frame_index"]) for r in rows]==[(c,f) for c in range(3) for f in range(5)]
    assert module.QUERIES==(("human","person."),("object","bottle."))
    assert module.masks.SCHEMA=="world-reward-hands-rgb-inputs-v1"


@pytest.mark.parametrize("fault",["schema","top_private","record_private","extra","count","order","booldim","sha","tamper","symlink"])
def test_input_firewall_before_native_assets_or_private(module,inputs,monkeypatch,fault):
    directory,manifest,save=inputs
    if fault=="schema":manifest["schema"]="world-reward-joint-rgb-v1"
    elif fault=="top_private":manifest["truth_K"]=[[1,0,0]]
    elif fault=="record_private":manifest["images"][0]["identity_shape"]=[0.]
    elif fault=="extra":(directory/"truth.npz").write_bytes(b"forbidden")
    elif fault=="count":manifest["images"].pop()
    elif fault=="order":manifest["images"][0],manifest["images"][1]=manifest["images"][1],manifest["images"][0]
    elif fault=="booldim":manifest["images"][0]["width"]=True
    elif fault=="sha":manifest["images"][0]["sha256"]="not-sha"
    elif fault=="tamper":(directory/manifest["images"][0]["file"]).write_bytes(b"changed")
    else:
        path=directory/manifest["images"][0]["file"];path.unlink();path.symlink_to(directory/manifest["images"][1]["file"])
    save()
    with pytest.raises(ValueError):module.validate_inputs(directory)


def complete_rows():
    rows=[]
    for index in range(15):
        clip,frame=divmod(index,5);stem=f"clip_{clip:02d}_frame_{frame:03d}"
        row=dict(file=stem+".png",clip_index=clip,frame_index=frame)
        for label,query in (("human","person."),("object","bottle.")):
            row.update({label+"_query":query,label+"_mask_file":stem+"_"+label+".png",label+"_mask_sha256":"a"*64,label+"_mask_pixels":42})
        rows.append(row)
    return dict(records=rows,actual_detector_calls=30,actual_sam2_calls=30,actual_sam2_image_encoder_calls=15)


def test_all_thirty_automatic_masks_required(module):
    module.completed(complete_rows())


@pytest.mark.parametrize("fault",["calls","partial","order","query","filename","hash","empty","oversize","boolpixels"])
def test_partial_or_fallback_masks_cannot_pass(module,fault):
    report=complete_rows();row=report["records"][0]
    if fault=="calls":report["actual_sam2_calls"]=29
    elif fault=="partial":report["records"].pop()
    elif fault=="order":row["frame_index"]=1
    elif fault=="query":row["object_query"]="a bottle according to truth"
    elif fault=="filename":row["human_mask_file"]="../private/mask.png"
    elif fault=="hash":row["human_mask_sha256"]="bad"
    elif fault=="empty":row["human_mask_pixels"]=0
    elif fault=="oversize":row["human_mask_pixels"]=1024*768+1
    else:row["human_mask_pixels"]=True
    with pytest.raises(ValueError):module.completed(report)


def test_reused_selection_abstains_without_oracle(module):
    with pytest.raises(ValueError):module.masks.select_person([],1024,768)
    candidate=module.BoxDetection((10.,20.,100.,200.),.95)
    assert module.masks.select_person([candidate],1024,768)==candidate


def test_reused_sam_binary_mask_requires_original_full_support(module):
    predicted=np.zeros((1,768,1024),np.float32);predicted[:,30:60,20:50]=1
    mask=module.masks.binary_mask(predicted,np.array([.9]),1024,768)
    assert mask.dtype==np.uint8 and set(np.unique(mask))=={0,255} and np.count_nonzero(mask)==900
    with pytest.raises(RuntimeError):module.masks.binary_mask(predicted[:,:,:100],np.array([.9]),1024,768)
    with pytest.raises(RuntimeError):module.masks.binary_mask(predicted*0,np.array([.9]),1024,768)


def test_runtime_lazy_model_import_and_no_private_paths(module):
    source=Path(module.__file__).read_text();tree=ast.parse(source)
    topimports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    names={n.module for n in topimports if isinstance(n,ast.ImportFrom)}|{a.name for n in topimports if isinstance(n,ast.Import) for a in n.names}
    assert not {"torch","sam2","transformers"}&names
    assert "eval_private" not in source and "truth.npz" not in source and "mask_fallback" not in source
    assert 'local_files_only=True' in source and '180' in source and 'target.chmod(0o444)' in source
    assert 'masks.validate_assets(root, report["image_id"])' in source
    assert 'masks.sam2_source_identity' in source


def test_unknown_args_rejected_before_runtime(module):
    with pytest.raises(SystemExit):module.main(["--ground-truth-masks"])


def test_source_closure_keeps_only_native_helper_not_truth_renderer(module):
    import azure_job
    root=Path(module.__file__).resolve().parents[1]
    files={str(p.relative_to(root)):p.read_bytes() for base in (root/"infra",root/"src",root/"configs") for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"]=(root/"pyproject.toml").read_bytes()
    files["infra/run_identity_mask_test.sh"]=b'python "$CODE/infra/identity_rgb_masks.py"\n'
    closure=azure_job.runtime_bundle_paths(files,"infra/run_identity_mask_test.sh")
    assert "infra/hand_synthetic_masks.py" in closure and "src/world_reward/prompt_selection.py" in closure
    assert "infra/identity_rgb_render.py" not in closure and "infra/hand_synthetic_render.py" not in closure


def test_image_encoder_count_cannot_be_missing(module):
    report=complete_rows();del report["actual_sam2_image_encoder_calls"]
    with pytest.raises(ValueError):module.completed(report)
