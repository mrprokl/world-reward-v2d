"""Tiny independent byte/proof fixtures; no media, model, Docker or Azure."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def binding(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"))
    spec=importlib.util.spec_from_file_location("bridge_binding_test",ROOT/"infra/bridge_frontend_bindings.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_bytes(raw);path.chmod(0o400)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


@pytest.fixture
def public(binding,tmp_path):
    directory=tmp_path/binding.BASE/"inputs"; rows=[]; files={}
    for clip,name in enumerate(binding.FILENAMES):
        files[name]=write(directory/name,b"tiny metadata-only fixture "+str(clip).encode())
        rows.append(dict(clip_id=clip,frame_id=0,file=name,width=640,height=480,**files[name]))
    manifest=dict(schema="world_reward.bridge_rgb_anchor_public.v1",images=rows)
    files["manifest.json"]=write(directory/"manifest.json",json.dumps(manifest).encode())
    path=tmp_path/"code"/binding.INPUT_PINS; pins=dict(schema="world_reward.bridge_rgb_anchor_input_pins.v1",public_files=files)
    write(path,json.dumps(pins).encode());return tmp_path,path,directory,manifest,pins


def test_public_reader_pins_all_five_before_decode_and_no_private_inputs(binding,public):
    root,path,directory,manifest,pins=public
    records,frozen=binding.public_inputs(root,path)
    assert len(records)==4 and len(frozen)==6 and [r["clip_id"] for r in records]==list(range(4))
    assert all(r["path"]==directory/r["file"] for r in records)
    assert not any("eval_private" in str(p) or "render" in str(p) or "pose" in str(p) for p in frozen)
    binding.recheck(frozen)


@pytest.mark.parametrize("fault",["privatefield","camera","clipbool","frame","width","order","name","extra","alias","RGB","pinbool","pinextra"])
def test_rgb_metadata_or_file_change_rejected_even_rehashed_manifest(binding,public,fault):
    root,path,directory,manifest,pins=public
    if fault=="privatefield":manifest["recipe"]={"seed":1}
    elif fault=="camera":manifest["images"][0]["K"]=[1,2,3]
    elif fault=="clipbool":manifest["images"][0]["clip_id"]=False
    elif fault=="frame":manifest["images"][0]["frame_id"]=1
    elif fault=="width":manifest["images"][0]["width"]=639
    elif fault=="order":manifest["images"].reverse()
    elif fault=="name":manifest["images"][0]["file"]="../truth.png"
    elif fault=="extra":write(directory/"depth.npy",b"forbidden")
    elif fault=="alias":
        p=directory/binding.FILENAMES[0];p.unlink();p.symlink_to(directory/binding.FILENAMES[1])
    elif fault=="RGB":write(directory/binding.FILENAMES[0],b"changed")
    elif fault=="pinbool":pins["public_files"][binding.FILENAMES[0]]["bytes"]=True
    else:pins["public_files"][binding.FILENAMES[0]]["oracle"]=False
    pins["public_files"]["manifest.json"]=write(directory/"manifest.json",json.dumps(manifest).encode())
    write(path,json.dumps(pins).encode())
    with pytest.raises(ValueError):binding.public_inputs(root,path)


@pytest.mark.parametrize("fault",["mode","alias","size","sha"])
def test_independent_pin_must_precede_JSON_trust(binding,tmp_path,fault):
    path=tmp_path/"old-report.json";pin=write(path,b'{"status":"pass"}')
    if fault=="mode":path.chmod(0o600)
    elif fault=="alias":link=tmp_path/"alias";link.symlink_to(path);path=link
    elif fault=="size":pin["bytes"]+=1
    else:pin["sha256"]="a"*64
    with pytest.raises(ValueError):binding.pinned(path,pin)


def test_selected_assets_read_only_whitelist_receipt_and_models(binding,tmp_path,monkeypatch):
    monkeypatch.setattr(binding,"DEST",tmp_path)
    files={"results/weights-acquisition.json":b'{"assets":[]}',"weights/example/model.pt":b"tiny learned-weight fixture"}
    entries={}
    for name,raw in files.items():entries[name]=dict(type="file",role="source_receipt" if name.startswith("results/") else "asset",**write(tmp_path/name,raw))
    contract=dict(entries=entries)
    constants={"example/model.pt":(entries["weights/example/model.pt"]["sha256"],entries["weights/example/model.pt"]["bytes"])}
    assert binding.selected_assets(contract,constants)=={"example/model.pt":{k:entries["weights/example/model.pt"][k] for k in ("bytes","sha256")}}
    write(tmp_path/"weights/example/model.pt",b"wrong")
    with pytest.raises(ValueError):binding.selected_assets(contract,constants)


@pytest.fixture
def proofs(binding,tmp_path,monkeypatch):
    root=tmp_path/"root";code=root/"jobs"/("a"*40)/"run_bridge_rgb_anchor_masks/code"
    buildcode=root/"jobs"/binding.BUILD_REV/"run_frontend_grounding_build/code"
    kernelcode=root/"jobs"/binding.KERNEL_REV/"run_frontend_sam2_kernel_gate/code"
    for key,path in (("ROOT",root),("BUILD_CODE",buildcode),("KERNEL_CODE",kernelcode),
        ("BUILD_REPORT",root/"results/build.json"),("KERNEL_REPORT",root/"results/kernel.json")):
        monkeypatch.setattr(binding,key,path)
    cfg=dict(bytes=1,sha256="c"*64)
    own=dict(closure_sha256="a"*64,helpers={binding.CONFIG:cfg})
    oldbuild=dict(closure_sha256="b"*64,helpers={binding.CONFIG:cfg,"infra/frontend_grounding_build.py":dict(bytes=27520,sha256=binding.BUILD_SHA)})
    oldkernel=dict(closure_sha256="d"*64,helpers={binding.CONFIG:cfg,"infra/frontend_sam2_kernel_gate.py":binding.KERNEL_SOURCE_PIN})
    closures={code:own,buildcode:oldbuild,kernelcode:oldkernel};calls=[]
    kernel=SimpleNamespace(BASE="sha256:"+"7"*64,BUILD_HELPERS=(),closure=lambda c,*args:calls.append(c) or copy.deepcopy(closures[c]))
    monkeypatch.setattr(binding,"kernel_helper",lambda:kernel)
    probe=write(root/"results/frontend-grounding-build-v6/child-CPU-probe.log",b'private CPU probe only\n')
    parent=dict(Id=kernel.BASE,RootFS=dict(Layers=["same"]*44));child=dict(Id=binding.IMAGE,RootFS=dict(Layers=["same"]*44+["new"]))
    build=dict(schema="world_reward.frontend_grounding_build.v6",stage="frontend_grounding_build",status="pass",phase="complete",
        producer_revision=binding.BUILD_REV,offline_build_exit_code=0,child_probe_exit_code=0,parent_unchanged_verified=True,
        source_rechecked_before_and_after=True,extension_import_verified=True,CUDA_execution_verified=False,replica_ready=False,
        license_eligibility_verified=False,training_overlap_verified=False,source_binding=oldbuild,child_image=child,parent_image=parent,
        owner=hashlib.sha256((binding.BUILD_REV+oldbuild["closure_sha256"]).encode()).hexdigest(),source_files={},
        private_child_probe_log=dict(relative_path="results/frontend-grounding-build-v6/child-CPU-probe.log",**probe))
    buildpin=write(binding.BUILD_REPORT,json.dumps(build).encode());monkeypatch.setattr(binding,"BUILD_PIN",buildpin)
    receipt=dict(schema="world_reward.frontend_sam2_kernel_gate.v1",stage="frontend_sam2_kernel_gate",status="pass",
        producer_revision=binding.KERNEL_REV,script_sha256=binding.KERNEL_SOURCE_PIN["sha256"],image_id=binding.IMAGE,
        models_loaded=False,challenge_data_read=False,CUDA_operator_execution_verified=True,build_report_identity=buildpin,
        replica_ready=False,source_binding=oldkernel,operator_source_identities=dict(extension=binding.EXTENSION_PIN))
    kernelpin=write(binding.KERNEL_REPORT,json.dumps(receipt).encode());monkeypatch.setattr(binding,"KERNEL_PIN",kernelpin)
    contract=dict(entries={},manifest_identity={})
    monkeypatch.setattr(binding.selected,"load_contract",lambda r,m:contract)
    return root,code,build,receipt,closures,calls,contract


def test_proof_authentication_distinguishes_three_genuine_source_namespaces(binding,proofs):
    root,code,build,receipt,closures,calls,contract=proofs
    value=binding.authenticate(root,code,"run_bridge_rgb_anchor_masks")
    assert value["selected_contract"] is contract and value["child_image"]["Id"]==binding.IMAGE
    assert calls==[code,binding.BUILD_CODE,binding.KERNEL_CODE]


@pytest.mark.parametrize("fault",["foreignentry","buildpin","kernelpin","oldbuild","oldkernel","config","status","CUDA","extension","parent","owner","probe"])
def test_provenance_failure_precedes_any_model_or_live_job(binding,proofs,fault):
    root,code,build,receipt,closures,calls,contract=proofs
    entry="run_bridge_rgb_anchor_masks"
    if fault=="foreignentry":entry="run_bridge_rgb_anchor_render"
    elif fault=="buildpin":binding.BUILD_REPORT.chmod(0o600);binding.BUILD_REPORT.write_bytes(b'{}');binding.BUILD_REPORT.chmod(0o400)
    elif fault=="kernelpin":binding.KERNEL_REPORT.chmod(0o600);binding.KERNEL_REPORT.write_bytes(b'{}');binding.KERNEL_REPORT.chmod(0o400)
    elif fault=="oldbuild":closures[binding.BUILD_CODE]["closure_sha256"]="e"*64
    elif fault=="oldkernel":closures[binding.KERNEL_CODE]["helpers"]["infra/frontend_sam2_kernel_gate.py"]=dict(bytes=1,sha256="e"*64)
    elif fault=="config":closures[code]["helpers"][binding.CONFIG]=dict(bytes=2,sha256="e"*64)
    elif fault=="status":build["status"]="fail"
    elif fault=="CUDA":receipt["CUDA_operator_execution_verified"]=False
    elif fault=="extension":receipt["operator_source_identities"]["extension"]=dict(bytes=1,sha256="e"*64)
    elif fault=="parent":build["child_image"]["RootFS"]["Layers"][0]="changed"
    elif fault=="owner":build["owner"]="e"*64
    else:write(root/"results/frontend-grounding-build-v6/child-CPU-probe.log",b"changed")
    if fault not in ("buildpin","kernelpin"):
        binding.BUILD_PIN=write(binding.BUILD_REPORT,json.dumps(build).encode())
        receipt["build_report_identity"]=binding.BUILD_PIN
        binding.KERNEL_PIN=write(binding.KERNEL_REPORT,json.dumps(receipt).encode())
    with pytest.raises(ValueError):binding.authenticate(root,code,entry)


def test_live_verifier_uses_actual_old_job_environment_no_credentials(binding,proofs,monkeypatch):
    root,code,*_=proofs;calls=[]
    monkeypatch.setattr(binding.os,"uname",lambda:SimpleNamespace(nodename="world-reward-ncc-h100-02"))
    monkeypatch.setenv("SECRET_FIXTURE","must_not_propagate")
    def run(args,**kwargs):calls.append((args,kwargs));return SimpleNamespace(returncode=0,stdout=b"kernel_gate_bindings_verified\n")
    monkeypatch.setattr(binding.subprocess,"run",run)
    binding.authenticate(root,code,"run_bridge_rgb_anchor_masks",live=True)
    args,kwargs=calls[0]
    assert str(binding.KERNEL_CODE/"infra/frontend_sam2_kernel_gate.py") in args and "--verify" in args
    assert kwargs["env"]["WR_CODE"]==str(binding.KERNEL_CODE) and kwargs["env"]["WR_CODE_REVISION"]==binding.KERNEL_REV
    assert "SECRET_FIXTURE" not in kwargs["env"] and not any("--run"==v for v in args)


def test_current_kernel_import_requires_actual_original_bytes(binding,monkeypatch):
    fake=binding.KERNEL_SOURCE_PIN|dict(sha256="a"*64);monkeypatch.setattr(binding,"KERNEL_SOURCE_PIN",fake)
    with pytest.raises(ValueError):binding.kernel_helper()


@pytest.fixture
def masks(binding,public):
    root,path,directory,manifest,pins=public;records,_=binding.public_inputs(root,path)
    folder=root/binding.BASE/"automatic_masks";files={};rows=[]
    for original in records:
        row={k:v for k,v in original.items() if k!="path"}
        for label,query in (("person","person."),("object","bottle.")):
            name=label+"_"+row["file"];files[name]=write(folder/name,b"tiny automatic mask metadata "+name.encode())
            row[label]=dict(query=query,mask_file=name,mask_pixels=100,mask_bytes=files[name]["bytes"],mask_sha256=files[name]["sha256"])
        rows.append(row)
    receipt=dict(schema="world_reward.bridge_rgb_anchor_masks.v1",stage="bridge_rgb_anchor_automatic_masks",status="pass",phase="complete",
        image_id=binding.IMAGE,frames=4,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],network="none",
        confidence=.3,text_threshold=.25,nms_iou=.7,ambiguity_margin=.05,detector_calls=8,sam2_calls=8,images=rows,
        build_report_identity=binding.BUILD_PIN,kernel_report_identity=binding.KERNEL_PIN,producer_revision="a"*40,script_sha256="b"*64)
    files["report.json"]=write(folder/"report.json",json.dumps(receipt).encode())
    pinpath=root/"code/configs/bridge_rgb_anchor_mask_pins.json"
    maskpins=dict(schema="world_reward.bridge_rgb_anchor_mask_pins.v1",producer_revision="a"*40,script_sha256="b"*64,files=files)
    write(pinpath,json.dumps(maskpins).encode());return root,pinpath,records,folder,receipt,maskpins


def test_mask_consumer_pins_all_nine_without_decoding_or_current_source_relabel(binding,masks):
    root,path,records,folder,receipt,pins=masks
    rows,frozen=binding.load_masks(root,path,records)
    assert rows==receipt["images"] and len(frozen)==10
    assert not any("body_smoke" in str(p) or "eval_private" in str(p) for p in frozen)


@pytest.mark.parametrize("fault",["producer","script","status","calls","query","frames","framebool","support","SHA","extra","missing"])
def test_mask_consumer_requires_complete_real_automatic_provenance(binding,masks,fault):
    root,path,records,folder,receipt,pins=masks
    if fault=="producer":receipt["producer_revision"]="c"*40
    elif fault=="script":receipt["script_sha256"]="c"*64
    elif fault=="status":receipt["status"]="fail"
    elif fault=="calls":receipt["sam2_calls"]=7
    elif fault=="query":receipt["images"][0]["object"]["query"]="manual prompt"
    elif fault=="frames":receipt["images"].pop()
    elif fault=="framebool":receipt["images"][0]["frame_id"]=False
    elif fault=="support":receipt["images"][0]["person"]["mask_pixels"]=0
    elif fault=="SHA":receipt["images"][0]["person"]["mask_sha256"]="c"*64
    elif fault=="extra":write(folder/"private_camera.json",b"forbidden")
    else:(folder/("person_"+binding.FILENAMES[0])).unlink()
    pins["files"]["report.json"]=write(folder/"report.json",json.dumps(receipt).encode());write(path,json.dumps(pins).encode())
    with pytest.raises((ValueError,FileNotFoundError)):binding.load_masks(root,path,records)


def test_actual_independent_old_helper_pin_and_public_src_closure(binding):
    raw=(ROOT/"infra/frontend_sam2_kernel_gate.py").read_bytes()
    assert dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==binding.KERNEL_SOURCE_PIN
    source=(ROOT/"infra/bridge_frontend_bindings.py").read_text()
    assert "bridge_rgb_anchor_render" not in source and "triangle_ray_gate" not in source
    assert "torch" not in sys.modules or "import torch" not in source
