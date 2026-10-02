"""Tiny own camera-cohort archives; no cloud/media/model downloads."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


@pytest.fixture
def gate(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/"infra";monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location("camera_transfer_test",infra/"perspective_rgb_transfer.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fixture(gate, root):
    files={gate.BASE+f"/inputs/case_{i:02d}.png":f"tinyRGB{i}".encode() for i in range(9)}
    public={"schema":"world-reward-perspective-rgb-v1","images":[{"file":f"case_{i:02d}.png","sha256":hashlib.sha256(files[gate.BASE+f"/inputs/case_{i:02d}.png"]).hexdigest(),"width":1024,"height":768} for i in range(9)]}
    image="sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
    semantic=dict(source_image_id=image,stage="own_reference_mhr_finger_semantics_support",status="pass",phase_A_verified=True,
        phase_B_verified=True,named_finger_joint_partition_verified=True,excluded_joint_invariance_verified=True,
        correctives_skeleton_bit_identical=True,model_sha256=gate.MODEL_SHA,
        body_checkpoint_sha256=gate.BODY_SHA,body_revision=gate.BODY_REV,
        network="none",challenge_inputs_used=False,script_sha256=gate.digest(Path(gate.__file__).with_name("mhr_finger_semantics_gate.py")),
        deterministic_algorithms=True,CUBLAS_WORKSPACE_CONFIG=":4096:8",TF32=False,jit_optimized_execution=False)
    files[gate.NAMES[0]]=json.dumps(public).encode(); files[gate.SEMANTIC]=json.dumps(semantic).encode()
    truthname=gate.BASE+"/eval_private/calibration_truth.npz";files[truthname]=b"tiny own camera truth"
    report={"stage":"own_procedural_perspective_rgb_render","status":"pass","challenge_inputs_used":False,
        "inference_performed":False,"actual_MHR_reference_used":True,"actual_reference_forward_calls":2,
        "synthetic_truth_used_for_rendering_only":True,"public_manifest_sha256":hashlib.sha256(files[gate.NAMES[0]]).hexdigest(),
        "code_revision":"a"*40,"truth_sha256":hashlib.sha256(files[truthname]).hexdigest(),
        "semantic_report_sha256":hashlib.sha256(files[gate.SEMANTIC]).hexdigest(),"image_id":image,
        "model_sha256":gate.MODEL_SHA,"script_sha256":gate.digest(Path(gate.__file__).with_name("perspective_rgb_render.py")),
        "render_helper_sha256":gate.digest(Path(gate.__file__).with_name("hand_synthetic_render.py")),"joint_helper_sha256":gate.digest(Path(gate.__file__).with_name("joint_rgb_render.py")),
        "camera_helper_sha256":gate.digest(Path(gate.__file__).with_name("camera_render.py"))}
    files[gate.BASE+"/eval_private/render-report.json"]=json.dumps(report).encode()
    for name,data in files.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    return files


def test_exact_roundtrip_no_models_or_challenge(gate,tmp_path):
    source=tmp_path/"source";source.mkdir();files=fixture(gate,source)
    archive=tmp_path/"cohort.tar";sha=gate.export(source,archive)
    assert archive.stat().st_mode&0o777==0o600
    destination=tmp_path/"dest";destination.mkdir();manifest=gate.extract(destination,archive,sha)
    assert len(manifest["entries"])==13 and manifest["private_camera_evaluation_included"]
    assert manifest["models_included"] is manifest["challenge_inputs_included"] is manifest["predictions_included"] is False
    for name,data in files.items():assert (destination/name).read_bytes()==data
    assert (destination/gate.BASE/"eval_private").stat().st_mode&0o777==0o700
    assert (destination/gate.NAMES[0]).stat().st_mode&0o777==0o444
    with pytest.raises(FileExistsError):gate.extract(destination,archive,sha)


@pytest.mark.parametrize("fault",["rgb","truth","private_extra","public_extra","symlink","secret","revision","image","model","script","semantic"])
def test_source_tamper_rejected(gate,tmp_path,fault):
    root=tmp_path/"source";root.mkdir();fixture(gate,root)
    if fault=="rgb":(root/gate.NAMES[1]).write_bytes(b"changed")
    elif fault=="truth":(root/gate.BASE/"eval_private/calibration_truth.npz").write_bytes(b"changed")
    elif fault.endswith("extra"):
        folder="eval_private" if fault=="private_extra" else "inputs";(root/gate.BASE/folder/"unexpected").write_bytes(b"noise")
    elif fault=="symlink":
        p=root/gate.NAMES[1];p.unlink();p.symlink_to(root/gate.NAMES[2])
    elif fault=="semantic":
        p=root/gate.SEMANTIC;r=json.loads(p.read_text());r["status"]="fail";p.write_text(json.dumps(r))
    else:
        p=root/gate.BASE/"eval_private/render-report.json";r=json.loads(p.read_text())
        if fault=="secret":r["HF_TOKEN"]="forbidden"
        elif fault=="image":r["image_id"]=None
        elif fault=="model":r["model_sha256"]="0"*64
        elif fault=="script":r["script_sha256"]="0"*64
        else:r["code_revision"]="not-a-commit"
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError):gate.inventory(root)


@pytest.mark.parametrize("fault",["sha","order","traversal","duplicate","symlink","wrongbytes","wrongmode","flags"])
def test_hostile_archive_no_destination_writes(gate,tmp_path,fault):
    root=tmp_path/"source";root.mkdir();fixture(gate,root);original=tmp_path/"a.tar";sha=gate.export(root,original)
    with tarfile.open(original) as tar:values=[(m,tar.extractfile(m).read()) for m in tar.getmembers()]
    if fault=="order":values[1],values[2]=values[2],values[1]
    elif fault=="traversal":values[1][0].name="../escape"
    elif fault=="duplicate":values.append(values[1])
    elif fault=="symlink":values[1][0].type=tarfile.SYMTYPE;values[1][0].linkname="/outside";values[1][0].size=0;values[1]=(values[1][0],None)
    elif fault=="wrongbytes":values[1]=(values[1][0],b"x"*len(values[1][1]))
    elif fault=="wrongmode":values[1][0].mode=0o777
    elif fault=="flags":
        m,d=values[0];r=json.loads(d);r["models_included"]=True;d=json.dumps(r).encode();m.size=len(d);values[0]=(m,d)
    bad=tmp_path/"b.tar"
    with tarfile.open(bad,"w",format=tarfile.USTAR_FORMAT) as tar:
        for m,d in values:tar.addfile(m,io.BytesIO(d) if d is not None else None)
    destination=tmp_path/"dest";destination.mkdir()
    with pytest.raises(ValueError):gate.extract(destination,bad,"0"*64 if fault=="sha" else gate.digest(bad))
    assert not list(destination.iterdir())


def test_destination_existing_semantic_not_overwritten(gate,tmp_path):
    root=tmp_path/"source";root.mkdir();fixture(gate,root);archive=tmp_path/"a.tar";sha=gate.export(root,archive)
    dest=tmp_path/"dest";(dest/"results").mkdir(parents=True);(dest/gate.SEMANTIC).write_bytes(b"user work")
    with pytest.raises(FileExistsError):gate.extract(dest,archive,sha)
    assert (dest/gate.SEMANTIC).read_bytes()==b"user work" and not (dest/gate.BASE).exists()


def test_source_only_transfer_closure_and_wrappers(gate):
    import ast
    import subprocess
    import azure_job
    tree=ast.parse(Path(gate.__file__).read_text())
    imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert imports=={"pathlib","research_transfer"}
    root=Path(gate.__file__).resolve().parents[1]
    files={str(p.relative_to(root)):p.read_bytes() for base in (root/"infra",root/"src",root/"configs") for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files["pyproject.toml"]=(root/"pyproject.toml").read_bytes()
    for mode in ("export","import"):
        path="infra/run_perspective_rgb_"+mode+".sh";subprocess.run(["bash","-n",str(root/path)],check=True)
        closure=azure_job.runtime_bundle_paths(files,path)
        assert all(name.lstrip('/') in closure for name in (*gate.SOURCE_FILES.values(),gate.SEMANTIC_SOURCE))
        assert "--gpus" not in (root/path).read_text()


def test_transfer_parent_traversal_precedes_unprivileged_work(gate):
    root=Path(gate.__file__).resolve().parents[1]
    for mode in ("export","import"):
        text=(root/("infra/run_perspective_rgb_"+mode+".sh")).read_text()
        assert text.index('chmod 711 "$ROOT/transfer"') < text.index('timeout --signal=TERM')
        assert 'runuser -u scenesmith -- test -' in text
    export=(root/"infra/run_perspective_rgb_export.sh").read_text()
    assert 'OUT="$ROOT/transfer/perspective-v2"' in export
    assert 'chmod 700 "$OUT"' in export
