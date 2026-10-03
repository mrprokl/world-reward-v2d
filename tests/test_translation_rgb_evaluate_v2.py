"""H97 v2 native face ABI fixtures; no private/model/challenge data."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

INFRA=Path(__file__).parents[1]/"infra"


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA));monkeypatch.syspath_prepend(str(INFRA.parent/"src"))
    return load("own_translation_quality_v2",INFRA/"translation_rgb_evaluate_v2.py")


def truth_fixture(gate,monkeypatch):
    fixture=load("old_quality_fixtures",Path(__file__).with_name("test_translation_rgb_evaluate.py"))
    for module in (gate,gate.old):
        monkeypatch.setattr(module,"VERTICES",120);monkeypatch.setattr(module,"WIDTH",16);monkeypatch.setattr(module,"HEIGHT",16)
    record=dict(clip_index=0,frame_index=0);truth=fixture.own_truth(gate.old,record);faces=truth["human_faces"].copy()
    truth["human_faces"]=truth["human_faces"].astype(np.int32)
    monkeypatch.setattr(gate,"HUMAN_FACES_SHA",hashlib.sha256(truth["human_faces"].tobytes()).hexdigest())
    return truth,record,faces


def test_native_int32_truth_matches_int64_baseline_without_repair(gate,monkeypatch):
    truth,record,faces=truth_fixture(gate,monkeypatch);before={k:v.tobytes()for k,v in truth.items()}
    with pytest.raises(ValueError,match="Truth topology differs"):gate.old.validate_truth(truth,record,faces)
    gate.validate_truth(truth,record,faces)
    assert before=={k:v.tobytes()for k,v in truth.items()} and truth["human_faces"].dtype==np.int32


@pytest.mark.parametrize("fault",["int64","float","bool","uint","masked","flipped","reordered","index","negative","objectdtype","objectface","baseline","typedsha"])
def test_no_other_topology_or_dtype_relaxation(gate,monkeypatch,fault):
    truth,record,faces=truth_fixture(gate,monkeypatch)
    if fault in("int64","float","bool","uint"):
        truth["human_faces"]=truth["human_faces"].astype({"int64":np.int64,"float":np.float32,"bool":np.bool_,"uint":np.uint32}[fault])
    elif fault=="masked":truth["human_faces"]=np.ma.array(truth["human_faces"],mask=False)
    elif fault=="flipped":truth["human_faces"][0]=[0,2,1]
    elif fault=="reordered":
        truth["human_faces"][0]=[1,2,0];truth["human_faces"][[0,1]]=truth["human_faces"][[1,0]]
    elif fault=="index":truth["human_faces"][0,0]=119
    elif fault=="negative":truth["human_faces"][0,0]=-1
    elif fault=="objectdtype":truth["object_faces"]=truth["object_faces"].astype(np.int32)
    elif fault=="objectface":truth["object_faces"][0]=[0,0,2]
    elif fault=="baseline":faces=faces.astype(np.int32)
    else:monkeypatch.setattr(gate,"HUMAN_FACES_SHA","a"*64)
    with pytest.raises(ValueError):gate.validate_truth(truth,record,faces)


def failed_fixture(gate,tmp_path,monkeypatch,fit_sha=None):
    path=tmp_path/gate.BASE/"translation_quality_v1/report.json";path.parent.mkdir(parents=True,exist_ok=True)
    data=dict(stage=gate.old.STAGE,status="fail",phase="public_predictions_audited",producer_revision=gate.FIT_REVISION,
        image_id=gate.IMAGE,network="none",script_sha256=gate.OLD_SOURCE_SHA,error_type="ValueError",error="Truth topology differs",
        predictions_frozen_before_private=True,private_truth_used_for_evaluation_only=True,fit_report_sha256=fit_sha or gate.FIT_REPORT_SHA,
        challenge_inputs_used=False,accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,
        methodological_followup_not_independent_replication=True)
    def save():
        if path.exists():path.chmod(0o644)
        path.write_text(json.dumps(data));path.chmod(0o444)
        monkeypatch.setattr(gate,"FAILED_REPORT_SHA",gate.sha256(path));monkeypatch.setattr(gate,"FAILED_REPORT_BYTES",path.stat().st_size)
    save();return path,data,save


@pytest.mark.parametrize("fault",[None,"pass","phase","decision","scores","fitsha","error","revision","private","size","extra","changed"])
def test_actual_stopped_v1_receipt_shape_preserved(gate,tmp_path,monkeypatch,fault):
    path,data,save=failed_fixture(gate,tmp_path,monkeypatch)
    if fault=="pass":data["status"]="pass"
    elif fault=="phase":data["phase"]="complete"
    elif fault=="decision":data["decision"]={}
    elif fault=="scores":data["frame_metrics"]=[]
    elif fault=="fitsha":data["fit_report_sha256"]="a"*64
    elif fault=="error":data["error"]="other failure"
    elif fault=="revision":data["producer_revision"]="c"*40
    elif fault=="private":data["private_truth_used_for_evaluation_only"]=False
    save()
    if fault=="size":monkeypatch.setattr(gate,"FAILED_REPORT_BYTES",path.stat().st_size+1)
    elif fault=="extra":(path.parent/"prediction.npz").write_bytes(b"hidden")
    elif fault=="changed":path.chmod(0o644);path.write_bytes(b"changed");path.chmod(0o444)
    if fault:
        with pytest.raises(ValueError):gate.previous_failure(tmp_path)
    else:assert gate.previous_failure(tmp_path)["sha256"]==gate.FAILED_REPORT_SHA


def complete_public_fixture(gate,tmp_path,monkeypatch):
    fixture=load("translation_900state_fixture",Path(__file__).with_name("test_translation_rgb_evaluate.py"))
    public,report,save,path=fixture.complete_public_fit_fixture(gate.old,tmp_path,monkeypatch)
    monkeypatch.setattr(gate,"FIT_REVISION",report["producer_revision"])
    monkeypatch.setattr(gate,"FIT_REPORT_SHA",gate.sha256(path))
    failed_fixture(gate,tmp_path,monkeypatch,gate.FIT_REPORT_SHA)
    return public,report,save,path


def test_frozen_old_fit_revision_distinct_new_evaluator_full900_replay(gate,tmp_path,monkeypatch):
    _,producer,_,_=complete_public_fixture(gate,tmp_path,monkeypatch)
    monkeypatch.setenv("WR_CODE_REVISION","b"*40)
    result=gate.public_fit(tmp_path)
    assert all(len(x)==15 for x in result[:5]) and result[5]["counters"]["evaluated_states"]==900
    assert result[5]["producer_revision"]!=__import__("os").environ["WR_CODE_REVISION"]
    assert result[6][0]["sha256"]==gate.FIT_REPORT_SHA


@pytest.mark.parametrize("fault",["state","fitbytes","failure","revision","candidate"])
def test_public_failures_stop_before_any_private_read(gate,tmp_path,monkeypatch,fault):
    _,report,save,path=complete_public_fixture(gate,tmp_path,monkeypatch)
    if fault=="state":report["fit_records"][2]["evaluated_latents"][28][0]+=.01;save();monkeypatch.setattr(gate,"FIT_REPORT_SHA",gate.sha256(path))
    elif fault=="fitbytes":path.chmod(0o644);path.write_bytes(b"changed");path.chmod(0o444)
    elif fault=="failure":
        fail=tmp_path/gate.BASE/"translation_quality_v1/report.json";fail.chmod(0o644);fail.write_bytes(b"changed");fail.chmod(0o444)
    elif fault=="revision":monkeypatch.setattr(gate,"FIT_REVISION","c"*40)
    else:
        p=path.parent/report["candidate_outputs"][0]["artifact"];p.chmod(0o644);p.write_bytes(b"changed");p.chmod(0o444)
    monkeypatch.setattr(gate,"source_bindings",lambda root:{})
    monkeypatch.setattr(gate.old,"private_truth",lambda *_:pytest.fail("private read before complete public/failure freeze"))
    with pytest.raises(ValueError):gate.run(tmp_path,{})


def full_quality_fixture(gate,tmp_path,monkeypatch):
    fixture=load("translation_private_15fixture",Path(__file__).with_name("test_translation_rgb_evaluate.py"))
    private,public,records=fixture.full_quality_fixture(gate.old,tmp_path,monkeypatch)
    render_path=private/"render-report.json";render=json.loads(render_path.read_text())
    for record,case in zip(records,render["cases"]):
        path=private/(Path(record["file"]).stem+".npz")
        with np.load(path,allow_pickle=False)as saved:truth={k:saved[k]for k in saved.files}
        truth["human_faces"]=truth["human_faces"].astype(np.int32)
        path.chmod(0o644)
        with path.open("wb")as handle:np.savez_compressed(handle,**truth)
        path.chmod(0o444);case["truth_sha256"]=gate.sha256(path)
    render_path.chmod(0o644);render_path.write_text(json.dumps(render));render_path.chmod(0o444)
    monkeypatch.setattr(gate.old,"RENDER_SHA",gate.sha256(render_path))
    monkeypatch.setattr(gate,"HUMAN_FACES_SHA",hashlib.sha256(truth["human_faces"].tobytes()).hexdigest())
    for name in("VERTICES","WIDTH","HEIGHT"):monkeypatch.setattr(gate,name,getattr(gate.old,name))
    monkeypatch.setattr(gate,"public_fit",gate.old.public_fit)
    monkeypatch.setattr(gate,"source_bindings",lambda root:{})
    monkeypatch.setattr(gate,"fit_source",gate.old.fit_source)
    return private,public,records


def test_all15_first_scores_exact_same_metrics_no_dtype_cast(gate,tmp_path,monkeypatch):
    _,_,_=full_quality_fixture(gate,tmp_path,monkeypatch);report={};gate.run(tmp_path,report)
    assert report["status"]=="pass" and len(report["frame_metrics"])==15
    assert report["clip_metrics"][0]["baseline"]["human_pve_cm"]==pytest.approx(20.)
    assert report["clip_metrics"][0]["fitted"]["human_pve_cm"]==pytest.approx(10.)
    assert not report["decision"]["adoption_authorized"] and report["dtype_abi_repair_only"]
    assert report["metric_or_threshold_changed"] is False and report["optimizer_rerun"] is False


@pytest.mark.parametrize("fault",["truthbytes","publicbytes","sources"])
def test_post_freeze_changes_do_not_pass(gate,tmp_path,monkeypatch,fault):
    private,public,records=full_quality_fixture(gate,tmp_path,monkeypatch)
    if fault=="truthbytes":
        p=private/(Path(records[0]["file"]).stem+".npz");p.chmod(0o644);p.write_bytes(b"changed");p.chmod(0o444)
    elif fault=="publicbytes":
        original=gate.old.frame_metrics
        def mutate(*args):
            public.chmod(0o644);public.write_bytes(b"changed");public.chmod(0o444);return original(*args)
        monkeypatch.setattr(gate.old,"frame_metrics",mutate)
    else:
        calls=0
        def sources(root):
            nonlocal calls
            calls+=1;return {}if calls==1 else {"changed":{}}
        monkeypatch.setattr(gate,"source_bindings",sources)
    with pytest.raises(ValueError):gate.run(tmp_path,{})


def test_only_validate_human_face_dtype_changed_from_frozen_v1(gate):
    source=Path(gate.__file__).read_text();tree=ast.parse(source)
    validator=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="validate_truth")
    original=next(n for n in ast.parse(Path(gate.old.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="validate_truth")
    # Remove only the two additional structural/hash guards, then undo I32 spelling.
    normalized=ast.parse(ast.unparse(validator)).body[0]
    normalized.body=[node for node in normalized.body if not(isinstance(node,ast.If)and
        ("Original baseline int64 topology required"in ast.unparse(node)or"Original int32 face bytes changed"in ast.unparse(node)))]
    for node in ast.walk(normalized):
        if isinstance(node,ast.Attribute)and node.attr=="int32":node.attr="int64"
    assert ast.dump(normalized,include_attributes=False)==ast.dump(original,include_attributes=False)
    for name in("frame_metrics","aggregate","heldout_diagnostics","visible_object_median","private_truth"):
        assert not any(isinstance(n,ast.FunctionDef)and n.name==name for n in tree.body)
    assert "old.frame_metrics("in source and"old.aggregate("in source and"old.private_truth("in source
    assert "astype(np.int32)"not in source and"astype(np.int64)"not in source


def test_wrapper_old_fit_canonical_revision_new_output_and_cpu(gate):
    path=INFRA/"run_translation_rgb_evaluate_v2.sh";subprocess.run(["bash","-n",str(path)],check=True);text=path.read_text()
    assert 'OUT="$BASE/translation_quality_v2"'in text and"--gpus"not in text
    assert 'jobs/2799e8d2c468aff9230b422beea33779e640f325/run_translation_rgb_fit/code/infra/$NAME'in text
    assert '"$BASE/translation_quality_v1/report.json"'in text
    assert 'src=$SOURCE,dst=$SOURCE,readonly'in text and"--network none --cpus 4 --memory 8g"in text


def test_cpu_no_torch_or_existing_output_reuse(gate,tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace())
    with pytest.raises(ValueError,match="must not import Torch"):gate.run(tmp_path,{})
    monkeypatch.setenv("WR_ROOT",str(tmp_path));monkeypatch.setenv("WR_CODE_REVISION","b"*40)
    with pytest.raises(ValueError,match="Fresh canonical"):gate.main([])
    assert not(tmp_path/gate.OUT).exists()


@pytest.mark.parametrize("fault",[None,"old_quality","consumer","policy","fit","own_writable","symlink"])
def test_exact_source_bindings_historical_and_v2_all_readonly(gate,tmp_path,monkeypatch,fault):
    import translation_rgb_public as public
    directory=tmp_path/"source";directory.mkdir()
    modules=[(gate.old,"translation_rgb_evaluate.py"),(public,"translation_rgb_public.py"),
             (gate.old.policy,"translation_refit.py"),(gate,"translation_rgb_evaluate_v2.py")]
    for module,name in modules:
        path=directory/name;path.write_bytes(Path(module.__file__).read_bytes());path.chmod(0o444)
        monkeypatch.setattr(module,"__file__",str(path))
    source=gate.fit_source(tmp_path);source.parent.mkdir(parents=True);source.write_bytes((INFRA/"translation_rgb_fit.py").read_bytes());source.chmod(0o444)
    if fault in("old_quality","consumer","policy","fit"):
        path={"old_quality":directory/"translation_rgb_evaluate.py","consumer":directory/"translation_rgb_public.py",
              "policy":directory/"translation_refit.py","fit":source}[fault]
        path.chmod(0o644);path.write_bytes(b"changed source");path.chmod(0o444)
    elif fault=="own_writable":(directory/"translation_rgb_evaluate_v2.py").chmod(0o644)
    elif fault=="symlink":
        alias=directory/"alias.py";alias.symlink_to(source);monkeypatch.setattr(gate,"fit_source",lambda root:alias)
    if fault:
        with pytest.raises(ValueError):gate.source_bindings(tmp_path)
    else:
        rows=gate.source_bindings(tmp_path)
        assert set(rows)=={"v1_quality","public_consumer","policy","frozen_fit","v2_quality"}
        assert rows["v1_quality"]["sha256"]==gate.OLD_SOURCE_SHA and rows["frozen_fit"]["sha256"]==gate.FIT_SOURCE_SHA


def test_full_static_v2_code_only_closure_budget(gate):
    import io
    import tarfile
    import azure_job
    root=INFRA.parent
    files={p.relative_to(root).as_posix():p.read_bytes()for directory in("infra","src","configs")
        for p in(root/directory).rglob("*")if p.is_file()and(p.suffix in(".py",".sh",".cpp",".json",".yaml",".toml")or p.name.startswith("Dockerfile"))}
    files["pyproject.toml"]=(root/"pyproject.toml").read_bytes()
    paths=azure_job.runtime_bundle_paths(files,"infra/run_translation_rgb_evaluate_v2.sh")
    assert "infra/translation_rgb_evaluate.py"in paths and"infra/translation_rgb_public.py"in paths
    assert "infra/translation_rgb_fit.py"in paths  # frozen source-hash dependency, never executed
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode="w",format=tarfile.PAX_FORMAT)as archive:
        for path in paths:
            info=tarfile.TarInfo(path);info.size=len(files[path]);archive.addfile(info,io.BytesIO(files[path]))
    encoded,_=azure_job.encoded_runtime_archive(stream.getvalue())
    assert len(encoded)<=128000
