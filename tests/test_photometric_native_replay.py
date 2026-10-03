"""Tiny source/provenance tests for the empirical replay, not CUDA inference."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"infra"));monkeypatch.syspath_prepend(str(ROOT/"src"))
    spec=importlib.util.spec_from_file_location("photometric_replay_test",ROOT/"infra/photometric_native_replay.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_exact_source_and_failed_report_pins(replay,tmp_path):
    path=tmp_path/"report.json";path.write_text("{}");path.chmod(0o444)
    from world_reward.data import sha256
    assert replay.pinned(path,sha256(path),2)["bytes"]==2
    for digest,size in [("0"*64,2),(sha256(path),3)]:
        with pytest.raises(ValueError):replay.pinned(path,digest,size)
    path.chmod(0o644)
    with pytest.raises(ValueError):replay.pinned(path,sha256(path))
    assert replay.REVISION=="8c68fba8d4f9034c2e19ed2fa9eac394c9f64758"
    assert replay.FAIL_SHA=="e136b3cd417fd72d54368885e32163250514b692acda9c808f6ea98207a0caaf"


def test_distinct_runtime_mode_not_model_patch_or_tolerance(replay):
    source=Path(replay.__file__).read_text();tree=ast.parse(source)
    calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
    assert "core.run_body"in calls and "deterministic_algorithms=False"in source
    assert 'exact_SHAM_gate_relaxed=False'in source and 'hidden_warmup_calls=0'in source
    assert 'official_cumsum_modified=False'in source
    assert not any(name in calls for name in("torch.cumsum","core.main","core.manufacture.main","setattr","np.load"))
    assert replay.OUT=="validation/photometric_native_v1/capability_replay_v1"
    assert replay.BUDGET==180


def test_original_failure_validated_before_loading_source(replay,tmp_path,monkeypatch):
    base=tmp_path/replay.core.BASE;(base/"automatic_masks").mkdir(parents=True);(base/"capability_v1").mkdir()
    source=replay.historical_source(tmp_path);source.parent.mkdir(parents=True);source.write_text("raise RuntimeError('never execute')")
    for p,row in [(base/"render-report.json",dict(status="pass",phase="complete",code_revision=replay.REVISION,frames=1,private_arrays_created=False,truth_arrays_exported=False)),
                  (base/"automatic_masks/report.json",dict(status="pass",phase="complete",producer_revision=replay.REVISION,script_sha256=replay.SOURCE_SHA)),
                  (base/"capability_v1/report.json",dict(status="pass"))]:p.write_text(json.dumps(row))
    monkeypatch.setattr(replay,"pinned",lambda *a:{"sha256":"0"*64,"bytes":1})
    with pytest.raises(ValueError,match="contract"):replay.historical_inputs(tmp_path,"a"*40)


def test_bad_reader_fails_before_torch_runtime(replay,tmp_path):
    def reject(*a):raise ValueError("bad reader")
    with pytest.raises(ValueError,match="bad reader"):
        replay.core.run_body(tmp_path,tmp_path,{},lambda:None,"a"*40,deterministic_algorithms=False,public_input_reader=reject)
    with pytest.raises(ValueError,match="runtime mode"):
        replay.core.run_body(tmp_path,tmp_path,{},lambda:None,"a"*40,deterministic_algorithms=0,public_input_reader=reject)


def test_native_runtime_guard_explicit_per_branch(replay):
    source=Path(replay.core.__file__).read_text()
    assert "torch.use_deterministic_algorithms(deterministic_algorithms,warn_only=False)"in source
    assert "torch.are_deterministic_algorithms_enabled()!=deterministic_algorithms"in source
    assert "torch.is_deterministic_algorithms_warn_only_enabled()"in source
    assert 'parity(proposal,anchor,exact=True)'in source and 'parity(candidate,fixed[0],exact=True)'in source


def test_wrapper_no_private_mesh_or_failed_arrays(replay):
    p=ROOT/"infra/run_photometric_native_replay.sh";text=p.read_text()
    subprocess.run(["rtk","proxy","bash","-n",str(p)],check=True)
    assert text.count("docker run")==1 and "183s docker run"in text
    assert "eval_private"not in text and "weights/mhr"not in text
    assert 'src=$BASE/capability_v1/report.json,dst=$BASE/capability_v1/report.json,readonly'in text
    assert 'src=$BASE/capability_v1,dst='not in text
    assert 'src=$BASE,dst='not in text and 'src=$OLD,dst=$OLD,readonly'in text
    assert subprocess.run(["rtk","proxy","bash",str(p),"bad"],capture_output=True).returncode==2


def test_cli_no_unregistered_runtime_tuning(replay):
    with pytest.raises(SystemExit):replay.main(["--warn-only"])
