"""Tiny mocked provenance/load routing; no models, media, GPUs or remote calls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_final_episode_gate", infra / "final_episode_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_runtime(gate, monkeypatch, tmp_path):
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("lo")]))
    monkeypatch.setenv("WR_ROOT", str(tmp_path))
    monkeypatch.setenv("WR_CODE_REVISION", "a" * 40)
    calls = []
    def inputs(root, *, episode_index):
        calls.append(("inputs", root, episode_index))
        return {"total_frames": 3, "video_sha256": "b" * 64, "dataset_revision": "c" * 40,
                "mask_report_sha256": "d" * 64}
    def load(root, episode, frames, digest):
        calls.append(("load", root, episode, frames, digest))
        return SimpleNamespace(episode=SimpleNamespace(validate=lambda: calls.append(("validate",))),
                               manifest={"integrity_and_schema_verified": True,
                                         "submission_eligibility_verified": False,
                                         "challenge_performance_verified": False})
    monkeypatch.setattr(gate, "_validate_inputs", inputs)
    monkeypatch.setattr(gate, "load_track1_episode", load)
    return tmp_path, calls


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_original_manifest_then_loader_bound_selected_episode_and_cpu_schema_only(gate, fake_runtime, episode):
    root, calls = fake_runtime
    (root / f"outputs/episode_{episode:06d}").mkdir(parents=True)
    gate.main(["--episode", str(episode)])
    assert calls == [("inputs", root, episode), ("load", root, episode, 3, "b" * 64), ("validate",)]
    report = json.loads((root / f"outputs/episode_{episode:06d}/final_schema/report.json").read_text())
    assert report["stage"] == "world_reward_final_episode_integrity_schema" and report["status"] == "pass"
    assert report["episode_index"] == episode and report["frames"] == 3
    assert report["input_sha256"] == "b" * 64 and report["input_track"] == "track_1"
    assert report["integrity_and_schema_verified"] is True and report["execution_device"] == "CPU"
    assert report["gate_code_revision"] == "a" * 40 and report["all_stage_producer_revisions_same"] is False
    for key in ("numerical_truth_independently_reverified", "submission_eligibility_verified",
                "license_eligibility_verified", "challenge_performance_verified", "complete_challenge_submission_created"):
        assert report[key] is False
    assert report["ground_truth_used"] is False and report["hand_labeled_test"] is False and report["oracle_modes"] == []


@pytest.mark.parametrize("arguments", [[], ["--episode"], ["--episode", "-1"], ["--episode", "30"],
                                      ["--episode", "true"], ["--episode", "15.0"], ["--episode", "../track_2"],
                                      ["--episode", "0", "--root", "/tmp"], ["--ep", "0"], ["--gpu"]])
def test_required_strict_cli_before_input_io(gate, fake_runtime, arguments):
    _, calls = fake_runtime
    with pytest.raises(SystemExit): gate.main(arguments)
    assert calls == []


@pytest.mark.parametrize("revision", ["", "a" * 39, "a" * 41, "A" * 40, "z" * 40])
def test_immutable_revision_required_before_input_reads(gate, fake_runtime, monkeypatch, revision):
    _, calls = fake_runtime
    monkeypatch.setenv("WR_CODE_REVISION", revision)
    with pytest.raises(RuntimeError, match="source revision"): gate.main(["--episode", "0"])
    assert calls == []


@pytest.mark.parametrize("surface", ["Darwin", "Windows"])
def test_never_runs_gate_on_local_nonlinux(gate, fake_runtime, monkeypatch, surface):
    _, calls = fake_runtime
    monkeypatch.setattr(gate.platform, "system", lambda: surface)
    with pytest.raises(RuntimeError, match="remote Linux"): gate.main(["--episode", "0"])
    assert calls == []


def test_network_interface_other_than_loopback_rejected(gate, fake_runtime, monkeypatch):
    _, calls = fake_runtime
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("lo"), Path("eth0")]))
    with pytest.raises(RuntimeError, match="network none"): gate.main(["--episode", "0"])
    assert calls == []


@pytest.mark.parametrize("failure", ["inputs", "loader", "validate"])
def test_any_failed_integrity_or_schema_never_writes_passing_report(gate, fake_runtime, monkeypatch, failure):
    root, _ = fake_runtime
    def fail(*args, **kwargs): raise ValueError("tiny deliberate failure")
    if failure == "inputs": monkeypatch.setattr(gate, "_validate_inputs", fail)
    elif failure == "loader": monkeypatch.setattr(gate, "load_track1_episode", fail)
    else:
        monkeypatch.setattr(gate, "load_track1_episode", lambda *args: SimpleNamespace(episode=SimpleNamespace(validate=fail), manifest={}))
    with pytest.raises(ValueError, match="deliberate"): gate.main(["--episode", "0"])
    assert not (root / "outputs/episode_000000/final_schema").exists()


@pytest.mark.parametrize("kind", ["directory", "report", "broken_symlink"])
def test_frozen_target_never_overwritten_and_rejected_before_reads(gate, fake_runtime, kind):
    root, calls = fake_runtime
    output = root / "outputs/episode_000000/final_schema"
    output.parent.mkdir(parents=True)
    if kind == "broken_symlink": output.symlink_to("absent")
    else:
        output.mkdir()
        if kind == "report": (output / "report.json").write_text("frozen")
    with pytest.raises(FileExistsError): gate.main(["--episode", "0"])
    assert calls == []
    if kind == "report": assert (output / "report.json").read_text() == "frozen"


def test_wrapper_cpu_no_models_gpu_or_writable_input_mount(gate):
    source = Path(gate.__file__).with_name("run_final_episode_gate.sh").read_text()
    assert "--network none" in source and "--gpus" not in source
    assert '"$CODE/infra/final_episode_gate.py" "$@"' in source
    for name in ("data", "results"):
        assert f"src=$ROOT/{name},dst=$ROOT/{name},readonly" in source
    assert "src=$ROOT/outputs,dst=$ROOT/outputs" in source
    assert "src=$ROOT/weights" not in source and "src=$ROOT/vendor" not in source


def test_immutable_runtime_bundle_includes_loader_contract_not_gpu_entrypoints(gate):
    repository = Path(gate.__file__).parents[1]
    spec = importlib.util.spec_from_file_location("world_reward_test_final_gate_bundle", repository / "infra/azure_job.py")
    launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
    files = {str(path.relative_to(repository)): path.read_bytes()
             for base in ("infra", "src", "configs") for path in (repository / base).rglob("*")
             if path.is_file() and "__pycache__" not in path.parts}
    files["pyproject.toml"] = (repository / "pyproject.toml").read_bytes()
    selected = launcher.runtime_bundle_paths(files, "infra/run_final_episode_gate.sh")
    assert set(selected) >= {"infra/final_episode_gate.py", "infra/track1_episode_loader.py", "infra/cari_converter.py", "infra/body_smoke.py"}
    assert "infra/run_cari_forward.sh" not in selected and "infra/run_cari_converter.sh" not in selected


def test_refined_schema_uses_explicit_loader_source_and_fresh_namespace(gate, fake_runtime, monkeypatch):
    root, calls = fake_runtime
    base = root / "outputs/episode_000015"
    (base / "final_schema").mkdir(parents=True)
    (base / "final_schema/report.json").write_text("frozen forward")
    def load(root, episode, frames, digest, *, bundle_source):
        calls.append(("load_refined", root, episode, frames, digest, bundle_source))
        return SimpleNamespace(episode=SimpleNamespace(validate=lambda: None), manifest={"bundle_source": bundle_source})
    monkeypatch.setattr(gate, "load_track1_episode", load)
    gate.main(["--episode", "15", "--bundle-source", "refined"])
    assert calls[-1] == ("load_refined", root, 15, 3, "b" * 64, "refined")
    report = json.loads((base / "final_schema_refined/report.json").read_text())
    assert report["bundle_source"] == "refined"
    assert (base / "final_schema/report.json").read_text() == "frozen forward"
