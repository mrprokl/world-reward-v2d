"""Pure CLI/routing for native stages; no weights, data, torch or Azure calls."""

import builtins
import importlib.util
import json
from pathlib import Path
import sys

import pytest


STAGES = ("cari_body_adapter_smoke", "cari_prepare", "cari_forward", "cari_converter")


@pytest.fixture(params=STAGES)
def stage(request, monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location(f"world_reward_test_{request.param}_routing", infra / f"{request.param}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return request.param, module


def test_native_default_episode_and_all_track1_choices(stage):
    name, module = stage
    defaults = module._argument_parser().parse_args([])
    assert defaults.episode == 15
    if name == "cari_forward": assert defaults.kernel_only is False
    for episode in range(30):
        assert module._argument_parser().parse_args(["--episode", str(episode)]).episode == episode
    if name == "cari_forward":
        for episode in (0, 15, 29):
            args = module._argument_parser().parse_args(["--episode", str(episode), "--kernel-only"])
            assert args.episode == episode and args.kernel_only


@pytest.mark.parametrize("episode", ["-1", "30", "15.0", "true", "../track_3", ""])
def test_invalid_native_episode_fails_before_model_import(stage, episode):
    _, module = stage
    with pytest.raises(SystemExit): module._argument_parser().parse_args(["--episode", episode])


@pytest.mark.parametrize("argument", ["--full-video", "--root", "--manual-mask", "--no-gt"])
def test_native_new_cli_exposes_no_extra_algorithm_or_input_modes(stage, argument):
    _, module = stage
    with pytest.raises(SystemExit): module._argument_parser().parse_args([argument])


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_main_routes_selected_episode_before_torch_or_heavy_io(stage, monkeypatch, episode):
    name, module = stage
    monkeypatch.setenv("WR_ROOT", "/srv/synthetic-root")
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("lo")]))
    monkeypatch.setattr(sys, "argv", [name + ".py", "--episode", str(episode)])
    received = []
    original_import = builtins.__import__
    def no_heavy_import(package, *args, **kwargs):
        if package.split(".")[0] in {"torch", "cv2", "h5py", "trimesh", "PIL"}:
            raise AssertionError("Heavy import occurred before selected input/report gate")
        return original_import(package, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_heavy_import)
    if name == "cari_prepare":
        def input_gate(root, *, episode_index):
            received.append((root, episode_index))
            raise RuntimeError("synthetic routing stop")
        monkeypatch.setattr(module, "_validate_inputs", input_gate)
        expected = (Path("/srv/synthetic-root"), episode)
    else:
        def read_gate(path, *args, **kwargs):
            received.append(path)
            raise RuntimeError("synthetic routing stop")
        monkeypatch.setattr(Path, "read_text", read_gate)
        filename = "cari_forward/report.json" if name == "cari_converter" else "body_full/report.json"
        expected = Path(f"/srv/synthetic-root/outputs/episode_{episode:06d}/{filename}")
    with pytest.raises(RuntimeError, match="synthetic routing stop"): module.main()
    assert received == [expected]


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_checkpoint_only_routes_body_not_prepared_episode_inputs(monkeypatch, episode):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_cari_kernel_routing", infra / "cari_forward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("WR_ROOT", "/srv/synthetic-root")
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("lo")]))
    monkeypatch.setattr(sys, "argv", ["cari_forward.py", "--episode", str(episode), "--kernel-only"])
    reads = []
    def read_gate(path, *args, **kwargs):
        reads.append(path)
        raise RuntimeError("synthetic kernel routing stop")
    monkeypatch.setattr(Path, "read_text", read_gate)
    with pytest.raises(RuntimeError, match="synthetic kernel routing stop"): module.main()
    assert reads == [Path(f"/srv/synthetic-root/outputs/episode_{episode:06d}/body_full/report.json")]


def test_selected_episode_paths_sequences_reports_and_no_algorithm_changes(stage):
    name, module = stage
    text = Path(module.__file__).read_text()
    assert "outputs/episode_000015" not in text
    assert "export/episode_000015" not in text
    assert '"episode_index": args.episode' in text
    if name == "cari_prepare":
        assert 'sequence = f"episode_{args.episode:06d}"' in text
        assert '_validate_inputs(root, episode_index=args.episode)' in text
        assert "writer.mark_complete()" in text
    elif name == "cari_converter":
        assert 'base / f"cari_inputs/export/episode_{args.episode:06d}"' in text
        assert "max_mean_vertex_error_mm=MAX_MEAN_VERTEX_ERROR_MM" in text
        assert "source_poses, object_report = restore_object_source_frame" in text
    elif name == "cari_forward":
        assert 'sys.path[:0] = environment["PYTHONPATH"].split(":")' in text
        assert "stride=96, render_batch_size=32" in text
        assert "Native preparation paths belong to another episode" in text


def test_converter_episode_metadata_guard_legacy_absent_but_present_strict(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_cari_converter_legacy_routing", infra / "cari_converter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for episode in (0, 15, 29):
        module._require_report_episode({}, episode)  # Identity still bound by selected path/hashes in main.
        module._require_report_episode({"episode_index": episode}, episode)
        for bad in (-1, 30, None, True, str(episode), float(episode), (episode + 1) % 30):
            with pytest.raises(ValueError, match="another episode"):
                module._require_report_episode({"episode_index": bad}, episode)


def test_legacy_native_report_stage_and_provenance_not_relaxed_for_episode(monkeypatch, tmp_path):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_cari_converter_legacy_report", infra / "cari_converter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "report.json"
    report = {"stage": "world_reward_native_cari_inputs", "status": "pass", "input_track": "track_1",
              "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": []}
    path.write_text(json.dumps(report))
    assert module._read_report(path, report["stage"], episode_index=15) == report
    report["ground_truth_used"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError): module._read_report(path, report["stage"], episode_index=15)


def test_forward_pass_report_records_enforced_network_namespace_contract():
    import ast
    tree = ast.parse((Path(__file__).resolve().parents[1] / "infra/cari_forward.py").read_text())
    result = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "result" for t in node.targets))
    fields = {key.value: value for key, value in zip(result.keys, result.values) if isinstance(key, ast.Constant)}
    assert isinstance(fields["network"], ast.Constant) and fields["network"].value == "none"
    # The producer guard is before imports/model/forward and no report can be
    # written by the non-loopback branch. This is a contract, not attestation.
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert isinstance(main.body[0], ast.If)
    assert "/sys/class/net" in ast.unparse(main.body[0].test)
    assert isinstance(main.body[0].body[0], ast.Raise)
