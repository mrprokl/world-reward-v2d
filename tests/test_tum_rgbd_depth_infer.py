"""Tiny blind driver/control fixtures; no real media, model or GPU execution."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
from test_tum_rgbd_depth_inputs import cohort as input_cohort


@pytest.fixture
def gate(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'src'));monkeypatch.syspath_prepend(str(root/'infra'))
    return importlib.import_module('tum_rgbd_depth_infer')


def cohort(gate, tmp_path): return input_cohort(gate.inputs, tmp_path)


def test_genuine_original_loader_and_grid_bytes_no_TUDL_namespace_patch(gate):
    root = Path(__file__).resolve().parents[1]
    assert gate.BUDGET == 300 and gate.IMAGE_ID == "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
    for name, digest in gate.FROZEN_HELPERS.items(): assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert gate.native.inputs.FRAME_IDS == ((1788, 3235, 5138, 6925), (1566, 3137, 4894, 6490), (1512, 3227, 4819, 6647))
    assert gate.inputs.ordered_frames(dict(ordered_frames=[[s, f] for s in (1, 2, 3) for f in (40, 80, 120, 160)])) == tuple((s, f) for s in (1, 2, 3) for f in (40, 80, 120, 160))
    assert gate.GRID.shape == (480, 640) and gate.GRID.K.tolist() == gate.native.K.tolist()
    source = (root / "infra/tum_rgbd_depth_infer.py").read_text()
    assert "native.da3_frame(torch" in source and "native.load_da3(root" in source
    assert "contract.checked_moge_infer(torch" in source and "contract.scene_anchors(GRID" in source
    assert all(text not in source for text in ("native.run(", "native.load_public(", "native.bound_source(", "eval_private", "tum_rgbd_depth_render"))


def test_wrapper_exposes_only_infra_src_single_pin_and_public_files_never_recipe(gate):
    path = Path(__file__).resolve().parents[1] / "infra/run_tum_rgbd_depth_infer.sh"; source = path.read_text()
    assert "src=$CODE,dst=$CODE" not in source and "src=$CODE/configs,dst=" not in source
    assert "src=$CODE/infra,dst=$CODE/infra,readonly" in source and "src=$CODE/src,dst=$CODE/src,readonly" in source
    assert "src=$CODE/configs/tum_rgbd_depth_input_pins.json,dst=$CODE/configs/tum_rgbd_depth_input_pins.json,readonly" in source
    assert 'src=$path,dst=$path,readonly' in source and 'path="$BASE/inputs/$name"' in source
    assert all(text not in source for text in ("eval_private", "tum_rgbd_depth_protocol", "tum_rgbd_depth_transport_v2.json", "MHR", 'src=$BASE,dst=$BASE'))
    assert "--network none" in source and "flock --nonblock 9" in source
    assert "303s docker run" in source and "--kill-after=10s" in source and "AFTER=\"$(integrity)\"" in source
    assert '--name "$CONTAINER"' in source and 'CONTAINER="world-reward-tum-depth-anchor-$REV"' in source
    assert "trap finish EXIT" in source and "cleanup_container || STATUS=1" in source


def test_actual_archive_keeps_all_configs_but_numerical_closure_excludes_manufacture(gate):
    root = Path(__file__).resolve().parents[1]
    launcher = importlib.import_module("azure_job")
    files = {str(path.relative_to(root)): path.read_bytes() for directory in ("infra", "src", "configs")
        for path in (root / directory).rglob("*") if path.is_file() and path.suffix in (".py", ".sh", ".json", ".toml", ".cpp")}
    files["pyproject.toml"] = (root / "pyproject.toml").read_bytes()
    files[gate.PIN_FILE] = b"{}"
    selected = set(launcher.runtime_bundle_paths(files, "infra/run_tum_rgbd_depth_infer.sh"))
    assert {name for name in selected if name.startswith(("infra/", "src/"))} == set(gate.SOURCE_FILES)
    assert {name for name in files if name.startswith("configs/")} <= selected
    assert not any("render" in name or "reference" in name or "evaluate" in name for name in selected if name.startswith("infra/"))


def test_bounded_container_cleanup_executes_owned_only_stop_kill_rm(gate):
    source = (Path(__file__).resolve().parents[1] / "infra/run_tum_rgbd_depth_infer.sh").read_text()
    function = source[source.index("cleanup_container() {"):source.index("finish() {")]
    harness = r'''
set -euo pipefail
OWNED_CONTAINER=0;CONTAINER=world-reward-tum-depth-anchor-test;PRESENT=1;CALLS=""
timeout(){ while [[ "$1" != docker ]];do shift;done;"$@"; }
docker(){
 CALLS+="$*|"
 if [[ "$1" == ps ]];then [[ "$PRESENT" == 0 ]] || echo owned;return 0;fi
 if [[ "$1" == rm ]];then PRESENT=0;fi
}
''' + function + r'''
cleanup_container
[[ -z "$CALLS" ]]
OWNED_CONTAINER=1
cleanup_container
[[ "$PRESENT" == 0 && "$CALLS" == *"stop --time 5 $CONTAINER"* && "$CALLS" == *"kill $CONTAINER"* && "$CALLS" == *"rm --force $CONTAINER"* ]]
'''
    result = subprocess.run(["rtk", "proxy", "bash", "-c", harness], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_source_public_models_dependency_all_postrechecked(gate, monkeypatch):
    report = dict(source_helpers={"source": "old"}, input_pins={"pin": "old"}, public_records=[],
        MoGe_bindings={"moge": "old"}, DA3_assets={"da3": "old"}, DA3_dependency={"wheel": "old"}); seen = []
    monkeypatch.setattr(gate, "bound_source", lambda *_: seen.append("source") or report["source_helpers"])
    monkeypatch.setattr(gate, "load_public", lambda *_: ([], dict(input_pins=report["input_pins"], public_records=[])))
    monkeypatch.setattr(gate.native, "moge_bindings", lambda *_: seen.append("moge") or (Path("/model"), report["MoGe_bindings"]))
    monkeypatch.setattr(gate.native.da3, "pinned_assets", lambda *_: seen.append("da3") or (report["DA3_assets"], Path("/source")))
    monkeypatch.setattr(gate.native, "da3_dependency", lambda *_: seen.append("wheel") or report["DA3_dependency"])
    gate.verify_bindings(Path("/remote"), Path("/code"), "a" * 40, report)
    assert seen == ["source", "moge", "da3", "wheel"] and report["source_assets_after_reverified"] is True
    monkeypatch.setattr(gate.native.da3, "pinned_assets", lambda *_: ({"da3": "changed"}, Path("/source")))
    with pytest.raises(ValueError, match="DA3"): gate.verify_bindings(Path("/remote"), Path("/code"), "a" * 40, report)


@pytest.mark.parametrize("support_failure", [False, True])
def test_tiny_simulated_control_chain_completes_12_native_calls_and_11_saved_arrays(gate, tmp_path, monkeypatch, support_failure):
    """Mock control flow only, not inference/geometry/quality evidence."""
    directory, pins, _ = cohort(gate, tmp_path)
    records, receipt = gate.inputs.public_inputs(directory, pins)
    output = directory.parent / gate.OUTPUT; output.mkdir(); (output / "report.json").write_text("mock only")
    report = {key: 0 for key in ("MoGe_attempts", "MoGe_returns", "MoGe_calls_completed", "DA3_attempts", "DA3_returns", "DA3_calls_completed")}
    report.update(native_focal_solver_calls=[], outputs=[], whole_support_diagnostics=[]); events = []
    class Tensor:
        def __init__(self, value): self.value = value
        def cuda(self): return self
        def permute(self, *_): return self
        def float(self): return self
        def __truediv__(self, _): return self
        def __getitem__(self, _): return self
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.value.copy()
    class Network:
        def cuda(self): return self
        def eval(self): return self
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None, empty_cache=lambda: events.append("released"), max_memory_allocated=lambda: 0)
    torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()), cudnn=types.SimpleNamespace())
    torch.set_num_threads = lambda _: None; torch.from_numpy = Tensor; torch.__version__ = "mock"; torch.version = types.SimpleNamespace(cuda="mock")
    alias = object(); module = types.SimpleNamespace(recover_focal_shift=alias,
        MoGeModel=types.SimpleNamespace(from_pretrained=lambda _: Network()))
    model = types.ModuleType("moge.model"); model.v2 = module
    utils = types.ModuleType("moge.utils"); utils.geometry_torch = types.SimpleNamespace(recover_focal_shift=alias)
    for name, value in (("torch", torch), ("moge", types.ModuleType("moge")), ("moge.model", model), ("moge.utils", utils)):
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(gate, "load_public", lambda *_: (records, dict(input_pins=pins, input_manifest=receipt)))
    monkeypatch.setattr(gate, "read_rgb", lambda row: (np.array([[[1, 2, 3]]], np.uint8), row["sha256"]))
    monkeypatch.setattr(gate.native, "moge_bindings", lambda *_: (Path("/mock-model"), {}))
    monkeypatch.setattr(gate.native.da3, "pinned_assets", lambda *_: ({}, Path("/mock-source")))
    monkeypatch.setattr(gate.native, "da3_dependency", lambda *_: {})
    monkeypatch.setattr(gate.native, "seed", lambda *_: None)
    baseline = dict(depth=np.array([[2.]], np.float32), points=np.array([[[0., 0., 2.]]], np.float32),
        validity=np.array([[True]], np.bool_), K=gate.GRID.K)
    def checked(torch, grid, module, original, network, tensor, diagnostics, context):
        assert original is alias
        diagnostics.append(dict(**context, native_nearest64_valid_pixels=2, focal_prior_supplied=True, original_solver_returned=True))
        events.append("moge")
        return {key: Tensor(baseline[source]) for key, source in (("depth", "depth"), ("points", "points"), ("mask", "validity"), ("intrinsics", "K"))}, {}
    monkeypatch.setattr(gate.contract, "checked_moge_infer", checked)
    monkeypatch.setattr(gate.contract, "camera_arrays", lambda grid, depth, points, valid, K: (dict(depth=depth, points=points, validity=valid, K=K), {}))
    monkeypatch.setattr(gate.contract, "validate_baseline", lambda *_: {})
    def load_da3(*_): assert events == ["moge"] * 12 + ["released"]; events.append("da3_load"); return object(), object(), object()
    monkeypatch.setattr(gate.native, "load_da3", load_da3)
    monkeypatch.setattr(gate.native, "da3_frame", lambda *_: (events.append("da3") or np.array([[1.]], np.float32), {}))
    monkeypatch.setattr(gate.contract, "whole_support_ratio", lambda *_: dict(ratio_median=2., passed=not support_failure))
    persisted = []
    def require(row):
        assert persisted[-1]["whole_support_diagnostics"][-1] == row
        if support_failure: raise ValueError("declared support gate failed")
    monkeypatch.setattr(gate.contract, "require_support", require)
    def anchors(grid, rows, ordered):
        assert [(row["scene_id"], row["frame_id"]) for row in rows] == list(ordered)
        assert len(rows) == 12; return {1: 2., 2: 2., 3: 2.}
    monkeypatch.setattr(gate.contract, "scene_anchors", anchors)
    def candidate(grid, base, raw, coefficient, scene, frame):
        return dict(moge_depth=base["depth"].copy(), moge_points=base["points"].copy(), moge_validity=base["validity"].copy(),
            da3_depth=raw.copy(), anchored_depth=raw * coefficient, anchored_points=base["points"].copy(),
            anchored_validity=base["validity"].copy(), K=base["K"].copy(), scene_id=np.array(scene, np.int64),
            frame_id=np.array(frame, np.int64), scene_anchor=np.array(coefficient, np.float64))
    monkeypatch.setattr(gate.contract, "candidate_arrays", candidate)
    monkeypatch.setattr(gate.contract, "validate_prediction_arrays", lambda *_: {})
    monkeypatch.setattr(gate, "verify_bindings", lambda *_: None)
    def persist(): persisted.append(json.loads(json.dumps(report)))
    if support_failure:
        with pytest.raises(ValueError, match="declared support gate"):
            gate.run(tmp_path, tmp_path / "code", "a" * 40, output, report, persist)
        assert report["MoGe_calls_completed"] == 12 and report["DA3_calls_completed"] == 1
        assert report["whole_support_diagnostics"] == [dict(ratio_median=2., passed=False, scene_id=1, frame_id=40)]
        assert "scene_anchor_coefficients" not in report and report["outputs"] == []
        assert {p.name for p in output.iterdir()} == {"report.json"}
        return
    gate.run(tmp_path, tmp_path / "code", "a" * 40, output, report, persist)
    assert report["status"] == "pass" and report["outputs_completed"] == 12
    assert report["scene_anchor_coefficients"] == [2., 2., 2.] and len(report["whole_support_diagnostics"]) == 12
    assert len(report["native_focal_solver_calls"]) == 12
    assert all(set(row["arrays"]) == gate.native.ARRAY_KEYS for row in report["outputs"])


@pytest.mark.parametrize("fault", [None, "unexpected", "symlink"])
def test_failed_cleanup_scoped_preserves_receipt_and_unknown_files(gate, tmp_path, fault):
    directory, pins, _ = cohort(gate, tmp_path); output = directory.parent / gate.OUTPUT; output.mkdir()
    receipt = output / "report.json"; receipt.write_bytes(b"original failure"); receipt.chmod(0o400)
    name = Path(gate.inputs.filenames(pins)[0]).stem + ".npz"; partial = output / name
    if fault == "symlink": partial.symlink_to(receipt)
    else: partial.write_bytes(b"partial predicted arrays"); partial.chmod(0o444)
    if fault == "unexpected": (output / "unknown.npz").write_bytes(b"do not delete")
    report = dict(status="fail", input_pins=pins)
    if fault:
        with pytest.raises(ValueError, match="preserved"): gate.cleanup_failed_outputs(output, report)
        assert (output / (name if fault == "symlink" else "unknown.npz")).exists()
    else:
        gate.cleanup_failed_outputs(output, report)
        assert report["partial_prediction_arrays_removed"] is True and report["partial_prediction_array_count_removed"] == 1
        assert {p.name for p in output.iterdir()} == {"report.json"}
    assert receipt.read_bytes() == b"original failure" and receipt.stat().st_mode & 0o777 == 0o400


@pytest.mark.parametrize("owned", [False, True])
def test_container_query_failure_is_closed_without_unrelated_stop(gate, owned):
    source = (Path(__file__).resolve().parents[1] / "infra/run_tum_rgbd_depth_infer.sh").read_text()
    function = source[source.index("cleanup_container() {"):source.index("finish() {")]
    harness = 'set -euo pipefail\nOWNED_CONTAINER=' + str(int(owned)) + ';CONTAINER=owned-test\n'
    harness += 'timeout(){ while [[ "$1" != docker ]];do shift;done;"$@"; }\n'
    harness += 'docker(){ [[ "$1" == ps ]] || exit 88;return 7; }\n' + function + '\ncleanup_container\n'
    result = subprocess.run(["rtk", "proxy", "bash", "-c", harness], capture_output=True, text=True)
    assert result.returncode == (1 if owned else 0)


@pytest.mark.parametrize("query", ["empty", "busy", "failed"])
def test_wrapper_finish_requires_actual_empty_GPU_without_killing_foreign_PID(gate, query):
    source = (Path(__file__).resolve().parents[1] / "infra/run_tum_rgbd_depth_infer.sh").read_text()
    function = source[source.index("finish() {"):source.index("trap finish EXIT")]
    harness = r'''
set -euo pipefail
OWNED_CONTAINER=1;BEFORE=""
cleanup_container(){ return 0; }
timeout(){ while [[ "$1" != nvidia-smi ]];do shift;done;"$@"; }
''' + ({"empty": 'nvidia-smi(){ return 0; }\n', "busy": 'nvidia-smi(){ echo 99999;return 0; }\n',
    "failed": 'nvidia-smi(){ return 7; }\n'}[query]) + function + '\nfinish\n'
    result = subprocess.run(["rtk", "proxy", "bash", "-c", harness], capture_output=True, text=True)
    assert result.returncode == (0 if query == "empty" else 1)
    assert ("actual_GPU_compute_apps_empty_after=true" in result.stdout) == (query == "empty")
    assert "no unrelated process killed" in result.stderr if query != "empty" else not result.stderr


@pytest.mark.parametrize("failure", ["timeout", "postcheck"])
def test_main_seals_failure_receipt_and_restores_300s_signal(gate, tmp_path, monkeypatch, failure):
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); native_iter = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda p: iter([Path("lo")]) if str(p) == "/sys/class/net" else native_iter(p))
    output = tmp_path / gate.BASE / gate.OUTPUT; output.mkdir(parents=True)
    for name, value in dict(WR_ROOT=str(tmp_path), WR_CODE=str(tmp_path / "code"), WR_CODE_REVISION="a" * 40, WR_IMAGE_ID=gate.IMAGE_ID).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(gate, "bound_source", lambda *_: {gate.SOURCE_FILES[0]: dict(sha256="a" * 64, bytes=1)})
    handlers = {gate.signal.SIGALRM: object(), gate.signal.SIGTERM: object()}; before = handlers.copy(); alarms = []
    def signal(number, handler): previous = handlers[number]; handlers[number] = handler; return previous
    monkeypatch.setattr(gate.signal, "signal", signal); monkeypatch.setattr(gate.signal, "alarm", alarms.append)
    def run(root, code, revision, output, report, persist):
        if failure == "timeout": handlers[gate.signal.SIGALRM]()
        report.update(status="pass", phase="complete")
    monkeypatch.setattr(gate, "run", run)
    def verify(*_):
        if failure == "postcheck": raise ValueError("changed source")
    monkeypatch.setattr(gate, "verify_bindings", verify)
    with pytest.raises((TimeoutError, ValueError)): gate.main([])
    assert handlers == before and alarms == [300, 0]
    path = output / "report.json"; report = json.loads(path.read_text())
    assert path.stat().st_mode & 0o777 == 0o400 and report["status"] == "fail"
    assert report["ground_truth_used"] is False and report["private_truth_read"] is False
    assert report["adoption_performed"] is False and report["real_world_generalization_verified"] is False
    assert report["training_overlap_verified"] is False and report["support_diagnostics_recorded_before_gate"] is True
    assert report["minimum_pairs"] == 1024 and report["minimum_grid_coverage"] == .25
    assert report["partial_prediction_arrays_removed"] is True
