"""Tiny CPU stubs verify native ABI binding and sealed fail-closed routing.

No Torch, model assets, real-video values, CUDA, optimizer efficacy or adoption.
The NumPy callbacks are procedural linear fixtures, not MHR accuracy proof.
"""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runtime(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"; monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("wr_test_diagnostic_runtime", infra/"cari_converter.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class Tensor(np.ndarray):
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)


def bindings(runtime, monkeypatch, tmp_path):
    tool, model = tmp_path/"official.py", tmp_path/"model.pt"
    tool.write_text("own stub verified only in tests"); model.write_bytes(b"own tiny asset")
    checks = []; calls = []
    monkeypatch.setattr(runtime, "_require_hash", lambda path, expected: checks.append((path, expected)))
    torch = SimpleNamespace(float64=np.float64, float32=np.float32,
        tensor=lambda value, *, dtype, device: np.array(value, dtype=dtype, copy=True).view(Tensor))
    namespace = dict(np=np, Tensor=Tensor, calls=calls, SimpleNamespace=SimpleNamespace)
    source = '''
class MHR:
    def __init__(self, path, device, chunk, precision):
        calls.append(("MHR", path, device, chunk, precision))
        self.dtype = np.float64
        self.mdtype = np.float64 if precision == "float64" else np.float32
        self.fd_step = 1e-6 if precision == "float64" else 1e-3
        self.chunk = chunk; self.device = SimpleNamespace(type=device)
    def run(self, p, z):
        calls.append(("run", str(self.mdtype), p.copy(), z.copy()))
        v = np.zeros((len(p), 18439, 3), np.float64).view(Tensor)
        v[:, :, 0] = p[:, :1]
        return v, np.zeros((len(p), 127, 3), np.float64).view(Tensor)
def lm_pose(mhr, tgt, pose, z, *, iters, tol, frames_per_batch):
    calls.append(("lm_pose", iters, tol, frames_per_batch, pose.copy(), z.copy(), tgt.copy()))
    proposal = pose.copy(); proposal[:, 0] = 0
    return proposal, np.zeros(len(pose), np.float64).view(Tensor)
'''
    exec(compile(source, str(tool), "exec"), namespace)
    converter = SimpleNamespace(__file__=str(tool), MHR=namespace["MHR"], lm_pose=namespace["lm_pose"])
    return torch, converter, tool, model, checks, calls


def data(runtime):
    errors = np.array([.5, 3., 1., 2., 4.], np.float32); count = len(errors)
    pose = np.zeros((count, 136), np.float32); pose[:, 0] = errors
    converted = dict(pose=pose, scales=np.zeros(68, np.float32), shape=np.zeros(45, np.float32),
        valid_input=np.ones(count, bool), per_frame_vertex_error_mm=errors,
        report=dict(frames=count, fitted_frames=count, invalid_input_frames=[], precision="float32",
                    vertex_error_mm=dict(mean=float(errors.mean()), worst_frame_mean=float(errors.max()))))
    params = {k: np.zeros((count, d), np.float32) for k, d in runtime.PARAMETER_DIMS.items()}
    native = np.zeros((count, 18439, 3), np.float32)
    provenance = dict(native_bundle_sha256="a"*64, ground_truth_used=False, hand_labeled_test=False,
                      oracle_modes=[], input_track="track_1")
    return converted, params, native, provenance


def test_original_cli_and_names_unchanged_optional_branch(runtime):
    args = runtime._argument_parser().parse_args([])
    assert args.episode == 15 and args.bundle_source == "forward" and args.diagnostic_only is False
    assert runtime.conversion_directory("forward") == "cari_conversion"
    assert runtime.conversion_directory("refined") == "cari_conversion_refined"
    assert runtime.diagnostic_directory("forward") == "cari_converter_diagnostic_forward_v1"
    assert runtime.diagnostic_directory("refined") == "cari_converter_diagnostic_refined_v1"
    args = runtime._argument_parser().parse_args(["--episode", "0", "--bundle-source", "refined", "--diagnostic-only"])
    assert args.episode == 0 and args.bundle_source == "refined" and args.diagnostic_only is True
    with pytest.raises(ValueError): runtime.diagnostic_directory("fallback")


def test_canonical_native_hash_c_order_little_endian_no_mutation(runtime):
    array = np.arange(12, dtype=np.float32).reshape(2, 2, 3); original = array.copy()
    identities = [runtime.canonical_array_identity(value) for value in
                  (array, np.asfortranarray(array), array.astype(">f4"))]
    assert identities[0] == identities[1] == identities[2]
    assert identities[0]["shape"] == [2, 2, 3] and identities[0]["bytes"] == 48
    assert np.array_equal(array, original)
    with pytest.raises(ValueError): runtime.canonical_array_identity(np.ma.array(array))
    with pytest.raises(ValueError): runtime.canonical_array_identity(np.array([np.nan]))


def test_real_binding_protocol_owned_copies_and_execution_counters(runtime, monkeypatch, tmp_path):
    torch, converter, tool, model, checks, calls = bindings(runtime, monkeypatch, tmp_path); report = {}
    polish, replay = runtime.bind_official_diagnostic_callbacks(torch, converter, tool, model, report)
    p = np.zeros((3, 136), np.float64); p[:, 0] = 2.; z = np.zeros((1, 113), np.float64)
    target = np.zeros((3, 18439*3), np.float64); originals = [x.copy() for x in (p, z, target)]
    q, errors = polish(target, p, z, iters=60, tol=1e-5)
    v = replay(p.astype(np.float32), z.astype(np.float32))
    assert q.dtype == errors.dtype == v.dtype == np.float64 and np.all(errors == 0)
    assert report["runtime_bindings"]["float64_pose_polish_calls_completed"] == 1
    assert report["runtime_bindings"]["float32_reference_replay_calls_completed"] == 1
    assert report["runtime_bindings"]["float64_initial_mean_vertex_error_mm"] == [2.]*3
    assert [c[1:4] for c in calls if c[0] == "lm_pose"] == [(60, 1e-5, 3)]
    assert checks == [(tool, runtime.CONVERTER_SHA256), (model, runtime.REFERENCE_MODEL_SHA256)]
    assert not any(k in report for k in ("source_bindings_runtime_verified", "actual_official_polish_verified"))
    q[:] = 10
    assert all(np.array_equal(a, b) for a, b in zip((p, z, target), originals))


@pytest.mark.parametrize("fault", ["module_path", "function_path", "fd64", "fd32", "chunk", "dtype", "device"])
def test_binding_rejects_source_precision_or_fd_drift(runtime, monkeypatch, tmp_path, fault):
    torch, converter, tool, model, _, _ = bindings(runtime, monkeypatch, tmp_path)
    if fault == "module_path": converter.__file__ = str(tmp_path/"wrong.py")
    elif fault == "function_path": converter.lm_pose = lambda *a, **k: None
    else:
        original = converter.MHR.__init__
        def changed(self, *args, **kwargs):
            original(self, *args, **kwargs)
            if fault == "fd64" and self.mdtype == np.float64: self.fd_step = 1e-3
            elif fault == "fd32" and self.mdtype == np.float32: self.fd_step = 1e-6
            elif fault == "chunk": self.chunk = 4
            elif fault == "dtype": self.dtype = np.float32
            elif fault == "device": self.device = SimpleNamespace(type="cpu")
        converter.MHR.__init__ = changed
    with pytest.raises(RuntimeError): runtime.bind_official_diagnostic_callbacks(torch, converter, tool, model, {})


@pytest.mark.parametrize("fault", ["iterations", "tolerance", "count", "dtype", "shape", "nan"])
def test_polish_abi_is_fixed_not_an_optimizer_tuning_interface(runtime, monkeypatch, tmp_path, fault):
    torch, converter, tool, model, _, _ = bindings(runtime, monkeypatch, tmp_path)
    polish, _ = runtime.bind_official_diagnostic_callbacks(torch, converter, tool, model, {})
    p = np.zeros((3, 136), np.float64); z = np.zeros((1, 113), np.float64); target = np.zeros((3, 18439*3), np.float64)
    iters, tol = 60, 1e-5
    if fault == "iterations": iters = 61
    elif fault == "tolerance": tol = 1e-4
    elif fault == "count": p = np.zeros((4, 136), np.float64)
    elif fault == "dtype": p = p.astype(np.float32)
    elif fault == "shape": z = np.zeros((3, 113), np.float64)
    else: p[0, 0] = np.nan
    with pytest.raises(ValueError): polish(target, p, z, iters=iters, tol=tol)


def test_full_original_fail_is_sealed_and_only_stable_error_probes_polished(runtime, monkeypatch, tmp_path):
    torch, converter, tool, model, _, calls = bindings(runtime, monkeypatch, tmp_path)
    converted, params, native, provenance = data(runtime); original = copy.deepcopy(converted)
    output = tmp_path/"new_diagnostic"; output.mkdir(); report = {}; phases = []
    runtime.finish_diagnostic(output, converted, params, native, torch=torch, converter=converter, tool=tool, model=model,
        episode=0, frame_indices=np.arange(5), provenance=provenance, report=report, persist=lambda: phases.append(report["phase"]))
    assert phases == ["original_sealed", "probe_polish", "complete"]
    assert report["status"] == "pass" and report["original_gate_pass"] is False
    assert report["original"]["gate_failure_count"] == 2 and report["original"]["probe_indices"] == [0, 3, 4]
    assert report["actual_official_polish_verified"] is True and report["source_bindings_runtime_verified"] is True
    assert report["full_frame_fidelity_verified"] is False
    assert report["probe_report"]["adoption_authorized"] is False
    assert report["probe_report"]["full_frame_fidelity_verified"] is False
    assert report["probe_report"]["probe_gate_pass"] is True
    assert report["probe_report"]["before_mean_vertex_error_mm"] == [.5, 2., 4.]
    assert report["provenance"]["native_vertices_sha256"] == runtime.canonical_array_identity(native)["sha256"]
    assert len([c for c in calls if c[0] == "lm_pose"]) == 1
    with np.load(output/"sealed_original/original_converter.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["pose"], original["pose"])
    with np.load(output/"native_parameters.npz", allow_pickle=False) as saved:
        assert set(saved) == set(runtime.PARAMETER_DIMS) | {"frame_index"}
        assert np.array_equal(saved["mhr_shape"], params["mhr_shape"])
    with np.load(output/"probes.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["frame_index"], [0, 3, 4]) and len(saved["quantized_probe_pose"]) == 3
    frozen = json.loads((output/"sealed_original/original_report.json").read_text())
    assert frozen["status"] == "sealed" and frozen["source_bindings_runtime_verified"] is False
    assert np.array_equal(converted["pose"], original["pose"])
    assert not (output/"episode.npz").exists() and not (output/"params.npz").exists()
    with pytest.raises(ValueError):
        runtime.finish_diagnostic(output, converted, params, native, torch=torch, converter=converter, tool=tool, model=model,
            episode=0, frame_indices=np.arange(5), provenance=provenance, report={}, persist=lambda: None)


def test_late_native_failure_keeps_original_archive_not_false_bindings(runtime, monkeypatch, tmp_path):
    torch, converter, tool, model, _, _ = bindings(runtime, monkeypatch, tmp_path)
    converted, params, native, provenance = data(runtime); output = tmp_path/"failed_diagnostic"; output.mkdir(); report = {}
    monkeypatch.setattr(runtime, "bind_official_diagnostic_callbacks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("actual native failure")))
    with pytest.raises(RuntimeError, match="actual native"):
        runtime.finish_diagnostic(output, converted, params, native, torch=torch, converter=converter, tool=tool, model=model,
            episode=15, frame_indices=np.arange(5), provenance=provenance, report=report, persist=lambda: None)
    assert (output/"sealed_original/original_converter.npz").is_file()
    assert (output/"native_parameters.npz").is_file() and report["phase"] == "original_sealed"
    assert report.get("actual_official_polish_verified") is not True and not (output/"probes.npz").exists()


def test_diagnostic_wrapper_nested_output_only_rw_no_wait_or_default_episode(runtime):
    wrapper = Path(runtime.__file__).with_name("run_cari_converter_diagnostic.sh").read_text()
    mounts = [line for line in wrapper.splitlines() if "--mount" in line]
    assert len(mounts) == 6 and sum("readonly" in line for line in mounts) == 5
    assert 'src=$ROOT/outputs,dst=$ROOT/outputs,readonly' in wrapper and 'src=$OUT,dst=$OUT' in wrapper
    assert "--network none --memory 32g --cpus 4" in wrapper and "903s" in wrapper
    assert '"$2" == 0 || "$2" == 15' in wrapper and "world-reward-cari-forward.service" not in wrapper
    assert "--diagnostic-only" in wrapper and "chown -R" not in wrapper


def main_environment(runtime, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: iter([Path("lo")]) if path == Path("/sys/class/net") else original(path))
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "b"*40)
    monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"c"*64)
    for episode in (0, 15): (tmp_path/f"outputs/episode_{episode:06d}").mkdir(parents=True)


def test_main_legacy_route_does_not_reserve_diagnostic_outputs(runtime, monkeypatch, tmp_path):
    main_environment(runtime, monkeypatch, tmp_path); calls = []
    monkeypatch.setattr(runtime, "_convert_episode", lambda args, diagnostic=None: calls.append((args, diagnostic)))
    runtime.main(["--episode", "15"])
    assert len(calls) == 1 and calls[0][1] is None and calls[0][0].bundle_source == "forward"
    assert list((tmp_path/"outputs/episode_000015").iterdir()) == []


def test_main_diagnostic_failure_persists_and_rejects_overwrite(runtime, monkeypatch, tmp_path):
    main_environment(runtime, monkeypatch, tmp_path)
    def fails(args, diagnostic):
        diagnostic["report"]["phase"] = "actual_failed_call"
        raise RuntimeError("own simulated native failure")
    monkeypatch.setattr(runtime, "_convert_episode", fails)
    with pytest.raises(RuntimeError, match="own simulated"):
        runtime.main(["--episode", "0", "--diagnostic-only"])
    path = tmp_path/"outputs/episode_000000/cari_converter_diagnostic_forward_v1/report.json"
    report = json.loads(path.read_text()); original = path.read_bytes()
    assert report["status"] == "fail" and report["phase"] == "actual_failed_call"
    assert report["source_bindings_runtime_verified"] is False and report["submission_produced"] is False
    assert report["budget_seconds"] == 900 and report["adoption_authorized"] is False
    with pytest.raises(FileExistsError): runtime.main(["--episode", "0", "--diagnostic-only"])
    assert path.read_bytes() == original


def test_main_diagnostic_restricts_cohort_before_artifact_writes(runtime, monkeypatch, tmp_path):
    main_environment(runtime, monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="restricted"):
        runtime.main(["--episode", "29", "--diagnostic-only"])
    assert not (tmp_path/"outputs/episode_000029").exists()
