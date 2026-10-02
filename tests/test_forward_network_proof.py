"""Tiny legacy source/report binding fixtures, no actual model or Azure calls."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def proof(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_forward_proof", infra / "forward_network_proof.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def records(proof):
    forward = {"stage": "world_reward_native_cari_full_forward", "status": "pass", "input_track": "track_1",
               "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
               "script_sha256": proof.LEGACY_SCRIPT_SHA256, "bundle_sha256": proof.LEGACY_BUNDLE_SHA256,
               "episode_inputs_used": True, "checkpoint_sha256": "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3",
               "metadata": {"actual_network_forward_verified": True, "full_original_frame_coverage_verified": True}}
    sidecar = {"stage": proof.PROOF_STAGE, "status": "pass", "episode_index": 15,
               "legacy_revision": proof.LEGACY_REVISION, "producer_script_sha256": proof.LEGACY_SCRIPT_SHA256,
               "wrapper_sha256": proof.LEGACY_WRAPPER_SHA256, "forward_report_sha256": proof.LEGACY_REPORT_SHA256,
               "bundle_sha256": proof.LEGACY_BUNDLE_SHA256, "basis": proof.PROOF_BASIS, "runtime_guard_source_verified": True,
               "immutable_wrapper_source_verified": True, "immutable_wrapper_launch_bound": False, "network_security_attestation": False,
               "producer_report_modified": False, "source_archive_sha256": proof.LEGACY_ARCHIVE_SHA256,
               "source_archive_rehashed": False}
    return forward, sidecar


def test_allowlisted_legacy_sources_match_actual_committed_bytes_and_guards(proof):
    repository = Path(proof.__file__).parents[1]
    for relative, expected in (("infra/cari_forward.py", proof.LEGACY_SCRIPT_SHA256),
                               ("infra/run_cari_forward.sh", proof.LEGACY_WRAPPER_SHA256)):
        source = subprocess.check_output(["rtk", "proxy", "git", "show", f"{proof.LEGACY_REVISION}:{relative}"], cwd=repository)
        assert hashlib.sha256(source).hexdigest() == expected
        if relative.endswith(".py"):
            assert b'{p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}' in source
            assert source.index(b'Path("/sys/class/net")') < source.index(b"import torch")
        else:
            assert b"--gpus all --network none" in source and b"src=$CODE,dst=$CODE,readonly" in source


def test_original_report_not_changed_and_proof_not_security_attestation(proof, records):
    forward, sidecar = records
    before = copy.deepcopy(records)
    proof.validate_proof(sidecar, forward, proof.LEGACY_REPORT_SHA256, proof.LEGACY_BUNDLE_SHA256, 15)
    assert records == before and "network" not in forward
    assert sidecar["network_security_attestation"] is False


@pytest.mark.parametrize("field,value", [("status", "fail"), ("episode_index", 0), ("episode_index", True),
                                        ("producer_script_sha256", "0" * 64), ("wrapper_sha256", "0" * 64),
                                        ("forward_report_sha256", "0" * 64), ("bundle_sha256", "0" * 64),
                                        ("runtime_guard_source_verified", 1), ("immutable_wrapper_launch_bound", True),
                                        ("immutable_wrapper_source_verified", False), ("source_archive_rehashed", True),
                                        ("network_security_attestation", True), ("producer_report_modified", True),
                                        ("source_archive_sha256", ""), ("source_archive_sha256", "a" * 64),
                                        ("launch_exec_start_sha256", "A" * 64),
                                        ("basis", "assumed_from_missing_field")])
def test_bad_proof_cannot_fill_legacy_network_field(proof, records, field, value):
    records[1][field] = value
    with pytest.raises(ValueError): proof.validate_proof(records[1], records[0], proof.LEGACY_REPORT_SHA256, proof.LEGACY_BUNDLE_SHA256, 15)


@pytest.mark.parametrize("change", ["explicit_none", "explicit_default", "wrong_source", "wrong_episode"])
def test_proof_only_for_exact_missing_field_producer(proof, records, change):
    forward, sidecar = records
    if change.startswith("explicit"): forward["network"] = "none" if change == "explicit_none" else "default"
    elif change == "wrong_source": forward["script_sha256"] = "0" * 64
    else: forward["episode_index"] = 0
    with pytest.raises(ValueError): proof.validate_proof(sidecar, forward, proof.LEGACY_REPORT_SHA256, proof.LEGACY_BUNDLE_SHA256, 15)


@pytest.mark.parametrize("episode", [0, 29, True, "15", 15.])
def test_no_generic_legacy_episode_exception(proof, records, episode):
    with pytest.raises(ValueError): proof.validate_proof(records[1], records[0], proof.LEGACY_REPORT_SHA256, proof.LEGACY_BUNDLE_SHA256, episode)


@pytest.mark.parametrize("report_hash,bundle_hash", [("a" * 64, None), (None, "b" * 64)])
def test_even_valid_other_report_and_bundle_hashes_are_not_allowlisted(proof, records, report_hash, bundle_hash):
    report_hash = report_hash or proof.LEGACY_REPORT_SHA256
    bundle_hash = bundle_hash or proof.LEGACY_BUNDLE_SHA256
    records[1]["forward_report_sha256"], records[1]["bundle_sha256"] = report_hash, bundle_hash
    records[0]["bundle_sha256"] = bundle_hash
    with pytest.raises(ValueError): proof.validate_proof(records[1], records[0], report_hash, bundle_hash, 15)


@pytest.fixture
def fake_runtime(proof, records, monkeypatch, tmp_path):
    root = tmp_path; base = root / "outputs/episode_000015"
    job = root / f"jobs/{proof.LEGACY_REVISION}/run_cari_forward"
    (job / "code/infra").mkdir(parents=True)
    (job / "code/infra/cari_forward.py").write_text("synthetic source hash mocked")
    (job / "code/infra/run_cari_forward.sh").write_text("synthetic wrapper hash mocked")
    (job / "code/infra/cari_forward.py").chmod(0o444)
    (job / "code/infra/run_cari_forward.sh").chmod(0o444)
    (job / "revision").write_text(proof.LEGACY_REVISION + "\n")
    (job / "source-sha256").write_text(proof.LEGACY_ARCHIVE_SHA256 + "\n")
    (base / "cari_forward").mkdir(parents=True); (base / "cari_conversion").mkdir()
    bundle = base / "cari_forward/coconet.pth"; bundle.write_bytes(b"tiny synthetic not a model")
    forward, _ = records
    report_path = base / "cari_forward/report.json"; report_path.write_text(json.dumps(forward))
    final = {"stage": "world_reward_native_cari_official_conversion", "status": "pass", "input_track": "track_1",
             "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
             "producer_revision": proof.LEGACY_REVISION,
             "script_sha256": "d2642f9816a6c7d6146b550a4b6500e108a33ff48ce55f54dec9bd7deddd1bc3",
             "bundle_sha256": proof.LEGACY_BUNDLE_SHA256, "input_report_sha256": {"forward": proof.LEGACY_REPORT_SHA256}}
    (base / "cari_conversion/report.json").write_text(json.dumps(final))
    real_hash = proof.sha256
    def fake_source_hash(path):
        if path == job / "code/infra/cari_forward.py": return proof.LEGACY_SCRIPT_SHA256
        if path == job / "code/infra/run_cari_forward.sh": return proof.LEGACY_WRAPPER_SHA256
        if path == report_path: return proof.LEGACY_REPORT_SHA256
        if path == bundle and path.read_bytes() == b"tiny synthetic not a model": return proof.LEGACY_BUNDLE_SHA256
        return real_hash(path)
    monkeypatch.setattr(proof, "sha256", fake_source_hash)
    monkeypatch.setenv("WR_ROOT", str(root)); monkeypatch.setenv("WR_CODE_REVISION", "a" * 40)
    monkeypatch.setattr(proof.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("lo")]))
    return root, base, job, report_path


def test_cpu_driver_writes_exclusive_sidecar_not_original_report(proof, fake_runtime):
    root, base, _, report_path = fake_runtime
    original = report_path.read_bytes()
    proof.main([])
    sidecar = json.loads((base / "cari_forward/network_proof.json").read_text())
    assert report_path.read_bytes() == original
    assert sidecar["source_archive_rehashed"] is False and sidecar["network_security_attestation"] is False
    assert sidecar["immutable_wrapper_source_verified"] is True and sidecar["immutable_wrapper_launch_bound"] is False
    assert "launch_exec_start_sha256" not in sidecar
    with pytest.raises(FileExistsError): proof.main([])


@pytest.mark.parametrize("failure", ["source", "revision", "archive", "bundle", "final_link", "symlink", "writable_script", "writable_wrapper"])
def test_driver_failed_binding_never_publishes_sidecar(proof, fake_runtime, monkeypatch, failure):
    root, base, job, report_path = fake_runtime
    if failure == "source": monkeypatch.setattr(proof, "sha256", lambda path: "0" * 64)
    elif failure == "revision": (job / "revision").write_text("0" * 40)
    elif failure == "archive": (job / "source-sha256").write_text("not-hash")
    elif failure == "bundle": (base / "cari_forward/coconet.pth").write_bytes(b"changed")
    elif failure == "final_link":
        path = base / "cari_conversion/report.json"; record = json.loads(path.read_text())
        record["input_report_sha256"]["forward"] = "0" * 64; path.write_text(json.dumps(record))
    elif failure == "symlink":
        source = job / "revision"; moved = job / "outside"; source.rename(moved); source.symlink_to(moved)
    else:
        source = job / "code/infra" / ("cari_forward.py" if failure == "writable_script" else "run_cari_forward.sh")
        source.chmod(0o644)
    with pytest.raises(ValueError): proof.main([])
    assert not (base / "cari_forward/network_proof.json").exists()


def test_wrapper_cpu_source_audit_no_systemctl_mounts_jobs_readonly(proof):
    source = Path(proof.__file__).with_name("run_forward_network_proof.sh").read_text()
    assert "systemctl" not in source and "ExecStart" not in source
    assert "--network none" in source and "--gpus" not in source
    assert "src=$ROOT/jobs,dst=$ROOT/jobs,readonly" in source
    assert "src=$ROOT/weights" not in source and "src=$ROOT/data" not in source
