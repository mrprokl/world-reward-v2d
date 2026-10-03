"""Tiny procedural wrapper controls only, no Azure/assets/archive acquisition."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "infra/run_frontend_asset_archive.sh"
CONFIG = ROOT / "configs/frontend_replica_inventory_pins.json"
OLD = "40fdc2780076d3f83632dc20c7926a3a6763e9d7"


def block(name):
    return WRAPPER.read_text().split("<<'" + name + "'\n", 1)[1].split("\n" + name, 1)[0]


def pin(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), git_blob_sha1=hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest())


def execute(source, args):
    return subprocess.run(["rtk", "proxy", sys.executable, "-I", "-B", "-", *map(str, args)],
        input=source, capture_output=True, text=True, timeout=10)


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "runtime"; revision = "a" * 40
    code = root / "jobs" / revision / "run_frontend_asset_archive/code"; (code / "infra").mkdir(parents=True)
    (root / "results").mkdir()
    (code.parent / "revision").write_text(revision + "\n"); (code.parent / "source-sha256").write_text("b" * 64 + "\n")
    oldcode = root / "jobs" / OLD / "run_frontend_replica_inventory/code"; (oldcode / "infra").mkdir(parents=True)
    oldraw = {"infra/frontend_replica_inventory.py": b"# original procedural inventory only\n", "infra/run_frontend_replica_inventory.sh": b"# original procedural wrapper only\n"}
    config = json.loads(CONFIG.read_bytes()); config["inventory_source_pins"] = {n: pin(raw) for n, raw in oldraw.items()}
    original = root / config["inventory_report"]["pathrelative"]; original.parent.mkdir(); original.write_bytes(b'{"status":"pass","tiny_procedural_receipt":true}\n'); original.chmod(0o400)
    config["inventory_report"]["pin"].update({k: pin(original.read_bytes())[k] for k in ("bytes", "sha256")})
    config_raw = json.dumps(config, sort_keys=True, indent=2).encode(); config_path = code / "configs/frontend_replica_inventory_pins.json"; config_path.parent.mkdir(); config_path.write_bytes(config_raw)
    fixture_driver = '''from pathlib import Path
import hashlib,os
def _source(code,pins):return True
def _read_pinned(path,pin):
 raw=Path(path).read_bytes()
 if len(raw)!=pin['bytes']or hashlib.sha256(raw).hexdigest()!=pin['sha256']:raise ValueError('Old receipt changed')
 return raw
def build_archive(root,report,oldcode,pin,helpers,out,exporter):
 with out.open('xb')as h:h.write(b'tiny procedural archive only')
 out.chmod(0o400)
 return dict(files=448,links=3,total_bytes=19910803804,archive_identity=dict(bytes=28,sha256='c'*64),manifest_identity=dict(bytes=1,sha256='d'*64))
def verify_archive(path,archive,manifest,oldcode,pins):
 return dict(status='pass',extraction_performed=False)
'''
    (code / "infra/frontend_asset_archive.py").write_text(fixture_driver)
    (code / "infra/frontend_replica_inventory.py").write_text("# current provenance-only helper\n")
    (code / "infra/run_frontend_replica_inventory.sh").write_text("# current provenance-only wrapper\n")
    for name, raw in oldraw.items(): (oldcode / name).write_bytes(raw)
    config_identity = {k: pin(config_raw)[k] for k in ("bytes", "sha256")}
    text = WRAPPER.read_text().replace("/srv/scenesmith/world-reward", str(root))
    text = text.replace("CONFIG_BYTES=1155", "CONFIG_BYTES=" + str(config_identity["bytes"]))
    text = text.replace("CONFIG_SHA256=312a10621c02c02449d0786649eac137abc6d9d314236f851848942fb4bff72e", "CONFIG_SHA256=" + config_identity["sha256"])
    # Production fixed original receipt identity is replaced only in this tiny
    # test copy; the original production wrapper remains independently pinned.
    text = text.replace("pin['bytes']!=149203", "pin['bytes']!=" + str(original.stat().st_size))
    text = text.replace("95d09454133e481324380e9253d235ba3bbdd3657e1fac25f97e8f7c4f1741f1", pin(original.read_bytes())["sha256"])
    wrapper = code / "infra/run_frontend_asset_archive.sh"; wrapper.write_text(text)
    for base in (code, oldcode):
        for path in (base, *base.rglob("*")): path.chmod(0o555 if path.is_dir() else 0o444)
    out = root / "results" / ("frontend-asset-archive-" + revision)
    integrity = text.split("<<'PYINTEGRITY'\n")[1].split("\nPYINTEGRITY")[0]
    args = [root, code, revision, wrapper, OLD, config_identity["bytes"], config_identity["sha256"]]
    def fingerprint(): return execute(integrity, args)
    def reserve(): return execute(block("PYRESERVE"), [root, out, revision])
    def control(before): return execute(block("PYCONTROL"), [root, code, revision, out, config_identity["bytes"], config_identity["sha256"], before])
    # Never carry local credentials into pytest fixtures/failure representations.
    env = dict(PATH=os.environ.get("PATH", "/usr/bin:/bin"), HOME=str(tmp_path),
        WR_ROOT=str(root), WR_CODE=str(code), WR_CODE_REVISION=revision)
    def run(*args): return subprocess.run(["rtk", "proxy", "bash", str(wrapper), *args], env=env, text=True, capture_output=True, timeout=10)
    return dict(root=root, code=code, out=out, wrapper=wrapper, oldcode=oldcode, oldreceipt=original,
        config=config_path, fingerprint=fingerprint, reserve=reserve, control=control, env=env, run=run)


def test_frozen_production_config_and_cpu_shell_contract():
    raw = CONFIG.read_bytes(); assert len(raw) == 1155
    assert hashlib.sha256(raw).hexdigest() == "312a10621c02c02449d0786649eac137abc6d9d314236f851848942fb4bff72e"
    subprocess.run(["rtk", "proxy", "bash", "-n", str(WRAPPER)], check=True)
    text = WRAPPER.read_text()
    assert "1800s nice -n 15 ionice -c 3" in text and "ulimit -v 4194304" in text
    assert "env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B" in text
    assert "operational_same_run_archive_verification_is_not_independent_external_validation=True" in text
    assert not any(word in text for word in ("--gpus", "docker run", "docker save", "docker load", "nvidia-smi", "flock", "systemctl", "runuser", "urlopen", "snapshot_download"))
    assert not re.search(r"(?m)^\s*(?:rm|rmdir)\s", text)
    assert "PENDING" not in text and "declare -A" not in text and "mapfile" not in text


def test_pure_controls_genuine_namespace_fingerprint_archive_selfverify_receipt(runtime):
    before = runtime["fingerprint"](); assert before.returncode == 0, before.stderr
    assert re.fullmatch(r"[0-9a-f]{64}\n", before.stdout)
    assert runtime["reserve"]().returncode == 0
    result = runtime["control"](before.stdout.strip()); assert result.returncode == 0, result.stderr
    report = json.loads((runtime["out"] / "report.json").read_bytes())
    assert report["status"] == "pass" and report["replica_ready"] is report["license_eligibility_verified"] is report["CUDA_verified"] is False
    assert report["operational_same_run_archive_verification_is_not_independent_external_validation"] is True
    assert {p.name for p in runtime["out"].iterdir()} == {"archive.tar", "report.json"}
    assert runtime["out"].stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o400 for p in runtime["out"].iterdir())
    assert "inventory_source_pins" not in result.stdout and "entries" not in result.stdout and len(result.stdout) < 1000
    assert runtime["fingerprint"]().stdout == before.stdout


@pytest.mark.parametrize("args", [["--help"], ["--resume"], ["--root", "/tmp"], ["--budget", "100"]])
def test_no_cli_controls_before_source_or_output(runtime, args):
    assert runtime["run"](*args).returncode == 2
    assert not runtime["out"].exists()


@pytest.mark.parametrize("fault", ["root", "namespace", "revision", "darwin"])
def test_shell_rejects_wrong_host_or_namespace_without_work(runtime, fault, tmp_path):
    if fault == "root": runtime["env"]["WR_ROOT"] = str(tmp_path)
    elif fault == "namespace": runtime["env"]["WR_CODE"] = str(tmp_path)
    elif fault == "revision": runtime["env"]["WR_CODE_REVISION"] = "A" * 40
    else:
        # Current developer host is Darwin; preserve a positive control by
        # testing the Linux guard text instead of emulating a production run.
        assert '"$(uname -s)" == Linux' in WRAPPER.read_text(); return
    assert runtime["run"]().returncode == 2 and not runtime["out"].exists()


@pytest.mark.parametrize("fault", ["writable", "symlink", "hardlink", "marker", "config", "oldhelper", "oldreceipt", "missing"])
def test_integrity_checks_current_old_source_and_frozen_config_before_output(runtime, fault, tmp_path):
    code = runtime["code"]
    if fault == "marker": (code.parent / "revision").write_text("c" * 40 + "\n")
    elif fault == "oldreceipt": runtime["oldreceipt"].chmod(0o600); runtime["oldreceipt"].write_bytes(b"changed"); runtime["oldreceipt"].chmod(0o400)
    elif fault == "config": runtime["config"].chmod(0o600); runtime["config"].write_text("{}"); runtime["config"].chmod(0o444)
    else:
        path = (runtime["oldcode"] if fault == "oldhelper" else code) / "infra/frontend_replica_inventory.py"
        if fault == "writable": path.chmod(0o644)
        elif fault == "hardlink": os.link(path, tmp_path / "alias")
        elif fault == "oldhelper": path.chmod(0o600); path.write_bytes(b"changed"); path.chmod(0o444)
        else:
            path.parent.chmod(0o755); path.unlink()
            if fault == "symlink": path.symlink_to("frontend_asset_archive.py")
    assert runtime["fingerprint"]().returncode != 0 and not runtime["out"].exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_exclusive_result_no_overwrite_or_cleanup(runtime, kind):
    out = runtime["out"]
    if kind == "directory": out.mkdir(); (out / "userwork").write_text("retained")
    elif kind == "file": out.write_text("retained")
    else: out.symlink_to("missing")
    assert runtime["reserve"]().returncode != 0
    assert out.exists() or out.is_symlink()
    if kind == "directory": assert (out / "userwork").read_text() == "retained"
    elif kind == "file": assert out.read_text() == "retained"


@pytest.mark.parametrize("fault", ["current", "config", "old", "marker"])
def test_control_rejects_post_fingerprint_mutation(runtime, fault):
    before = runtime["fingerprint"](); assert before.returncode == 0
    assert runtime["reserve"]().returncode == 0
    path = runtime["code"] / "infra/frontend_asset_archive.py" if fault == "current" else runtime["config"] if fault == "config" else runtime["oldcode"] / "infra/frontend_replica_inventory.py" if fault == "old" else runtime["code"].parent / "revision"
    path.chmod(0o600); path.write_text("changed"); path.chmod(0o444)
    assert runtime["control"](before.stdout.strip()).returncode != 0
    assert not any(runtime["out"].iterdir())


def test_before_receipt_complete_markers_rechecked(runtime):
    code = runtime["code"]; driver = code / "infra/frontend_asset_archive.py"; driver.chmod(0o600)
    source = driver.read_text().replace("return dict(status='pass',extraction_performed=False)", "(Path(__file__).parent.parent.parent/'revision').write_text('c'*40+'\\n')\n return dict(status='pass',extraction_performed=False)")
    driver.write_text(source); driver.chmod(0o444)
    before = runtime["fingerprint"](); assert before.returncode == 0, before.stderr
    assert runtime["reserve"]().returncode == 0
    result = runtime["control"](before.stdout.strip()); assert result.returncode != 0
    report = json.loads((runtime["out"] / "report.json").read_bytes())
    assert report["status"] == "fail" and report["error"] == "Immutable archive source/provenance changed"


@pytest.mark.parametrize("fault", ["secret_exception", "unknown_report", "out_replaced"])
def test_control_failure_sanitized_and_unknown_results_not_overwritten(runtime, fault):
    driver = runtime["code"] / "infra/frontend_asset_archive.py"; driver.chmod(0o600); text = driver.read_text()
    if fault == "secret_exception":
        text = text.replace("with out.open('xb')as h:h.write(b'tiny procedural archive only')", "raise ValueError('hf_'+'x'*30+' private credential must not print')")
    elif fault == "unknown_report":
        text = text.replace("return dict(status='pass',extraction_performed=False)", "(path.parent/'report.json').write_bytes(b'user receipt must remain')\n return dict(status='pass',extraction_performed=False)")
    else:
        text = text.replace("return dict(status='pass',extraction_performed=False)", "out=path.parent;out.rename(out.with_name('retained-owned-original'));out.mkdir(mode=0o700);(out/'unknown').write_bytes(b'user work')\n return dict(status='pass',extraction_performed=False)")
    driver.write_text(text); driver.chmod(0o444)
    before = runtime["fingerprint"](); assert before.returncode == 0
    assert runtime["reserve"]().returncode == 0
    result = runtime["control"](before.stdout.strip()); assert result.returncode != 0
    assert "private credential" not in result.stderr + result.stdout and "hf_" not in result.stderr + result.stdout
    if fault == "secret_exception":
        report = json.loads((runtime["out"] / "report.json").read_bytes())
        assert report["status"] == "fail" and report["error_type"] == "ValueError"
        assert {p.name for p in runtime["out"].iterdir()} == {"report.json"}
    elif fault == "unknown_report": assert (runtime["out"] / "report.json").read_bytes() == b"user receipt must remain"
    else:
        assert (runtime["out"] / "unknown").read_bytes() == b"user work" and not (runtime["out"] / "report.json").exists()


def test_bundle_retains_complete_literal_provenance_closure(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "infra")); import azure_job
    files = {str(p.relative_to(ROOT)): p.read_bytes() for folder in ("infra", "src", "configs") for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    selected = azure_job.runtime_bundle_paths(files, "infra/run_frontend_asset_archive.sh")
    assert {n for n in selected if n.startswith("infra/")} == {"infra/run_frontend_asset_archive.sh", "infra/frontend_asset_archive.py", "infra/frontend_replica_inventory.py", "infra/run_frontend_replica_inventory.sh"}
    assert "configs/frontend_replica_inventory_pins.json" in selected
