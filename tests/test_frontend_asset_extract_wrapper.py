"""Tiny isolated wrapper controls; no real disk, Azure, SSH or asset reads."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "infra/run_frontend_asset_extract.sh"


def block(name):
    return WRAPPER.read_text().split("<<'" + name + "'\n", 1)[1].split("\n" + name, 1)[0]


def execute(source, args):
    return subprocess.run(["rtk", "proxy", sys.executable, "-I", "-B", "-", *map(str, args)],
        input=source, capture_output=True, text=True, timeout=10)


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "root"; rev = "a" * 40
    code = root / "jobs" / rev / "run_frontend_asset_extract/code"; (code / "infra").mkdir(parents=True)
    (root / "results").mkdir()
    for name, raw in (("revision", rev + "\n"), ("source-sha256", "b" * 64 + "\n")):
        (code.parent / name).write_text(raw)
    for name in ("frontend_replica_inventory.py", "run_frontend_replica_inventory.sh"):
        (code / "infra" / name).write_bytes((ROOT / "infra" / name).read_bytes())
    (code / "infra/frontend_asset_archive.py").write_text("# tiny authenticated fixture helper\n")
    fixture = '''from pathlib import Path
import hashlib,json
def extract_archive(archive,pin,manifest,oldcode,helpers,dest,sources):
 assert pin=={'bytes':19911464960,'sha256':'5b817ea15e98f1f18165529fcac3ca0b7fc9f88b96d22f2195396db6ffbf8342'}
 assert manifest=={'bytes':131666,'sha256':'16a6b5314b205c5ad526df2340ab99e46873fd5d62b1c8811f9a210184bfb0f8'}
 for name,expected in helpers.items():
  raw=(oldcode/name).read_bytes()
  assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\\0'+raw).hexdigest()}==expected
 for name,expected in sources.items():
  raw=(oldcode/name).read_bytes();assert {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==expected
 if dest.exists()or dest.is_symlink():return dict(stage='frontend_asset_extract',status='fail',replica_ready=False,promotion_performed=False,partial_output_retained=False)
 dest.mkdir(mode=0o700)
 (dest/'called.json').write_text(json.dumps(dict(archive=str(archive),oldcode=str(oldcode),helpers=list(helpers),sources=list(sources))))
 return dict(stage='frontend_asset_extract',status='pass',replica_ready=False,promotion_performed=False,imported_files=448,imported_links=3)
'''
    (code / "infra/frontend_asset_extract.py").write_text(fixture)
    (code / "infra/run_frontend_asset_extract.sh").write_bytes(WRAPPER.read_bytes())
    disk = tmp_path / "data"; incoming = disk / "frontend-assets-v1"; incoming.mkdir(parents=True, mode=0o700)
    archive = incoming / "archive.tar"; archive.write_bytes(b"tiny procedural archive only"); archive.chmod(0o400)
    report = dict(schema="world_reward.frontend_peer_receive.v1", status="pass", bytes_received=19911464960,
        receipt_written=True, archive_identity=dict(bytes=19911464960, sha256="5b817ea15e98f1f18165529fcac3ca0b7fc9f88b96d22f2195396db6ffbf8342"))
    receipt = incoming / "report.json"; receipt.write_text(json.dumps(report)); receipt.chmod(0o400)
    for path in (code, *code.rglob("*")): path.chmod(0o555 if path.is_dir() else 0o444)
    out = root / "results" / ("frontend-asset-extract-" + rev); out.mkdir(mode=0o700)
    dest = disk / ("frontend-assets-extracted-" + rev)
    settings = dict(uuid="24df126a-5f5f-41d8-801c-9ddaa7a582d8", free=30_000_000_000)
    def control():
        # The temporary test copy changes only platform/disk reality. The real
        # source-pinned API is independently covered by its procedural tests.
        source = block("PY").replace("Path('/srv/world-reward-data')", "Path(" + repr(str(disk)) + ")")
        source = source.replace("os.geteuid()!=0", "False").replace(".st_uid!=0", ".st_uid!=os.getuid()")
        prelude = "import os,subprocess,types\nsubprocess.check_output=lambda *a,**k:" + repr(settings["uuid"]) + "\nos.statvfs=lambda *a:types.SimpleNamespace(f_bavail=" + str(settings["free"]) + ",f_frsize=1)\n"
        return execute(prelude + source, (root, code, rev, out))
    return dict(root=root, code=code, disk=disk, incoming=incoming, archive=archive, receipt=receipt,
        report=report, out=out, dest=dest, rev=rev, settings=settings, control=control)


def test_shell_syntax_ast_source_closure_and_isolation():
    subprocess.run(["rtk", "proxy", "bash", "-n", str(WRAPPER)], check=True)
    ast.parse(block("PY")); ast.parse(block("PYPREFLIGHT"))
    text = WRAPPER.read_text()
    assert "1200s nice -n 15 ionice -c 3" in text and "ulimit -v 4194304" in text
    assert text.count("env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B") == 2
    assert text.index("PYPREFLIGHT\n") < text.index('mkdir -m 700 "$OUT"')
    assert not any(x in text for x in ("--gpus", "docker run", "docker load", "docker save", "snapshot_download", "urlopen", "rmtree", "os.environ"))


def test_positive_control_fixed_pins_original_helpers_and_private_receipt(runtime):
    result = runtime["control"](); assert result.returncode == 0, result.stderr
    report = json.loads((runtime["out"] / "report.json").read_bytes())
    assert report["status"] == "pass" and report["replica_ready"] is report["promotion_performed"] is False
    assert report["imported_files"] == 448 and report["imported_links"] == 3
    call = json.loads((runtime["dest"] / "called.json").read_bytes())
    assert call["oldcode"] == str(runtime["code"])
    assert set(call["helpers"]) == {"infra/frontend_replica_inventory.py", "infra/run_frontend_replica_inventory.sh"}
    assert set(call["sources"]) == {"infra/frontend_asset_extract.py", "infra/frontend_asset_archive.py"}
    assert (runtime["out"] / "report.json").stat().st_mode & 0o777 == 0o400
    assert len(result.stdout) < 1000 and "helpers" not in result.stdout and "sources" not in result.stdout


@pytest.mark.parametrize("fault", ["uuid", "space", "transport", "archive_pin", "receipt_mode", "foreign_input"])
def test_disk_and_completed_transport_fail_before_extraction(runtime, fault):
    if fault == "uuid": runtime["settings"]["uuid"] = "different disk"
    elif fault == "space": runtime["settings"]["free"] = 20_999_999_999
    elif fault == "receipt_mode": runtime["receipt"].chmod(0o644)
    elif fault == "foreign_input": (runtime["incoming"] / "unexpected").write_text("user work")
    else:
        report = runtime["report"]
        if fault == "transport": report["status"] = "fail"
        else: report["archive_identity"]["sha256"] = "0" * 64
        runtime["receipt"].chmod(0o600); runtime["receipt"].write_text(json.dumps(report)); runtime["receipt"].chmod(0o400)
    assert runtime["control"]().returncode != 0
    assert not runtime["dest"].exists() and not any(runtime["out"].iterdir())


@pytest.mark.parametrize("fault", ["source", "marker", "oldhelper", "source_alias"])
def test_immutable_source_and_original_helper_pins_fail_before_output(runtime, fault):
    path = runtime["code"] / "infra/frontend_asset_archive.py"
    if fault == "marker": (runtime["code"].parent / "revision").write_text("c" * 40 + "\n")
    elif fault == "source": path.chmod(0o644)
    elif fault == "oldhelper":
        path = runtime["code"] / "infra/frontend_replica_inventory.py"; path.chmod(0o600); path.write_text("changed"); path.chmod(0o444)
    else:
        path.parent.chmod(0o755); path.unlink(); path.symlink_to("frontend_asset_extract.py")
    assert runtime["control"]().returncode != 0 and not runtime["dest"].exists()


def test_existing_destination_not_merged_or_removed(runtime):
    runtime["dest"].mkdir(mode=0o700); (runtime["dest"] / "user").write_bytes(b"retained")
    assert runtime["control"]().returncode != 0
    assert (runtime["dest"] / "user").read_bytes() == b"retained"
    assert json.loads((runtime["out"] / "report.json").read_bytes())["status"] == "fail"


@pytest.mark.parametrize("fault", ["source", "transport"])
def test_post_api_source_and_transport_rechecked_before_receipt(runtime, fault):
    path = runtime["code"] / "infra/frontend_asset_extract.py"; path.chmod(0o600)
    operation = "(oldcode.parent/'source-sha256').write_text('c'*64+'\\n')" if fault == "source" else "(archive.parent/'report.json').chmod(0o600);(archive.parent/'report.json').write_text('{}');(archive.parent/'report.json').chmod(0o400)"
    source = path.read_text().replace("return dict(stage='frontend_asset_extract',status='pass'", operation + "\n return dict(stage='frontend_asset_extract',status='pass'")
    path.write_text(source); path.chmod(0o444)
    assert runtime["control"]().returncode != 0
    assert json.loads((runtime["out"] / "report.json").read_bytes())["status"] == "fail"


def test_preflight_rejects_output_parent_alias_before_shell_reservation(runtime, tmp_path):
    alias = tmp_path / "alias"; alias.symlink_to(runtime["root"] / "results")
    out = alias / "unwritten"
    result = execute(block("PYPREFLIGHT"), (runtime["root"], runtime["code"], out))
    assert result.returncode != 0 and not out.exists()


def test_complete_bundle_retains_current_original_byteidentical_helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "infra")); import azure_job
    files = {str(p.relative_to(ROOT)): p.read_bytes() for folder in ("infra", "src", "configs")
        for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    selected = azure_job.runtime_bundle_paths(files, "infra/run_frontend_asset_extract.sh")
    assert {p for p in selected if p.startswith("infra/")} == {"infra/run_frontend_asset_extract.sh", "infra/frontend_asset_extract.py",
        "infra/frontend_asset_archive.py", "infra/frontend_replica_inventory.py", "infra/run_frontend_replica_inventory.sh"}
    expected = {"infra/frontend_replica_inventory.py": "f9cbb398a53beb257c580df0959c47c707e978afb653ea411382ae8e205821ee",
        "infra/run_frontend_replica_inventory.sh": "db01c510368134cb0dcac6fc10cc73d8e18f689d635c7fd62248b0fd11727fc3"}
    assert all(hashlib.sha256(files[n]).hexdigest() == sha for n, sha in expected.items())
    assert all(n in selected for n in files if n.startswith("configs/"))
