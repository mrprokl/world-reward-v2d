"""Small CPU wrapper/firewall fixtures, no RGB, checkpoint or private labels."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "infra/run_tudl_anchor_evaluate_v2.sh"
ORIGINAL_SHA = "079bf028d2cc8f8b219b7ac726429b1402a4475f58b3e1647e51b2dad795a4ef"
IMAGE = "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
REQUIRED = ("infra/tudl_anchor_evaluate.py", "infra/tudl_anchor_infer.py",
    "infra/tudl_holdout_inputs.py", "infra/tudl_evaluate.py",
    "configs/tudl_frame_holdout_input_pins.json", "configs/tudl_anchor_prediction_pins.json")


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "runtime"; revision = "a" * 40
    code = root / "jobs" / revision / "run_tudl_anchor_evaluate_v2/code"
    (code / "infra").mkdir(parents=True)
    (code.parent / "revision").write_text(revision + "\n")
    (code.parent / "source-sha256").write_text("b" * 64 + "\n")
    wrapper = code / "infra/run_tudl_anchor_evaluate_v2.sh"
    original=root/"validation/tudl_frame_holdout_v1/quality_anchor_v1/report.json"
    original.parent.mkdir(parents=True);original.write_bytes(b"tiny original failed fixture".ljust(2928,b" "));original.chmod(0o400)
    fixture_sha=hashlib.sha256(original.read_bytes()).hexdigest()
    wrapper.write_text(WRAPPER.read_text().replace("/srv/scenesmith/world-reward", str(root)).replace(ORIGINAL_SHA,fixture_sha))
    for name in REQUIRED:
        path = code / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tiny own fixture; not executed\n")
    for path in (code, *code.rglob("*")): path.chmod(0o555 if path.is_dir() else 0o444)
    base = root / "validation/tudl_frame_holdout_v1"
    for name in ("inputs", "anchor_predictions_v1", "eval_private"):
        (base / name).mkdir(parents=True); (base / name / "unchanged").write_text("tiny original")
    bin_dir = tmp_path / "bin"; bin_dir.mkdir(); log = tmp_path / "calls.jsonl"
    scripts = {
        "chown": "#!/bin/sh\nexit 0\n",
        "timeout": "#!/bin/sh\nprintf '%s\\n' \"$*\" > \"$FAKE_TIMEOUT\"\nshift 3;exec \"$@\"\n",
        "docker": f'''#!{sys.executable}
import json,os,pathlib,sys
args=sys.argv[1:]
with open(os.environ['FAKE_LOG'],'a') as stream: stream.write(json.dumps(args)+'\\n')
if args[:2]==['image','inspect']:
 counter=pathlib.Path(os.environ['FAKE_COUNTER']); n=int(counter.read_text()) if counter.exists() else 0
 counter.write_text(str(n+1)); print(os.environ['FAKE_IMAGE_AFTER'] if n else os.environ['FAKE_IMAGE']);sys.exit(0)
if args[0]!='run':sys.exit(9)
code=pathlib.Path(os.environ['WR_CODE']);root=pathlib.Path(os.environ['WR_ROOT'])
out=root/'validation/tudl_frame_holdout_v1/quality_anchor_v2'
report=out/'report.json';report.write_text('{{"status":"pass","adoption_performed":false}}');report.chmod(0o400)
fault=os.environ.get('FAKE_MUTATION','')
if fault=='revision':(code.parent/'revision').write_text('c'*40+'\\n')
elif fault=='archive':(code.parent/'source-sha256').write_text('d'*64+'\\n')
elif fault in ('source','pins'):
 p=code/('infra/tudl_anchor_evaluate.py' if fault=='source' else 'configs/tudl_anchor_prediction_pins.json')
 p.chmod(0o644);p.write_text('changed');p.chmod(0o444)
sys.exit(int(os.environ.get('FAKE_STATUS','0')))
''',
    }
    for name, source in scripts.items():
        path = bin_dir / name; path.write_text(source); path.chmod(0o755)
    env = dict(os.environ, WR_ROOT=str(root), WR_CODE=str(code), WR_CODE_REVISION=revision,
        FAKE_IMAGE=IMAGE, FAKE_IMAGE_AFTER=IMAGE, FAKE_LOG=str(log),
        FAKE_COUNTER=str(tmp_path / "counter"), FAKE_TIMEOUT=str(tmp_path / "timeout"),
        PATH=str(bin_dir) + ":" + os.environ["PATH"])
    def run(*args):
        return subprocess.run(["rtk", "proxy", "bash", str(wrapper), *args],
            env=env, capture_output=True, text=True, timeout=8)
    return root, code, base, wrapper, env, log, run


@pytest.mark.parametrize("status", [0, 7, 124, 137])
def test_exact_cpu_eight_mounts_private_readonly_and_status(runtime, status):
    root, code, base, wrapper, env, log, run = runtime; env["FAKE_STATUS"] = str(status)
    before = {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}
    result = run(); assert result.returncode == status, result.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 3 and rows[0] == rows[-1] == ["image", "inspect", IMAGE, "--format", "{{.Id}}"]
    args = rows[1]; out = base / "quality_anchor_v2"
    assert args[-3:] == [IMAGE, "-B", str(code / "infra/tudl_anchor_evaluate.py")]
    assert "--gpus" not in args and args[args.index("--network") + 1] == "none"
    assert args[args.index("--cpus") + 1] == "4" and args[args.index("--memory") + 1] == "8g"
    assert args[args.index("--user") + 1] == "1000"
    mounts = [args[i + 1] for i, item in enumerate(args) if item == "--mount"]
    paths = [code, code.parent / "revision", code.parent / "source-sha256",
        base / "inputs", base / "anchor_predictions_v1", base / "eval_private", base / "quality_anchor_v1/report.json"]
    assert mounts == [f"type=bind,src={p},dst={p},readonly" for p in paths] + [f"type=bind,src={out},dst={out}"]
    assert all("weights" not in mount and "vendor" not in mount for mount in mounts)
    assert out.stat().st_mode & 0o777 == 0o700
    assert {p.name for p in out.iterdir()} == {"report.json"}
    assert (out / "report.json").stat().st_mode & 0o777 == 0o400
    assert all(p.read_bytes() == data for p, data in before.items())
    assert Path(env["FAKE_TIMEOUT"]).read_text().startswith("--signal=TERM --kill-after=10s 183s docker run")


@pytest.mark.parametrize("args", [["--help"], ["--resume"], ["--episode", "1"], ["--budget", "181"]])
def test_no_cli_override(runtime, args):
    _, _, base, _, _, log, run = runtime
    assert run(*args).returncode == 2 and not log.exists() and not (base / "quality_anchor_v2").exists()


@pytest.mark.parametrize("name", REQUIRED)
def test_complete_source_and_independently_committed_pins_required(runtime, name):
    _, code, base, _, _, log, run = runtime
    path = code / name; path.parent.chmod(0o755); path.unlink()
    assert run().returncode != 0 and not log.exists() and not (base / "quality_anchor_v2").exists()


@pytest.mark.parametrize("fault", ["revision", "archive", "writable", "symlink", "root", "namespace"])
def test_immutable_canonical_dispatch_before_docker(runtime, fault, tmp_path):
    _, code, _, _, env, log, run = runtime
    if fault == "revision": (code.parent / "revision").write_text("c" * 40 + "\n")
    elif fault == "archive": (code.parent / "source-sha256").write_text("bad")
    elif fault == "writable": (code / REQUIRED[0]).chmod(0o644)
    elif fault == "symlink":
        p = code / REQUIRED[0]; p.parent.chmod(0o755); p.unlink(); p.symlink_to("missing")
    elif fault == "root": env["WR_ROOT"] = str(tmp_path / "wrong")
    else: env["WR_CODE"] = str(tmp_path / "wrong")
    assert run().returncode != 0 and not log.exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_result_never_overwritten_or_deleted(runtime, kind):
    _, _, base, _, _, log, run = runtime; out = base / "quality_anchor_v2"
    if kind == "directory": out.mkdir(); (out / "retained").write_text("retained")
    elif kind == "file": out.write_text("retained")
    else: out.symlink_to("missing")
    assert run().returncode != 0 and not log.exists() and (out.exists() or out.is_symlink())


@pytest.mark.parametrize("split", ["inputs", "anchor_predictions_v1", "eval_private"])
def test_original_splits_must_exist_and_not_alias(runtime, split, tmp_path):
    _, _, base, _, _, log, run = runtime
    p = base / split; (p / "unchanged").unlink(); p.rmdir(); p.symlink_to(tmp_path)
    assert run().returncode != 0 and not log.exists()


@pytest.mark.parametrize("image", ["", "sha256:" + "0" * 64,
    "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"])
def test_exact_vm02_image_not_source_vm01_image(runtime, image):
    _, _, base, _, env, _, run = runtime; env["FAKE_IMAGE"] = image
    assert run().returncode != 0 and not (base / "quality_anchor_v2").exists()


@pytest.mark.parametrize("mutation", ["source", "pins", "revision", "archive", "image"])
def test_source_pins_markers_and_image_rechecked_after_child(runtime, mutation):
    _, _, base, _, env, _, run = runtime
    if mutation == "image": env["FAKE_IMAGE_AFTER"] = "sha256:" + "0" * 64
    else: env["FAKE_MUTATION"] = mutation
    assert run().returncode != 0 and (base / "quality_anchor_v2/report.json").exists()


def test_production_shell_has_no_gpu_models_cleanup_or_retry():
    subprocess.run(["rtk", "proxy", "bash", "-n", str(WRAPPER)], check=True)
    text = WRAPPER.read_text()
    for forbidden in ("--gpus", "nvidia-smi", "flock", "systemctl", "src=$ROOT/data"):
        assert forbidden not in text
    assert not re.search(r"(?m)^\s*(?:rm|rmdir)\s", text)
    assert "183s" in text and "STATUS=$?" in text and IMAGE in text and ORIGINAL_SHA in text


@pytest.mark.parametrize("fault",["missing","writable","wrongbytes","wrongsha","symlink"])
def test_original_failure_identity_before_any_output_or_docker(runtime,fault):
    _,_,base,_,_,log,run=runtime;path=base/"quality_anchor_v1/report.json"
    if fault=="missing":path.unlink()
    elif fault=="writable":path.chmod(0o600)
    elif fault=="symlink":path.unlink();path.symlink_to("missing")
    else:
        path.chmod(0o600);data=path.read_bytes();path.write_bytes(data+b" " if fault=="wrongbytes" else b"x"+data[1:]);path.chmod(0o400)
    assert run().returncode!=0 and not log.exists() and not(base/"quality_anchor_v2").exists()
