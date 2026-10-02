"""The cloud chain preserves each stage's isolation and failure boundary."""
import base64
import lzma
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra"))
import azure_job


def test_chain_order_and_code_closure():
    entry = "infra/run_perspective_rgb_validate.sh"
    source = (ROOT / entry).read_text()
    subprocess.run(["bash", "-n", str(ROOT / entry)], check=True)
    assert "set -euo pipefail" in source
    positions = [source.index("/infra/run_perspective_rgb_" + stage + ".sh")
                 for stage in ("import", "infer", "evaluate")]
    assert positions == sorted(positions)
    assert "docker run" not in source and "eval_private" not in source
    files = {str(p.relative_to(ROOT)): p.read_bytes()
             for base in ("infra", "src", "configs") for p in (ROOT / base).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    closure = azure_job.runtime_bundle_paths(files, entry)
    assert all("infra/run_perspective_rgb_" + stage + ".sh" in closure
               for stage in ("import", "infer", "evaluate"))
    # Conservative compressed code-only size; the launcher also checks its TAR.
    assert len(base64.b64encode(lzma.compress(b"".join(files[p] for p in closure)))) < 90_000


def test_chain_rejects_missing_untrusted_sha_before_work():
    entry = ROOT / "infra/run_perspective_rgb_validate.sh"
    for arguments in ([], ["not-a-sha"], ["0" * 64, "extra"]):
        result = subprocess.run(["bash", str(entry), *arguments], capture_output=True)
        assert result.returncode == 2
