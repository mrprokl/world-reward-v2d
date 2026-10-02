import importlib.util
import io
from pathlib import Path
import tarfile

import pytest


spec = importlib.util.spec_from_file_location("wr_azure_bundle_test", Path(__file__).resolve().parents[1] / "infra/azure_job.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def files():
    return {"infra/run_smoke.sh": b'python "$CODE/infra/smoke.py"\n',
            "infra/smoke.py": b"from helper import validate\nimport external\n",
            "infra/helper.py": b"from nested import func\n", "infra/nested.py": b"VALUE=1\n",
            "infra/unused.py": b"raise Exception('not this job')\n",
            "src/world_reward/__init__.py": b"", "src/world_reward/check.py": b"x=1\n",
            "configs/sources.json": b"{}", "pyproject.toml": b""}


def test_transitive_python_import_closure_and_package_kept_unused_tooling_excluded():
    selected = launcher.runtime_bundle_paths(files(), "infra/run_smoke.sh")
    assert selected == sorted(set(files()) - {"infra/unused.py"})


def test_shell_dockerfile_and_child_references_included():
    source = files()
    source["infra/run_smoke.sh"] += b'bash "$CODE/infra/run_child.sh"\ndocker build --file "$CODE/infra/Dockerfile.runtime" "$CODE/infra"\n'
    source["infra/run_child.sh"] = b"echo OK\n"
    source["infra/Dockerfile.runtime"] = b"FROM existing:pinned\n"
    selected = launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")
    assert "infra/run_child.sh" in selected and "infra/Dockerfile.runtime" in selected


def test_import_cycles_do_not_duplicate_paths():
    source = files()
    source["infra/nested.py"] = b"import smoke\n"
    selected = launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")
    assert len(selected) == len(set(selected))


def test_missing_referenced_script_rejected_not_silently_omitted():
    source = files()
    source["infra/run_smoke.sh"] += b'python "$CODE/infra/missing.py"\n'
    with pytest.raises(ValueError, match="not committed"):
        launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


@pytest.mark.parametrize("entrypoint", ["../secret", "infra/missing.sh", "infra/smoke.py"])
def test_invalid_entrypoint(entrypoint):
    with pytest.raises(ValueError, match="entrypoint"):
        launcher.runtime_bundle_paths(files(), entrypoint)


def full_archive(source, *, symlink=False):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for path, value in source.items():
            member = tarfile.TarInfo(path)
            member.size, member.mode, member.mtime = len(value), 0o755 if path.endswith(".sh") else 0o644, 17
            archive.addfile(member, io.BytesIO(value))
        if symlink:
            member = tarfile.TarInfo("infra/alias.py")
            member.type, member.linkname = tarfile.SYMTYPE, "smoke.py"
            archive.addfile(member)
    return output.getvalue()


def test_filtered_archive_keeps_git_metadata_and_exact_bytes_deterministic():
    source = files()
    original = full_archive(source)
    filtered, selected = launcher.runtime_archive(original, "infra/run_smoke.sh")
    assert launcher.runtime_archive(original, "infra/run_smoke.sh")[0] == filtered
    with tarfile.open(fileobj=io.BytesIO(filtered)) as archive:
        assert archive.getnames() == selected
        for member in archive:
            assert member.mtime == 17
            assert member.mode == (0o755 if member.name.endswith(".sh") else 0o644)
            assert archive.extractfile(member).read() == source[member.name]


def test_symlink_runtime_aliases_rejected():
    with pytest.raises(ValueError, match="symlinks"):
        launcher.runtime_archive(full_archive(files(), symlink=True), "infra/run_smoke.sh")
