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


def test_transitive_infra_import_closure_keeps_initializer_but_excludes_unused_source():
    selected = launcher.runtime_bundle_paths(files(), "infra/run_smoke.sh")
    assert selected == sorted(set(files()) - {"infra/unused.py", "src/world_reward/check.py"})


def test_static_transitive_package_imports_inside_functions_and_src_to_infra():
    source = files()
    source["infra/smoke.py"] += b"\ndef later():\n import world_reward.check\n from world_reward.other import VALUE\n"
    source["src/world_reward/check.py"] = b"from world_reward.nested import child\n"
    source["src/world_reward/other.py"] = b"from helper import validate\n"
    source["src/world_reward/nested/__init__.py"] = b"from . import child\n"
    source["src/world_reward/nested/child.py"] = b"from ..other import VALUE\n"
    selected = launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")
    assert selected == sorted(set(source) - {"infra/unused.py"})


@pytest.mark.parametrize("statement", [
    "from world_reward import check", "from world_reward import check as alias",
    "import world_reward.check as alias", "import world_reward\nworld_reward.check.VALUE",
    "import world_reward as wr\nwr.check.VALUE",
    "import world_reward as wr\ngetattr(wr, 'check').VALUE",
    "from importlib import import_module\nimport_module('world_reward.check')",
    "from importlib import import_module as load\nload('world_reward.check')",
    "import importlib\nimportlib.import_module(name='world_reward.check')",
    "import importlib\nimportlib.import_module('.check', package='world_reward')",
    "__import__('world_reward.check')",
])
def test_package_child_forms_and_literal_dynamic_imports(statement):
    source = files(); source["infra/smoke.py"] += (statement + "\n").encode()
    assert "src/world_reward/check.py" in launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


def test_from_initializer_symbol_is_not_invented_submodule_and_initializer_imports_scanned():
    source = files()
    source["infra/smoke.py"] = b"from world_reward import VERSION\n"
    source["src/world_reward/__init__.py"] = b"from .check import VALUE\nVERSION=1\n"
    selected = launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")
    assert "src/world_reward/check.py" in selected
    assert "src/world_reward/VERSION.py" not in selected


@pytest.mark.parametrize("statement", [
    "import world_reward.missing", "from world_reward.missing import symbol",
    "__import__('world_reward.missing')", "import importlib\nimportlib.import_module('world_reward.missing')",
])
def test_missing_explicit_own_module_fails_closed(statement):
    source = files(); source["infra/smoke.py"] += (statement + "\n").encode()
    with pytest.raises(ValueError, match="missing or ambiguous"):
        launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


def test_nested_package_requires_committed_parent_initializers():
    source = files(); source["infra/smoke.py"] += b"import world_reward.nested.child\n"
    source["src/world_reward/nested/child.py"] = b"VALUE=1\n"
    with pytest.raises(ValueError, match="initializer is not committed"):
        launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


def test_ambiguous_module_and_package_fail_closed():
    source = files(); source["infra/smoke.py"] += b"import world_reward.check\n"
    source["src/world_reward/check/__init__.py"] = b""
    with pytest.raises(ValueError, match="ambiguous"):
        launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


@pytest.mark.parametrize("statement", [
    "import importlib\nimportlib.import_module('world_reward.' + name)",
    "__import__(f'world_reward.{name}')",
    "from importlib import import_module as load\nload('world_reward.{}'.format(name))",
    "import importlib\nmodule = 'world_reward.' + name\nimportlib.import_module(module)",
])
def test_computed_obvious_own_package_names_fail_closed(statement):
    source = files(); source["infra/smoke.py"] += (statement + "\n").encode()
    with pytest.raises(ValueError, match="Computed own-package"):
        launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


def test_computed_import_inside_own_package_unsupported_but_vendor_loader_unaffected():
    source = files()
    source["infra/smoke.py"] += b"import world_reward.check\nimport importlib.util\nimportlib.util.spec_from_file_location('official_converter', tool)\n"
    source["src/world_reward/check.py"] = b"import importlib\nimportlib.import_module(module_name)\n"
    with pytest.raises(ValueError, match="Computed own-package"):
        launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")
    source["src/world_reward/check.py"] = b"import importlib\nimportlib.import_module('external')\n"
    assert "src/world_reward/check.py" in launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


def test_computed_import_relative_to_own_package_and_dynamic_attribute_fail():
    source = files()
    for statement in ("import importlib\nimportlib.import_module(name, package='world_reward')",
                      "import world_reward as wr\ngetattr(wr, name)"):
        source["infra/smoke.py"] = (statement + "\n").encode()
        with pytest.raises(ValueError, match="Computed own-package"):
            launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


def test_relative_sibling_package_cycle_and_unresolved_relative_import():
    source = files(); source["infra/smoke.py"] += b"import world_reward.check\n"
    source["src/world_reward/check.py"] = b"from .other import VALUE\n"
    source["src/world_reward/other.py"] = b"from . import check\nVALUE=1\n"
    selected = launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")
    assert {"src/world_reward/check.py", "src/world_reward/other.py"} <= set(selected)
    assert len(selected) == len(set(selected))
    source["src/world_reward/check.py"] = b"from ..outside import VALUE\n"
    with pytest.raises(ValueError, match="relative runtime import"):
        launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


def test_literal_source_file_reference_includes_unimported_hash_dependency():
    source = files(); source["infra/smoke.py"] += b"path = '/src/world_reward/check.py'\n"
    assert "src/world_reward/check.py" in launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


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
