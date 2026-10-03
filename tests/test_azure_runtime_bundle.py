import importlib.util
import io
import base64
import hashlib
import lzma
from pathlib import Path
import tarfile

import pytest


spec = importlib.util.spec_from_file_location("wr_azure_bundle_test", Path(__file__).resolve().parents[1] / "infra/azure_job.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_explicit_code_only_control_ceiling_preserves_full_import_closure(monkeypatch):
    source = b"tiny committed code"
    encoded, digest = launcher.encoded_runtime_archive(source)
    compressed = base64.b64decode(encoded)
    assert lzma.decompress(compressed) == source
    assert digest == hashlib.sha256(compressed).hexdigest()
    assert launcher.MAX_CODE_CONTROL_BYTES == 160_000
    monkeypatch.setattr(launcher, "MAX_CODE_CONTROL_BYTES", len(encoded))
    assert launcher.encoded_runtime_archive(source)[0] == encoded
    monkeypatch.setattr(launcher, "MAX_CODE_CONTROL_BYTES", len(encoded)-1)
    with pytest.raises(RuntimeError, match="code-only control budget"):
        launcher.encoded_runtime_archive(source)


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


def test_literal_sibling_source_hash_inputs_include_transitive_shell_closure():
    source = files()
    source["infra/helper.py"] += b'from pathlib import Path\nPath(__file__).with_name("run_audit.sh").read_bytes()\n'
    source["infra/run_audit.sh"] = b'python "$CODE/infra/audit.py"\n'
    source["infra/audit.py"] = b'from world_reward import check\n'
    selected = launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")
    assert {"infra/run_audit.sh", "infra/audit.py", "src/world_reward/check.py"} <= set(selected)
    source.pop("infra/run_audit.sh")
    with pytest.raises(ValueError, match="Literal sibling source dependency is not committed"):
        launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


def test_finite_sibling_hash_list_is_frozen_without_inventing_vendor_files():
    source = files()
    source["infra/helper.py"] += b'from pathlib import Path\nfor name in ("run_audit.sh", "audit.py", "build_sam.py"):\n Path(__file__).with_name(name)\n'
    source["infra/run_audit.sh"] = b'echo audited\n'
    source["infra/audit.py"] = b'VALUE=1\n'
    selected = launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")
    assert {"infra/run_audit.sh", "infra/audit.py"} <= set(selected)
    assert "infra/build_sam.py" not in selected


def test_h98_native_fit_bundle_includes_all_DW_source_only_wrappers():
    root = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(root)): p.read_bytes()
              for folder in ("infra", "src", "configs") for p in (root/folder).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    source["pyproject.toml"] = (root/"pyproject.toml").read_bytes()
    selected = launcher.runtime_bundle_paths(source, "infra/run_root5_rgb_fit.sh")
    assert {"infra/run_dwpose_smoke.sh", "infra/run_keypoint_rgb_dwpose.sh",
            "infra/dwpose_smoke.py", "infra/keypoint_rgb_dwpose.py"} <= set(selected)
    assert not any(path.endswith((".npz", ".png", ".onnx")) for path in selected)


def test_shell_dockerfile_and_child_references_included():
    source = files()
    source["infra/run_smoke.sh"] += b'bash "$CODE/infra/run_child.sh"\ndocker build --file "$CODE/infra/Dockerfile.runtime" "$CODE/infra"\n'
    source["infra/run_child.sh"] = b"echo OK\n"
    source["infra/Dockerfile.runtime"] = b"FROM existing:pinned\n"
    selected = launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")
    assert "infra/run_child.sh" in selected and "infra/Dockerfile.runtime" in selected


def test_literal_cpp_build_input_is_frozen_without_parsing_cpp_as_python():
    source = files()
    source["infra/run_smoke.sh"] += b'c++ "$CODE/infra/kernel.cpp" -o /tmp/kernel\n'
    source["infra/kernel.cpp"] = b'int main() { return 0; }\n'
    assert "infra/kernel.cpp" in launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")
    source.pop("infra/kernel.cpp")
    with pytest.raises(ValueError, match="not committed"):
        launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


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


def launch_stub(monkeypatch, args, *, returncode=0):
    """Capture argv and frozen remote script; no Git/Azure subprocess executes."""
    calls = []; revision = "a"*40
    def git(command):
        assert command[:3] == ["rtk", "proxy", "git"]
        if command[3] == "status": return b""
        if command[3] == "rev-parse": return revision.encode()+b"\n"
        if command[3] == "cat-file": return b""
        if command[3] == "archive": return full_archive(files())
        raise AssertionError(command)
    monkeypatch.setattr(launcher.subprocess, "check_output", git)
    def azure(command, **kwargs):
        calls.append((command, kwargs))
        return type('Result', (), {'returncode': returncode})()
    monkeypatch.setattr(launcher.subprocess, "run", azure)
    launcher.main(["--name", "unit-test", "--script", "infra/run_smoke.sh", *args])
    assert len(calls) == 1 and calls[0][1] == {"check": False}
    return calls[0][0]


def test_failed_dispatch_does_not_echo_frozen_payload_or_retry(monkeypatch):
    with pytest.raises(RuntimeError,match='payload omitted') as caught:
        launch_stub(monkeypatch,[],returncode=1)
    assert 'base64' not in str(caught.value) and '--scripts' not in str(caught.value)


def test_legacy_defaults_identical_to_explicit_target_and_snapshot(monkeypatch):
    default = launch_stub(monkeypatch, [])
    explicit = launch_stub(monkeypatch, ["--resource-group", "SCENESMITH-H100", "--vm-name", "scenesmith-ncc-h100-01"])
    assert default == explicit
    assert default[:10] == ["rtk", "proxy", "az", "vm", "run-command", "invoke", "--resource-group", "SCENESMITH-H100", "--name", "scenesmith-ncc-h100-01"]


def test_second_target_changes_only_azure_argv_not_remote_root_or_job(monkeypatch):
    original = launch_stub(monkeypatch, [])
    second = launch_stub(monkeypatch, ["--resource-group", "WORLD-REWARD-RESEARCH", "--vm-name", "scenesmith-ncc-h100-02"])
    assert second[7] == "WORLD-REWARD-RESEARCH" and second[9] == "scenesmith-ncc-h100-02"
    assert second[:7] == original[:7] and second[10:] == original[10:]
    remote = second[second.index("--scripts")+1]
    assert "ROOT=/srv/scenesmith/world-reward" in remote and "UNIT=world-reward-unit-test" in remote
    assert "scenesmith-ncc-h100-02" not in remote and "WORLD-REWARD-RESEARCH" not in remote


def test_job_name_does_not_choose_vm_and_script_arguments_remain_quoted(monkeypatch):
    command = launch_stub(monkeypatch, ["--", "--episode", "0", "literal;not-shell"])
    assert command[9] == "scenesmith-ncc-h100-01"
    remote = command[command.index("--scripts")+1]
    assert "UNIT=world-reward-unit-test" in remote and "--episode 0 'literal;not-shell'" in remote


@pytest.mark.parametrize("field,value", [
    ("--resource-group", ""), ("--resource-group", "--other"), ("--resource-group", "group;echo-secret"),
    ("--resource-group", "group name"), ("--resource-group", "../group"), ("--resource-group", "-group"),
    ("--resource-group", "group-"), ("--resource-group", "a"*91), ("--resource-group", "$(cmd)"),
    ("--vm-name", ""), ("--vm-name", "vm/02"), ("--vm-name", "vm_02"), ("--vm-name", "vm.02"),
    ("--vm-name", "vm\ncommand"), ("--vm-name", "vm;command"), ("--vm-name", "-vm"),
    ("--vm-name", "vm-"), ("--vm-name", "a"*65), ("--vm-name", "équipe"),
])
def test_invalid_target_rejected_before_git_or_azure(monkeypatch, field, value):
    def forbidden(*a, **k): raise AssertionError("No subprocess on invalid target")
    monkeypatch.setattr(launcher.subprocess, "check_output", forbidden)
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    with pytest.raises(SystemExit): launcher.main(["--name", "test", "--script", "infra/run_smoke.sh", field, value])


@pytest.mark.parametrize("args", [["--vm", "vm02"], ["--resource", "group"], ["--root", "/tmp/other"], ["--unknown", "x"]])
def test_unknown_abbreviated_targets_and_root_override_fail(monkeypatch, args):
    def forbidden(*a, **k): raise AssertionError("No subprocess on invalid option")
    monkeypatch.setattr(launcher.subprocess, "check_output", forbidden)
    with pytest.raises(SystemExit): launcher.main(["--name", "test", "--script", "infra/run_smoke.sh", *args])
