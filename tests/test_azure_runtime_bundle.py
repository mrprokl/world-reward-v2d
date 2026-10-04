import importlib.util
import io
import base64
import hashlib
import lzma
from pathlib import Path
import tarfile
import subprocess
import types
import sys
import random
import shlex

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
    assert launcher.MAX_CODE_CONTROL_BYTES == 256_000
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


@pytest.mark.parametrize("name", ["infra/run_audit.sh", "src/world_reward/other.py"])
def test_code_relative_receipt_helper_includes_complete_closure_and_missing_fails(name):
    source = files()
    source["infra/helper.py"] += f'helpers = {{{name!r}}}\n'.encode()
    source[name] = b'python "$CODE/infra/audit.py"\n' if name.endswith('.sh') else b'from world_reward import check\n'
    source["infra/audit.py"] = b'from world_reward import check\n'
    selected = launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")
    assert {name, "src/world_reward/check.py"} <= set(selected)
    if name.endswith('.sh'): assert "infra/audit.py" in selected
    source.pop(name)
    with pytest.raises(ValueError, match="Literal code-relative source dependency is not committed"):
        launcher.runtime_bundle_paths(source, "infra/run_smoke.sh")


def test_h102_closures_keep_their_actual_producer_provenance_helpers():
    root = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(root)): p.read_bytes()
              for folder in ("infra", "src", "configs") for p in (root / folder).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    source["pyproject.toml"] = (root / "pyproject.toml").read_bytes()
    for stage in ("forward", "refine", "export"):
        required = {"infra/cari96_inputs.py", "infra/body_smoke.py"}
        required |= ({"infra/cari96_refine.py", "infra/run_cari96_refine.sh", "infra/cari_refine.py"}
                     if stage == "refine" else {"infra/cari96_prepare.py", "infra/run_cari96_prepare.sh"})
        assert required <= set(launcher.runtime_bundle_paths(source, f"infra/run_cari96_{stage}.sh"))


def test_full_shared_prepare_closure_preserves_all_source_bound_helpers():
    root = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(root)): p.read_bytes()
              for folder in ("infra", "src", "configs") for p in (root / folder).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    source["pyproject.toml"] = (root / "pyproject.toml").read_bytes()
    required = {"infra/cari_shared_prepare.py", "infra/run_cari_shared_prepare.sh", "infra/cari_clip_inputs.py",
                "infra/cari96_prepare.py", "infra/run_cari96_prepare.sh", "infra/cari96_inputs.py", "infra/body_smoke.py",
                "src/world_reward/shared_identity.py", "src/world_reward/timeline.py", "src/world_reward/data.py"}
    selected = launcher.runtime_bundle_paths(source, "infra/run_cari_shared_prepare.sh")
    assert required <= set(selected)
    assert "configs/cari_clip_000015_input_pins.json" in selected
    assert "infra/cari96_forward.py" not in selected


@pytest.mark.parametrize("stage", ["forward", "refine", "export"])
def test_full_video_runtime_closures_keep_previous_producers_and_core_contracts(stage):
    root = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(root)): p.read_bytes()
              for folder in ("infra", "src", "configs") for p in (root / folder).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    source["pyproject.toml"] = (root / "pyproject.toml").read_bytes()
    selected = set(launcher.runtime_bundle_paths(source, f"infra/run_cari_full_{stage}.sh"))
    required = {"infra/cari_full_forward.py", "infra/run_cari_full_forward.sh",
                "infra/cari_shared_prepare.py", "infra/run_cari_shared_prepare.sh",
                "infra/cari_clip_inputs.py", "src/world_reward/shared_identity.py",
                "src/world_reward/timeline.py", "configs/cari_clip_000015_shared_prepare_pins.json"}
    if stage in {"refine", "export"}:
        required |= {"infra/cari_full_refine.py", "infra/run_cari_full_refine.sh", "infra/cari_refine.py"}
    if stage == "export":
        required |= {"infra/cari_full_export.py", "src/world_reward/submission.py", "src/world_reward/contracts.py"}
    assert required <= selected


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


def remote_ack(command):
    script=command[command.index('--scripts')+1]
    return script.rsplit("printf '%s\\n' '",1)[1].split("'",1)[0].encode()+b'\n'


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
        return types.SimpleNamespace(returncode=returncode,stdout=remote_ack(command),stderr=b'')
    monkeypatch.setattr(launcher.subprocess, "run", azure)
    launcher.main(["--name", "unit-test", "--script", "infra/run_smoke.sh", *args])
    assert len(calls) == 1 and calls[0][1] == {"check": False,"capture_output":True,"timeout":300}
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


@pytest.mark.parametrize("revision", ["", "HEAD", "main", "v1", "a" * 39, "a" * 41, "A" * 40, "g" * 40, "a" * 40 + "^", "--all", "a;echo x", "a" * 40 + "\n"])
def test_explicit_revision_invalid_before_any_git_or_azure(monkeypatch, revision):
    def forbidden(*_, **__): raise AssertionError("Invalid revision reached subprocess")
    monkeypatch.setattr(launcher.subprocess, "check_output", forbidden)
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    with pytest.raises(SystemExit):
        launcher.main(["--name", "test", "--script", "infra/run_smoke.sh", "--revision", revision])


def exact_revision_stub(monkeypatch, *, explicit=True, kind=b"commit\n", resolved=None, missing=False):
    revision = "b" * 40; calls = []; archives = []; git_calls = []
    committed = files(); committed["infra/smoke.py"] = b"COMMITTED_VALUE = 1\n"
    def git(command):
        assert command[:3] == ["rtk", "proxy", "git"]
        args = command[3:]; git_calls.append(args)
        if args[0] == "status": return b" M infra/smoke.py\n?? disjoint.py\n"
        if args[:2] == ["cat-file", "-t"]:
            assert args[2] == revision
            if missing: raise launcher.subprocess.CalledProcessError(128, command)
            return kind
        if args[0] == "rev-parse":
            assert args == ["rev-parse", "--verify", revision + "^{commit}"]
            return (resolved or revision).encode() + b"\n"
        if args[:2] == ["cat-file", "-e"]:
            assert args[2] == revision + ":infra/run_smoke.sh"; return b""
        if args[0] == "archive":
            assert args == ["archive", "--format=tar", revision, "infra", "src", "configs", "pyproject.toml"]
            archives.append(args); return full_archive(committed)
        raise AssertionError(command)
    monkeypatch.setattr(launcher.subprocess, "check_output", git)
    def azure(command, **kwargs):
        calls.append(command); return types.SimpleNamespace(returncode=0,stdout=remote_ack(command),stderr=b'')
    monkeypatch.setattr(launcher.subprocess, "run", azure)
    args = ["--name", "test", "--script", "infra/run_smoke.sh"]
    if explicit: args += ["--revision", revision]
    return args, revision, calls, archives, git_calls


def test_explicit_commit_dispatch_ignores_dirty_worktree_and_freezes_commit_bytes(monkeypatch, capsys):
    args, revision, calls, archives, git_calls = exact_revision_stub(monkeypatch)
    launcher.main(args)
    assert len(calls) == len(archives) == 1 and not any(call[0] == "status" for call in git_calls)
    assert "revision=" + revision in capsys.readouterr().out
    remote = calls[0][calls[0].index("--scripts") + 1]
    assert f"jobs/{revision}/run_smoke" in remote and f"WR_CODE_REVISION='{revision}'" in remote
    encoded = remote.split("printf '%s' '", 1)[1].split("' | base64", 1)[0]
    frozen = lzma.decompress(base64.b64decode(encoded))
    with tarfile.open(fileobj=io.BytesIO(frozen), mode="r:") as archive:
        assert archive.extractfile("infra/smoke.py").read() == b"COMMITTED_VALUE = 1\n"
        assert not any("disjoint" in member.name for member in archive)


def test_default_still_rejects_dirty_tree_before_archive_or_azure(monkeypatch):
    args, _, calls, archives, _ = exact_revision_stub(monkeypatch, explicit=False)
    with pytest.raises(RuntimeError, match="clean"): launcher.main(args)
    assert not calls and not archives


@pytest.mark.parametrize("kind", [b"tree\n", b"blob\n", b"tag\n", b""])
def test_explicit_noncommit_object_rejected_before_archive_or_azure(monkeypatch, kind):
    args, _, calls, archives, _ = exact_revision_stub(monkeypatch, kind=kind)
    with pytest.raises(ValueError, match="Git commit"): launcher.main(args)
    assert not calls and not archives


def test_explicit_resolved_mismatch_or_missing_commit_never_calls_azure(monkeypatch):
    args, _, calls, archives, _ = exact_revision_stub(monkeypatch, resolved="c" * 40)
    with pytest.raises(ValueError, match="differs"): launcher.main(args)
    assert not calls and not archives
    args, _, calls, archives, _ = exact_revision_stub(monkeypatch, missing=True)
    with pytest.raises(launcher.subprocess.CalledProcessError): launcher.main(args)
    assert not calls and not archives


@pytest.mark.parametrize('output',[b'',b'EnableSucceeded\n',b'[stdout]\n',b'WORLD_REWARD_DISPATCH_ACK_V1:wrong\n',
    b'WORLD_REWARD_DISPATCH_ACK_V1:expected\nWORLD_REWARD_DISPATCH_ACK_V1:expected\n'])
def test_azure_exit_zero_without_exact_unique_stdout_sentinel_is_not_ack(monkeypatch,output):
    calls=[]
    def fake(command,**kwargs):
        calls.append(command);return types.SimpleNamespace(returncode=0,stdout=output,stderr=b'PRIVATE_ERROR_MUST_NOT_ESCAPE')
    monkeypatch.setattr(launcher.subprocess,'run',fake)
    with pytest.raises(RuntimeError,match='inspect target before any retry')as caught:
        launcher.invoke_transport('CODE_PAYLOAD_MUST_NOT_ESCAPE','WORLD_REWARD_DISPATCH_ACK_V1:expected','rg','vm')
    assert len(calls)==1 and 'CODE_PAYLOAD'not in str(caught.value)and 'PRIVATE_ERROR'not in str(caught.value)


def test_exact_ack_capture_does_not_print_payload_or_remote_errors(monkeypatch,capsys):
    ack=launcher.ACK_PREFIX+'a'*64+':phase'
    monkeypatch.setattr(launcher.subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=0,stdout=('[stdout]\n'+ack+'\n[stderr]\n').encode(),stderr=b'not-printed'))
    launcher.invoke_transport('not-printed-code',ack,'rg','vm')
    assert capsys.readouterr().out==''


def staged_fixture():
    payload=base64.b64encode(random.Random(310427).randbytes(155_000))
    source=files();source['configs/transport_fixture.json']=b'{"code_only":"'+payload+b'"}'
    archive=launcher.runtime_archive(full_archive(source),'infra/run_smoke.sh')[0]
    encoded,sha=launcher.encoded_runtime_archive(archive)
    commands=launcher.transport_commands(encoded,sha,archive,'b'*40,'infra/run_smoke.sh','stage-test',['--episode','8'])
    assert len(commands)>2
    return archive,encoded,sha,commands


def test_staged_script_caps_full_exact_reassembly_and_only_final_can_launch():
    archive,encoded,sha,commands=staged_fixture()
    assert launcher.INLINE_SCRIPT_BYTES==180_000 and launcher.STAGED_SCRIPT_BYTES==120_000
    assert all(len(script.encode())<=120_000 for _,_,script in commands)
    parts=[script.split("printf '%s' '",1)[1].split("' >",1)[0]for phase,_,script in commands if phase.startswith('chunk-')]
    assert ''.join(parts)==encoded
    compressed=base64.b64decode(''.join(parts),validate=True)
    assert hashlib.sha256(compressed).hexdigest()==sha and lzma.decompress(compressed)==archive
    assert all('systemd-run'not in script and 'WR_CODE='not in script for _,_,script in commands[:-1])
    assert 'renameat2'in commands[-1][2]and ',1)==0)'in commands[-1][2]
    assert commands[-1][2].index('hashlib.sha256(compressed)')<commands[-1][2].index('renameat2')<commands[-1][2].index('systemd-run')
    assert commands[-1][2].index('hashlib.sha256(raw)')<commands[-1][2].index('renameat2')
    assert 'stage-test'in commands[0][2]and '.runtime-stage-'in commands[0][2]
    assert len({ack for _,ack,_ in commands})==len(commands)


def local_transport_runtime(tmp_path):
    root=tmp_path/'remote';(root/'jobs').mkdir(parents=True);(root/'results').mkdir()
    bindir=tmp_path/'bin';bindir.mkdir()
    for name in('systemctl','systemd-run'):
        path=bindir/name
        path.write_text('#!/bin/bash\n'+('printf "%s\\n" "$*" >> "$HOME/launched"\n'if name=='systemd-run'else'exit 0\n'))
        path.chmod(0o755)
    # macOS lacks coreutils; this tiny shim checks the actual bytes, never a fake PASS.
    checksum=bindir/'sha256sum'
    checksum.write_text("#!/usr/bin/python3\nimport hashlib,sys\nfrom pathlib import Path\nsha,path=sys.stdin.read().strip().split(maxsplit=1)\nraise SystemExit(0 if hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha else 1)\n")
    checksum.chmod(0o755)
    return root,{'PATH':str(bindir)+':/usr/bin:/bin','HOME':str(tmp_path)}


def run_local_phase(script,root,environment):
    script=script.replace('ROOT=/srv/scenesmith/world-reward','ROOT='+shlex.quote(str(root)))
    if sys.platform!='linux'and "<<'PY_PUBLISH'"in script:
        # Only the Linux NOREPLACE syscall is emulated; all assembly/TAR gates run unchanged.
        old="libc=ctypes.CDLL(None,use_errno=True);rename=getattr(libc,'renameat2',None);require(rename is not None)"
        new="""def rename(a,old,b,new,flags):
  require(flags==1 and not Path(os.fsdecode(new)).exists());os.rename(old,new);return 0"""
        assert old in script;script=script.replace(old,new)
    return subprocess.run(['bash','-c',script],env=environment,capture_output=True)


def test_staged_execution_exact_code_publish_no_unit_before_final_and_no_resume(tmp_path):
    archive,_,_,commands=staged_fixture();root,environment=local_transport_runtime(tmp_path)
    for _,ack,script in commands[:-1]:
        result=run_local_phase(script,root,environment)
        assert result.returncode==0 and ack.encode()in result.stdout
        assert not(tmp_path/'launched').exists()and not(root/'jobs'/('b'*40)/'run_smoke').exists()
    duplicate=run_local_phase(commands[-2][2],root,environment)
    assert duplicate.returncode!=0 and commands[-2][1].encode()not in duplicate.stdout
    result=run_local_phase(commands[-1][2],root,environment)
    assert result.returncode==0 and commands[-1][1].encode()in result.stdout
    job=root/'jobs'/('b'*40)/'run_smoke'
    assert (tmp_path/'launched').exists()and len((tmp_path/'launched').read_text().splitlines())==1
    with tarfile.open(fileobj=io.BytesIO(archive))as tar:
        assert {str(p.relative_to(job/'code'))for p in(job/'code').rglob('*')if p.is_file()}==set(tar.getnames())
        for member in tar:
            assert (job/'code'/member.name).read_bytes()==tar.extractfile(member).read()
            assert (job/'code'/member.name).stat().st_mode&0o777==member.mode&0o555
    assert not list((root/'jobs').glob('.runtime-stage-*'))
    assert run_local_phase(commands[0][2],root,environment).returncode!=0  # Existing JOB is never resumed.
    assert len((tmp_path/'launched').read_text().splitlines())==1


@pytest.mark.parametrize('fault',['chunk_tamper','chunk_missing','extra_stage_file','job_exists','stage_symlink','identity_tamper','identity_writable'])
def test_staged_unknown_or_tampered_state_never_publishes_or_starts(tmp_path,fault):
    _,_,_,commands=staged_fixture();root,environment=local_transport_runtime(tmp_path)
    for _,_,script in commands[:-1]:assert run_local_phase(script,root,environment).returncode==0
    stage=next((root/'jobs').glob('.runtime-stage-*'));job=root/'jobs'/('b'*40)/'run_smoke'
    if fault=='chunk_tamper':
        path=stage/'chunk_0000';path.chmod(0o600);path.write_bytes(b'x'*path.stat().st_size);path.chmod(0o400)
    elif fault=='chunk_missing':(stage/'chunk_0000').unlink()
    elif fault=='extra_stage_file':(stage/'unexpected').write_text('unowned')
    elif fault=='job_exists':job.mkdir(parents=True)
    elif fault=='stage_symlink':
        saved=stage.with_name(stage.name+'-saved');stage.rename(saved);stage.symlink_to(saved,target_is_directory=True)
    elif fault=='identity_tamper':
        path=stage/'identity';path.chmod(0o600);path.write_bytes(b'changed');path.chmod(0o400)
    else:(stage/'identity').chmod(0o600)
    result=run_local_phase(commands[-1][2],root,environment)
    assert result.returncode!=0 and commands[-1][1].encode()not in result.stdout
    assert not(tmp_path/'launched').exists()and not(job/'code').exists()


def test_staging_duplicate_initial_namespace_never_resumes(tmp_path):
    *_,commands=staged_fixture();root,environment=local_transport_runtime(tmp_path)
    assert run_local_phase(commands[0][2],root,environment).returncode==0
    assert run_local_phase(commands[0][2],root,environment).returncode!=0
    assert not(tmp_path/'launched').exists()


def test_transport_main_serializes_phases_and_stops_immediately_on_unknown_ack(monkeypatch):
    source=files();source['configs/big.json']=base64.b64encode(random.Random(310427).randbytes(155_000))
    revision='a'*40;calls=[]
    def git(argv):
        if argv[3]=='status':return b''
        if argv[3]=='rev-parse':return (revision+'\n').encode()
        if argv[3]=='cat-file':return b''
        if argv[3]=='archive':return full_archive(source)
        raise AssertionError('Unexpected Git fixture call')
    def azure(argv,**kwargs):
        calls.append(argv)
        return types.SimpleNamespace(returncode=0,stdout=remote_ack(argv)if len(calls)==1 else b'',stderr=b'')
    monkeypatch.setattr(launcher.subprocess,'check_output',git);monkeypatch.setattr(launcher.subprocess,'run',azure)
    with pytest.raises(RuntimeError,match='unacknowledged'):launcher.main(['--name','test','--script','infra/run_smoke.sh'])
    assert len(calls)==2 and not any('systemd-run'in x[x.index('--scripts')+1]for x in calls)


@pytest.mark.parametrize('fault',[None,'markers','symlink'])
def test_inline_existing_snapshot_reused_only_exact_markers_and_canonical_job(tmp_path,fault):
    source=launcher.runtime_archive(full_archive(files()),'infra/run_smoke.sh')[0]
    encoded,sha=launcher.encoded_runtime_archive(source);revision='b'*40
    commands=launcher.transport_commands(encoded,sha,source,revision,'infra/run_smoke.sh','reuse-test',[])
    assert len(commands)==1 and len(commands[0][2].encode())<=180_000
    root,environment=local_transport_runtime(tmp_path);job=root/'jobs'/revision/'run_smoke';job.mkdir(parents=True)
    (job/'revision').write_text(revision+'\n');(job/'source-sha256').write_text((sha if fault!='markers'else'c'*64)+'\n')
    with tarfile.open(fileobj=io.BytesIO(source))as archive:
        for member in archive:
            target=job/'code'/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(member).read());target.chmod(member.mode&0o555)
    for path in(job/'code',*(job/'code').rglob('*')):
        if path.is_dir():path.chmod(0o555)
    original=job/'code/infra/run_smoke.sh';original_bytes=original.read_bytes()
    if fault=='symlink':
        saved=job.with_name('saved');job.rename(saved);job.symlink_to(saved,target_is_directory=True)
    result=run_local_phase(commands[0][2],root,environment)
    assert (result.returncode==0)is(fault is None)
    assert (tmp_path/'launched').exists()is(fault is None)
    assert original.read_bytes()==original_bytes


def test_staged_published_source_rehash_stops_tamper_before_systemd(tmp_path):
    *_,commands=staged_fixture();root,environment=local_transport_runtime(tmp_path)
    for _,_,script in commands[:-1]:assert run_local_phase(script,root,environment).returncode==0
    publish=commands[-1][2]
    old=' syncdir(job.parent)\n'
    new=" syncdir(job.parent)\n p=job/'code'/'infra/run_smoke.sh';p.chmod(0o600);p.write_bytes(b'changed at publication');p.chmod(0o555)\n"
    assert old in publish;result=run_local_phase(publish.replace(old,new),root,environment)
    assert result.returncode!=0 and not(tmp_path/'launched').exists()
    assert commands[-1][1].encode()not in result.stdout


@pytest.mark.parametrize('staged',[False,True])
def test_publication_under_umask077_all_new_ancestors_traversable_code_readonly(tmp_path,staged):
    if staged:archive,_,_,commands=staged_fixture()
    else:
        archive=launcher.runtime_archive(full_archive(files()),'infra/run_smoke.sh')[0]
        encoded,sha=launcher.encoded_runtime_archive(archive)
        commands=launcher.transport_commands(encoded,sha,archive,'b'*40,'infra/run_smoke.sh','stage-test',[])
    root,environment=local_transport_runtime(tmp_path)
    shared_before={p:(p.stat().st_ino,p.stat().st_mode)for p in(root,root/'jobs',root/'results')}
    for _,ack,script in commands:
        result=run_local_phase('umask 077\n'+script,root,environment)
        assert result.returncode==0 and ack.encode()in result.stdout
    job=root/'jobs'/('b'*40)/'run_smoke'
    assert all(p.stat().st_mode&0o777==0o755 for p in(job.parent,job))
    assert all(p.stat().st_mode&0o777==0o555 for p in(job/'code',*(p for p in(job/'code').rglob('*')if p.is_dir())))
    with tarfile.open(fileobj=io.BytesIO(archive))as tar:
        for m in tar:
            p=job/'code'/m.name;assert p.read_bytes()==tar.extractfile(m).read()and p.stat().st_mode&0o777==m.mode&0o555
    assert {p:(p.stat().st_ino,p.stat().st_mode)for p in shared_before}==shared_before


@pytest.mark.parametrize('staged',[False,True])
def test_existing_private_revision_ancestor_fails_without_permission_repair(tmp_path,staged):
    if staged:*_,commands=staged_fixture()
    else:
        archive=launcher.runtime_archive(full_archive(files()),'infra/run_smoke.sh')[0]
        encoded,sha=launcher.encoded_runtime_archive(archive)
        commands=launcher.transport_commands(encoded,sha,archive,'b'*40,'infra/run_smoke.sh','stage-test',[])
    root,environment=local_transport_runtime(tmp_path);parent=root/'jobs'/('b'*40);parent.mkdir(mode=0o700)
    before=parent.stat().st_mode
    result=run_local_phase(commands[0][2],root,environment)
    assert result.returncode!=0 and not(tmp_path/'launched').exists()
    assert parent.stat().st_mode==before and not(parent/'run_smoke').exists()


@pytest.mark.parametrize('fault',['private_job','private_code','source_tamper'])
def test_existing_inline_snapshot_not_silently_fixed_or_accepted_by_markers(tmp_path,fault):
    archive=launcher.runtime_archive(full_archive(files()),'infra/run_smoke.sh')[0]
    encoded,sha=launcher.encoded_runtime_archive(archive);revision='b'*40
    first=launcher.transport_commands(encoded,sha,archive,revision,'infra/run_smoke.sh','first',[])
    root,environment=local_transport_runtime(tmp_path)
    assert run_local_phase(first[0][2],root,environment).returncode==0
    job=root/'jobs'/revision/'run_smoke';code=job/'code'
    if fault=='private_job':job.chmod(0o700)
    elif fault=='private_code':code.chmod(0o500)
    else:
        path=code/'infra/run_smoke.sh';path.chmod(0o755);path.write_bytes(b'changed source with exact old markers');path.chmod(0o555)
    modes={p:p.stat().st_mode for p in(job,code,code/'infra/run_smoke.sh')}
    second=launcher.transport_commands(encoded,sha,archive,revision,'infra/run_smoke.sh','second',[])
    result=run_local_phase(second[0][2],root,environment)
    assert result.returncode!=0 and len((tmp_path/'launched').read_text().splitlines())==1
    assert {p:p.stat().st_mode for p in modes}==modes
