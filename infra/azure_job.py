"""Launch one immutable, committed code snapshot via Azure Run Command.

Only a small Git archive is sent from the laptop. Jobs/data/model artifacts stay
remote. Existing units and snapshots are never replaced or restarted implicitly.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import lzma
import re
import shlex
import subprocess
import tarfile


def runtime_bundle_paths(files: dict[str, bytes], script: str) -> list[str]:
    """Select committed entrypoint/import closure, never data or unused tooling.

    Follow all static imports, including inside functions, for infra and the
    world_reward package. Keep package initializers, config and pyproject; omit
    unrelated experiments. Literal own-package dynamic imports are supported;
    computed own-package names and dynamic infra plugins are unsupported. This
    is not a general Python dependency resolver: external/vendor imports stay
    in the audited image. Each entrypoint retains its own immutable directory.
    """
    if not re.fullmatch(r"infra/[a-z0-9_]+\.sh", script) or script not in files:
        raise ValueError("Require a committed infra shell entrypoint")
    selected = {path for path in files if path.startswith("configs/") or path == "pyproject.toml"}
    pending = [script]
    if "src/world_reward/__init__.py" in files:
        pending.append("src/world_reward/__init__.py")

    def module_paths(module: str, *, required: bool = True) -> set[str]:
        if module != "world_reward" and not module.startswith("world_reward."):
            path = f"infra/{module.split('.')[0]}.py"
            return {path} if path in files else set()
        if not all(part.isidentifier() for part in module.split(".")):
            raise ValueError(f"Invalid own-package module: {module}")
        stem = "src/" + module.replace(".", "/")
        matches = {path for path in (stem + ".py", stem + "/__init__.py") if path in files}
        if not matches and not required:
            return set()  # A from-import name may be a symbol, not a submodule.
        if len(matches) != 1:
            raise ValueError(f"Own-package dependency is missing or ambiguous: {module}")
        parents = module.split(".")[:-1]
        for length in range(1, len(parents) + 1):
            initializer = "src/" + "/".join(parents[:length]) + "/__init__.py"
            if initializer not in files:
                raise ValueError(f"Own-package initializer is not committed: {initializer}")
            matches.add(initializer)
        return matches

    while pending:
        path = pending.pop()
        if path in selected:
            continue
        selected.add(path)
        source = files[path].decode("utf-8")
        dependencies = set()
        if path.endswith(".py"):
            tree = ast.parse(source)
            package = ""
            if path.startswith("src/"):
                module = path[4:-3].replace("/", ".")
                package = module.removesuffix(".__init__") if path.endswith("/__init__.py") else module.rpartition(".")[0]
            dynamic_names = {"__import__"}
            own_names = set()
            package_aliases = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "importlib":
                    dynamic_names.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
                if isinstance(node, ast.Assign) and any(
                    isinstance(value, ast.Constant) and isinstance(value.value, str)
                    and "world_reward" in value.value for value in ast.walk(node.value)
                ):
                    own_names.update(target.id for target in node.targets if isinstance(target, ast.Name))
                if isinstance(node, ast.Import):
                    package_aliases.update(alias.asname or alias.name for alias in node.names if alias.name == "world_reward")
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.update(module_paths(alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        parts = package.split(".") if package else []
                        if node.level > len(parts):
                            raise ValueError(f"Unresolved relative runtime import in {path}")
                        module = ".".join(parts[:len(parts) - node.level + 1] + ([module] if module else []))
                    dependencies.update(module_paths(module))
                    if module == "world_reward" or module.startswith("world_reward."):
                        for alias in node.names:
                            if alias.name != "*":
                                dependencies.update(module_paths(module + "." + alias.name, required=False))
                elif isinstance(node, ast.Call) and (
                    (isinstance(node.func, ast.Name) and node.func.id in dynamic_names)
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                ):
                    argument = node.args[0] if node.args else next((value.value for value in node.keywords if value.arg == "name"), None)
                    dynamic_package = node.args[1] if len(node.args) > 1 else next((value.value for value in node.keywords if value.arg == "package"), None)
                    own_dynamic_package = (isinstance(dynamic_package, ast.Constant) and isinstance(dynamic_package.value, str)
                                           and (dynamic_package.value == "world_reward" or dynamic_package.value.startswith("world_reward.")))
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        module = argument.value
                        if own_dynamic_package and module.startswith("."):
                            level = len(module) - len(module.lstrip("."))
                            parts = dynamic_package.value.split(".")
                            if level > len(parts):
                                raise ValueError(f"Unresolved relative dynamic runtime import in {path}")
                            module = ".".join(parts[:len(parts) - level + 1] + ([module[level:]] if module[level:] else []))
                        if module == "world_reward" or module.startswith("world_reward."):
                            dependencies.update(module_paths(module))
                    elif argument is not None and (own_dynamic_package or package or any(
                        (isinstance(value, ast.Constant) and isinstance(value.value, str) and "world_reward" in value.value)
                        or (isinstance(value, ast.Name) and value.id in own_names) for value in ast.walk(argument)
                    )):
                        raise ValueError(f"Computed own-package dynamic imports are unsupported: {path}")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and node.args:
                    if isinstance(node.args[0], ast.Name) and node.args[0].id in package_aliases:
                        name = node.args[1] if len(node.args) > 1 else None
                        if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
                            raise ValueError(f"Computed own-package attribute imports are unsupported: {path}")
                        dependencies.update(module_paths("world_reward." + name.value, required=False))
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in package_aliases:
                    dependencies.update(module_paths("world_reward." + node.attr, required=False))
        # Literal $CODE/infra/foo, including Python subprocess child entrypoints.
        dependencies.update("infra/" + name for name in re.findall(
            r"/infra/([a-z0-9_]+\.(?:py|sh)|Dockerfile\.[a-z0-9_]+)", source,
        ))
        dependencies.update("src/world_reward/" + name for name in re.findall(
            r"/src/world_reward/([a-zA-Z0-9_/]+\.py)", source,
        ))
        for dependency in sorted(dependencies):
            if dependency not in files:
                raise ValueError(f"Entrypoint dependency is not committed: {dependency}")
            pending.append(dependency)
    if any(not path.startswith(("infra/", "src/", "configs/")) and path != "pyproject.toml" for path in selected):
        raise ValueError("Runtime bundle contains non-code paths")
    return sorted(selected)


def runtime_archive(full_archive: bytes, script: str) -> tuple[bytes, list[str]]:
    """Keep original Git member metadata while dropping irrelevant source files."""
    with tarfile.open(fileobj=io.BytesIO(full_archive), mode="r:") as source:
        members = {member.name: member for member in source.getmembers() if member.isfile()}
        files = {name: source.extractfile(member).read() for name, member in members.items()}
        # Reject symlinks/hardlinks rather than silently invent a runtime alias.
        if any(not member.isfile() and not member.isdir() for member in source.getmembers()):
            raise ValueError("Code snapshots cannot contain symlinks or special files")
        selected = runtime_bundle_paths(files, script)
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as target:
            for path in selected:
                target.addfile(members[path], io.BytesIO(files[path]))
    return output.getvalue(), selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--script", required=True, help="Committed infra/*.sh entrypoint")
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,50}", args.name):
        raise ValueError("Job name must be a short lowercase slug")
    if not re.fullmatch(r"infra/[a-z0-9_]+\.sh", args.script):
        raise ValueError("Only explicit committed infra shell entrypoints are allowed")
    # RTK also wraps child shell-facing CLIs on the local host.
    def git(*arguments: str) -> bytes:
        return subprocess.check_output(["rtk", "proxy", "git", *arguments])
    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise RuntimeError("Commit/clean the worktree before launching a reproducible job")
    revision = git("rev-parse", "HEAD").decode().strip()
    git("cat-file", "-e", f"{revision}:{args.script}")
    source_archive, paths = runtime_archive(git("archive", "--format=tar", revision,
                                               "infra", "src", "configs", "pyproject.toml"), args.script)
    archive = lzma.compress(source_archive, preset=6)
    if len(archive) > 2_000_000:
        raise RuntimeError("Code archive unexpectedly large; no heavy artifacts may transit locally")
    archive_hash = hashlib.sha256(archive).hexdigest()
    encoded = base64.b64encode(archive).decode()
    if len(encoded) > 100_000:
        raise RuntimeError("Run Command payload exceeds the small code-only control budget")
    unit = "world-reward-" + args.name
    command_arguments = " ".join(shlex.quote(value) for value in args.arguments)
    bundle_id = args.script.removeprefix("infra/").removesuffix(".sh")
    command = f"""set -eu
ROOT=/srv/scenesmith/world-reward
UNIT={shlex.quote(unit)}
test -z "$(systemctl list-units --all --plain --no-legend "$UNIT.service")"
JOB="$ROOT/jobs/{revision}/{bundle_id}"
test ! -e "$ROOT/results/{args.name}.log"
if test ! -d "$JOB"; then
  mkdir -p "$JOB"
  printf '%s' '{encoded}' | base64 -d > "$JOB/source.tar.xz"
  echo '{archive_hash}  '"$JOB/source.tar.xz" | sha256sum -c - >/dev/null
  mkdir "$JOB/code"
  tar -xJf "$JOB/source.tar.xz" -C "$JOB/code"
  rm "$JOB/source.tar.xz"
  printf '%s\\n' '{revision}' > "$JOB/revision"
  printf '%s\\n' '{archive_hash}' > "$JOB/source-sha256"
  chmod -R a-w "$JOB/code"
fi
test "$(cat "$JOB/revision")" = '{revision}'
test "$(cat "$JOB/source-sha256")" = '{archive_hash}'
systemd-run --unit "$UNIT" --property=Type=exec \\
  --property=StandardOutput=append:"$ROOT/results/{args.name}.log" \\
  --property=StandardError=append:"$ROOT/results/{args.name}.log" \\
  /usr/bin/env WR_ROOT="$ROOT" WR_CODE="$JOB/code" WR_CODE_REVISION='{revision}' \\
  /bin/bash "$JOB/code/{args.script}" {command_arguments}
systemctl show "$UNIT" -p ActiveState -p ExecMainStatus -p MainPID
"""
    print(f"immutable_runtime_bundle_files={len(paths)} encoded_bytes={len(encoded)} revision={revision}", flush=True)
    subprocess.run(["rtk", "proxy", "az", "vm", "run-command", "invoke",
                    "--resource-group", "SCENESMITH-H100", "--name", "scenesmith-ncc-h100-01",
                    "--command-id", "RunShellScript", "--scripts", command,
                    "--query", "value[0].message", "-o", "tsv"], check=True)


if __name__ == "__main__":
    main()
