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

    Include the whole lightweight project package and config. Infra dependencies
    are direct shell file references and static Python imports; dynamic infra
    plugins are not supported. Each entrypoint gets its own immutable directory
    so different closures at the same commit cannot alias.
    """
    if not re.fullmatch(r"infra/[a-z0-9_]+\.sh", script) or script not in files:
        raise ValueError("Require a committed infra shell entrypoint")
    selected = {path for path in files if path.startswith(("src/", "configs/")) or path == "pyproject.toml"}
    pending = [script]
    while pending:
        path = pending.pop()
        if path in selected:
            continue
        selected.add(path)
        source = files[path].decode("utf-8")
        dependencies = set()
        if path.endswith(".py"):
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [(node.module or "").split(".")[0]]
                else:
                    continue
                dependencies.update(f"infra/{module}.py" for module in modules if f"infra/{module}.py" in files)
        # Literal $CODE/infra/foo, including Python subprocess child entrypoints.
        dependencies.update("infra/" + name for name in re.findall(
            r"/infra/([a-z0-9_]+\.(?:py|sh)|Dockerfile\.[a-z0-9_]+)", source,
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
