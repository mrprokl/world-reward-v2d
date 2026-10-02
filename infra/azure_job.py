"""Launch one immutable, committed code snapshot via Azure Run Command.

Only a small Git archive is sent from the laptop. Jobs/data/model artifacts stay
remote. Existing units and snapshots are never replaced or restarted implicitly.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import shlex
import subprocess


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
    archive = git("archive", "--format=tar.gz", revision,
                  "infra", "src", "configs", "pyproject.toml")
    if len(archive) > 2_000_000:
        raise RuntimeError("Code archive unexpectedly large; no heavy artifacts may transit locally")
    archive_hash = hashlib.sha256(archive).hexdigest()
    encoded = base64.b64encode(archive).decode()
    if len(encoded) > 100_000:
        raise RuntimeError("Run Command payload exceeds the small code-only control budget")
    unit = "world-reward-" + args.name
    command_arguments = " ".join(shlex.quote(value) for value in args.arguments)
    command = f"""set -eu
ROOT=/srv/scenesmith/world-reward
UNIT={shlex.quote(unit)}
test -z "$(systemctl list-units --all --plain --no-legend "$UNIT.service")"
JOB="$ROOT/jobs/{revision}"
test ! -e "$ROOT/results/{args.name}.log"
if test ! -d "$JOB"; then
  mkdir -p "$JOB"
  printf '%s' '{encoded}' | base64 -d > "$JOB/source.tar.gz"
  echo '{archive_hash}  '"$JOB/source.tar.gz" | sha256sum -c - >/dev/null
  mkdir "$JOB/code"
  tar -xzf "$JOB/source.tar.gz" -C "$JOB/code"
  rm "$JOB/source.tar.gz"
  printf '%s\\n' '{revision}' > "$JOB/revision"
  chmod -R a-w "$JOB/code"
fi
test "$(cat "$JOB/revision")" = '{revision}'
systemd-run --unit "$UNIT" --property=Type=exec \\
  --property=StandardOutput=append:"$ROOT/results/{args.name}.log" \\
  --property=StandardError=append:"$ROOT/results/{args.name}.log" \\
  /usr/bin/env WR_ROOT="$ROOT" WR_CODE="$JOB/code" WR_CODE_REVISION='{revision}' \\
  /bin/bash "$JOB/code/{args.script}" {command_arguments}
systemctl show "$UNIT" -p ActiveState -p ExecMainStatus -p MainPID
"""
    subprocess.run(["rtk", "proxy", "az", "vm", "run-command", "invoke",
                    "--resource-group", "SCENESMITH-H100", "--name", "scenesmith-ncc-h100-01",
                    "--command-id", "RunShellScript", "--scripts", command,
                    "--query", "value[0].message", "-o", "tsv"], check=True)


if __name__ == "__main__":
    main()
