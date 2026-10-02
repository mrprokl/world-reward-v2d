"""Shell preflight stubs only; never install/start daemons on the test host."""
import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/"infra/research_runtime_bootstrap.sh"


@pytest.fixture
def shell(tmp_path):
    binary = tmp_path/"bin"; binary.mkdir(); log = tmp_path/"commands"
    commands = {
        "uname": 'printf "%s\\n" "${FAKE_OS:-Linux}"',
        "id": '''case "$*" in
 "-u") echo 0 ;; "-u scenesmith") echo "${FAKE_UID:-1000}" ;; "-g scenesmith") id -g ;; "-gn scenesmith") echo scenesmith ;; *) exit 2 ;; esac'''.replace('id -g', '/usr/bin/id -g'),
        "systemctl": 'printf "%s\\n" "${FAKE_LOAD:-not-found}"',
        "containerd": 'printf "containerd containerd.io %s abcdef\\n" "${FAKE_CONTAINERD_VERSION:-1.7.27}"',
        "dockerd": 'echo "Docker version 26.1.5, build fixture"',
        "docker": 'echo "Docker version 26.1.5, build fixture"',
        "nvidia-container-runtime": 'echo "NVIDIA Container Runtime version1.19.1"',
        "systemd-run": 'echo "unexpected systemd-run" >&2; exit 77',
        "chown": ':',
        "readlink": '''python3 - "$2" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY''',
    }
    for name, text in commands.items():
        path = binary/name
        path.write_text('#!/usr/bin/env bash\nset -euo pipefail\nprintf "%s\\n" "${0##*/} $*" >> "$FAKE_LOG"\n'+text+'\n')
        path.chmod(0o755)
    # Replace only absolute host probes; the production script has no test knobs.
    config = tmp_path/"nvidia.toml"; config.write_text("# own tiny VMI config fixture\n")
    runtime = tmp_path/"host-runtime"; copied = tmp_path/"bootstrap.sh"
    root = tmp_path/"fresh-world-reward"
    source = SCRIPT.read_text().replace("NVIDIA_CONFIG=/etc/nvidia-container-runtime/config.toml", f"NVIDIA_CONFIG={config}")
    source = source.replace('"$ROOT" == /srv/scenesmith/world-reward', f'"$ROOT" == {root}')
    source = source.replace('s.st_uid!=1000', f's.st_uid!={os.getuid()}')
    source = source.replace("/run/world-reward-vm02-docker", str(runtime))
    copied.write_text(source)
    env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ["PATH"], WR_ROOT=str(root), WR_CODE_REVISION="a"*40, FAKE_LOG=str(log))
    def run(*args): return subprocess.run(["bash", str(copied), *args], env=env, text=True, capture_output=True, timeout=5)
    return env, run, log, root, runtime, config, binary


@pytest.mark.parametrize("version", ["1.7.27", "v1.6.33", "2.2.4", "v2.0.0"])
def test_preflight_has_zero_writes_or_daemon_actions(shell, version):
    env, run, log, root, _, _, _ = shell; env["FAKE_CONTAINERD_VERSION"] = version
    result = run()
    assert result.returncode == 0, result.stderr
    assert "runtime_preflight=pass" in result.stdout and not root.exists()
    assert "systemd-run" not in log.read_text() and "--version" in log.read_text()
    assert " start " not in log.read_text() and " stop " not in log.read_text()


@pytest.mark.parametrize("fault", ["arguments", "mode", "os", "uid", "revision", "version", "unit", "root_relative", "root_traversal", "root_symlink", "results_symlink", "config_symlink", "runtime", "world_writable", "unwritable"])
def test_preflight_refuses_unsafe_or_unknown_state(shell, fault, tmp_path):
    env, run, log, root, runtime, config, _ = shell; args = []
    if fault == "arguments": args = ["--preflight", "extra"]
    elif fault == "mode": args = ["--resume"]
    elif fault == "os": env["FAKE_OS"] = "Darwin"
    elif fault == "uid": env["FAKE_UID"] = "1001"
    elif fault == "revision": env["WR_CODE_REVISION"] = "bad"
    elif fault == "version": env["FAKE_CONTAINERD_VERSION"] = "3.0.0"
    elif fault == "unit": env["FAKE_LOAD"] = "loaded"
    elif fault == "root_relative": env["WR_ROOT"] = "relative"
    elif fault == "root_traversal": env["WR_ROOT"] = str(tmp_path/"unused/../fresh-world-reward")
    elif fault == "root_symlink":
        actual = tmp_path/"actual"; actual.mkdir(); root.symlink_to(actual, target_is_directory=True)
    elif fault == "results_symlink":
        root.mkdir(); (root/"results").symlink_to(tmp_path/"outside", target_is_directory=True)
    elif fault == "config_symlink":
        saved = tmp_path/"realconfig"; config.rename(saved); config.symlink_to(saved)
    elif fault == "world_writable": root.mkdir(); root.chmod(0o777)
    elif fault == "unwritable": root.mkdir(); root.chmod(0o555)
    else: runtime.mkdir()
    result = run(*args)
    assert result.returncode != 0 and "systemd-run" not in (log.read_text() if log.exists() else "")


@pytest.mark.parametrize("target", ["docker", "docker-exec", "containerd", "runtime", "docker.sock", "results/research-runtime.json"])
@pytest.mark.parametrize("symlink", [False, True])
def test_existing_task_outputs_never_reused_or_overwritten(shell, target, symlink):
    _, run, log, root, _, _, _ = shell; path = root/target; path.parent.mkdir(parents=True, exist_ok=True)
    if symlink: path.symlink_to(root/"does-not-exist")
    else: path.write_bytes(b"frozen fixture")
    result = run("--install")
    assert result.returncode != 0 and "Fresh runtime path occupied" in result.stderr
    assert path.is_symlink() if symlink else path.read_bytes() == b"frozen fixture"
    assert "systemd-run" not in log.read_text()


@pytest.mark.parametrize("version,expected", [("1.7.27", "version = 2"), ("2.2.4", "version = 3")])
def test_explicit_install_records_failure_no_retry_and_new_config_only(shell, version, expected):
    env, run, log, root, runtime, _, _ = shell; env["FAKE_CONTAINERD_VERSION"] = version
    result = run("--install")
    assert result.returncode == 77 and "unexpected systemd-run" in result.stderr
    config = (root/"runtime/containerd.toml").read_text()
    assert expected in config and f'address = "{runtime}/containerd.sock"' in config
    assert str(root/"containerd") in config and "imports" not in config
    assert "io.containerd.cri.v1.runtime" in config if version.startswith("2") else "io.containerd.grpc.v1.cri" in config
    assert log.read_text().count("systemd-run --unit") == 1
    report = (root/"results/research-runtime.json").read_text()
    assert '"status":"fail"' in report and '"phase":"containerd_start"' in report
    frozen = (root/"runtime/containerd.toml").read_bytes()
    assert run("--install").returncode != 0 and (root/"runtime/containerd.toml").read_bytes() == frozen
    assert [l for l in log.read_text().splitlines() if l.startswith('chown ')] == [f'chown 1000:{os.getgid()} {root}', f'chown 1000:{os.getgid()} {root}/results']
    assert root.stat().st_mode & 0o777 == 0o755 and (root/'results').stat().st_mode & 0o777 == 0o755


def test_daemon_isolation_and_no_host_mutation_static():
    source = SCRIPT.read_text()
    assert "--containerd /run/world-reward-vm02-docker/containerd.sock" in source
    assert "--containerd-namespace world-reward" in source and "--containerd-plugins-namespace world-reward-plugins" in source
    assert "--data-root \"$ROOT/docker\" --exec-root \"$ROOT/docker-exec\"" in source
    assert "--pidfile \"$ROOT/docker-exec/dockerd.pid\" --host \"unix://$ROOT/docker.sock\"" in source
    for flag in ("--bridge none", "--iptables=false", "--ip6tables=false", "--ip-masq=false", "--ip-forward=false", "--userland-proxy=false"):
        assert flag in source
    assert source.count("--property=Restart=no") == 2
    assert "RuntimeDirectory=world-reward-vm02-docker" in source and "--property=Delegate=yes" in source
    assert not any(command in source for command in ("apt ", "pip ", "curl ", "useradd", "usermod", "chown -R", "systemctl restart", "systemctl stop", "systemctl enable", "daemon-reload", "iptables -"))
    assert '/etc/docker/daemon.json' not in source and '/etc/containerd/config.toml' not in source
    assert "GPU_execution_verified':False" in source and "image_import_verified':False" in source
    assert "containerd-snapshotter" in source and "overlay2" in source
    assert 'for p in (root,*root.parents' in source and 's.st_uid!=1000' in source
    assert 'chown "1000:$GID" "$path"' in source and 'mkdir -m 0755 "$path"' in source


def test_shell_syntax():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
