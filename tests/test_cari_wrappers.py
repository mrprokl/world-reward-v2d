"""Run wrappers against tiny fake shell commands, never Docker/Azure or media."""

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    "prepare": ("run_cari_prepare.sh", "cari_prepare.py", "object_pose_full", "world-reward-object-pose-full.service"),
    "forward": ("run_cari_forward.sh", "cari_forward.py", "cari_inputs", "world-reward-cari-prepare-v2.service"),
    "converter": ("run_cari_converter.sh", "cari_converter.py", "cari_forward", "world-reward-cari-forward.service"),
    "adapter": ("run_cari_body_adapter_smoke.sh", "cari_body_adapter_smoke.py", "body_full", None),
}


@pytest.fixture
def fake_shell(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    log, state = tmp_path / "commands.log", tmp_path / "state"
    state.write_text("inactive")
    commands = {
        "docker": 'printf "docker\\0" >> "$FAKE_LOG"; printf "%s\\0" "$@" >> "$FAKE_LOG"',
        "id": 'printf "123\\n"',
        "systemctl": '''printf "systemctl %s\\n" "$*" >> "$FAKE_CTL_LOG"
case "$3" in
  --property=LoadState) printf '%s\\n' "$FAKE_LOAD" ;;
  --property=ActiveState) cat "$FAKE_STATE" ;;
  *) exit 9 ;;
esac''',
        "sleep": '''printf "sleep %s\\n" "$*" >> "$FAKE_CTL_LOG"
printf '%s' "$FAKE_STOP_STATE" > "$FAKE_STATE"
if [[ -n "${FAKE_CREATE_REPORT:-}" ]]; then mkdir -p "$(dirname "$FAKE_CREATE_REPORT")"; printf '{}' > "$FAKE_CREATE_REPORT"; fi''',
    }
    for name, source in commands.items():
        path = binary / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + source + "\n")
        path.chmod(0o755)
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                       WR_ROOT=str(tmp_path / "runtime"), WR_CODE=str(ROOT), WR_CODE_REVISION="a" * 40,
                       FAKE_LOG=str(log), FAKE_CTL_LOG=str(tmp_path / "ctl.log"), FAKE_STATE=str(state),
                       FAKE_LOAD="loaded", FAKE_STOP_STATE="inactive")
    def run(mode, *arguments):
        return subprocess.run(["bash", str(ROOT / "infra" / WRAPPERS[mode][0]), *arguments],
                              env=environment, text=True, capture_output=True, timeout=5)
    def report(mode, episode=15, stage=None):
        path = Path(environment["WR_ROOT"]) / f"outputs/episode_{episode:06d}" / (stage or WRAPPERS[mode][2]) / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path
    return environment, run, report, log, state


def docker_arguments(log):
    assert log.exists()
    return log.read_bytes().decode().strip("\0").split("\0")


@pytest.mark.parametrize("mode", WRAPPERS)
def test_default15_preserves_original_stage_dependencies_and_passes_explicit_episode(fake_shell, mode):
    env, run, report, log, _ = fake_shell
    report(mode)
    result = run(mode)
    assert result.returncode == 0, result.stderr
    arguments = docker_arguments(log)
    assert arguments[-3:] == [str(ROOT / "infra" / WRAPPERS[mode][1]), "--episode", "15"]
    assert "--network" in arguments and arguments[arguments.index("--network") + 1] == "none"
    expected_unit = WRAPPERS[mode][3]
    ctl = Path(env["FAKE_CTL_LOG"])
    if expected_unit:
        assert expected_unit in ctl.read_text()
    else:
        assert not ctl.exists()


@pytest.mark.parametrize("mode", WRAPPERS)
@pytest.mark.parametrize("episode", [0, 1, 29])
def test_nondefault_episode_requires_own_report_without_implicit_episode15_wait(fake_shell, mode, episode):
    env, run, report, log, _ = fake_shell
    report(mode, episode)
    result = run(mode, "--episode", str(episode))
    assert result.returncode == 0, result.stderr
    assert docker_arguments(log)[-2:] == ["--episode", str(episode)]
    assert not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("mode", WRAPPERS)
def test_other_episode_report_cannot_satisfy_selected_clip(fake_shell, mode):
    env, run, report, log, _ = fake_shell
    report(mode, 15)
    result = run(mode, "--episode", "0")
    assert result.returncode != 0 and "episode_000000" in result.stderr
    assert not log.exists() and not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("mode", WRAPPERS)
@pytest.mark.parametrize("arguments", [
    ("--episode",), ("--episode", "-1"), ("--episode", "30"), ("--episode", "true"),
    ("--episode", "False"), ("--episode", "15.0"), ("--episode", "01"), ("--episode", "+1"),
    ("--episode", "1", "--episode", "2"), ("--unknown",), ("--episode=1",),
    ("--wait-for",), ("--wait-for", "other-service"), ("--wait-for", "world-reward-x;evil"),
    ("--wait-for", "world-reward-../x"), ("--wait-for", "world-reward-x", "--wait-for", "world-reward-y"),
])
def test_invalid_unknown_duplicate_alias_or_injection_arguments_fail_before_commands(fake_shell, mode, arguments):
    env, run, _, log, _ = fake_shell
    result = run(mode, *arguments)
    assert result.returncode == 2
    assert not log.exists() and not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("mode", ["prepare", "converter", "adapter"])
def test_kernel_only_not_accepted_by_nonforward_wrappers(fake_shell, mode):
    env, run, _, log, _ = fake_shell
    result = run(mode, "--kernel-only")
    assert result.returncode == 2 and not log.exists()
    assert not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_checkpoint_only_has_body_dependency_but_never_waits_for_prepare(fake_shell, episode):
    env, run, report, log, _ = fake_shell
    report("forward", episode, "body_full")
    result = run("forward", "--episode", str(episode), "--kernel-only")
    assert result.returncode == 0, result.stderr
    assert docker_arguments(log)[-3:] == ["--episode", str(episode), "--kernel-only"]
    assert not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("arguments", [("--kernel-only", "--kernel-only"),
                                        ("--kernel-only", "--wait-for", "world-reward-prepare")])
def test_checkpoint_only_rejects_duplicate_or_preparation_wait(fake_shell, arguments):
    env, run, _, log, _ = fake_shell
    assert run("forward", *arguments).returncode == 2
    assert not log.exists() and not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("mode", WRAPPERS)
@pytest.mark.parametrize("suffix", ["", ".service"])
def test_explicit_wait_unit_normalized_not_forwarded_to_python(fake_shell, mode, suffix):
    env, run, report, log, _ = fake_shell
    report(mode, 0)
    result = run(mode, "--wait-for", "world-reward-own-episode0" + suffix, "--episode", "0")
    assert result.returncode == 0, result.stderr
    assert "world-reward-own-episode0.service" in Path(env["FAKE_CTL_LOG"]).read_text()
    assert "--wait-for" not in docker_arguments(log)


@pytest.mark.parametrize("mode", WRAPPERS)
def test_active_explicit_producer_waits_once_then_requires_selected_report(fake_shell, mode):
    env, run, _, log, state = fake_shell
    state.write_text("active")
    target = Path(env["WR_ROOT"]) / "outputs/episode_000029" / WRAPPERS[mode][2] / "report.json"
    env["FAKE_CREATE_REPORT"] = str(target)
    result = run(mode, "--episode", "29", "--wait-for", "world-reward-own29")
    assert result.returncode == 0, result.stderr
    assert target.exists() and log.exists()
    assert Path(env["FAKE_CTL_LOG"]).read_text().count("sleep 30") == 1


@pytest.mark.parametrize("mode", WRAPPERS)
@pytest.mark.parametrize("state_value", ["inactive", "failed", "mystery"])
def test_stopped_failed_unknown_producer_without_report_fails_without_waiting(fake_shell, mode, state_value):
    env, run, _, log, state = fake_shell
    state.write_text(state_value)
    result = run(mode, "--episode", "0", "--wait-for", "world-reward-no-report")
    assert result.returncode != 0 and not log.exists()
    assert "sleep" not in Path(env["FAKE_CTL_LOG"]).read_text()


@pytest.mark.parametrize("mode", WRAPPERS)
def test_missing_unit_cannot_wait_forever_or_claim_success_without_report(fake_shell, mode):
    env, run, _, log, _ = fake_shell
    env["FAKE_LOAD"] = "not-found"
    result = run(mode, "--episode", "0", "--wait-for", "world-reward-missing")
    assert result.returncode != 0 and "absent" in result.stderr and not log.exists()
    assert "sleep" not in Path(env["FAKE_CTL_LOG"]).read_text()


def test_collected_old_default_unit_can_use_frozen_report(fake_shell):
    env, run, report, log, _ = fake_shell
    report("prepare")
    env["FAKE_LOAD"] = "not-found"
    assert run("prepare").returncode == 0
    assert log.exists()


def test_failed_unit_is_not_rehabilitated_by_existing_report(fake_shell):
    _, run, report, log, state = fake_shell
    report("forward")
    state.write_text("failed")
    result = run("forward")
    assert result.returncode != 0 and "failed" in result.stderr and not log.exists()


def test_successful_process_completion_without_report_still_fails(fake_shell):
    env, run, _, log, state = fake_shell
    state.write_text("active")
    result = run("prepare", "--episode", "29", "--wait-for", "world-reward-own29")
    assert result.returncode != 0 and "stopped without" in result.stderr and not log.exists()
    assert Path(env["FAKE_CTL_LOG"]).read_text().count("sleep 30") == 1


def test_existing_report_does_not_bypass_active_producer(fake_shell):
    env, run, report, log, state = fake_shell
    report("converter", 0)
    state.write_text("active")
    result = run("converter", "--episode", "0", "--wait-for", "world-reward-own0")
    assert result.returncode == 0, result.stderr
    assert log.exists() and Path(env["FAKE_CTL_LOG"]).read_text().count("sleep 30") == 1


def test_wait_is_bounded_without_real_sleep_or_subprocess_systemctl(tmp_path):
    helper = ROOT / "infra/cari_wrapper_common.sh"
    script = f'''set -eu
source "{helper}"
systemctl() {{ if [[ "$3" == --property=LoadState ]]; then echo loaded; else echo active; fi; }}
sleep() {{ :; }}
wr_require_dependency_report "{tmp_path}/missing.json" world-reward-running.service
'''
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=5)
    assert result.returncode != 0 and "12 hours" in result.stderr


def test_immutable_launcher_import_closure_includes_common_shell_helper_and_python_entrypoint():
    spec = importlib.util.spec_from_file_location("world_reward_test_wrapper_launcher", ROOT / "infra/azure_job.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    files = {str(path.relative_to(ROOT)): path.read_bytes() for path in (ROOT / "infra").glob("*") if path.is_file()}
    for shell, python, _, _ in WRAPPERS.values():
        closure = launcher.runtime_bundle_paths(files, "infra/" + shell)
        assert "infra/cari_wrapper_common.sh" in closure
        assert "infra/" + python in closure
