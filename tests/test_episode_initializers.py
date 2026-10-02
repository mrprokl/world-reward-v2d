"""Initializer orchestration against tiny fake shell children, never GPU/data."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
WRAPPER = REPOSITORY / "infra/run_episode_initializers.sh"
STAGES = (
    "body_smoke", "depth_smoke", "scale_smoke", "object_grounded",
    "body_full", "depth_full", "cari_adapter",
)
TARGETS = (*STAGES[:-1], "body_full/cari_adapter")


@pytest.fixture
def fake_runtime(tmp_path):
    runtime, code, binary = (tmp_path / name for name in ("runtime", "code", "bin"))
    (code / "infra").mkdir(parents=True)
    binary.mkdir()
    (code / "infra/cari_wrapper_common.sh").write_text(
        (REPOSITORY / "infra/cari_wrapper_common.sh").read_text())
    state = tmp_path / "state"
    state.write_text("inactive")
    log, ctl = tmp_path / "children.log", tmp_path / "ctl.log"
    commands = {
        "systemctl": '''printf 'systemctl %s\\n' "$*" >> "$FAKE_CTL_LOG"
case "$3" in
  --property=LoadState) printf '%s\\n' "$FAKE_LOAD" ;;
  --property=ActiveState) cat "$FAKE_STATE" ;;
  *) exit 9 ;;
esac''',
        "sleep": '''printf 'sleep %s\\n' "$*" >> "$FAKE_CTL_LOG"
printf '%s' "$FAKE_STOP_STATE" > "$FAKE_STATE"
if [[ -n "${FAKE_CREATE_REPORT:-}" ]]; then
  mkdir -p "$(dirname "$FAKE_CREATE_REPORT")"; printf '{}' > "$FAKE_CREATE_REPORT"
fi''',
    }
    for name, source in commands.items():
        path = binary / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + source + "\n")
        path.chmod(0o755)
    child = '''#!/usr/bin/env bash
set -euo pipefail
name="$(basename "$0")"
episode="$2"
printf -v padded '%06d' "$episode"
case "$name" in
  run_body_smoke.sh) if [[ " $* " == *" --full-video "* ]]; then stage=body_full; else stage=body_smoke; fi ;;
  run_depth_smoke.sh) if [[ " $* " == *" --full-video "* ]]; then stage=depth_full; else stage=depth_smoke; fi ;;
  run_scale_smoke.sh) stage=scale_smoke ;;
  run_object_smoke.sh) stage=object_grounded ;;
  run_cari_body_adapter_smoke.sh) stage=cari_adapter ;;
  *) exit 9 ;;
esac
printf '%s|%s\\n' "$stage" "$*" >> "$FAKE_CHILD_LOG"
if [[ "$stage" == "${FAKE_FAIL_STAGE:-}" ]]; then echo 'synthetic child failure' >&2; exit 7; fi
target="$stage"
if [[ "$stage" == cari_adapter ]]; then target=body_full/cari_adapter; fi
mkdir -p "$WR_ROOT/outputs/episode_$padded/$target"
printf '{}' > "$WR_ROOT/outputs/episode_$padded/$target/report.json"
'''
    for name in ("run_body_smoke.sh", "run_depth_smoke.sh", "run_scale_smoke.sh",
                 "run_object_smoke.sh", "run_cari_body_adapter_smoke.sh"):
        (code / "infra" / name).write_text(child)
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                       WR_ROOT=str(runtime), WR_CODE=str(code), FAKE_CHILD_LOG=str(log),
                       FAKE_CTL_LOG=str(ctl), FAKE_STATE=str(state), FAKE_LOAD="loaded",
                       FAKE_STOP_STATE="inactive")
    gate = runtime / "results/camera-render.json"
    gate.parent.mkdir(parents=True)
    gate.write_text('{"status":"pass"}')
    def report(episode=0):
        path = runtime / f"outputs/episode_{episode:06d}/automatic_masks/report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path
    def run(*arguments):
        return subprocess.run(["bash", str(WRAPPER), *arguments], env=environment,
                              capture_output=True, text=True, timeout=5)
    return environment, run, report, log, ctl, state


def phases(result):
    return [json.loads(line) for line in result.stdout.splitlines()]


@pytest.mark.parametrize("episode", [0, 1, 15, 29])
def test_serial_exact_stage_commands_selected_episode_and_no_implicit_wait(fake_runtime, episode):
    env, run, report, log, ctl, _ = fake_runtime
    report(episode)
    result = run("--episode", str(episode))
    assert result.returncode == 0, result.stderr
    expected = [
        f"body_smoke|--episode {episode} --inference-type body",
        f"depth_smoke|--episode {episode}",
        f"scale_smoke|--episode {episode}",
        f"object_grounded|--episode {episode} --aligned-pointmap",
        f"body_full|--episode {episode} --full-video --inference-type body",
        f"depth_full|--episode {episode} --full-video",
        f"cari_adapter|--episode {episode}",
    ]
    assert log.read_text().splitlines() == expected
    assert not ctl.exists()
    assert [(row["stage"], row["phase"]) for row in phases(result)] == [
        (stage, phase) for stage in ("preflight", *STAGES) for phase in ("start", "pass")]
    assert all(set(row) == {"stage", "phase", "timestamp_utc"} for row in phases(result))
    assert all(row["timestamp_utc"].endswith("Z") for row in phases(result))
    for target in TARGETS:
        assert (Path(env["WR_ROOT"]) / f"outputs/episode_{episode:06d}/{target}/report.json").is_file()


@pytest.mark.parametrize("arguments", [
    (), ("--episode",), ("--episode", "-1"), ("--episode", "30"),
    ("--episode", "true"), ("--episode", "False"), ("--episode", "15.0"),
    ("--episode", "00"), ("--episode", "01"), ("--episode", "+1"),
    ("--episode", "1", "--episode", "2"), ("--unknown",), ("--episode=1",),
    ("--wait-for",), ("--wait-for", "world-reward-valid"),
    ("--episode", "0", "--wait-for", "other-service"),
    ("--episode", "0", "--wait-for", "world-reward-x;evil"),
    ("--episode", "0", "--wait-for", "world-reward-../x"),
    ("--episode", "0", "--wait-for", "world-reward-x", "--wait-for", "world-reward-y"),
    ("--episode", "../track_3"), ("--episode", "0", "--full-video"),
    ("--episode", "0", "--kernel-only"), ("--episode", "0", "--root", "/tmp"),
])
def test_invalid_missing_duplicate_alias_or_injection_fails_before_io(fake_runtime, arguments):
    _, run, _, log, ctl, _ = fake_runtime
    result = run(*arguments)
    assert result.returncode == 2
    assert not log.exists() and not ctl.exists()
    assert result.stdout == ""


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("kind", ["directory", "report", "file", "broken_symlink"])
def test_any_existing_target_aborts_before_consuming_any_gpu_stage(fake_runtime, target, kind):
    env, run, report, log, _, _ = fake_runtime
    report()
    path = Path(env["WR_ROOT"]) / "outputs/episode_000000" / target
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind in ("directory", "report"):
        path.mkdir()
        if kind == "report": (path / "report.json").write_text('{"status":"pass"}')
    elif kind == "file": path.write_text("tiny frozen artifact")
    else: path.symlink_to("does-not-exist")
    result = run("--episode", "0")
    assert result.returncode != 0 and "already exists" in result.stderr
    assert not log.exists()
    assert [(row["stage"], row["phase"]) for row in phases(result)] == [("preflight", "start"), ("preflight", "fail")]
    assert path.is_symlink() if kind == "broken_symlink" else path.exists()


def test_another_episode_masks_report_cannot_satisfy_selected_episode(fake_runtime):
    _, run, report, log, _, _ = fake_runtime
    report(15)
    result = run("--episode", "0")
    assert result.returncode != 0 and "episode_000000" in result.stderr
    assert not log.exists()


@pytest.mark.parametrize("suffix", ["", ".service"])
def test_wait_for_exact_selected_masks_producer_is_normalized_not_forwarded(fake_runtime, suffix):
    _, run, report, log, ctl, _ = fake_runtime
    report()
    result = run("--wait-for", "world-reward-masks-own0" + suffix, "--episode", "0")
    assert result.returncode == 0, result.stderr
    assert "world-reward-masks-own0.service" in ctl.read_text()
    assert "--wait-for" not in log.read_text()


def test_active_masks_producer_waits_before_preflight_and_child_gpu(fake_runtime):
    env, run, _, log, ctl, state = fake_runtime
    state.write_text("active")
    env["FAKE_CREATE_REPORT"] = str(Path(env["WR_ROOT"]) / "outputs/episode_000000/automatic_masks/report.json")
    result = run("--episode", "0", "--wait-for", "world-reward-masks-own0")
    assert result.returncode == 0, result.stderr
    assert log.exists() and ctl.read_text().count("sleep 30") == 1


@pytest.mark.parametrize("state_value", ["failed", "inactive", "mystery"])
def test_terminal_failed_or_unknown_producer_never_starts_child_gpu(fake_runtime, state_value):
    _, run, _, log, ctl, state = fake_runtime
    state.write_text(state_value)
    result = run("--episode", "0", "--wait-for", "world-reward-masks-own0")
    assert result.returncode != 0 and not log.exists()
    assert "sleep" not in ctl.read_text()


def test_failed_producer_is_not_reused_even_when_a_report_exists(fake_runtime):
    _, run, report, log, _, state = fake_runtime
    report()
    state.write_text("failed")
    result = run("--episode", "0", "--wait-for", "world-reward-masks-own0")
    assert result.returncode != 0 and "failed" in result.stderr and not log.exists()


def test_missing_unit_and_report_fail_fast_and_collected_unit_with_report_is_allowed(fake_runtime):
    env, run, report, log, ctl, _ = fake_runtime
    env["FAKE_LOAD"] = "not-found"
    result = run("--episode", "0", "--wait-for", "world-reward-masks-collected")
    assert result.returncode != 0 and not log.exists() and "sleep" not in ctl.read_text()
    report()
    result = run("--episode", "0", "--wait-for", "world-reward-masks-collected")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_missing_or_empty_global_camera_gate_fails_before_gpu_children(fake_runtime, kind):
    env, run, report, log, _, _ = fake_runtime
    report()
    gate = Path(env["WR_ROOT"]) / "results/camera-render.json"
    if kind == "missing": gate.unlink()
    else: gate.write_text("")
    result = run("--episode", "0")
    assert result.returncode != 0 and "camera/depth gate" in result.stderr and not log.exists()


@pytest.mark.parametrize("stage", STAGES)
def test_child_failure_preserves_exit_code_and_never_runs_later_stages(fake_runtime, stage):
    env, run, report, log, _, _ = fake_runtime
    report()
    env["FAKE_FAIL_STAGE"] = stage
    result = run("--episode", "0")
    assert result.returncode == 7 and "synthetic child failure" in result.stderr
    assert [line.split("|", 1)[0] for line in log.read_text().splitlines()] == list(STAGES[:STAGES.index(stage) + 1])
    assert phases(result)[-1]["stage"] == stage and phases(result)[-1]["phase"] == "fail"


def test_literal_child_closure_includes_all_required_remote_sources():
    spec = importlib.util.spec_from_file_location("world_reward_test_initializers_bundle", REPOSITORY / "infra/azure_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {str(path.relative_to(REPOSITORY)): path.read_bytes()
             for base in ("infra", "src", "configs") for path in (REPOSITORY / base).rglob("*")
             if path.is_file() and "__pycache__" not in path.parts}
    files["pyproject.toml"] = (REPOSITORY / "pyproject.toml").read_bytes()
    selected = module.runtime_bundle_paths(files, "infra/run_episode_initializers.sh")
    assert set(selected) >= {
        "infra/run_episode_initializers.sh", "infra/cari_wrapper_common.sh",
        "infra/run_body_smoke.sh", "infra/body_smoke.py",
        "infra/run_depth_smoke.sh", "infra/depth_smoke.py",
        "infra/run_scale_smoke.sh", "infra/scale_smoke.py",
        "infra/run_object_smoke.sh", "infra/object_smoke.py",
        "infra/run_cari_body_adapter_smoke.sh", "infra/cari_body_adapter_smoke.py",
        "infra/cari_body_adapter.py", "infra/camera_render.py",
    }
    assert "infra/run_object_pose_smoke.sh" not in selected
    assert "infra/run_cari_prepare.sh" not in selected
    assert "infra/run_cari_forward.sh" not in selected
    assert "infra/run_cari_converter.sh" not in selected


def test_wrapper_uses_no_generic_reuse_dag_parser_or_historical_unit():
    text = WRAPPER.read_text()
    assert 'wr_require_dependency_report "$BASE/automatic_masks/report.json" "$WAIT_FOR"' in text
    assert "wr_parse_cari_arguments " not in text
    assert "world-reward-cari-prepare" not in text
    assert "world-reward-object-pose-full" not in text
    assert "eval " not in text
    assert "&\n" not in text
