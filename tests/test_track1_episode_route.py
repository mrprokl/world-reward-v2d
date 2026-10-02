"""Clean serial route against tiny shell children; no GPU or model loads."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "infra/run_track1_episode.sh"
STAGES = ("automatic_masks", "episode_initializers", "object_pose_full", "cari_inputs", "cari_forward", "cari_conversion")
TARGETS = ("automatic_masks", "body_smoke", "depth_smoke", "scale_smoke", "object_grounded",
           "body_full", "depth_full", "object_pose_full", "body_full/cari_adapter",
           "cari_inputs", "cari_forward", "cari_conversion")


@pytest.fixture
def runtime(tmp_path):
    root, code = tmp_path / "runtime", tmp_path / "code"
    (code / "infra").mkdir(parents=True)
    (code / "infra/cari_wrapper_common.sh").write_text((ROOT / "infra/cari_wrapper_common.sh").read_text())
    # Actual seven-stage initializer and preflight, not a success-only stub.
    (code / "infra/run_episode_initializers.sh").write_text((ROOT / "infra/run_episode_initializers.sh").read_text())
    log = tmp_path / "calls.log"
    child = '''#!/usr/bin/env bash
set -euo pipefail
name="$(basename "$0")"
[[ "$1" == --episode ]] || exit 9
episode="$2"
printf -v padded '%06d' "$episode"
case "$name" in
 run_automatic_masks.sh) stage=automatic_masks ;;
 run_body_smoke.sh) if [[ " $* " == *" --full-video "* ]]; then stage=body_full; else stage=body_smoke; fi ;;
 run_depth_smoke.sh) if [[ " $* " == *" --full-video "* ]]; then stage=depth_full; else stage=depth_smoke; fi ;;
 run_scale_smoke.sh) stage=scale_smoke ;;
 run_object_smoke.sh) stage=object_grounded ;;
 run_cari_body_adapter_smoke.sh) stage=cari_adapter ;;
 run_object_pose_smoke.sh) stage=object_pose_full ;;
 run_cari_prepare.sh) stage=cari_inputs ;;
 run_cari_forward.sh) stage=cari_forward ;;
 run_cari_converter.sh) stage=cari_conversion ;;
 *) exit 9 ;;
esac
printf '%s|%s\\n' "$stage" "$*" >> "$FAKE_LOG"
if [[ "$stage" == "${FAKE_FAIL_STAGE:-}" ]]; then echo 'synthetic child failure' >&2; exit 7; fi
if [[ "$stage" == cari_inputs || "$stage" == cari_forward || "$stage" == cari_conversion ]]; then
 [[ "$*" == "--episode $episode --no-wait" ]] || exit 9
 source "$WR_CODE/infra/cari_wrapper_common.sh"
 ROOT="$WR_ROOT"
 case "$stage" in cari_inputs) mode=prepare ;; cari_forward) mode=forward ;; cari_conversion) mode=converter ;; esac
 wr_parse_cari_arguments "$mode" "$@"
 [[ -z "$WR_WAIT_FOR" ]] || exit 9
 wr_cari_dependency "$mode"
fi
target="$stage"
[[ "$stage" != cari_adapter ]] || target=body_full/cari_adapter
mkdir -p "$WR_ROOT/outputs/episode_$padded/$target"
printf '{}' > "$WR_ROOT/outputs/episode_$padded/$target/report.json"
'''
    for name in ("run_automatic_masks.sh", "run_body_smoke.sh", "run_depth_smoke.sh", "run_scale_smoke.sh",
                 "run_object_smoke.sh", "run_cari_body_adapter_smoke.sh", "run_object_pose_smoke.sh",
                 "run_cari_prepare.sh", "run_cari_forward.sh", "run_cari_converter.sh"):
        (code / "infra" / name).write_text(child)
    gate = root / "results/camera-render.json"
    gate.parent.mkdir(parents=True); gate.write_text('{"status":"pass"}')
    env = dict(os.environ, WR_ROOT=str(root), WR_CODE=str(code), WR_CODE_REVISION="a" * 40, FAKE_LOG=str(log))
    def run(*args):
        return subprocess.run(["bash", str(WRAPPER), *args], env=env,
                              capture_output=True, text=True, timeout=5)
    return env, run, log


def records(result):
    return [json.loads(row) for row in result.stdout.splitlines()]


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_exact_full_serial_route_including_real_initializer_preflight(runtime, episode):
    env, run, log = runtime
    result = run("--episode", str(episode))
    assert result.returncode == 0, result.stderr
    expected = [
        f"automatic_masks|--episode {episode}", f"body_smoke|--episode {episode} --inference-type body",
        f"depth_smoke|--episode {episode}", f"scale_smoke|--episode {episode}",
        f"object_grounded|--episode {episode} --aligned-pointmap",
        f"body_full|--episode {episode} --full-video --inference-type body",
        f"depth_full|--episode {episode} --full-video", f"cari_adapter|--episode {episode}",
        f"object_pose_full|--episode {episode} --full-video",
        f"cari_inputs|--episode {episode} --no-wait", f"cari_forward|--episode {episode} --no-wait",
        f"cari_conversion|--episode {episode} --no-wait",
    ]
    assert log.read_text().splitlines() == expected
    assert all((Path(env["WR_ROOT"]) / f"outputs/episode_{episode:06d}/{target}/report.json").is_file() for target in TARGETS)
    rows = records(result)
    assert all(set(row) == {"stage", "phase", "timestamp_utc"} and row["timestamp_utc"].endswith("Z") for row in rows)
    route_rows = [row for row in rows if row["stage"] in STAGES]
    assert [(row["stage"], row["phase"]) for row in route_rows] == [(stage, phase) for stage in STAGES for phase in ("start", "pass")]


@pytest.mark.parametrize("args", [(), ("--episode",), ("--episode", "-1"), ("--episode", "30"),
                                   ("--episode", "true"), ("--episode", "15.0"), ("--episode", "00"),
                                   ("--episode", "+1"), ("--episode", "0", "--episode", "1"),
                                   ("--episode=15",), ("--episode", "0", "--wait-for", "world-reward-x"),
                                   ("--episode", "0", "--no-wait"), ("--episode", "0", "--full-video"),
                                   ("--episode", "0", "--resume"), ("--episode", "0", "--overwrite"),
                                   ("--episode", "../track_2"), ("--episode", "1;rm -rf /"),
                                   ("--root", "/tmp"), ("--episode", "0", "--kernel-only")])
def test_invalid_controls_fail_before_any_child_or_filesystem_mutation(runtime, args):
    env, run, log = runtime
    result = run(*args)
    assert result.returncode == 2 and not log.exists()
    assert not (Path(env["WR_ROOT"]) / "outputs").exists()


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("kind", ["directory", "report", "file", "broken_symlink"])
def test_all_twelve_outputs_preflight_before_first_gpu_child(runtime, target, kind):
    env, run, log = runtime
    frozen = Path(env["WR_ROOT"]) / f"outputs/episode_000015/{target}"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if kind in ("directory", "report"):
        frozen.mkdir()
        if kind == "report": (frozen / "report.json").write_text("frozen")
    elif kind == "file": frozen.write_text("frozen")
    else: frozen.symlink_to("missing-target")
    result = run("--episode", "15")
    assert result.returncode != 0 and "Frozen episode target" in result.stderr and not log.exists()
    assert [(row["stage"], row["phase"]) for row in records(result)] == [("preflight", "start"), ("preflight", "fail")]
    assert frozen.is_symlink() if kind == "broken_symlink" else frozen.exists()


@pytest.mark.parametrize("stage", ["automatic_masks", "body_smoke", "depth_smoke", "scale_smoke", "object_grounded",
                                   "body_full", "depth_full", "cari_adapter", "object_pose_full", "cari_inputs",
                                   "cari_forward", "cari_conversion"])
def test_first_child_failure_stops_route_preserves_exit_and_partial_artifacts(runtime, stage):
    env, run, log = runtime; env["FAKE_FAIL_STAGE"] = stage
    result = run("--episode", "0")
    assert result.returncode == 7 and "synthetic child failure" in result.stderr
    calls = [row.split("|")[0] for row in log.read_text().splitlines()]
    assert calls[-1] == stage and len(calls) == len(set(calls))
    assert records(result)[-1]["phase"] == "fail"
    assert not (Path(env["WR_ROOT"]) / f"outputs/episode_000000/{stage}/report.json").exists()


@pytest.mark.parametrize("variable,value", [("WR_ROOT", "relative"), ("WR_ROOT", "/"), ("WR_ROOT", "/tmp/../other"),
                                          ("WR_ROOT", "/tmp/./other"), ("WR_CODE", "relative"),
                                          ("WR_CODE_REVISION", "main"), ("WR_CODE_REVISION", "A" * 40),
                                          ("WR_CODE_REVISION", "a" * 39), ("WR_ROOT", "/tmp\nother")])
def test_runtime_source_revision_contract_invalid_before_children(runtime, variable, value):
    env, run, log = runtime; env[variable] = value
    assert run("--episode", "0").returncode == 2 and not log.exists()


@pytest.mark.parametrize("variable", ["WR_ROOT", "WR_CODE", "WR_CODE_REVISION"])
def test_missing_runtime_environment_not_defaulted(runtime, variable):
    env, run, log = runtime; env.pop(variable)
    assert run("--episode", "0").returncode != 0 and not log.exists()


def test_missing_global_camera_gate_and_missing_wrapper_fail_preflight(runtime):
    env, run, log = runtime
    gate = Path(env["WR_ROOT"]) / "results/camera-render.json"; gate.unlink()
    result = run("--episode", "0")
    assert result.returncode != 0 and "camera/depth gate" in result.stderr and not log.exists()
    gate.write_text("{}")
    (Path(env["WR_CODE"]) / "infra/run_cari_converter.sh").unlink()
    result = run("--episode", "0")
    assert result.returncode != 0 and "child wrapper" in result.stderr and not log.exists()


@pytest.mark.parametrize("target", ["outputs", "outputs/episode_000000"])
def test_output_root_symlink_rejected_before_children(runtime, target, tmp_path):
    env, run, log = runtime
    path = Path(env["WR_ROOT"]) / target; path.parent.mkdir(exist_ok=True)
    external = tmp_path / "external"; external.mkdir()
    path.symlink_to(external, target_is_directory=True)
    result = run("--episode", "0")
    assert result.returncode != 0 and "cannot be symlinks" in result.stderr and not log.exists()
    assert not list(external.iterdir())


def test_route_targets_match_single_existing_planner_without_other_producing_stages():
    from world_reward.pipeline import plan_episode
    planned = {stage.output_directory.removeprefix("outputs/episode_000000/") for stage in plan_episode(0)}
    assert planned == set(TARGETS)
    text = WRAPPER.read_text()
    assert "world-reward-cari-prepare-v2" not in text and "--wait-for" not in text
    assert "run_body_converter.sh" not in text and "--kernel-only" not in text
    assert "docker " not in text and "systemctl " not in text and "eval " not in text and "&\n" not in text
    assert 'run_cari_prepare.sh" --episode "$EPISODE" --no-wait' in text


def test_literal_runtime_closure_includes_all_actual_child_wrappers():
    spec = importlib.util.spec_from_file_location("wr_test_route_bundle", ROOT / "infra/azure_job.py")
    launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
    files = {str(path.relative_to(ROOT)): path.read_bytes() for base in ("infra", "src", "configs")
             for path in (ROOT / base).rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    files["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    selected = launcher.runtime_bundle_paths(files, "infra/run_track1_episode.sh")
    assert set(selected) >= {"infra/run_track1_episode.sh", "infra/run_automatic_masks.sh", "infra/automatic_masks.py",
                             "infra/run_episode_initializers.sh", "infra/run_object_pose_smoke.sh", "infra/object_pose_smoke.py",
                             "infra/run_cari_prepare.sh", "infra/cari_prepare.py", "infra/run_cari_forward.sh", "infra/cari_forward.py",
                             "infra/run_cari_converter.sh", "infra/cari_converter.py", "infra/cari_wrapper_common.sh"}
    assert "infra/run_body_converter.sh" not in selected and "infra/run_finger_transfer_smoke.sh" not in selected
