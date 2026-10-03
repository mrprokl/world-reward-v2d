"""Procedural stdlib archival tests; no challenge records or external assets."""

import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess

import pytest


@pytest.fixture
def driver():
    source = Path(__file__).resolve().parents[1] / "infra/failed_masks_transition.py"
    spec = importlib.util.spec_from_file_location("test_failed_masks_transition_driver", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, raw, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


@pytest.fixture
def case(driver, tmp_path):
    root = tmp_path.resolve()
    episode, old_revision, current_revision = 8, "a" * 40, "b" * 40
    old_code = root / "jobs" / old_revision / "run_track1_frontends/code"
    sources = {name: _write(old_code / name, ("# procedural " + name + "\n").encode(), 0o400)
               for name in driver.OLD_SOURCES}
    _write(old_code.parent / "revision", (old_revision + "\n").encode())
    _write(old_code.parent / "source-sha256", ("c" * 64 + "\n").encode())
    inputs = {
        "track_1/meta/episodes.jsonl": _write(root / "data/track_1/meta/episodes.jsonl",
                                               (json.dumps(dict(episode_index=episode, length=601)) + "\n").encode()),
        "track_1/meta/episodes_metadata.jsonl": _write(root / "data/track_1/meta/episodes_metadata.jsonl",
                (json.dumps(dict(episode_index=episode, object_prompt="procedural object")) + "\n").encode()),
        driver._video_name(episode): _write(root / "data" / driver._video_name(episode), b"not-a-real-video"),
    }
    manifest = dict(track="track_1", repo_id="nvidia/video_to_data_challenge", revision=driver.DATASET_REVISION,
                    files=[dict(path=name, **pin) for name, pin in inputs.items()])
    manifest_id = _write(root / "results/input-manifest.json", json.dumps(manifest).encode())
    diagnostic = dict(episode=episode, frames=601, observations=[], confidence=0.3, nms_iou=0.7,
                      detector_revision=driver.DETECTOR_REVISION, input_track="track_1",
                      ground_truth_used=False, hand_labeled_test=False)
    source = root / f"outputs/episode_{episode:06d}/automatic_masks"
    diagnostic_id = _write(source / "seed-diagnostics.json", json.dumps(diagnostic).encode())
    unit = f"world-reward-track1-episode{episode}-frontends.service"
    log_name = f"track1-episode{episode}-frontends.log"
    log_raw = b'Traceback: procedural failure\n' + json.dumps(dict(mode="public_frontends_only",
                stage="automatic_masks", phase="fail", timestamp_utc="2000-01-01T00:00:00Z"),
                separators=(",", ":")).encode() + b"\n"
    log_id = _write(root / "results" / log_name, log_raw)
    pins = dict(schema=driver.SCHEMA, episode_index=episode, old_producer_revision=old_revision,
                old_source_files=sources, failed_unit=unit, failed_log=dict(name=log_name, **log_id),
                diagnostic=diagnostic_id, input_manifest=manifest_id, track1_inputs=inputs)
    current_code = root / "jobs" / current_revision / "run_failed_masks_transition/code"
    current = {name: _write(current_code / name, ("# procedural current " + name + "\n").encode(), 0o400)
               for name in driver.CURRENT_SOURCES}
    _write(current_code.parent / "revision", (current_revision + "\n").encode())
    _write(current_code.parent / "source-sha256", ("d" * 64 + "\n").encode())
    _write(current_code / driver.CONFIG, json.dumps(pins).encode(), 0o400)
    return dict(root=root, pins=pins, revision=current_revision, current=current, source=source,
                old_code=old_code, current_code=current_code, diagnostic=diagnostic, manifest=manifest, log_raw=log_raw)


def _validate(driver, case, probe=None):
    return driver.validate(case["root"], case["pins"], case["revision"],
                           current_source_files=case["current"],
                           probe=probe or (lambda _unit: dict(driver.FAILED_UNIT_STATE)))


def _repin_config(driver, case):
    (case["current_code"] / driver.CONFIG).chmod(0o600)
    _write(case["current_code"] / driver.CONFIG, json.dumps(case["pins"]).encode(), 0o400)


def _rename_for_procedural_test(source, destination):
    # Explicit injected test double, NOT a production no-overwrite fallback.
    assert not destination.exists() and not destination.is_symlink()
    source.rename(destination)


def test_validation_has_no_mutation_and_transition_keeps_same_diagnostic_inode_bytes_and_log(driver, case):
    source = case["source"]
    before = (source / "seed-diagnostics.json").stat()
    plan = _validate(driver, case)
    assert source.is_dir() and not plan["destination"].exists() and not plan["receipt"].exists()
    assert not plan["destination"].parent.exists()
    log = case["root"] / "results" / case["pins"]["failed_log"]["name"]
    log_state = driver._state(log)
    record = driver.transition(plan, probe=lambda _unit: dict(driver.FAILED_UNIT_STATE),
                               rename=_rename_for_procedural_test)
    archived = plan["destination"] / "seed-diagnostics.json"
    assert not source.exists()
    assert archived.stat().st_ino == before.st_ino
    assert driver.identity(archived) == case["pins"]["diagnostic"]
    assert stat.S_IMODE(archived.stat().st_mode) == 0o400
    assert stat.S_IMODE(plan["destination"].stat().st_mode) == 0o500
    assert stat.S_IMODE(plan["receipt"].stat().st_mode) == 0o400
    assert driver._state(log) == log_state
    assert log.read_bytes() == case["log_raw"]
    assert record["before"]["sha256"] == record["after"]["sha256"]
    assert record["original_status"] == "fail" and record["original_failure_reinterpreted"] is False
    assert record["original_producer_revision"] == case["pins"]["old_producer_revision"]
    assert record["producer_revision"] == case["revision"]
    assert record["source_files_rechecked_before_thenafter"] is True
    assert record["source_rechecked_before_and_after"] is False  # no actual main callback in a procedural test
    assert record["transition_config_identity"] == driver.identity(case["current_code"] / driver.CONFIG)
    assert record["files_deleted"] is False and record["media_copied"] is False
    assert json.loads(plan["receipt"].read_bytes()) == record
    with pytest.raises((ValueError, FileNotFoundError)):
        _validate(driver, case)


@pytest.mark.parametrize("episode", [-1, 30, True, 8.0, "8", None])
def test_invalid_episode_rejected_before_any_paths_or_probe(driver, case, episode):
    case["pins"]["episode_index"] = episode
    with pytest.raises(ValueError):
        _validate(driver, case, lambda _unit: pytest.fail("invalid config must precede live query"))


@pytest.mark.parametrize("name", ["prompts.json", "report.json", "masks", "foreign.txt", "eval_private"])
def test_any_other_failed_folder_entry_is_blocking(driver, case, name):
    _write(case["source"] / name, b"foreign")
    with pytest.raises(ValueError, match="Only seed-diagnostics"):
        _validate(driver, case)


@pytest.mark.parametrize("name", ["body_smoke", "depth_smoke", "scale_smoke", "object_grounded", "body_full",
                                   "depth_full", "object_pose_full", "cari_inputs", "cari_forward", "foreign"])
def test_any_downstream_or_foreign_episode_output_is_blocking(driver, case, name):
    (case["source"].parent / name).mkdir()
    with pytest.raises(ValueError, match="no downstream or foreign"):
        _validate(driver, case)


@pytest.mark.parametrize("which", ["root", "episode", "source", "diagnostic", "old_source", "log", "video", "metadata"])
def test_symlinks_in_any_prerequisite_or_ancestor_are_rejected(driver, case, tmp_path, which):
    paths = dict(root=case["root"], episode=case["source"].parent, source=case["source"],
                 diagnostic=case["source"] / "seed-diagnostics.json",
                 old_source=case["old_code"] / driver.OLD_SOURCES[0],
                 log=case["root"] / "results" / case["pins"]["failed_log"]["name"],
                 video=case["root"] / "data" / driver._video_name(8),
                 metadata=case["root"] / "data/track_1/meta/episodes.jsonl")
    path = paths[which]
    target = case["root"].parent / (case["root"].name + "-" + which + "-actual")
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())
    with pytest.raises(ValueError, match="nonsymlink"):
        _validate(driver, case)


@pytest.mark.parametrize("mutation", ["private_input", "traversal_source", "private_source", "extra_key", "bad_log", "wrong_unit"])
def test_config_never_accepts_arbitrary_paths_private_inputs_or_unbound_unit(driver, case, mutation):
    pins = case["pins"]
    if mutation == "private_input": pins["track1_inputs"]["track_2/secret.mp4"] = dict(bytes=1, sha256="e" * 64)
    elif mutation == "traversal_source": pins["old_source_files"]["../secret"] = dict(bytes=1, sha256="e" * 64)
    elif mutation == "private_source": pins["old_source_files"]["validation/eval_private.py"] = dict(bytes=1, sha256="e" * 64)
    elif mutation == "extra_key": pins["resume"] = True
    elif mutation == "bad_log": pins["failed_log"]["name"] = "../foreign.log"
    elif mutation == "wrong_unit": pins["failed_unit"] = "world-reward-track1-episode9-frontends.service"
    with pytest.raises(ValueError):
        _validate(driver, case)


@pytest.mark.parametrize("which", ["diagnostic", "log", "input_manifest", "video", "metadata", "old_source"])
def test_all_pinned_prerequisite_byte_mismatches_fail(driver, case, which):
    paths = dict(diagnostic=case["source"] / "seed-diagnostics.json",
                 log=case["root"] / "results" / case["pins"]["failed_log"]["name"],
                 input_manifest=case["root"] / "results/input-manifest.json",
                 video=case["root"] / "data" / driver._video_name(8),
                 metadata=case["root"] / "data/track_1/meta/episodes.jsonl",
                 old_source=case["old_code"] / driver.OLD_SOURCES[0])
    path = paths[which]
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="pin mismatch"):
        _validate(driver, case)


@pytest.mark.parametrize("mutation", ["track2", "gt_true", "gt_integer", "wrong_episode", "extra_oracle", "frames", "detector"])
def test_diagnostic_provenance_is_checked_even_after_repinning(driver, case, mutation):
    diagnostic = case["diagnostic"]
    updates = dict(track2=("input_track", "track_2"), gt_true=("ground_truth_used", True),
                   gt_integer=("ground_truth_used", 0), wrong_episode=("episode", 9),
                   extra_oracle=("oracle_mode", True), frames=("frames", 600), detector=("detector_revision", "main"))
    key, value = updates[mutation]
    diagnostic[key] = value
    case["pins"]["diagnostic"] = _write(case["source"] / "seed-diagnostics.json", json.dumps(diagnostic).encode())
    _repin_config(driver, case)
    with pytest.raises(ValueError, match="diagnostic Track1/no-oracle"):
        _validate(driver, case)


@pytest.mark.parametrize("key,value", [("ActiveState", "active"), ("ActiveState", "inactive"), ("LoadState", "not-found"),
                                     ("MainPID", "42"), ("ExecMainStatus", "0"), ("ExecMainCode", "2"), ("Result", "success")])
def test_nonfailed_running_reset_or_successful_unit_never_archives(driver, case, key, value):
    projection = dict(driver.FAILED_UNIT_STATE, **{key: value})
    with pytest.raises(ValueError, match="failed exit1"):
        _validate(driver, case, lambda _unit: projection)
    assert case["source"].exists()


@pytest.mark.parametrize("which", ["destination", "receipt", "broken_archive_link"])
def test_occupied_destinations_or_receipts_never_overwrite(driver, case, which):
    plan = _validate(driver, case)
    if which == "destination": plan["destination"].mkdir(parents=True)
    elif which == "receipt": _write(plan["receipt"], b"original")
    else:
        plan["destination"].parent.mkdir(parents=True)
        plan["destination"].symlink_to(plan["destination"].parent / "missing")
    with pytest.raises(ValueError):
        _validate(driver, case)
    assert case["source"].exists()


def test_transition_rechecks_live_unit_immediately_before_rename(driver, case):
    plan = _validate(driver, case)
    count = 0
    def probe(_unit):
        nonlocal count
        count += 1
        return dict(driver.FAILED_UNIT_STATE, MainPID="45") if count == 2 else dict(driver.FAILED_UNIT_STATE)
    with pytest.raises(ValueError, match="failed exit1"):
        driver.transition(plan, probe=probe, rename=lambda *_args: pytest.fail("must not rename"))
    assert case["source"].exists() and not plan["receipt"].exists()


def test_transition_revalidates_pins_not_stale_mutable_plan_paths(driver, case):
    plan = _validate(driver, case)
    (case["source"] / "seed-diagnostics.json").write_bytes(b"changed-after-validation")
    with pytest.raises(ValueError, match="pin mismatch"):
        driver.transition(plan, probe=lambda _unit: dict(driver.FAILED_UNIT_STATE),
                          rename=lambda *_args: pytest.fail("must not rename"))
    assert case["source"].exists() and not plan["receipt"].exists()


def test_atomic_rename_error_preserves_original_folder_and_writes_no_receipt(driver, case):
    plan = _validate(driver, case)
    def blocked(_source, _destination):
        raise FileExistsError("injected no-replace failure")
    with pytest.raises(FileExistsError):
        driver.transition(plan, probe=lambda _unit: dict(driver.FAILED_UNIT_STATE), rename=blocked)
    assert case["source"].exists() and not plan["receipt"].exists()


def test_unit_projection_only_exact_scalar_fields_and_bounded_command(driver, monkeypatch):
    captured = []
    def run(argv, **kwargs):
        captured.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "\n".join(k + "=" + v for k, v in driver.FAILED_UNIT_STATE.items()) + "\n")
    monkeypatch.setattr(driver.subprocess, "run", run)
    assert driver.unit_projection("world-reward-track1-episode8-frontends.service") == driver.FAILED_UNIT_STATE
    argv, kwargs = captured[0]
    assert argv[:3] == ["systemctl", "show", "--no-pager"]
    assert {v.removeprefix("--property=") for v in argv[3:-1]} == set(driver.FAILED_UNIT_STATE)
    assert kwargs["timeout"] == 4 and kwargs["stderr"] == subprocess.DEVNULL
    assert "shell" not in kwargs


@pytest.mark.parametrize("text", ["", "MainPID=0\nMainPID=0\n", "Environment=SECRET\n", "x" * 2049])
def test_unexpected_duplicate_missing_or_oversized_projection_fails_without_echo(driver, monkeypatch, text):
    monkeypatch.setattr(driver.subprocess, "run", lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, text))
    with pytest.raises(ValueError) as error:
        driver.unit_projection("world-reward-track1-episode8-frontends.service")
    assert "SECRET" not in str(error.value)


def test_unit_timeout_fails_without_raw_subprocess_output(driver, monkeypatch):
    def run(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, 4, output="SECRET")
    monkeypatch.setattr(driver.subprocess, "run", run)
    with pytest.raises(ValueError, match="state query failed") as error:
        driver.unit_projection("world-reward-track1-episode8-frontends.service")
    assert "SECRET" not in str(error.value)


def test_production_rename_requires_linux_and_no_plain_rename_fallback(driver, monkeypatch):
    monkeypatch.setattr(driver.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="Linux renameat2"):
        driver.rename_noreplace(Path("/source"), Path("/destination"))


def test_production_renameat2_requests_noreplace_and_propagates_occupied_destination(driver, monkeypatch):
    class Rename:
        def __call__(self, *args):
            assert args == (-100, b"/source", -100, b"/destination", 1)
            driver.ctypes.set_errno(errno.EEXIST)
            return -1
    class Library:
        renameat2 = Rename()
    monkeypatch.setattr(driver.platform, "system", lambda: "Linux")
    monkeypatch.setattr(driver.ctypes, "CDLL", lambda *_args, **_kwargs: Library())
    with pytest.raises(FileExistsError):
        driver.rename_noreplace(Path("/source"), Path("/destination"))


def test_duplicate_and_nonfinite_json_are_rejected(driver):
    with pytest.raises(ValueError, match="Duplicate"):
        driver._json(b'{"episode":8,"episode":8}')
    with pytest.raises(ValueError, match="Nonfinite"):
        driver._json(b'{"x":NaN}')


@pytest.mark.parametrize("which", ["diagnostic", "log", "video"])
def test_hardlinked_prerequisite_is_not_mutated_or_claimed_unaliased(driver, case, which):
    path = {"diagnostic": case["source"] / "seed-diagnostics.json",
            "log": case["root"] / "results" / case["pins"]["failed_log"]["name"],
            "video": case["root"] / "data" / driver._video_name(8)}[which]
    alias = case["root"].parent / (case["root"].name + "-hardlink")
    alias.hardlink_to(path)
    with pytest.raises(ValueError, match="without hardlinks"):
        _validate(driver, case)


def test_wrong_actual_current_source_pin_is_rejected(driver, case):
    name = driver.CURRENT_SOURCES[0]
    case["current"][name] = dict(bytes=1, sha256="e" * 64)
    with pytest.raises(ValueError, match="pin mismatch"):
        _validate(driver, case)


def test_current_config_cannot_be_a_changed_writable_or_mismatched_contract(driver, case):
    path = case["current_code"] / driver.CONFIG
    path.chmod(0o600)
    with pytest.raises(ValueError, match="readonly"):
        _validate(driver, case)
    path.write_text(json.dumps(dict(case["pins"], diagnostic=dict(bytes=1, sha256="e" * 64))))
    path.chmod(0o400)
    with pytest.raises(ValueError, match="differs from requested pins"):
        _validate(driver, case)


@pytest.mark.parametrize("which", ["revision", "source-sha256"])
def test_original_dispatch_markers_must_match_exact_revision_and_hash_syntax(driver, case, which):
    path = case["old_code"].parent / which
    path.write_bytes(b"foreign\n")
    with pytest.raises(ValueError, match="dispatch markers"):
        _validate(driver, case)


def test_original_launcher_pass_is_rejected_even_with_new_log_pin(driver, case):
    log = case["root"] / "results" / case["pins"]["failed_log"]["name"]
    raw = json.dumps(dict(mode="public_frontends_only", stage="automatic_masks", phase="pass"),
                     separators=(",", ":")).encode() + b"\n"
    case["pins"]["failed_log"].update(_write(log, raw))
    _repin_config(driver, case)
    with pytest.raises(ValueError, match="failure, never pass"):
        _validate(driver, case)


def test_manifest_public_provenance_is_checked_before_any_media_open(driver, case, monkeypatch):
    case["manifest"]["track"] = "track_2"
    case["pins"]["input_manifest"] = _write(case["root"] / "results/input-manifest.json",
                                            json.dumps(case["manifest"]).encode())
    _repin_config(driver, case)
    original = driver.identity
    def identity(path, *args, **kwargs):
        assert "data" not in Path(path).relative_to(case["root"]).parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(driver, "identity", identity)
    with pytest.raises(ValueError, match="official Track1 manifest"):
        _validate(driver, case)


def test_atomic_no_overwrite_catches_destination_created_at_rename_boundary(driver, case):
    plan = _validate(driver, case)
    def contested(_source, destination):
        destination.mkdir()
        raise FileExistsError("procedural renameat2 EEXIST")
    with pytest.raises(FileExistsError):
        driver.transition(plan, probe=lambda _unit: dict(driver.FAILED_UNIT_STATE), rename=contested)
    assert case["source"].exists() and not plan["receipt"].exists()
    assert list(plan["destination"].iterdir()) == []


def test_partial_after_rename_failure_is_closed_without_receipt_or_implicit_retry(driver, case, monkeypatch):
    plan = _validate(driver, case)
    def crash(_path):
        raise OSError("procedural post-rename sync failure")
    monkeypatch.setattr(driver, "_sync_directory", crash)
    with pytest.raises(OSError, match="post-rename"):
        driver.transition(plan, probe=lambda _unit: dict(driver.FAILED_UNIT_STATE), rename=_rename_for_procedural_test)
    assert not case["source"].exists() and plan["destination"].is_dir() and not plan["receipt"].exists()
    assert driver.identity(plan["destination"] / "seed-diagnostics.json") == case["pins"]["diagnostic"]
    with pytest.raises(ValueError):
        _validate(driver, case)


def test_main_source_binding_requires_real_derived_checkout_no_env_path_override(driver, case, monkeypatch):
    monkeypatch.setattr(driver, "ROOT", case["root"])
    helpers, config_pin, pins = driver.bound_source(case["root"], case["current_code"], case["revision"],
                                                   case["current_code"] / driver.CURRENT_SOURCES[0])
    assert helpers == case["current"] and pins == case["pins"]
    assert config_pin == driver.identity(case["current_code"] / driver.CONFIG)
    with pytest.raises(ValueError, match="actual|Actual"):
        driver.bound_source(case["root"], case["old_code"], case["revision"],
                            case["old_code"] / driver.CURRENT_SOURCES[0])


@pytest.mark.parametrize("which", ["current_source", "config", "current_marker", "old_source"])
def test_postarchive_source_config_or_marker_mutation_cannot_write_pass_receipt(driver, case, which):
    plan = _validate(driver, case)
    paths = dict(current_source=case["current_code"] / driver.CURRENT_SOURCES[0],
                 config=case["current_code"] / driver.CONFIG,
                 current_marker=case["current_code"].parent / "source-sha256",
                 old_source=case["old_code"] / driver.OLD_SOURCES[0])
    def rename(source, destination):
        _rename_for_procedural_test(source, destination)
        path = paths[which]
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b"altered")
        path.chmod(0o400)
    with pytest.raises(ValueError):
        driver.transition(plan, probe=lambda _unit: dict(driver.FAILED_UNIT_STATE), rename=rename)
    assert not case["source"].exists() and plan["destination"].exists() and not plan["receipt"].exists()


def test_actual_main_source_callback_must_pass_before_any_rename(driver, case):
    plan = _validate(driver, case)
    with pytest.raises(ValueError, match="before archive"):
        driver.transition(plan, probe=lambda _unit: dict(driver.FAILED_UNIT_STATE),
                          rename=lambda *_args: pytest.fail("callback must abort before rename"),
                          verify_sources=lambda: False)
    assert case["source"].exists() and not plan["receipt"].exists()


def test_actual_main_source_callback_failure_after_archive_writes_no_pass_receipt(driver, case):
    plan = _validate(driver, case)
    calls = []
    def verify():
        calls.append(True)
        return len(calls) == 1
    with pytest.raises(ValueError, match="after archive before PASS"):
        driver.transition(plan, probe=lambda _unit: dict(driver.FAILED_UNIT_STATE),
                          rename=_rename_for_procedural_test, verify_sources=verify)
    assert len(calls) == 2 and not case["source"].exists() and plan["destination"].exists()
    assert not plan["receipt"].exists()


def test_receipt_claims_actual_callback_checks_only_when_both_succeeded(driver, case):
    plan = _validate(driver, case)
    calls = []
    def verify():
        calls.append(True)
        return True
    record = driver.transition(plan, probe=lambda _unit: dict(driver.FAILED_UNIT_STATE),
                               rename=_rename_for_procedural_test, verify_sources=verify)
    assert calls == [True, True]
    assert record["source_rechecked_before_and_after"] is True
