"""Archive one pinned, inactive automatic-mask failure; never resume or erase it.

Azure host stdlib only. No models, GPU calls, image decoding, network, shell
evaluation, or arbitrary input/output paths. A crash after the atomic rename is
fail-closed: the occupied archive prevents an implicit retry or rollback.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import time


ROOT = Path("/srv/scenesmith/world-reward")
CONFIG = "configs/failed_masks_transition_pins.json"
SCHEMA = "world_reward.failed_masks_transition_pins.v1"
DATASET_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
DETECTOR_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
OLD_SOURCES = (
    "infra/run_track1_frontends.sh", "infra/run_automatic_masks.sh", "infra/automatic_masks.py",
    "src/world_reward/__init__.py", "src/world_reward/actor_selection.py",
    "src/world_reward/prompt_selection.py", "src/world_reward/data.py", "infra/body_smoke.py",
)
CURRENT_SOURCES = ("infra/failed_masks_transition.py", "infra/run_failed_masks_transition.sh")
FAILED_UNIT_STATE = dict(LoadState="loaded", ActiveState="failed", SubState="failed",
                         Result="exit-code", ExecMainCode="1", ExecMainStatus="1", MainPID="0")
CONFIG_FIELDS = {"schema", "episode_index", "old_producer_revision", "old_source_files",
                 "failed_unit", "failed_log", "diagnostic", "input_manifest", "track1_inputs"}


def canonical(path):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path or
            any(parent.is_symlink() for parent in (path, *path.parents))):
        raise ValueError("Canonical absolute nonsymlink path required")
    return path


def _state(path):
    s = path.lstat()
    return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink


def identity(path, maximum=32 * 1024 ** 3):
    path = canonical(path)
    before = _state(path)
    if not stat.S_ISREG(before[2]) or before[6] != 1 or not 0 < before[3] <= maximum:
        raise ValueError("Bounded nonempty regular file without hardlinks required: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 ** 2), b""):
            digest.update(chunk)
    if _state(path) != before:
        raise ValueError("Pinned file changed while hashing: " + str(path))
    return dict(bytes=before[3], sha256=digest.hexdigest())


def _pin(value):
    if (type(value) is not dict or set(value) != {"bytes", "sha256"} or
            type(value["bytes"]) is not int or value["bytes"] <= 0 or
            type(value["sha256"]) is not str or re.fullmatch("[0-9a-f]{64}", value["sha256"]) is None):
        raise ValueError("Exact positive-byte SHA256 pin required")
    return value


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key prohibited")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("Nonfinite JSON prohibited")))


def _verified(path, pin, *, maximum=32 * 1024 ** 3, readonly=False):
    wanted = _pin(pin)
    if identity(path, maximum) != wanted:
        raise ValueError("Frozen file pin mismatch: " + str(path))
    if readonly and Path(path).stat().st_mode & 0o222:
        raise ValueError("Frozen source must be readonly: " + str(path))
    return dict(wanted)


def _verified_json(path, pin, *, maximum=2 * 1024 ** 2):
    wanted = _verified(path, pin, maximum=maximum)
    raw = Path(path).read_bytes()
    if dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) != wanted:
        raise ValueError("Pinned JSON changed before interpretation")
    return _json(raw)


def _video_name(episode):
    return f"track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4"


def _unit_name(unit, episode):
    pattern = rf"world-reward-track1-episode{episode}-[a-z][a-z0-9-]{{0,40}}\.service"
    if type(unit) is not str or re.fullmatch(pattern, unit) is None:
        raise ValueError("Episode-bound World Reward frontend unit required")
    return unit.removeprefix("world-reward-").removesuffix(".service") + ".log"


def _check_projection(value):
    if type(value) is not dict or value != FAILED_UNIT_STATE or any(type(v) is not str for v in value.values()):
        raise ValueError("Original unit must be loaded, failed exit1, and have MainPID0")
    return dict(value)


def unit_projection(unit):
    """One bounded, scalar systemd projection; never print raw process output."""
    try:
        result = subprocess.run(
            ["systemctl", "show", "--no-pager", *("--property=" + key for key in FAILED_UNIT_STATE), unit],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=4, check=False, text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("Bounded original-unit state query failed") from None
    if result.returncode or len(result.stdout) > 2048:
        raise ValueError("Original-unit state projection unavailable or oversized")
    values = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if not sep or key not in FAILED_UNIT_STATE or key in values:
            raise ValueError("Unexpected original-unit projection field")
        values[key] = value
    return _check_projection(values)


def _absent(path):
    canonical(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Frozen transition target already occupied: " + str(path))


def _directory(path):
    path = canonical(path)
    if not path.is_dir() or not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError("Canonical directory required: " + str(path))
    return path


def _markers(code, revision):
    code = _directory(code)
    for name in ("revision", "source-sha256"):
        path = canonical(code.parent / name)
        before = identity(path, 128)
        raw = path.read_bytes()
        if (name == "revision" and raw != (revision + "\n").encode() or
                name == "source-sha256" and re.fullmatch(b"[0-9a-f]{64}\n", raw) is None):
            raise ValueError("Exact original dispatch markers required")
        if identity(path, 128) != before:
            raise ValueError("Dispatch marker changed during interpretation")


def _config(pins):
    if type(pins) is not dict or set(pins) != CONFIG_FIELDS or pins["schema"] != SCHEMA:
        raise ValueError("Exact failed-mask transition schema required")
    episode, revision = pins["episode_index"], pins["old_producer_revision"]
    if type(episode) is not int or not 0 <= episode < 30:
        raise ValueError("Integer Track1 episode in 0..29 required")
    if type(revision) is not str or re.fullmatch("[0-9a-f]{40}", revision) is None:
        raise ValueError("Exact old producer revision required")
    if type(pins["old_source_files"]) is not dict or set(pins["old_source_files"]) != set(OLD_SOURCES):
        raise ValueError("Exactly eight selected old source pins required")
    for pin in pins["old_source_files"].values():
        _pin(pin)
    names = {"track_1/meta/episodes.jsonl", "track_1/meta/episodes_metadata.jsonl", _video_name(episode)}
    if type(pins["track1_inputs"]) is not dict or set(pins["track1_inputs"]) != names:
        raise ValueError("Only exact selected Track1 RGB and two public metadata pins permitted")
    for pin in pins["track1_inputs"].values():
        _pin(pin)
    for name in ("diagnostic", "input_manifest"):
        _pin(pins[name])
    log_name = _unit_name(pins["failed_unit"], episode)
    log = pins["failed_log"]
    if type(log) is not dict or set(log) != {"name", "bytes", "sha256"} or log["name"] != log_name:
        raise ValueError("Original unit-bound log basename required; no arbitrary paths")
    _pin({key: log[key] for key in ("bytes", "sha256")})


def _episode_metadata(path, episode):
    records = [_json(line) for line in Path(path).read_bytes().splitlines() if line.strip()]
    if any(type(row) is not dict or type(row.get("episode_index")) is not int or
           not 0 <= row["episode_index"] < 30 for row in records):
        raise ValueError("Public metadata needs integer Track1 episode identities")
    matches = [row for row in records if row["episode_index"] == episode]
    if len(matches) != 1:
        raise ValueError("Exactly one selected public episode record required")
    return matches[0]


def _current_binding(root, revision, source_files, pins):
    code = root / "jobs" / revision / "run_failed_masks_transition/code"
    _markers(code, revision)
    for name, pin in source_files.items():
        _verified(code / name, pin, maximum=2 * 1024 ** 2, readonly=True)
    config_path = canonical(code / CONFIG)
    if config_path.stat().st_mode & 0o222:
        raise ValueError("Frozen transition config must be readonly")
    config_id = identity(config_path, 32 * 1024)
    if _verified_json(config_path, config_id, maximum=32 * 1024) != pins:
        raise ValueError("Actual frozen transition config differs from requested pins")
    return dict(config_identity=config_id, dispatch_markers={
        name: identity(code.parent / name, 128) for name in ("revision", "source-sha256")})


def validate(root, pins, current_revision, *, current_source_files, probe=unit_projection):
    """Pure path/hash/JSON checks plus caller-injectable bounded unit projection.

    No mutation. Main additionally binds current source files to its actual
    immutable dispatch checkout; tests supply procedural provenance identities.
    """
    _config(pins)
    root = _directory(root)
    if type(current_revision) is not str or re.fullmatch("[0-9a-f]{40}", current_revision) is None:
        raise ValueError("Exact actual current producer revision required")
    if type(current_source_files) is not dict or set(current_source_files) != set(CURRENT_SOURCES):
        raise ValueError("Exact current transition driver and wrapper identities required")
    for pin in current_source_files.values():
        _pin(pin)
    current_binding = _current_binding(root, current_revision, current_source_files, pins)
    episode, old_revision = pins["episode_index"], pins["old_producer_revision"]
    base = _directory(root / f"outputs/episode_{episode:06d}")
    if {p.name for p in base.iterdir()} != {"automatic_masks"}:
        raise ValueError("Only failed automatic_masks may exist; no downstream or foreign outputs")
    source = _directory(base / "automatic_masks")
    if {p.name for p in source.iterdir()} != {"seed-diagnostics.json"}:
        raise ValueError("Only seed-diagnostics.json permitted; no prompts, masks, pass report, or foreign files")
    destination = root / f"outputs/failures/episode_{episode:06d}/automatic_masks_{old_revision}"
    receipt = root / f"results/failed_masks_transition_episode_{episode:06d}_{old_revision}.json"
    _directory(root / "results")
    _absent(destination)
    _absent(receipt)
    old_code = root / "jobs" / old_revision / "run_track1_frontends/code"
    _markers(old_code, old_revision)
    source_ids = {name: _verified(old_code / name, pins["old_source_files"][name], maximum=2 * 1024 ** 2,
                                  readonly=True) for name in OLD_SOURCES}
    manifest = _verified_json(root / "results/input-manifest.json", pins["input_manifest"])
    if (type(manifest) is not dict or
            (manifest.get("track"), manifest.get("repo_id"), manifest.get("revision")) !=
            ("track_1", "nvidia/video_to_data_challenge", DATASET_REVISION) or
            type(manifest.get("files")) is not list):
        raise ValueError("Pinned official Track1 manifest required")
    input_ids = {}
    for name, pin in pins["track1_inputs"].items():
        records = [row for row in manifest["files"] if type(row) is dict and row.get("path") == name]
        if len(records) != 1 or {key: records[0].get(key) for key in ("bytes", "sha256")} != pin:
            raise ValueError("Selected Track1 manifest record differs from frozen pin")
        input_ids[name] = _verified(root / "data" / name, pin, maximum=(32 * 1024 ** 3 if name.endswith(".mp4") else 2 * 1024 ** 2))
    metadata = _episode_metadata(root / "data/track_1/meta/episodes.jsonl", episode)
    total = metadata.get("length")
    if type(total) is not int or total < 3:
        raise ValueError("Original full frame count required")
    prompt = _episode_metadata(root / "data/track_1/meta/episodes_metadata.jsonl", episode).get("object_prompt")
    if type(prompt) is not str or not prompt.strip():
        raise ValueError("Original public object prompt required")
    diagnostic = _verified_json(source / "seed-diagnostics.json", pins["diagnostic"])
    required = dict(episode=episode, frames=total, input_track="track_1", ground_truth_used=False,
                    hand_labeled_test=False, detector_revision=DETECTOR_REVISION)
    diagnostic_fields = {"episode", "frames", "observations", "confidence", "nms_iou", "detector_revision",
                         "input_track", "ground_truth_used", "hand_labeled_test"}
    if (type(diagnostic) is not dict or set(diagnostic) != diagnostic_fields or
            type(diagnostic["observations"]) is not list or
            any(type(diagnostic.get(k)) is not type(v) or diagnostic[k] != v
                                         for k, v in required.items())):
        raise ValueError("Original diagnostic Track1/no-oracle provenance differs")
    log_pin = {key: pins["failed_log"][key] for key in ("bytes", "sha256")}
    log_path = root / "results" / pins["failed_log"]["name"]
    _verified(log_path, log_pin, maximum=2 * 1024 ** 2)
    raw = log_path.read_bytes()
    if dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) != log_pin:
        raise ValueError("Original failed log changed before interpretation")
    phases = []
    for line in raw.splitlines():
        if line.startswith(b'{"mode":"public_frontends_only"'):
            phases.append(_json(line))
    if not phases or any(type(phases[-1].get(k)) is not str or phases[-1].get(k) != v for k, v in
                         dict(mode="public_frontends_only", stage="automatic_masks", phase="fail").items()):
        raise ValueError("Original launcher must end with automatic_masks failure, never pass")
    return dict(root=root, pins=pins, current_revision=current_revision,
                current_source_files=current_source_files, source=source, destination=destination, receipt=receipt,
                old_source_files=source_ids, track1_inputs=input_ids, original_log=log_pin,
                current_binding=current_binding, unit_state=_check_projection(probe(pins["failed_unit"])))


def rename_noreplace(source, destination):
    """Linux atomic rename with no overwrite, including an empty occupied dir."""
    if platform.system() != "Linux":
        raise ValueError("Linux renameat2 no-overwrite transition required")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        call = libc.renameat2
    except AttributeError:
        raise ValueError("Atomic renameat2 unavailable; refusing a non-atomic fallback") from None
    call.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    call.restype = ctypes.c_int
    if call(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code in (errno.ENOSYS, errno.EINVAL):
            raise ValueError("Atomic no-overwrite rename unsupported; no fallback")
        raise OSError(code, "Atomic no-overwrite archive rename failed")


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def transition(plan, *, probe=unit_projection, rename=rename_noreplace, verify_sources=None):
    """Revalidate everything before mutation; archive, pin after, write 0400 receipt."""
    fresh = validate(plan["root"], plan["pins"], plan["current_revision"],
                     current_source_files=plan["current_source_files"], probe=probe)
    source_callback = verify_sources is not None
    if source_callback and verify_sources() is not True:
        raise ValueError("Caller actual source binding must verify before archive")
    destination = fresh["destination"]
    for parent in (destination.parent.parent, destination.parent):
        canonical(parent)
        if parent.exists():
            _directory(parent)
        else:
            parent.mkdir(mode=0o700)
    _absent(destination)
    _absent(fresh["receipt"])
    _check_projection(probe(fresh["pins"]["failed_unit"]))
    diagnostic = fresh["source"] / "seed-diagnostics.json"
    _verified(diagnostic, fresh["pins"]["diagnostic"], maximum=2 * 1024 ** 2)
    if {p.name for p in fresh["source"].iterdir()} != {"seed-diagnostics.json"}:
        raise ValueError("Failed folder changed before atomic rename")
    rename(fresh["source"], destination)
    _sync_directory(destination.parent)
    _sync_directory(fresh["source"].parent)
    if fresh["source"].exists() or fresh["source"].is_symlink():
        raise ValueError("Original failed namespace still occupied after rename")
    _directory(destination)
    if {p.name for p in destination.iterdir()} != {"seed-diagnostics.json"}:
        raise ValueError("Archived failure has unexpected entries")
    after = _verified(destination / "seed-diagnostics.json", fresh["pins"]["diagnostic"], maximum=2 * 1024 ** 2)
    (destination / "seed-diagnostics.json").chmod(0o400)
    destination.chmod(0o500)
    after = _verified(destination / "seed-diagnostics.json", after, maximum=2 * 1024 ** 2)
    log = fresh["root"] / "results" / fresh["pins"]["failed_log"]["name"]
    _verified(log, fresh["original_log"], maximum=2 * 1024 ** 2)
    # A PASS receipt is not written until actual helper/config/marker identities
    # survive the archive operation. Never manufacture postcheck proof from pins.
    if _current_binding(fresh["root"], fresh["current_revision"], fresh["current_source_files"],
                        fresh["pins"]) != fresh["current_binding"]:
        raise ValueError("Actual current source/config/dispatch changed after archive")
    old_code = fresh["root"] / "jobs" / fresh["pins"]["old_producer_revision"] / "run_track1_frontends/code"
    _markers(old_code, fresh["pins"]["old_producer_revision"])
    for name, pin in fresh["old_source_files"].items():
        _verified(old_code / name, pin, maximum=2 * 1024 ** 2, readonly=True)
    if source_callback and verify_sources() is not True:
        raise ValueError("Caller actual source binding must verify after archive before PASS receipt")
    receipt = dict(schema="world_reward.failed_masks_transition_receipt.v1", stage="failed_masks_transition",
                   status="pass", original_status="fail", original_failure_reinterpreted=False,
                   episode_index=fresh["pins"]["episode_index"], original_producer_revision=fresh["pins"]["old_producer_revision"],
                   producer_revision=fresh["current_revision"], source_helpers=fresh["current_source_files"],
                   transition_config_identity=fresh["current_binding"]["config_identity"],
                   dispatch_markers=fresh["current_binding"]["dispatch_markers"],
                   source_files_rechecked_before_thenafter=True,
                   source_rechecked_before_and_after=source_callback,
                   original_source_files=fresh["old_source_files"], input_manifest=fresh["pins"]["input_manifest"],
                   track1_inputs=fresh["track1_inputs"], original_failed_unit=fresh["pins"]["failed_unit"],
                   unit_state=fresh["unit_state"], original_log=dict(path=str(log.relative_to(fresh["root"])), **fresh["original_log"]),
                   before=dict(path=str(fresh["source"].relative_to(fresh["root"]) / "seed-diagnostics.json"), **fresh["pins"]["diagnostic"]),
                   after=dict(path=str((destination / "seed-diagnostics.json").relative_to(fresh["root"])), **after),
                   atomic_rename_noreplace=True, files_deleted=False, media_copied=False, inference_performed=False,
                   input_track="track_1", ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
                   completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    raw = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(fresh["receipt"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(fresh["receipt"].parent)
    return receipt


def bound_source(root, code, revision, executing):
    root, code = canonical(root), canonical(code)
    if (root != ROOT or re.fullmatch("[0-9a-f]{40}", revision) is None or
            code != root / "jobs" / revision / "run_failed_masks_transition/code" or
            canonical(executing) != code / CURRENT_SOURCES[0]):
        raise ValueError("Actual immutable Azure transition dispatch source required")
    _markers(code, revision)
    source_ids = {name: identity(code / name, 2 * 1024 ** 2) for name in CURRENT_SOURCES}
    for name in (*CURRENT_SOURCES, CONFIG):
        if canonical(code / name).stat().st_mode & 0o222:
            raise ValueError("Current transition sources and pinned config must be readonly")
    config_id = identity(code / CONFIG, 32 * 1024)
    pins = _verified_json(code / CONFIG, config_id, maximum=32 * 1024)
    return source_ids, config_id, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args()
    if platform.system() != "Linux":
        raise ValueError("Azure Linux-only archival driver")
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"])
    revision = os.environ["WR_CODE_REVISION"]
    source_ids, config_id, pins = bound_source(root, code, revision, Path(__file__))
    plan = validate(root, pins, revision, current_source_files=source_ids)
    def verify_sources():
        if bound_source(root, code, revision, Path(__file__)) != (source_ids, config_id, pins):
            raise ValueError("Actual main source/config/dispatch changed; no PASS receipt")
        return True
    receipt = transition(plan, verify_sources=verify_sources)
    print(json.dumps(dict(stage=receipt["stage"], status=receipt["status"], original_status="fail",
                          episode_index=receipt["episode_index"], producer_revision=revision,
                          archive=receipt["after"]["path"], receipt=identity(plan["receipt"], 32 * 1024))))


if __name__ == "__main__":
    main()
