"""Azure stdlib-only Track 1 input readiness, not reconstruction or quality.

Only thirty original RGB files and two public metadata files are opened. Other
legitimate manifest entries are name-checked, never read. ffprobe reads container
metadata without decoding/counting frames. Existing frontend outputs are opaque.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import subprocess
import time

ROOT = Path("/srv/scenesmith/world-reward")
REPOSITORY = "nvidia/video_to_data_challenge"
DATASET_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
STAGE = "world_reward_track1_all_original_inputs_structural_readiness"
EPISODES, FRAMES, WIDTH, HEIGHT, FPS, BUDGET_SECONDS = 30, 16563, 1536, 1152, 30, 120
METADATA = ("track_1/meta/episodes.jsonl", "track_1/meta/episodes_metadata.jsonl")
FRONTENDS = ("automatic_masks", "body_smoke", "depth_smoke", "scale_smoke", "object_grounded",
             "body_full", "depth_full", "body_full/cari_adapter", "object_pose_full", "cari_inputs")
GIB = 1024 ** 3
ALL_WORKING_BUDGET_BYTES = 250 * GIB
REFERENCE_EPISODE_FRAMES = 501
REFERENCE_WORKING_BYTES = (94 * GIB + 9) // 10  # 1.1 + 8.3 GiB planning assumption.
MAX_JSON_BYTES = 16 * 1024 ** 2


def video_relative(episode):
    if type(episode) is not int or not 0 <= episode < EPISODES:
        raise ValueError("Exact bounded integer Track 1 episode required")
    return f"track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4"


SELECTED_PATHS = (*METADATA, *(video_relative(i) for i in range(EPISODES)))


def allowed_manifest_path(value):
    """Pinned acquisition allowlist; no imports of data/model stage drivers."""
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(p in {".", ".."} for p in path.parts):
        return False
    return bool(value in {"track_1/README.md", "track_1/.gitignore"}
        or re.fullmatch(r"track_1/meta/(info\.json|(?:episodes|episodes_metadata|episodes_stats|tasks)\.jsonl)", value)
        or re.fullmatch(r"track_1/data/chunk-000/episode_\d{6}\.parquet", value)
        or re.fullmatch(r"track_1/videos/chunk-000/observation\.images\.exo_camera/episode_\d{6}\.mp4", value))


def strict_json(text):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON keys forbidden")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("Nonfinite JSON constants forbidden")

    def finite(value):
        if type(value) is dict:
            for child in value.values():
                finite(child)
        elif type(value) is list:
            for child in value:
                finite(child)
        elif type(value) is float and not math.isfinite(value):
            raise ValueError("Nonfinite JSON numbers forbidden")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    finite(result)
    return result


def canonical_path(path):
    path = Path(path)
    if not path.is_absolute() or str(path) != os.path.abspath(path):
        raise ValueError("Canonical absolute path required")
    for item in (*reversed(path.parents), path):
        mode = item.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("No symlink file or ancestor allowed")
        if item != path and not stat.S_ISDIR(mode):
            raise ValueError("Regular directory ancestors required")
    return path


def _snapshot(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def file_identity(path, *, capture=False, max_bytes=None, immutable=False):
    path = canonical_path(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_size <= 0
            or (immutable and before.st_mode & 0o222)):
        raise ValueError("Nonempty regular file required; runtime sources immutable")
    if max_bytes is not None and before.st_size > max_bytes:
        raise ValueError("Public JSON exceeds bounded metadata size")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    digest, chunks = hashlib.sha256(), []
    with os.fdopen(descriptor, "rb") as stream:
        if _snapshot(os.fstat(stream.fileno())) != _snapshot(before):
            raise ValueError("Input changed before opening")
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
            if capture:
                chunks.append(block)
        if _snapshot(os.fstat(stream.fileno())) != _snapshot(before):
            raise ValueError("Input changed during hashing")
    if _snapshot(canonical_path(path).lstat()) != _snapshot(before):
        raise ValueError("Input path changed during hashing")
    identity = {"sha256": digest.hexdigest(), "bytes": before.st_size}
    return (identity, b"".join(chunks)) if capture else identity


def validate_manifest(manifest):
    if type(manifest) is not dict or (manifest.get("track"), manifest.get("repo_id"), manifest.get("revision")) != (
            "track_1", REPOSITORY, DATASET_REVISION):
        raise ValueError("Pinned official Track 1 manifest required")
    for key, expected in (("episodes", EPISODES), ("frames", FRAMES)):
        if key in manifest and (type(manifest[key]) is not int or manifest[key] != expected):
            raise ValueError("Manifest episode/frame totals differ from pinned source")
    if type(manifest.get("files")) is not list:
        raise ValueError("Manifest file inventory required")
    records = {}
    for row in manifest["files"]:
        if type(row) is not dict or not allowed_manifest_path(row.get("path")):
            raise ValueError("Only canonical acquisition-allowlisted Track 1 paths allowed")
        name = row["path"]
        if name in records:
            raise ValueError("Duplicate manifest paths forbidden")
        if (type(row.get("bytes")) is not int or row["bytes"] <= 0
                or type(row.get("sha256")) is not str or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise ValueError("Typed positive file bytes and SHA256 required")
        records[name] = {"sha256": row["sha256"], "bytes": row["bytes"]}
    if not set(SELECTED_PATHS) <= records.keys():
        raise ValueError("Exactly one of each selected two metadata and thirty videos required")
    videos = {name for name in records if name.startswith("track_1/videos/")}
    if videos != set(SELECTED_PATHS[2:]):
        raise ValueError("Official thirty-video inventory required")
    return records


def parse_episode_metadata(episodes_text, prompts_text):
    def rows(text, label):
        result = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            row = strict_json(line)
            if type(row) is not dict or type(row.get("episode_index")) is not int or not 0 <= row["episode_index"] < EPISODES:
                raise ValueError(f"{label}: bounded integer episode identities required")
            if row["episode_index"] in result:
                raise ValueError(f"{label}: duplicate episode identities forbidden")
            result[row["episode_index"]] = row
        if set(result) != set(range(EPISODES)):
            raise ValueError(f"{label}: all thirty distinct episodes required")
        return result

    episodes, prompts = rows(episodes_text, "length metadata"), rows(prompts_text, "object metadata")
    lengths = []
    for index in range(EPISODES):
        length, prompt = episodes[index].get("length"), prompts[index].get("object_prompt")
        if type(length) is not int or length < 96:
            raise ValueError("Native original clip requires integer >=96 frames; no padding")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("Official nonempty object prompt required")
        lengths.append(length)
    if sum(lengths) != FRAMES:
        raise ValueError("Pinned all-original-frame total must equal 16563")
    return lengths


def probe_video(path, timeout):
    executable = shutil.which("ffprobe")
    if executable is None:
        raise ValueError("ffprobe unavailable; no decoder or metadata fallback")
    result = subprocess.run([executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if len(result.stdout) > 65536:
        raise ValueError("ffprobe metadata output exceeds limit")
    return strict_json(result.stdout.decode("utf-8"))


def validate_video_metadata(value, length):
    # Azure ffprobe emits an empty programs envelope for this same explicit
    # selected-stream query. Permit that exact non-stream envelope only;
    # unknown fields/nonempty programs remain rejected, without a fallback.
    if (type(value) is not dict or set(value) not in ({"streams"}, {"streams", "programs"})
            or ("programs" in value and (type(value["programs"]) is not list or value["programs"]))
            or type(value["streams"]) is not list or len(value["streams"]) != 1):
        raise ValueError("Exactly one selected RGB video stream metadata required")
    stream = value["streams"][0]
    if type(stream) is not dict or set(stream) != {"width", "height", "nb_frames", "r_frame_rate"}:
        raise ValueError("Exact ffprobe container metadata fields required")
    if (type(stream["width"]) is not int or type(stream["height"]) is not int
            or (stream["width"], stream["height"]) != (WIDTH, HEIGHT)):
        raise ValueError("Original integer 1536x1152 RGB dimensions required")
    if (type(stream["nb_frames"]) is not str or not re.fullmatch(r"[1-9][0-9]*", stream["nb_frames"])
            or int(stream["nb_frames"]) != length):
        raise ValueError("Exact original nb_frames required; no counting or fallback")
    rate = stream["r_frame_rate"]
    if type(rate) is not str or not re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", rate):
        raise ValueError("Positive rational container frame rate required")
    numerator, denominator = map(int, rate.split("/"))
    if numerator != FPS * denominator:
        raise ValueError("Original frame rate must equal exactly 30 Hz")
    return {"width": WIDTH, "height": HEIGHT, "nb_frames": length, "fps": FPS}


def frontend_state(path):
    """Metadata-only opaque inventory: no predictions/receipts are opened."""
    path = Path(path)
    try:
        for item in (*reversed(path.parents), path):
            mode = item.lstat().st_mode
            if stat.S_ISLNK(mode) or (item != path and not stat.S_ISDIR(mode)):
                return {"state": "unverified", "kind": "unsafe_path", "trusted": False}
        if not stat.S_ISDIR(mode):
            return {"state": "unverified", "kind": "not_directory", "trusted": False}
        with os.scandir(path) as entries:
            empty = next(entries, None) is None
        return {"state": "occupied", "kind": "directory", "empty": empty, "trusted": False}
    except FileNotFoundError:
        return {"state": "absent", "trusted": False}
    except OSError:
        return {"state": "unverified", "kind": "inaccessible", "trusted": False}


def resource_plan(clips, free_bytes):
    if type(free_bytes) is not int or free_bytes < 0:
        raise ValueError("Nonnegative integer disk-free bytes required")
    candidates = [clip for clip in clips if all(row["state"] == "absent"
        for row in clip["frontend_outputs"].values())]
    next_clip = candidates[0] if candidates else None
    estimate = ((REFERENCE_WORKING_BYTES * next_clip["length"] + REFERENCE_EPISODE_FRAMES - 1)
        // REFERENCE_EPISODE_FRAMES) if next_clip else None
    threshold = max(20 * GIB, 2 * estimate) if estimate is not None else None
    return {"disk_free_bytes": free_bytes, "all_30_working_budget_bytes": ALL_WORKING_BUDGET_BYTES,
        "all_30_budget_available": free_bytes >= ALL_WORKING_BUDGET_BYTES,
        "next_clean_frontend_episode_index": next_clip["episode_index"] if next_clip else None,
        "next_clip_estimated_working_bytes": estimate, "next_clip_required_free_bytes": threshold,
        "next_clip_resource_ready": threshold is not None and free_bytes >= threshold,
        "estimate_reference_frames": REFERENCE_EPISODE_FRAMES,
        "estimate_reference_working_bytes": REFERENCE_WORKING_BYTES,
        "estimate_definition": "planning_only_1.1GiB_inputs_plus_8.3GiB_depth_scaled_by_original_length",
        "measured_disk_requirement_verified": False, "occupied_outputs_validated": False,
        "execution_authorized": False, "model_and_runtime_readiness_verified": False}


def audit(root, *, probe=probe_video, disk_usage=shutil.disk_usage, budget_seconds=BUDGET_SECONDS):
    started = time.monotonic()

    def remaining():
        left = budget_seconds - (time.monotonic() - started)
        if left <= 0:
            raise TimeoutError("Readiness whole-stage deadline exceeded")
        return left

    root = canonical_path(root)
    if not root.is_dir():
        raise ValueError("Regular isolated Azure root required")
    manifest_path = root / "results/input-manifest.json"
    manifest_id, payload = file_identity(manifest_path, capture=True, max_bytes=MAX_JSON_BYTES)
    records = validate_manifest(strict_json(payload.decode("utf-8")))
    actual, snapshots = {}, {}
    for relative in SELECTED_PATHS:
        remaining()
        path = canonical_path(root / "data" / relative)
        snapshots[relative] = _snapshot(path.lstat())
        actual[relative] = file_identity(path)
        if actual[relative] != records[relative] or _snapshot(path.lstat()) != snapshots[relative]:
            raise ValueError("Selected official input hash/bytes mismatch")
    metadata = []
    for relative in METADATA:
        identity, payload = file_identity(root / "data" / relative, capture=True, max_bytes=MAX_JSON_BYTES)
        if identity != actual[relative]:
            raise ValueError("Metadata changed after all selected input hashes passed")
        metadata.append(payload.decode("utf-8"))
    lengths = parse_episode_metadata(*metadata)
    clips = []
    for index, length in enumerate(lengths):
        relative = video_relative(index)
        path = canonical_path(root / "data" / relative)
        before = _snapshot(path.lstat())
        if before != snapshots[relative]:
            raise ValueError("Original video changed after selected input integrity gate")
        container = validate_video_metadata(probe(path, remaining()), length)
        if _snapshot(canonical_path(path).lstat()) != before:
            raise ValueError("Original video changed during metadata probe")
        clips.append({"episode_index": index, "length": length, "video_relative_path": relative,
            "video": actual[relative], **container,
            "frontend_outputs": {name: frontend_state(root / f"outputs/episode_{index:06d}" / name)
                for name in FRONTENDS}})
    if file_identity(manifest_path) != manifest_id:
        raise ValueError("Input manifest changed during audit")
    remaining()
    resources = resource_plan(clips, disk_usage(root).free)
    return {"stage": STAGE, "status": "pass", "structural_ready": True,
        "resource_ready": resources["next_clip_resource_ready"], "input_track": "track_1",
        "repo_id": REPOSITORY, "dataset_revision": DATASET_REVISION,
        "input_manifest": manifest_id, "manifest_files": len(records), "selected_files_verified": len(actual),
        "unselected_allowlisted_files_not_read": len(records) - len(actual),
        "metadata_files": {name: actual[name] for name in METADATA},
        "episodes": EPISODES, "frames": FRAMES, "clips": clips, "resources": resources,
        "all_selected_hashes_verified_before_metadata_parse": True, "ffprobe_metadata_calls": EPISODES,
        "video_frames_decoded": 0, "video_count_frames_used": False, "fallback_frame_count_used": False,
        "model_files_read": 0, "inference_calls": 0, "private_truth_read": False,
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
        "other_tracks_used": False, "frontend_outputs_verified": False,
        "challenge_performance_verified": False, "submission_eligible": False,
        "budget_seconds": budget_seconds, "elapsed_seconds": time.monotonic() - started}


def main():
    report = {"stage": STAGE, "status": "fail", "structural_ready": False,
        "resource_ready": False, "budget_seconds": BUDGET_SECONDS,
        "input_track": "track_1", "repo_id": REPOSITORY, "dataset_revision": DATASET_REVISION,
        "model_files_read": 0, "inference_calls": 0, "video_frames_decoded": 0,
        "private_truth_read": False, "ground_truth_used": False, "hand_labeled_test": False,
        "oracle_modes": [], "other_tracks_used": False, "challenge_performance_verified": False,
        "submission_eligible": False}
    started = time.monotonic()

    def expired(_signum, _frame):
        raise TimeoutError("Readiness hard120s deadline exceeded")

    try:
        if platform.system() != "Linux" or os.environ.get("WR_ROOT") != str(ROOT):
            raise ValueError("Only canonical Azure Linux root may be audited")
        code = canonical_path(Path(os.environ["WR_CODE"]))
        revision = os.environ["WR_CODE_REVISION"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Immutable full producer revision required")
        script = code / "infra/track1_readiness.py"
        wrapper = code / "infra/run_track1_readiness.sh"
        if Path(__file__).absolute() != script:
            raise ValueError("Exact canonical source script required")
        source_id, wrapper_id = file_identity(script, immutable=True), file_identity(wrapper, immutable=True)
        report.update(producer_revision=revision, script_sha256=source_id["sha256"],
            wrapper_sha256=wrapper_id["sha256"], runtime="Azure_python_stdlib_only")
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(BUDGET_SECONDS)
        report.update(audit(ROOT))
        if file_identity(script, immutable=True) != source_id or file_identity(wrapper, immutable=True) != wrapper_id:
            raise ValueError("Readiness source changed during audit")
        report.update(producer_revision=revision, script_sha256=source_id["sha256"],
            wrapper_sha256=wrapper_id["sha256"], runtime="Azure_python_stdlib_only")
    except Exception as exc:
        report.update(status="fail", structural_ready=False, resource_ready=False,
            error_type=type(exc).__name__, error="readiness_gate_failed_no_decoder_or_oracle_fallback")
        if isinstance(exc, (ValueError, TimeoutError)):
            report["gate_detail"] = str(exc)
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        print(json.dumps(report, allow_nan=False, separators=(",", ":")), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
