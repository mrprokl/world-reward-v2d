"""Fail-closed, metadata-only admission for an external FORM-HOI insight test.

This module never acquires archives or reads annotations. Sequence families
derived from filenames are only provisional strata, not verified object IDs.
Reference masks/geometry belong to a separate evaluator, never inference.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping


FRAMES = (0, 14, 29)
_SEQUENCE = re.compile(
    r"^(?P<recording>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_"
    r"(?P<body>[a-z0-9]+(?:_[a-z0-9]+)*)_(?P<take>\d+)$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
INFERENCE_KEYS = frozenset(
    {"sequence_id", "rgb_video", "rgb_sha256", "object_prompt", "action", "camera",
     "original_frame_indices"}
)
REQUIRED_GATES = (
    "rgb_content_duplicate_check_passed",
    "native_prompt_and_action_provenance_verified",
    "single_rgb_camera_verified",
    "official_object_group_assignment_verified",
    "reference_artifacts_isolated_from_inference",
    "original_frame_grid_verified",
    "cohort_frozen_before_inference",
)


def verify_metadata_bytes(data: bytes, *, expected_size: int, sha256: str) -> None:
    if not isinstance(data, bytes) or type(expected_size) is not int:
        raise ValueError("Metadata byte/size contract invalid")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ValueError("Invalid metadata SHA256")
    if len(data) != expected_size or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Metadata provenance mismatch")


def sequence_parts(sequence_id: str) -> tuple[str, str]:
    if not isinstance(sequence_id, str):
        raise ValueError("Sequence ID must be a string")
    match = _SEQUENCE.fullmatch(sequence_id)
    if match is None:
        raise ValueError("Unsupported or unsafe sequence ID")
    return match["recording"], match["body"]


def provisional_family(sequence_id: str) -> str:
    """Conservative first filename token, not an authoritative object label.

    Keeping all ``vacuum_*`` or ``toy_*`` together avoids splitting one object
    merely because an action word follows its name. Color prefixes may group
    multiple objects; official group verification remains mandatory.
    """
    _, body = sequence_parts(sequence_id)
    return body.split("_")[0]


def _object_slug(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Challenge target object missing")
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError("Challenge target object invalid")
    return slug


def _challenge_index(records: Iterable[Mapping[str, Any]]) -> tuple[set, set, set]:
    ids, recordings, objects = set(), set(), set()
    for record in records:
        sequence_id = record.get("sequence_id")
        recording, _ = sequence_parts(sequence_id)
        if sequence_id in ids:
            raise ValueError("Duplicate challenge sequence ID")
        ids.add(sequence_id)
        recordings.add(recording)
        objects.add(_object_slug(record.get("object")))
    if not ids:
        raise ValueError("Empty challenge exclusion index")
    return ids, recordings, objects


def _archive(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = {"sequence_id", "archive_path", "archive_size", "archive_sha256", "member_count"}
    if set(row) != keys:
        raise ValueError("Unexpected archive metadata schema")
    sequence_parts(row["sequence_id"])
    if row["archive_path"] != f"data/{row['sequence_id']}.tar":
        raise ValueError("Archive path does not match its sequence")
    if not isinstance(row["archive_sha256"], str) or not _SHA256.fullmatch(row["archive_sha256"]):
        raise ValueError("Archive SHA256 missing or invalid")
    for key in ("archive_size", "member_count"):
        if type(row[key]) is not int or row[key] <= 0:
            raise ValueError(f"Invalid archive metadata: {key}")
    return dict(row)


def audit_archives(
    rows: Iterable[Mapping[str, Any]], challenge_records: Iterable[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reject exact IDs, same-recording aliases and target-object slug aliases."""
    forbidden, recordings, objects = _challenge_index(challenge_records)
    admitted, seen_ids, seen_hashes = [], set(), set()
    counts = {"exact_sequence_overlap": 0, "recording_timestamp_overlap": 0,
              "challenge_object_slug_prefix_overlap": 0}
    unsupported = 0
    total = 0
    for raw in rows:
        total += 1
        # The manifest has a few legacy names; fail closed rather than infer IDs.
        try:
            row = _archive(raw)
        except ValueError as exc:
            if str(exc) == "Unsupported or unsafe sequence ID":
                unsupported += 1
                continue
            raise
        sid = row["sequence_id"]
        if sid in seen_ids or row["archive_sha256"] in seen_hashes:
            raise ValueError("Duplicate FORM-HOI sequence or archive digest")
        seen_ids.add(sid)
        seen_hashes.add(row["archive_sha256"])
        recording, body = sequence_parts(sid)
        exact = sid in forbidden
        same_recording = recording in recordings
        object_alias = any(body == slug or body.startswith(slug + "_") for slug in objects)
        counts["exact_sequence_overlap"] += exact
        counts["recording_timestamp_overlap"] += same_recording
        counts["challenge_object_slug_prefix_overlap"] += object_alias
        if not (exact or same_recording or object_alias):
            admitted.append(row)
    if not total:
        raise ValueError("Empty external archive manifest")
    return admitted, {
        "archive_rows": total,
        "challenge_unique_sequences": len(forbidden),
        **counts,
        "unsupported_sequence_names_excluded": unsupported,
        "admitted_archive_rows": len(admitted),
        "metadata_admission_passed": bool(admitted) and all(value == 0 for value in counts.values()),
        "content_duplicate_check_passed": False,
        "training_overlap_verified": False,
        "quality_verified": False,
    }


def freeze_cohort(
    admitted: Iterable[Mapping[str, Any]], *, seed: str, count: int = 8
) -> list[dict[str, Any]]:
    """One SHA-ranked sequence per provisional family, balanced DEV/RESERVED.

    Input ordering is irrelevant. Reserved filenames are frozen but RGB and
    references stay unopened. Actual object/person grouping remains a gate.
    """
    if not isinstance(seed, str) or not seed or type(count) is not int or count < 2 or count % 2:
        raise ValueError("Cohort requires a seed and an even count >= 2")
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids = set()
    for raw in admitted:
        row = _archive(raw)
        if row["sequence_id"] in ids:
            raise ValueError("Duplicate admitted sequence")
        ids.add(row["sequence_id"])
        families[provisional_family(row["sequence_id"])].append(row)
    # Exclude legacy singleton naming strata, not bad reconstruction outputs.
    groups = [group for group, members in families.items() if len(members) >= 3]
    rank = lambda namespace, value: hashlib.sha256(f"{seed}/{namespace}/{value}".encode()).hexdigest()
    groups.sort(key=lambda group: rank("family", group))
    if len(groups) < count:
        raise ValueError("Insufficient diverse admitted sequence families")
    cohort = []
    for index, group in enumerate(groups[:count]):
        chosen = min(families[group], key=lambda row: rank("sequence", row["sequence_id"]))
        cohort.append({
            **chosen,
            "provisional_sequence_family": group,
            "family_metadata_verified_object_id": False,
            "split": "development" if index < count // 2 else "reserved_unopened",
            "camera_policy": "front_stereo_camera_left_or_fail_no_substitution",
            "original_frame_indices": list(FRAMES),
        })
    return cohort


def require_ready_for_inference(admission: Mapping[str, Any], gates: Mapping[str, Any]) -> None:
    if admission.get("metadata_admission_passed") is not True:
        raise ValueError("External metadata admission did not pass")
    if set(gates) != set(REQUIRED_GATES) or any(gates.get(key) is not True for key in REQUIRED_GATES):
        raise ValueError("External inference blocked: qualification gate pending")


def validate_inference_package(package: Mapping[str, Any]) -> None:
    """Allow exactly one original RGB video, native text and frame IDs, no GT."""
    if set(package) != INFERENCE_KEYS:
        raise ValueError("Inference package contains missing or forbidden fields")
    sequence_parts(package["sequence_id"])
    frames = package["original_frame_indices"]
    if not isinstance(frames, list) or any(type(frame) is not int for frame in frames) or frames != list(FRAMES):
        raise ValueError("Initial-frame grid changed")
    if package["camera"] != "front_stereo_camera_left":
        raise ValueError("Unsupported camera or implicit substitution")
    for key in ("object_prompt", "action"):
        if not isinstance(package[key], str) or not package[key].strip():
            raise ValueError("Native prompt/action provenance required")
    if not isinstance(package["rgb_sha256"], str) or not _SHA256.fullmatch(package["rgb_sha256"]):
        raise ValueError("RGB provenance missing")
    if not isinstance(package["rgb_video"], str):
        raise ValueError("RGB input path required")
    path = PurePosixPath(package["rgb_video"])
    if not path.is_absolute() or ".." in path.parts or path.suffix != ".mp4":
        raise ValueError("Unsafe or non-RGB input path")
    forbidden = {"references", "reference", "ground_truth", "gt", "depth", "masks", "calibration"}
    if forbidden.intersection(part.lower() for part in path.parts):
        raise ValueError("Reference artifact routed into inference")
