"""Strict independent pins for new whole-support TUD-L RGB frames only.

All thirteen public files are hashed before JSON and afterward. Acquisition
receipt identity is metadata only; its private path is never opened. Ordered
actual frame IDs come from committed caller pins, never first-seen manifest.
"""
import copy
import hashlib
import re

from tudl_holdout_inputs import identity, strict_json, _canonical

BASE = "validation/tudl_whole_support_holdout_v1"
SCHEMA = "world-reward-tudl-whole-support-holdout-rgb-v1"
PINS_SCHEMA = "world_reward.tudl_whole_support_input_pins.v1"
REVISION = "6527f7d4b25d3e2e8dec84529284d9797b15f7b5"
LICENSE = "CC-BY-SA-4.0"
SELECTION = "sorted_RGB_indices_30_70_110_150_per_scene_before_private_values"
WIDTH, HEIGHT = 640, 480
OLD_FRAMES = ((0, 4074, 8227, 1788, 3235, 5138, 6925),
    (3, 4013, 7710, 1566, 3137, 4894, 6490), (4, 4028, 7969, 1512, 3227, 4819, 6647))


def ordered_frames(pins):
    pairs = pins.get("ordered_frames") if type(pins) is dict else None
    if type(pairs) is not list or len(pairs) != 12:
        raise ValueError("Twelve independently committed original frame pairs required")
    result = []
    for index, pair in enumerate(pairs):
        scene = index // 4 + 1
        if (type(pair) is not list or len(pair) != 2 or any(type(value) is not int for value in pair)
                or pair[0] != scene or pair[1] < 0 or pair[1] in OLD_FRAMES[scene - 1]
                or index % 4 and pair[1] <= result[-1][1]):
            raise ValueError("Exact ordered new four-frame groups disjoint from all old scene/frame pairs required")
        result.append(tuple(pair))
    return tuple(result)


def filenames(pins):
    return tuple(f"scene_{scene:06d}_frame_{frame:06d}.png" for scene, frame in ordered_frames(pins))


def _record(value):
    if (type(value) is not dict or set(value) != {"sha256", "bytes"}
            or type(value["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or type(value["bytes"]) is not int or value["bytes"] <= 0):
        raise ValueError("Exact independent positive byte/SHA256 identity required")


def validate_pins(pins):
    if (type(pins) is not dict or set(pins) != {"schema", "acquisition_report", "public_files", "ordered_frames"}
            or type(pins["schema"]) is not str or pins["schema"] != PINS_SCHEMA):
        raise ValueError("Independently committed whole-support RGB input pins required")
    receipt = pins["acquisition_report"]
    if type(receipt) is not dict or set(receipt) != {"sha256", "bytes", "producer_revision", "script_sha256"}:
        raise ValueError("Exact acquisition receipt metadata identity required; no private path")
    _record({key: receipt[key] for key in ("sha256", "bytes")})
    for key, size in (("producer_revision", 40), ("script_sha256", 64)):
        if type(receipt[key]) is not str or re.fullmatch(r"[0-9a-f]{%d}" % size, receipt[key]) is None:
            raise ValueError("Actual acquisition producer/source digest required")
    if type(pins["public_files"]) is not dict or set(pins["public_files"]) != {"manifest.json", *filenames(pins)}:
        raise ValueError("Exactly twelve new PNGs and one public manifest must be pinned")
    for value in pins["public_files"].values(): _record(value)


def public_inputs(directory, pins):
    validate_pins(pins); expected = copy.deepcopy(pins); directory = _canonical(directory)
    if directory.parts[-3:] != ("validation", "tudl_whole_support_holdout_v1", "inputs") or not directory.is_dir():
        raise ValueError("Exact new whole-support public namespace required")
    names = {"manifest.json", *filenames(pins)}
    if {path.name for path in directory.iterdir()} != names:
        raise ValueError("Public directory must expose exactly thirteen pinned files")
    before = {name: identity(directory / name) for name in sorted(names)}
    if before != expected["public_files"]: raise ValueError("Independent public SHA/bytes differ before JSON")
    raw = (directory / "manifest.json").read_bytes()
    if {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} != before["manifest.json"]:
        raise ValueError("Manifest changed before interpretation")
    manifest = strict_json(raw); fields = dict(schema=SCHEMA, revision=REVISION, license=LICENSE, selection=SELECTION)
    if (type(manifest) is not dict or set(manifest) != {*fields, "images"}
            or any(type(manifest.get(key)) is not str or manifest[key] != value for key, value in fields.items())
            or type(manifest["images"]) is not list or len(manifest["images"]) != 12):
        raise ValueError("Exact new RGB-only manifest required; private fields forbidden")
    records = []
    for (scene, frame), name, row in zip(ordered_frames(pins), filenames(pins), manifest["images"]):
        if (type(row) is not dict or set(row) != {"scene_id", "frame_id", "file", "sha256", "width", "height"}
                or any(type(row.get(key)) is not int for key in ("scene_id", "frame_id", "width", "height"))
                or (row["scene_id"], row["frame_id"], row["width"], row["height"]) != (scene, frame, WIDTH, HEIGHT)
                or type(row["file"]) is not str or row["file"] != name
                or type(row["sha256"]) is not str or row["sha256"] != before[name]["sha256"]):
            raise ValueError("Exact frozen new frame/order/grid/hash required; no private fields")
        records.append({**row, "path": directory / name})
    if ({path.name for path in directory.iterdir()} != names
            or {name: identity(directory / name) for name in sorted(names)} != before or pins != expected):
        raise ValueError("New public files/pins changed during interpretation")
    return records, dict(before["manifest.json"])
