"""Independent thirteen-file TUM RGB-only pins; never read private depth.

Original rank identities are fixed before acquisition; timestamp names come
from committed file pins, not first-seen manifest. Hash every public file before
JSON and again afterward. Genuine old stdlib hash/parser primitives unchanged.
"""
import copy
import hashlib
import re
from decimal import Decimal

from tudl_holdout_inputs import identity, strict_json, _canonical

BASE = "validation/tum_rgbd_depth_holdout_v1"
SCHEMA = "world_reward.tum_rgbd_depth_public.v1"
PINS_SCHEMA = "world_reward.tum_rgbd_depth_input_pins.v1"
DATASET, LICENSE = "TUM RGB-D", "CC-BY-4.0"
SELECTION = "sorted_RGB_indices_40_80_120_160_per_sequence_before_depth_values"
WIDTH, HEIGHT = 640, 480
ORDERED_FRAMES = tuple((scene, frame) for scene in (1, 2, 3) for frame in (40, 80, 120, 160))
NAME_PATTERN = r"scene_(00000[123])_rgb_([0-9]{10}\.[0-9]{6})\.png"


def ordered_frames(pins):
    pairs = pins.get("ordered_frames") if type(pins) is dict else None
    if (type(pairs) is not list or len(pairs) != 12 or any(type(pair) is not list or len(pair) != 2
            or any(type(value) is not int for value in pair) or tuple(pair) != expected for pair, expected in zip(pairs, ORDERED_FRAMES))):
        raise ValueError("Exact independently committed sequence/rank pairs 1..3 by40/80/120/160 required")
    return ORDERED_FRAMES


def filenames(pins):
    ordered_frames(pins); files = pins.get("public_files")
    if type(files) is not dict or len(files) != 13 or "manifest.json" not in files:
        raise ValueError("Exactly twelve independently pinned TUM timestamps and one manifest required")
    groups = {scene: [] for scene in (1, 2, 3)}
    for name in files:
        if name == "manifest.json": continue
        match = re.fullmatch(NAME_PATTERN, name) if type(name) is str else None
        if match is None: raise ValueError("Original timestamp-only TUM public PNG names required; no private paths")
        scene, timestamp = int(match[1]), match[2]
        if Decimal(timestamp) <= 0: raise ValueError("Original positive TUM timestamp required")
        groups[scene].append(name)
    if any(len(group) != 4 for group in groups.values()): raise ValueError("Exactly four ordered RGB timestamps per sequence required")
    return tuple(name for scene in (1, 2, 3) for name in sorted(groups[scene]))


def _record(value):
    if (type(value) is not dict or set(value) != {"sha256", "bytes"}
            or type(value["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or type(value["bytes"]) is not int or value["bytes"] <= 0):
        raise ValueError("Exact positive byte/SHA256 identity required")


def validate_pins(pins):
    if (type(pins) is not dict or set(pins) != {"schema", "acquisition_report", "public_files", "ordered_frames"}
            or type(pins["schema"]) is not str or pins["schema"] != PINS_SCHEMA):
        raise ValueError("Independent TUM RGB-only input pins required")
    receipt = pins["acquisition_report"]
    if type(receipt) is not dict or set(receipt) != {"sha256", "bytes", "producer_revision", "script_sha256"}:
        raise ValueError("Exact actual acquisition receipt metadata required; no private path")
    _record({key: receipt[key] for key in ("sha256", "bytes")})
    for key, size in (("producer_revision", 40), ("script_sha256", 64)):
        if type(receipt[key]) is not str or re.fullmatch(r"[0-9a-f]{%d}" % size, receipt[key]) is None:
            raise ValueError("Actual acquisition producer/source SHA required")
    filenames(pins)
    for record in pins["public_files"].values(): _record(record)


def public_inputs(directory, pins):
    validate_pins(pins); expected = copy.deepcopy(pins); directory = _canonical(directory)
    if directory.parts[-3:] != ("validation", "tum_rgbd_depth_holdout_v1", "inputs") or not directory.is_dir():
        raise ValueError("Exact new TUM public namespace required")
    names = {"manifest.json", *filenames(pins)}
    if {path.name for path in directory.iterdir()} != names: raise ValueError("Only thirteen independently pinned public files may be exposed")
    before = {name: identity(directory / name) for name in sorted(names)}
    if before != expected["public_files"]: raise ValueError("All original public file pins must match BEFORE JSON")
    raw = (directory / "manifest.json").read_bytes()
    if {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} != before["manifest.json"]: raise ValueError("Manifest changed before interpretation")
    manifest = strict_json(raw); fields = dict(schema=SCHEMA, dataset=DATASET, license=LICENSE, selection=SELECTION)
    if (type(manifest) is not dict or set(manifest) != {*fields, "images"}
            or any(type(manifest.get(key)) is not str or manifest[key] != value for key, value in fields.items())
            or type(manifest["images"]) is not list or len(manifest["images"]) != 12):
        raise ValueError("Exact TUM RGB-only manifest required; private fields forbidden")
    records = []
    for (scene, rank), name, row in zip(ordered_frames(pins), filenames(pins), manifest["images"]):
        timestamp = re.fullmatch(NAME_PATTERN, name)[2]
        if (type(row) is not dict or set(row) != {"scene_id", "frame_id", "file", "original_rgb_file", "timestamp", "sha256", "width", "height"}
                or any(type(row.get(key)) is not int for key in ("scene_id", "frame_id", "width", "height"))
                or (row["scene_id"], row["frame_id"], row["width"], row["height"]) != (scene, rank, WIDTH, HEIGHT)
                or any(type(row.get(key)) is not str or row[key] != value for key, value in
                    {"file": name, "timestamp": timestamp, "original_rgb_file": "rgb/" + timestamp + ".png", "sha256": before[name]["sha256"]}.items())):
            raise ValueError("Original RGB timestamp/rank/order/grid/hash mismatch; no private fields")
        records.append({**row, "path": directory / name})
    if ({path.name for path in directory.iterdir()} != names or pins != expected
            or {name: identity(directory / name) for name in sorted(names)} != before):
        raise ValueError("Original public TUM files/pins changed during interpretation")
    return records, dict(before["manifest.json"])
