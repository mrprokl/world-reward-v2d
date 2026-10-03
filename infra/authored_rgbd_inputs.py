"""Independent public pins for twelve new authored RGBs; never open truth.

Manufacture receipt identity is metadata supplied by committed caller pins.
All thirteen public files are hashed before manifest JSON and after reading.
Only the old stdlib identity/strict-JSON primitives are reused, not TUD-L IDs.
"""
import copy
import hashlib
from pathlib import Path
import re

from tudl_holdout_inputs import identity, strict_json, _canonical

BASE = "validation/authored_rgbd_holdout_v1"
SCHEMA = "world_reward.authored_rgbd_public.v1"
PINS_SCHEMA = "world_reward.authored_rgbd_input_pins.v1"
WIDTH, HEIGHT = 640, 480
ORDERED_FRAMES = tuple((scene, frame) for scene in (1, 2, 3) for frame in range(4))


def filenames():
    return tuple(f"scene_{scene:06d}_frame_{frame:06d}.png" for scene, frame in ORDERED_FRAMES)


def _record(value):
    if (type(value) is not dict or set(value) != {"sha256", "bytes"}
            or type(value["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or type(value["bytes"]) is not int or value["bytes"] <= 0):
        raise ValueError("Exact independent positive byte/SHA256 identity required")


def validate_pins(pins):
    if (type(pins) is not dict or set(pins) != {"schema", "manufacture_report", "public_files"}
            or type(pins["schema"]) is not str or pins["schema"] != PINS_SCHEMA):
        raise ValueError("Independently committed authored RGB input pins required")
    receipt = pins["manufacture_report"]
    if type(receipt) is not dict or set(receipt) != {"sha256", "bytes", "producer_revision", "script_sha256"}:
        raise ValueError("Exact manufacture receipt metadata identity required; no private path")
    _record({key: receipt[key] for key in ("sha256", "bytes")})
    for key, size in (("producer_revision", 40), ("script_sha256", 64)):
        if type(receipt[key]) is not str or re.fullmatch(r"[0-9a-f]{%d}" % size, receipt[key]) is None:
            raise ValueError("Actual manufacture producer/source digest required")
    files = pins["public_files"]
    if type(files) is not dict or set(files) != {"manifest.json", *filenames()}:
        raise ValueError("Exactly twelve authored PNGs and one public manifest must be pinned")
    for value in files.values(): _record(value)


def public_inputs(directory, pins):
    validate_pins(pins); expected = copy.deepcopy(pins); directory = _canonical(directory)
    if directory.parts[-3:] != ("validation", "authored_rgbd_holdout_v1", "inputs") or not directory.is_dir():
        raise ValueError("Exact new authored public namespace required")
    names = {"manifest.json", *filenames()}
    if {path.name for path in directory.iterdir()} != names:
        raise ValueError("Public directory must expose exactly thirteen pinned files")
    before = {name: identity(directory / name) for name in sorted(names)}
    if before != expected["public_files"]: raise ValueError("Independent public SHA/bytes differ before JSON")
    raw = (directory / "manifest.json").read_bytes()
    if {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} != before["manifest.json"]:
        raise ValueError("Manifest changed before interpretation")
    manifest = strict_json(raw)
    if (type(manifest) is not dict or set(manifest) != {"schema", "images"}
            or type(manifest["schema"]) is not str or manifest["schema"] != SCHEMA
            or type(manifest["images"]) is not list or len(manifest["images"]) != 12):
        raise ValueError("Exact authored RGB-only manifest required; private fields forbidden")
    records = []
    for (scene, frame), name, row in zip(ORDERED_FRAMES, filenames(), manifest["images"]):
        if (type(row) is not dict or set(row) != {"scene_id", "frame_id", "file", "sha256", "width", "height"}
                or any(type(row.get(key)) is not int for key in ("scene_id", "frame_id", "width", "height"))
                or (row["scene_id"], row["frame_id"], row["width"], row["height"]) != (scene, frame, WIDTH, HEIGHT)
                or type(row["file"]) is not str or row["file"] != name
                or type(row["sha256"]) is not str or row["sha256"] != before[name]["sha256"]):
            raise ValueError("Exact frozen authored frame/order/grid/hash required; no private fields")
        records.append({**row, "path": directory / name})
    if ({path.name for path in directory.iterdir()} != names
            or {name: identity(directory / name) for name in sorted(names)} != before or pins != expected):
        raise ValueError("Authored public files/pins changed during interpretation")
    return records, dict(before["manifest.json"])
